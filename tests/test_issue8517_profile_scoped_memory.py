"""#8517: a streaming turn must bind its memory provider to the session's OWN profile.

The Agent binds the external memory provider to whatever
``hermes_constants.get_hermes_home()`` answers while ``AIAgent.__init__`` runs
(agent_init._memory_provider_init_kwargs). Without a context-local override that
is the process-global ``HERMES_HOME`` env mirror, which other WebUI workers rewrite
for THEIR profile (api/profiles.profile_env_for_background_worker, used by
_save_streaming_checkpoint; the cron profile contexts; a process-wide switch).
A turn for profile alpha that is constructed while such a writer holds the mirror
for beta binds beta's memory bank, under beta's identity, and is then cached.

These tests drive the REAL ``api.streaming._run_agent_streaming`` turn path up to
and including agent construction and SESSION_AGENT_CACHE publication. Only the
Agent class is replaced, by a stand-in that binds memory exactly the way the real
Agent does (ambient ``get_hermes_home()`` at construction, ``_bound_scope`` dict with
``hermes_home`` / ``agent_identity`` / ``home_key``), plus the shared-singleton
provider from the #8517 probe harness (Mnemosyne 0.7.1 shape). The concurrent
writer is the REAL production writer, held open on another thread with events at
a fixed point, so nothing here depends on timing. No network, no real provider.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import queue
import re
import sys
import threading
import types
import uuid
from pathlib import Path

import pytest

import api.config as config_mod
import api.profiles as profiles_api
import api.streaming as streaming

REPO_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "issue8517 test sentinel"


# ── hermes_constants: the real one when it has the override API, else a twin ──

def _fake_hermes_constants(*, with_override: bool = True, delegate=None) -> types.ModuleType:
    """Same contract as hermes_constants (>= v0.18.0 when ``with_override``):
    override -> live ``HERMES_HOME`` env -> default. ``with_override=False`` is an
    Agent older than v0.18.0 (no context-local home at all); other names then
    resolve from ``delegate`` (the installed module) so unrelated Agent imports work."""
    mod = types.ModuleType("hermes_constants")
    if delegate is not None:
        def __getattr__(name):
            if name in _OVERRIDE_API or name.startswith("__"):
                raise AttributeError(name)
            return getattr(delegate, name)

        mod.__getattr__ = __getattr__
    unset = object()
    var = contextvars.ContextVar("issue8517_fake_hermes_home_override", default=unset)

    def get_hermes_home_override():
        value = var.get()
        return str(value) if value is not unset and value else None

    def get_hermes_home():
        override = get_hermes_home_override() if with_override else None
        if override:
            return Path(override)
        raw = os.environ.get("HERMES_HOME", "").strip()
        return Path(raw) if raw else Path.home() / ".hermes"

    mod.get_hermes_home = get_hermes_home
    if with_override:
        def set_hermes_home_override(path):
            return var.set(unset if path is None else str(path))

        def reset_hermes_home_override(token):
            var.reset(token)

        def hermes_home_key(path=None):
            candidate = Path(path) if path is not None else get_hermes_home()
            return os.path.normcase(str(candidate.expanduser().resolve(strict=False)))

        mod.set_hermes_home_override = set_hermes_home_override
        mod.reset_hermes_home_override = reset_hermes_home_override
        mod.get_hermes_home_override = get_hermes_home_override
        mod.hermes_home_key = hermes_home_key
    return mod


_OVERRIDE_API = (
    "set_hermes_home_override",
    "reset_hermes_home_override",
    "get_hermes_home_override",
    "get_hermes_home",
)


def _key(path) -> str:
    hc = sys.modules.get("hermes_constants")
    fn = getattr(hc, "hermes_home_key", None)
    if callable(fn):
        return str(fn(str(path)))
    return os.path.normcase(str(Path(str(path)).expanduser().resolve(strict=False)))


@pytest.fixture
def hc(monkeypatch):
    """hermes_constants with the context-local override API (real if installed)."""
    monkeypatch.setattr(streaming, "_streaming_hermes_home_override_available", None)
    monkeypatch.setattr(profiles_api, "_hermes_home_override_available", None)
    try:
        import hermes_constants as real  # noqa: PLC0415
    except Exception:
        real = None
    if real is not None and all(callable(getattr(real, n, None)) for n in _OVERRIDE_API):
        monkeypatch.setitem(sys.modules, "hermes_constants", real)
        return real
    fake = _fake_hermes_constants()
    monkeypatch.setitem(sys.modules, "hermes_constants", fake)
    return fake


@pytest.fixture
def legacy_hc(monkeypatch):
    """An Agent older than v0.18.0: hermes_constants without any home override."""
    monkeypatch.setattr(streaming, "_streaming_hermes_home_override_available", None)
    monkeypatch.setattr(profiles_api, "_hermes_home_override_available", None)
    try:
        import hermes_constants as real  # noqa: PLC0415
    except Exception:
        real = None
    fake = _fake_hermes_constants(with_override=False, delegate=real)
    monkeypatch.setitem(sys.modules, "hermes_constants", fake)
    return fake


# ── homes ────────────────────────────────────────────────────────────────────

@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "home" / ".hermes"
    out = {
        "default": root,
        "alpha": root / "profiles" / "alpha",
        "beta": root / "profiles" / "beta",
        "ws": tmp_path / "workspace",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    # The turn and the writer both mutate os.environ["HERMES_HOME"]; restore it.
    monkeypatch.setenv("HERMES_HOME", str(root))
    return out


def _profile_identity(home: str) -> str:
    p = Path(home)
    return p.name if p.parent.name == "profiles" else "default"


# ── the stand-in provider (port of probes/sbx_site/sbxmem_glue) ───────────────

class SharedInstanceMemoryProvider:
    """Mnemosyne 0.7.1 shape from the #8517 reports: ``register()`` hands every agent
    the SAME instance; ``initialize()`` trusts a supplied ``hermes_home``, else the
    raw ``HERMES_HOME`` env, else the root frozen at import; named bank per identity."""

    name = "sbxmem"

    def __init__(self, frozen_root):
        self.frozen_root = str(frozen_root)
        self.home = None
        self.db = None
        self.session = None
        self.inits: list[dict] = []

    def initialize(self, session_id, **kwargs):
        supplied = kwargs.get("hermes_home")
        home = supplied or os.environ.get("HERMES_HOME", "") or self.frozen_root
        ident = kwargs.get("agent_identity") or "default"
        self.home = str(home)
        self.session = session_id
        self.db = Path(home) / "sbxmem" / "banks" / ident / "bank.jsonl"
        self.inits.append({"session": session_id, "home": str(home), "identity": ident})

    def remember(self, marker):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"marker": marker, "stamp": self.session}) + "\n")


def bank_rows(home) -> list[dict]:
    rows = []
    for p in sorted(Path(home).glob("sbxmem/banks/*/bank.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


class FakeMemoryManager:
    """agent.memory_manager.MemoryManager as far as WebUI can see it. With
    ``bound_scope`` it carries the Agent's ``_bound_scope`` (agent/memory_caller_scope
    .bound_scope); without it, it is an Agent from before 2026-10-17."""

    def __init__(self, provider, home, identity, *, bound_scope=True):
        self.providers = [provider]
        if bound_scope:
            self._bound_scope = {
                "hermes_home": str(home),
                "agent_identity": identity,
                "home_key": _key(home),
            }


def make_agent_cls(provider, *, resolve_home="ambient", bound_scope=True, hooks=None):
    """Stand-in for run_agent.AIAgent's memory binding.

    ``resolve_home="ambient"`` is the real Agent (``get_hermes_home()``: override,
    then the env mirror). ``"raw_env"`` models a build that reads the raw env mirror
    even under an override (the defence-in-depth case for the post-build check).
    """
    hooks = hooks or {}

    class MemoryBindingAgent:
        built: list = []
        turns: list = []

        def __init__(self, session_id=None, skip_memory=False, **kwargs):
            self.session_id = session_id
            self.skip_memory = skip_memory
            self.kwargs = dict(kwargs)
            self._memory_manager = None
            if hooks.get("before_bind"):
                hooks["before_bind"]()
            try:
                if not skip_memory:
                    hc_mod = sys.modules["hermes_constants"]
                    if resolve_home == "ambient":
                        home = str(hc_mod.get_hermes_home())
                    else:
                        home = os.environ.get("HERMES_HOME", "")
                    identity = _profile_identity(home)
                    provider.initialize(session_id, hermes_home=home, agent_identity=identity)
                    self._memory_manager = FakeMemoryManager(
                        provider, home, identity, bound_scope=bound_scope
                    )
            finally:
                if hooks.get("after_bind"):
                    hooks["after_bind"]()
            MemoryBindingAgent.built.append(self)

        def run_conversation(self, *args, **kwargs):
            with config_mod.SESSION_AGENT_CACHE_LOCK:
                entry = config_mod.SESSION_AGENT_CACHE.get(self.session_id)
            MemoryBindingAgent.turns.append({
                "agent": self,
                "cached_agent": entry[0] if entry else None,
            })
            if self._memory_manager is not None:
                provider.remember(f"{self.session_id}:{kwargs.get('user_message')}")
            raise RuntimeError(SENTINEL)

    return MemoryBindingAgent


def bound_home(agent):
    mgr = getattr(agent, "_memory_manager", None)
    if mgr is None:
        return None
    return (getattr(mgr, "_bound_scope", None) or {}).get("hermes_home") or "<unverifiable>"


# ── the real production env writer, held open at a fixed point ───────────────

class HeldWriter:
    """``api.profiles.profile_env_for_background_worker(<profile>)`` held open on
    another thread: it sets ``os.environ["HERMES_HOME"]`` to that profile's home
    (profiles.py ``os.environ["HERMES_HOME"] = str(profile_home_path)``) and keeps it
    there until ``release()``. Same writer _save_streaming_checkpoint uses."""

    def __init__(self, profile="beta"):
        self.profile = profile
        self.entered = threading.Event()
        self._release = threading.Event()
        self.thread = None
        self.error = None
        self.env_seen = None

    def start(self):
        if self.thread is not None:
            return

        def _body():
            try:
                with profiles_api.profile_env_for_background_worker(
                    self.profile, "issue8517 test writer", scope_skill_modules=False
                ):
                    self.env_seen = os.environ.get("HERMES_HOME")
                    self.entered.set()
                    self._release.wait(15)
            except BaseException as exc:  # pragma: no cover - surfaced below
                self.error = exc
                self.entered.set()

        self.thread = threading.Thread(target=_body, name=f"issue8517-{self.profile}-writer", daemon=True)
        self.thread.start()
        assert self.entered.wait(15), "writer never entered"
        assert self.error is None, self.error

    def release(self):
        self._release.set()
        if self.thread is not None:
            self.thread.join(15)


@pytest.fixture
def writer():
    w = HeldWriter("beta")
    yield w
    w.release()


# ── turn driver (pattern of test_issue5567's streaming test) ──────────────────

@pytest.fixture
def run_turn(monkeypatch, homes):
    session_ids: list[str] = []

    def _run(*, profile, agent_cls, before_construct=None, before_turn=None, msg="hi"):
        session_id = f"issue8517-{profile or 'root'}-{uuid.uuid4().hex[:8]}"
        stream_id = f"{session_id}-stream"
        session_ids.append(session_id)
        workspace = homes["ws"]

        class _Session:
            def __init__(self):
                self.session_id = session_id
                self.workspace = str(workspace)
                self.profile = profile
                self.model = "gpt-4"
                self.model_provider = "hermes"
                self.messages = []
                self.context_messages = []
                self.path = str(workspace / "session.json")
                self.active_stream_id = stream_id
                self.pending_user_message = None
                self.pending_started_at = None
                self.pending_user_source = None
                self.pending_attachments = []

            def save(self, *args, **kwargs):
                return None

        class _FakeMeter:
            def __getattr__(self, name):
                if name == "get_interval":
                    return lambda: 11.0
                if name == "get_stats":
                    return lambda *a, **k: {}
                return lambda *a, **k: None

        def _get_ai_agent():
            if before_construct is not None:
                before_construct()
            return agent_cls

        def _home_for(name):
            return homes[name if name and name != "default" else "default"]

        noop = lambda *a, **k: None  # noqa: E731
        monkeypatch.setattr(streaming, "register_active_run", noop)
        monkeypatch.setattr(streaming, "update_active_run", noop)
        monkeypatch.setattr(streaming, "get_session", lambda sid: _Session())
        monkeypatch.setattr(streaming, "_get_session_agent_lock", lambda sid: contextlib.nullcontext())
        monkeypatch.setattr(streaming, "_prewarm_skill_tool_modules", lambda: None)
        monkeypatch.setattr(streaming, "_install_streaming_cronjob_profile_wrapper", lambda: None)
        monkeypatch.setattr(streaming, "_get_ai_agent", _get_ai_agent)
        monkeypatch.setattr(streaming, "_build_session_db_for_stream", lambda path: None)
        monkeypatch.setattr(streaming, "_materialize_pending_user_turn_before_error", noop)
        monkeypatch.setattr(streaming, "_snapshot_and_append_partial_on_error", noop)
        monkeypatch.setattr(streaming, "append_turn_journal_event_for_stream", noop)
        monkeypatch.setattr(streaming, "meter", lambda: _FakeMeter())
        monkeypatch.setattr(streaming, "RunJournalWriter", lambda *a, **k: None)
        monkeypatch.setattr(
            streaming,
            "resolve_model_provider",
            lambda model_with_provider_context, explicitly_picked=None: (model_with_provider_context, None, None),
        )
        monkeypatch.setattr(streaming, "_runtime_preferred_base_url", lambda rt, provider, configured_base_url: configured_base_url)
        monkeypatch.setattr(
            streaming,
            "_apply_profile_home_context_to_streaming_model",
            lambda model, provider_context, profile_home, has_profile: (model, provider_context, False),
        )
        monkeypatch.setattr(config_mod, "_resolve_cli_toolsets", lambda cfg: [])
        monkeypatch.setattr(config_mod, "get_config_for_profile_home", lambda profile_home: {})
        fake_mcp = types.ModuleType("tools.mcp_tool")
        fake_mcp.discover_mcp_tools = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mcp)
        monkeypatch.setattr(profiles_api, "get_hermes_home_for_profile", _home_for)
        monkeypatch.setattr(profiles_api, "get_profile_runtime_env", lambda home: {})
        monkeypatch.setattr(profiles_api, "filter_runtime_env_for_gateway_parity", lambda env: {})
        monkeypatch.setattr(profiles_api, "_skill_modules_support_profile_home", lambda profile_home: True)
        # The legacy skill-module patch path runs when no override is installed for
        # the turn; it is unrelated to memory binding, keep it inert here.
        monkeypatch.setattr(profiles_api, "snapshot_skill_home_modules", lambda: None)
        monkeypatch.setattr(profiles_api, "patch_skill_home_modules", noop)
        monkeypatch.setattr(profiles_api, "restore_skill_home_modules", noop)

        if before_turn is not None:
            before_turn()
        q = queue.Queue()
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id=session_id,
            msg_text=msg,
            model="gpt-4",
            workspace=str(workspace),
            stream_id=stream_id,
        )
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                break
        return session_id, events

    yield _run
    with config_mod.SESSION_AGENT_CACHE_LOCK:
        for sid in session_ids:
            config_mod.SESSION_AGENT_CACHE.pop(sid, None)


def _cached_agent(session_id):
    with config_mod.SESSION_AGENT_CACHE_LOCK:
        entry = config_mod.SESSION_AGENT_CACHE.get(session_id)
    return entry[0] if entry else None


def _assert_never_bound_to(home, agent_cls, provider, wrong_home):
    for agent in agent_cls.built:
        assert _key(bound_home(agent) or "/nonexistent-8517") != _key(wrong_home), (
            f"agent memory bound another profile's home: {bound_home(agent)}"
        )
    assert not bank_rows(wrong_home), f"wrote into another profile's bank: {bank_rows(wrong_home)}"
    assert all(_key(i["home"]) != _key(wrong_home) for i in provider.inits), provider.inits


def _memory_warnings(caplog, session_id):
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and session_id in r.getMessage() and "memory" in r.getMessage().lower()
    ]


# ── (a) failed / unavailable profile-home override must not bind the env mirror ─

def test_stale_unavailable_override_flag_does_not_bind_concurrent_writers_profile(
    hc, homes, run_turn, writer
):
    """fallback_clobber: WebUI's cached 'override unavailable' flag sends the turn to
    the shared env mirror; a beta writer holds it while alpha's agent is built."""
    streaming._streaming_hermes_home_override_available = False  # real cached-False state
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider, hooks={"before_bind": writer.start, "after_bind": writer.release})

    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls)

    assert writer.env_seen == str(homes["beta"]), "the clobber was not in effect"
    assert agent_cls.turns, "the turn never reached run_conversation"
    _assert_never_bound_to(homes["alpha"], agent_cls, provider, homes["beta"])
    # The Agent HAS the override API, so construction is scoped explicitly and memory
    # stays on, bound to alpha (a stale WebUI flag is not a reason to drop memory).
    turn_agent = agent_cls.turns[0]["agent"]
    assert _key(bound_home(turn_agent)) == _key(homes["alpha"])
    assert [r["stamp"] for r in bank_rows(homes["alpha"])] == [session_id]


def test_override_install_failure_builds_without_memory_and_is_not_cached(
    hc, homes, run_turn, writer, monkeypatch, caplog
):
    """fallback_clobber with set_hermes_home_override raising: fail closed."""
    def _raise(_path):
        raise RuntimeError("override unavailable (issue8517 test)")

    monkeypatch.setattr(hc, "set_hermes_home_override", _raise)
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider, hooks={"before_bind": writer.start, "after_bind": writer.release})
    caplog.set_level(logging.WARNING)

    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls)

    assert agent_cls.turns, "the turn never reached run_conversation"
    _assert_never_bound_to(homes["alpha"], agent_cls, provider, homes["beta"])
    turn = agent_cls.turns[0]
    assert turn["agent"]._memory_manager is None, "memory must be off when the profile scope cannot be set"
    assert turn["agent"].skip_memory is True
    assert turn["cached_agent"] is None, "a memory-disabled fail-closed agent must not be cached"
    assert _cached_agent(session_id) is None
    assert _memory_warnings(caplog, session_id), "fail-closed must log a visible WARNING"


# ── (b) post-construction bound-home check ───────────────────────────────────

def test_post_construction_mismatch_drops_provider_and_skips_cache(
    hc, homes, run_turn, writer, caplog
):
    """Even with the override installed, an agent whose memory manager reports a
    bound home other than the session's profile home must not keep the provider
    and must not be cached."""
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(
        provider,
        resolve_home="raw_env",
        hooks={"before_bind": writer.start, "after_bind": writer.release},
    )
    caplog.set_level(logging.WARNING)

    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls)

    assert agent_cls.turns, "the turn never reached run_conversation"
    turn = agent_cls.turns[0]
    assert bound_home(turn["agent"]) in (None,) or _key(bound_home(turn["agent"])) == _key(homes["alpha"]), (
        f"turn ran with memory bound to {bound_home(turn['agent'])}"
    )
    assert turn["cached_agent"] is None
    assert _cached_agent(session_id) is None
    assert not bank_rows(homes["beta"]), "the turn wrote into beta's bank"
    assert _memory_warnings(caplog, session_id), "a bound-home mismatch must log a visible WARNING"


# ── (c) the ImportError path must not take the home from the environment ─────

def test_profile_import_failure_does_not_take_home_from_env_mirror(
    hc, homes, run_turn, writer, monkeypatch
):
    """``from api.profiles import (...)`` failing used to set the turn's profile home
    from ``os.environ['HERMES_HOME']`` (another profile's, when a writer holds it)."""
    monkeypatch.delattr(profiles_api, "_SKILL_HOME_MODULE_PATCH_LOCK")
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider, hooks={"after_bind": writer.release})

    # beta holds the env mirror from BEFORE the turn resolves its profile home
    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls, before_turn=writer.start)

    assert writer.env_seen == str(homes["beta"]), "the clobber was not in effect"
    assert agent_cls.turns, "the turn never reached run_conversation"
    _assert_never_bound_to(homes["alpha"], agent_cls, provider, homes["beta"])
    home = bound_home(agent_cls.turns[0]["agent"])
    assert home is None or _key(home) == _key(homes["alpha"]), home


# ── single-profile and old-Agent behaviour is unchanged ──────────────────────

def test_single_profile_root_session_keeps_memory_and_cache(hc, homes, run_turn):
    """Root profile only, no switching, no writers: memory bound to root, cached."""
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider)

    session_id, _events = run_turn(profile=None, agent_cls=agent_cls)

    turn = agent_cls.turns[0]
    assert _key(bound_home(turn["agent"])) == _key(homes["default"])
    assert turn["agent"].skip_memory is False
    assert turn["cached_agent"] is turn["agent"]
    assert [r["stamp"] for r in bank_rows(homes["default"])] == [session_id]
    assert len(agent_cls.built) == 1, "the single-profile path must construct exactly once"


def test_named_profile_without_writers_keeps_memory_and_cache(hc, homes, run_turn):
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider)

    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls)

    turn = agent_cls.turns[0]
    assert _key(bound_home(turn["agent"])) == _key(homes["alpha"])
    assert turn["cached_agent"] is turn["agent"]
    assert len(agent_cls.built) == 1


def test_agent_without_bound_scope_keeps_memory_bound_by_the_override(
    hc, homes, run_turn, writer
):
    """Agents from before 2026-10-17 have no ``_bound_scope``: the binding cannot be
    read back, so the explicit override is what keeps it on alpha; memory stays on."""
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(
        provider,
        bound_scope=False,
        hooks={"before_bind": writer.start, "after_bind": writer.release},
    )

    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls)

    turn = agent_cls.turns[0]
    assert turn["agent"]._memory_manager is not None
    assert _key(provider.home) == _key(homes["alpha"])
    assert turn["cached_agent"] is turn["agent"]
    assert not bank_rows(homes["beta"])


def test_pre_override_agent_single_profile_keeps_memory(legacy_hc, homes, run_turn):
    """Agent < v0.18.0 (no context-local home): a root-only install keeps memory."""
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider, bound_scope=False)

    session_id, _events = run_turn(profile=None, agent_cls=agent_cls)

    turn = agent_cls.turns[0]
    assert turn["agent"]._memory_manager is not None
    assert _key(provider.home) == _key(homes["default"])
    assert turn["cached_agent"] is turn["agent"]


def test_pre_override_agent_with_env_mirror_on_another_profile_fails_closed(
    legacy_hc, homes, run_turn, writer
):
    """Agent < v0.18.0: memory can only follow the env mirror. When that names another
    profile at construction, memory must be off for the turn, not bound to beta."""
    provider = SharedInstanceMemoryProvider(homes["default"])
    agent_cls = make_agent_cls(provider, bound_scope=False, hooks={"after_bind": writer.release})

    session_id, _events = run_turn(profile="alpha", agent_cls=agent_cls, before_construct=writer.start)

    assert agent_cls.turns, "the turn never reached run_conversation"
    _assert_never_bound_to(homes["alpha"], agent_cls, provider, homes["beta"])
    turn = agent_cls.turns[0]
    assert turn["agent"]._memory_manager is None
    assert turn["cached_agent"] is None


# ── every streaming construction site goes through the scoped constructor ────

def test_every_streaming_agent_construction_is_profile_scoped():
    src = (REPO_ROOT / "api" / "streaming.py").read_text(encoding="utf-8")
    bare = [m.group(0) for m in re.finditer(r"_AIAgent\(\s*\*\*\w+\s*\)", src)]
    assert not bare, f"bare streaming agent construction(s) bypass the profile scope: {bare}"
    assert src.count("construct_profile_scoped_agent(") >= 4


# ── the Agent attribute WebUI reads is the one the installed Agent writes ────

def test_installed_agent_bound_scope_contract():
    mcs = pytest.importorskip("agent.memory_caller_scope")
    hc_real = pytest.importorskip("hermes_constants")
    if not hasattr(mcs, "bound_scope"):
        pytest.skip("installed Agent predates _bound_scope (older Agents are handled as unverifiable)")
    scope = mcs.bound_scope(str(REPO_ROOT), "alpha")
    assert scope["hermes_home"] == str(REPO_ROOT)
    assert scope["home_key"] == hc_real.hermes_home_key(str(REPO_ROOT))
