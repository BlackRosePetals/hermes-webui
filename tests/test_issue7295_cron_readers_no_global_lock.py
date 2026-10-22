"""#7295 / #6753: /api/crons* must not wait on the global cron env lock.

A long in-process cron job (scheduled fire-claim run via the WebUI
``run_job`` isolation wrapper, #7295, or the in-chat ``cronjob action=run``
handler, #6753) holds ``api.profiles._cron_env_lock`` and keeps the process
globals (``os.environ['HERMES_HOME']``, ``cron.jobs`` path constants,
``cron.scheduler._hermes_home``) pointed at its own profile for the whole run.

These tests hold that context in a real background thread and drive the real
route dispatchers for a DIFFERENT profile and for the SAME profile.  Every
converted endpoint must answer promptly with its own profile's data and must
not disturb the running job's globals.  ``/api/crons/create`` deliberately
stays on the lock (its selected-profile snapshot repair mutates process env),
so it is pinned as still waiting.  Older Agents without the per-thread helpers
fall back to the lock and stay correct.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

cron_jobs = pytest.importorskip("cron.jobs")  # needs the installed Agent

import api.profiles as profiles  # noqa: E402
import api.routes as routes  # noqa: E402

JOB_ID = "a1b2c3d4e5f6"  # identical id in both profiles: wrong-store reads are visible
RUN_FILE = "2026-10-01_00-00-00.md"
PROMPT_DEADLINE_S = 3.0
HOLD_MAX_S = 60.0

_HAS_PER_THREAD_HELPERS = callable(getattr(cron_jobs, "use_cron_store", None)) and callable(
    getattr(__import__("hermes_constants"), "set_hermes_home_override", None)
)
requires_per_thread_helpers = pytest.mark.skipif(
    not _HAS_PER_THREAD_HELPERS,
    reason="installed Agent predates cron.jobs.use_cron_store/set_hermes_home_override",
)


def _write_profile(home: Path, label: str) -> None:
    cron_dir = home / "cron"
    (cron_dir / "output" / JOB_ID).mkdir(parents=True, exist_ok=True)
    job = {
        "id": JOB_ID,
        "name": f"{label}-job",
        "prompt": f"{label} prompt",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "schedule_display": "every 60m",
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 1},
        "next_run_at": "2099-01-01T00:00:00+00:00",
        "last_run_at": "2026-10-01T00:00:00+00:00",
        "last_status": "ok",
        "deliver": "local",
        "profile": label if label != "default" else None,
    }
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": [job]}), encoding="utf-8")
    (cron_dir / "output" / JOB_ID / RUN_FILE).write_text(
        f"# Cron Job\n\n**Model:** {label}-model\n\n## Response\nOUTPUT-FROM-{label}\n",
        encoding="utf-8",
    )


def _jobs_on_disk(home: Path) -> list[dict]:
    return json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]


class _Handler:
    def __init__(self, body=None, path="/"):
        raw = json.dumps(body or {}).encode("utf-8")
        self.status = None
        self.response_headers = {}
        self.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.command = "POST" if body is not None else "GET"
        self.path = path
        self.client_address = ("127.0.0.1", 12345)

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers[key] = value

    def end_headers(self):
        pass

    def payload(self):
        raw = self.wfile.getvalue()
        return json.loads(raw.decode("utf-8")) if raw else {}


@pytest.fixture
def world(tmp_path, monkeypatch):
    base = (tmp_path / "base").resolve()
    alpha = base / "profiles" / "alpha"
    beta = base / "profiles" / "beta"
    for home, label in ((base, "default"), (alpha, "alpha"), (beta, "beta")):
        _write_profile(home, label)
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", base)
    monkeypatch.setattr(profiles, "_is_isolated_profile_mode", lambda: False)
    monkeypatch.setattr(routes, "_is_isolated_profile_mode", lambda: False)
    rows = [{"name": "default", "visible": True}, {"name": "alpha", "visible": True}, {"name": "beta", "visible": True}]
    monkeypatch.setattr(profiles, "list_profiles_api", lambda: [dict(r) for r in rows])
    monkeypatch.setattr(routes, "list_profiles_api", lambda: [dict(r) for r in rows])
    monkeypatch.setattr(profiles, "publish_session_list_changed", lambda *a, **k: None)
    monkeypatch.setattr(routes, "_latest_cron_session_info_for_jobs", lambda *a, **k: {})
    # The process default points at the server-default profile, as in production.
    monkeypatch.setenv("HERMES_HOME", str(base))
    for name, value in (
        ("HERMES_DIR", base),
        ("CRON_DIR", base / "cron"),
        ("JOBS_FILE", base / "cron" / "jobs.json"),
        ("OUTPUT_DIR", base / "cron" / "output"),
    ):
        monkeypatch.setattr(cron_jobs, name, value)
    try:
        import cron.scheduler as cron_scheduler
    except Exception:  # pragma: no cover - Agent without scheduler
        cron_scheduler = None
    if cron_scheduler is not None:
        monkeypatch.setattr(cron_scheduler, "_hermes_home", base, raising=False)
    assert not profiles._cron_env_lock.locked(), "cron env lock leaked from an earlier test"
    return {"base": base, "alpha": alpha, "beta": beta, "scheduler": cron_scheduler}


class _Holder:
    """Run a real long cron job body in the background holding the cron context."""

    def __init__(self, start_fn):
        self.started = threading.Event()
        self.release = threading.Event()
        self.observed = {}
        self.error = None
        self._start_fn = start_fn
        self.thread = threading.Thread(target=self._run, daemon=True)

    def body(self, *args, **kwargs):
        self.observed["env_home"] = os.environ.get("HERMES_HOME")
        self.observed["jobs_file"] = cron_jobs.JOBS_FILE
        self.started.set()
        self.release.wait(HOLD_MAX_S)
        # Globals must still be this job's after concurrent readers ran.
        self.observed["env_home_after"] = os.environ.get("HERMES_HOME")
        self.observed["jobs_file_after"] = cron_jobs.JOBS_FILE
        return "done"

    def _run(self):
        try:
            self._start_fn(self)
        except BaseException as exc:  # pragma: no cover - surfaced in asserts
            self.error = exc

    def __enter__(self):
        self.thread.start()
        assert self.started.wait(10), "slow job never entered its cron context"
        assert profiles._cron_env_lock.locked(), "slow job is not holding the cron env lock"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(15)
        assert not self.thread.is_alive()
        return False


def _scheduled_fire_holder(world, monkeypatch, profile_home):
    """#7295: scheduled fire-claim path through the WebUI run_job isolation wrapper."""
    cron_scheduler = world["scheduler"]
    if cron_scheduler is None:
        pytest.skip("cron.scheduler unavailable")

    def start(holder):
        monkeypatch.setattr(cron_scheduler, "run_job", holder.body)
        profiles.install_cron_scheduler_profile_isolation()
        wrapped = cron_scheduler.run_job
        assert getattr(wrapped, "_webui_profile_isolated", False)
        wrapped({"id": JOB_ID, "profile": profile_home.name})

    return _Holder(start)


def _in_chat_run_holder(world, monkeypatch, profile_home):
    """#6753: in-chat cronjob(action='run') through the streaming profile wrapper."""
    import api.streaming as streaming

    def start(holder):
        wrapper = streaming._streaming_cronjob_profile_handler(lambda args, **kw: holder.body())
        token = streaming._STREAMING_CRON_PROFILE_HOME.set(str(profile_home))
        try:
            wrapper({"action": "run", "job_id": JOB_ID})
        finally:
            streaming._STREAMING_CRON_PROFILE_HOME.reset(token)

    return _Holder(start)


HOLDERS = {"scheduled_fire_7295": _scheduled_fire_holder, "in_chat_run_6753": _in_chat_run_holder}


def _request(method, path, profile, body=None, deadline=PROMPT_DEADLINE_S):
    """Dispatch one request on a fresh server-like thread; return (elapsed, status, payload)."""
    out = {}

    def run():
        profiles.set_request_profile(profile)
        try:
            handler = _Handler(body=body if method == "POST" else None, path=path)
            parsed = urlparse(path)
            start = time.monotonic()
            if method == "GET":
                routes.handle_get(handler, parsed)
            else:
                routes.handle_post(handler, parsed)
            out["elapsed"] = time.monotonic() - start
            out["status"] = handler.status
            out["payload"] = handler.payload()
        except BaseException as exc:  # pragma: no cover - surfaced below
            out["error"] = exc
        finally:
            profiles.clear_request_profile()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(deadline)
    return t, out


def _finish(t, out):
    t.join(HOLD_MAX_S)
    if "error" in out:
        raise out["error"]
    return out


def _assert_job_untouched(world, profile):
    jobs = _jobs_on_disk(world[profile])
    assert [j["name"] for j in jobs] == [f"{profile}-job"], (profile, jobs)
    assert jobs[0].get("state") == "scheduled", (profile, jobs)


GET_READERS = [
    ("/api/crons", lambda body, me: [j["name"] for j in body["jobs"]] == [f"{me}-job"]),
    ("/api/crons?all_profiles=1", lambda body, me: sorted((j["owner_profile"], j["name"]) for j in body["jobs"])
     == sorted([("default", "default-job"), ("alpha", "alpha-job"), ("beta", "beta-job")])),
    ("/api/crons/recent?since=0", lambda body, me: [(c["job_id"], c["name"]) for c in body["completions"]]
     == [(JOB_ID, f"{me}-job")]),
    (f"/api/crons/output?job_id={JOB_ID}", lambda body, me: len(body["outputs"]) == 1
     and f"OUTPUT-FROM-{me}" in body["outputs"][0]["content"]),
    (f"/api/crons/history?job_id={JOB_ID}", lambda body, me: [r["filename"] for r in body["runs"]] == [RUN_FILE]
     and body["runs"][0]["usage"].get("model") == f"{me}-model"),
    (f"/api/crons/run?job_id={JOB_ID}&filename={RUN_FILE}", lambda body, me: f"OUTPUT-FROM-{me}" in body["content"]),
    ("/api/crons/status", lambda body, me: "running" in body),
    ("/api/crons/delivery-options", lambda body, me: any(p["value"] == "local" for p in body["platforms"])),
]


@requires_per_thread_helpers
@pytest.mark.parametrize("holder_kind", sorted(HOLDERS))
@pytest.mark.parametrize("reader_profile", ["beta", "alpha"])
def test_cron_readers_answer_promptly_with_own_profile_while_job_holds_lock(
    world, monkeypatch, holder_kind, reader_profile
):
    results = []
    with HOLDERS[holder_kind](world, monkeypatch, world["alpha"]) as holder:
        for path, check in GET_READERS:
            t, out = _request("GET", path, reader_profile)
            results.append((path, t.is_alive(), t, out, check))
        # Globals still belong to the running job while readers ran.
        assert os.environ.get("HERMES_HOME") == str(world["alpha"])
        assert cron_jobs.JOBS_FILE == world["alpha"] / "cron" / "jobs.json"
        blocked = [path for path, alive, *_ in results if alive]
    for path, _alive, t, out, check in results:
        _finish(t, out)
    assert blocked == [], f"readers waited on the running cron job's lock: {blocked}"
    assert holder.error is None
    assert holder.observed["env_home"] == holder.observed["env_home_after"] == str(world["alpha"])
    assert holder.observed["jobs_file"] == holder.observed["jobs_file_after"]
    for path, _alive, _t, out, check in results:
        assert out["status"] == 200, (path, out)
        assert check(out["payload"], reader_profile), (path, out["payload"])
    _assert_job_untouched(world, "alpha")
    _assert_job_untouched(world, "beta")


POST_WRITERS = [
    ("/api/crons/update", {"job_id": JOB_ID, "name": "beta-renamed"},
     lambda jobs: jobs[0]["name"] == "beta-renamed"),
    ("/api/crons/pause", {"job_id": JOB_ID, "reason": "t7295"},
     lambda jobs: jobs[0]["state"] == "paused"),
    ("/api/crons/resume", {"job_id": JOB_ID},
     lambda jobs: jobs[0]["state"] == "scheduled"),
    ("/api/crons/run", {"job_id": JOB_ID}, lambda jobs: jobs[0]["name"] == "beta-renamed"),
    ("/api/crons/delete", {"job_id": JOB_ID}, lambda jobs: jobs == []),
]


@requires_per_thread_helpers
@pytest.mark.parametrize("holder_kind", sorted(HOLDERS))
def test_cron_store_writers_do_not_wait_and_only_touch_own_profile(world, monkeypatch, holder_kind):
    launched = []
    monkeypatch.setattr(routes, "_RUNNING_CRON_JOBS", {})
    monkeypatch.setattr(routes, "_run_cron_tracked", lambda *a, **k: launched.append(a))
    with HOLDERS[holder_kind](world, monkeypatch, world["alpha"]) as holder:
        for path, body, check in POST_WRITERS:
            t, out = _request("POST", path, "beta", body=body)
            alive = t.is_alive()
            _finish(t, out)
            assert not alive, f"{path} waited on the running cron job's lock"
            assert out["status"] == 200, (path, out)
            assert out["payload"].get("ok") is True, (path, out)
            assert check(_jobs_on_disk(world["beta"])), (path, _jobs_on_disk(world["beta"]))
            _assert_job_untouched(world, "alpha")
        assert os.environ.get("HERMES_HOME") == str(world["alpha"])
    assert holder.error is None
    assert holder.observed["env_home_after"] == str(world["alpha"])
    assert len(launched) == 1
    job, owner_home, execution_home, _event_profile = launched[0]
    assert job["name"] == "beta-renamed"
    assert Path(owner_home) == world["beta"] and Path(execution_home) == world["beta"]
    _assert_job_untouched(world, "alpha")


@pytest.mark.parametrize("holder_kind", sorted(HOLDERS))
def test_cron_create_still_serializes_on_the_lock(world, monkeypatch, holder_kind):
    """Create keeps the lock: its selected-profile snapshot repair mutates os.environ."""
    with HOLDERS[holder_kind](world, monkeypatch, world["alpha"]):
        t, out = _request(
            "POST", "/api/crons/create", "beta",
            body={"prompt": "new beta job", "schedule": "every 2h", "name": "beta-created"},
            deadline=0.5,
        )
        assert t.is_alive(), "create unexpectedly bypassed the cron env lock"
    _finish(t, out)
    assert out["status"] == 200, out
    assert sorted(j["name"] for j in _jobs_on_disk(world["beta"])) == ["beta-created", "beta-job"]
    _assert_job_untouched(world, "alpha")


@requires_per_thread_helpers
def test_reader_scope_resets_per_thread_overrides_on_reused_threads(world):
    """Tokens are reset in finally: a reused server thread never inherits a prior profile."""
    import hermes_constants
    from concurrent.futures import ThreadPoolExecutor

    store_var = getattr(cron_jobs, "_cron_store_override", None)

    def serve(profile, path):
        profiles.set_request_profile(profile)
        try:
            handler = _Handler(path=path)
            routes.handle_get(handler, urlparse(path))
            return handler.status, handler.payload(), hermes_constants.get_hermes_home_override(), (
                store_var.get() if store_var is not None else None)
        finally:
            profiles.clear_request_profile()

    def boom(profile):
        profiles.set_request_profile(profile)
        try:
            with profiles.cron_store_scope():
                raise RuntimeError("handler failure")
        except RuntimeError:
            return hermes_constants.get_hermes_home_override(), (
                store_var.get() if store_var is not None else None)
        finally:
            profiles.clear_request_profile()

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(serve, "alpha", f"/api/crons/output?job_id={JOB_ID}").result()
        failed = pool.submit(boom, "alpha").result()
        second = pool.submit(serve, "beta", f"/api/crons/output?job_id={JOB_ID}").result()
        third = pool.submit(serve, "default", "/api/crons").result()
    assert "OUTPUT-FROM-alpha" in first[1]["outputs"][0]["content"]
    assert "OUTPUT-FROM-beta" in second[1]["outputs"][0]["content"]
    assert [j["name"] for j in third[1]["jobs"]] == ["default-job"]
    for leftover in (first[2:], failed, second[2:], third[2:]):
        assert leftover == (None, None), leftover


def _hide_per_thread_helpers(monkeypatch, which):
    if which == "use_cron_store":
        monkeypatch.delattr(cron_jobs, "use_cron_store", raising=False)
    else:
        import hermes_constants

        monkeypatch.delattr(hermes_constants, "set_hermes_home_override", raising=False)
        monkeypatch.setattr(profiles, "_hermes_home_override_available", None, raising=False)


@pytest.mark.parametrize("missing", ["use_cron_store", "set_hermes_home_override"])
def test_older_agent_without_helpers_falls_back_to_lock_and_stays_correct(world, monkeypatch, missing):
    _hide_per_thread_helpers(monkeypatch, missing)
    # No holder: lock-based fallback still reads the requesting profile.
    t, out = _request("GET", f"/api/crons/output?job_id={JOB_ID}", "beta")
    _finish(t, out)
    assert "OUTPUT-FROM-beta" in out["payload"]["outputs"][0]["content"]
    t, out = _request("GET", "/api/crons", "alpha")
    _finish(t, out)
    assert [j["name"] for j in out["payload"]["jobs"]] == ["alpha-job"]
    # With a running job holding the lock the fallback waits (master behavior)
    # and then answers with the requesting profile's data, not the job's.
    with _scheduled_fire_holder(world, monkeypatch, world["alpha"]):
        t, out = _request("GET", f"/api/crons/history?job_id={JOB_ID}", "beta", deadline=0.5)
        assert t.is_alive(), "older-Agent fallback must keep serializing on the cron env lock"
    _finish(t, out)
    assert out["payload"]["runs"][0]["usage"]["model"] == "beta-model"
    assert not profiles._cron_env_lock.locked()
    assert os.environ.get("HERMES_HOME") == str(world["base"])
    assert cron_jobs.JOBS_FILE == world["base"] / "cron" / "jobs.json"
