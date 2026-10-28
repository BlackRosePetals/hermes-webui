"""Profile-scoped construction of streaming agents (#8517).

The Agent binds its external memory provider to whatever
``hermes_constants.get_hermes_home()`` answers while ``AIAgent.__init__`` runs
(``agent/agent_init._memory_provider_init_kwargs``). Without a context-local
override that answer is the process-global ``HERMES_HOME`` env mirror, which
other WebUI workers rewrite for their own profile
(``api.profiles.profile_env_for_background_worker``, the cron profile contexts,
a process-wide profile switch). A turn built inside that window binds another
profile's memory bank, and the agent is then cached for every later turn.

``construct_profile_scoped_agent`` closes that window for the streaming path:

* it sets the context-local home override for the session's own profile home
  around the constructor, explicitly, and checks that ``get_hermes_home()``
  then answers that home;
* it fails closed: when the scope cannot be set, the agent is built with memory
  off for that turn (``skip_memory=True``), a WARNING is logged and the caller
  does not cache it, so the next turn tries again with memory;
* after construction it reads the home the memory manager says it bound
  (``_memory_manager._bound_scope``); on a mismatch it rebuilds the agent
  without memory, warns, and the caller does not cache it.

Every Agent attribute is read defensively. Agents from before the per-profile
memory scope (Agent commit 419e6627, 2026-10-17) have no ``_bound_scope``; their
binding cannot be read back and is trusted to the override (logged once).
Agents older than v0.18.0 have no context-local override at all; see
``_enter_profile_home``.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_logged_once: set[str] = set()
_logged_once_lock = threading.Lock()


def _log_once(key: str, message: str, *args: Any) -> None:
    with _logged_once_lock:
        if key in _logged_once:
            return
        _logged_once.add(key)
    logger.warning(message, *args)


def _hermes_constants():
    """The Agent's ``hermes_constants``, resolved on every call: a negative answer
    cached from an earlier failed import must never switch the scope off."""
    mod = sys.modules.get("hermes_constants")
    if mod is not None:
        return mod
    try:
        import hermes_constants as mod  # noqa: PLC0415
    except Exception:
        return None
    return mod


def profile_home_key(home: Any) -> Optional[str]:
    """Comparable key for a Hermes home: the Agent's ``hermes_home_key`` (the function
    that produced ``_bound_scope["home_key"]``) when available, else the resolved path."""
    raw = str(home or "").strip()
    if not raw:
        return None
    mod = _hermes_constants()
    key_fn = getattr(mod, "hermes_home_key", None) if mod is not None else None
    if callable(key_fn):
        try:
            return str(key_fn(raw))
        except Exception:
            logger.debug("hermes_home_key(%r) failed; using the resolved path", raw, exc_info=True)
    try:
        return os.path.normcase(str(Path(raw).expanduser().resolve(strict=False)))
    except Exception:
        return None


def session_profile_home_or_empty(session: Any) -> str:
    """The session's own profile home, never the shared ``HERMES_HOME`` env mirror.
    ``""`` when it cannot be resolved; memory then fails closed for the turn."""
    try:
        from api.profiles import get_hermes_home_for_profile  # noqa: PLC0415

        home = get_hermes_home_for_profile(getattr(session, "profile", None))
        return str(home) if home else ""
    except Exception:
        logger.warning(
            "[webui] #8517: could not resolve the profile home for session %s; "
            "its memory provider is disabled for this turn",
            getattr(session, "session_id", "?"),
            exc_info=True,
        )
        return ""


def _noop() -> None:
    return None


def _enter_profile_home(profile_home: str, target_key: Optional[str]) -> tuple[Callable[[], None], Optional[str]]:
    """Set the context-local Hermes home to ``profile_home``.

    Returns ``(restore, reason)``: ``reason`` is None when construction may keep
    memory, else why it must not.
    """
    if not target_key:
        return _noop, "the session's profile home could not be resolved"
    mod = _hermes_constants()
    setter = getattr(mod, "set_hermes_home_override", None) if mod is not None else None
    resetter = getattr(mod, "reset_hermes_home_override", None) if mod is not None else None
    if not (callable(setter) and callable(resetter)):
        # Hermes Agent < v0.18.0: no context-local home. The Agent can only read the
        # process env mirror, which this turn set to its own home; accept it while it
        # still names this profile (a single-profile install always does).
        if profile_home_key(os.environ.get("HERMES_HOME")) == target_key:
            _log_once(
                "legacy-env",
                "[webui] #8517: this Hermes Agent has no context-local home override "
                "(older than v0.18.0); memory providers follow the shared HERMES_HOME. "
                "Upgrade the Agent for per-profile memory isolation.",
            )
            return _noop, None
        return _noop, (
            "this Hermes Agent has no context-local home override (older than v0.18.0) "
            "and the shared HERMES_HOME names another profile"
        )
    try:
        token = setter(str(profile_home))
    except Exception as exc:
        return _noop, f"setting the profile home override failed ({type(exc).__name__})"

    def _restore() -> None:
        try:
            resetter(token)
        except Exception:
            logger.warning("[webui] #8517: resetting the profile home override failed", exc_info=True)

    getter = getattr(mod, "get_hermes_home", None)
    if callable(getter):
        try:
            effective = profile_home_key(getter())
        except Exception:
            effective = None
        if effective != target_key:
            _restore()
            return _noop, "the profile home override did not take effect"
    return _restore, None


def _bound_memory_home_key(agent: Any) -> tuple[str, Optional[str]]:
    """``("none", None)`` without an external memory manager, ``("bound", key)`` when
    the manager reports its home, ``("unverifiable", None)`` when it cannot say."""
    manager = getattr(agent, "_memory_manager", None)
    if manager is None:
        return "none", None
    scope = getattr(manager, "_bound_scope", None)
    if isinstance(scope, dict):
        key = scope.get("home_key")
        if isinstance(key, str) and key:
            return "bound", key
        key = profile_home_key(scope.get("hermes_home"))
        if key:
            return "bound", key
    # MemoryManager's own build-time home key (Agent 2026-10-12+), when there is no scope.
    key = getattr(manager, "_home_key", None)
    if isinstance(key, str) and key:
        return "bound", key
    return "unverifiable", None


def _detach_memory(agent: Any) -> None:
    """Drop the agent's reference to its memory manager without shutting the provider
    down: a shared provider instance may still serve another profile's live agent."""
    if getattr(agent, "_memory_manager", None) is None:
        return
    try:
        agent._memory_manager = None
    except Exception as exc:
        raise RuntimeError(
            "could not detach a memory provider bound to another profile from the agent"
        ) from exc


def _constructor_params(agent_cls: Any) -> set:
    try:
        return set(inspect.signature(agent_cls.__init__).parameters)
    except (TypeError, ValueError):
        return set()


def _build_without_memory(agent_cls: Any, agent_kwargs: dict, params: set) -> Any:
    kwargs = dict(agent_kwargs)
    if "skip_memory" in params:
        kwargs["skip_memory"] = True
    agent = agent_cls(**kwargs)
    _detach_memory(agent)
    return agent


def construct_profile_scoped_agent(
    agent_cls: Any,
    agent_kwargs: dict,
    *,
    profile_home: str,
    agent_params: Optional[set] = None,
    session_id: str = "",
) -> tuple[Any, bool]:
    """Build ``agent_cls(**agent_kwargs)`` for the session whose profile lives at
    ``profile_home`` so its memory provider can only bind that profile.

    Returns ``(agent, cacheable)``. ``cacheable`` is False when memory was turned
    off or dropped for this turn; the caller must not put that agent in
    SESSION_AGENT_CACHE, so the next turn builds again with memory.
    ``agent_kwargs`` is never mutated.
    """
    params = _constructor_params(agent_cls) if agent_params is None else set(agent_params)
    target_key = profile_home_key(profile_home)
    restore, reason = _enter_profile_home(profile_home, target_key)
    try:
        if reason is not None:
            logger.warning(
                "[webui] #8517: memory provider disabled for session %s this turn: %s "
                "(profile home %r). The agent is built without memory and not cached.",
                session_id, reason, profile_home,
            )
            return _build_without_memory(agent_cls, agent_kwargs, params), False
        agent = agent_cls(**agent_kwargs)
        state, bound_key = _bound_memory_home_key(agent)
        if state == "bound" and bound_key != target_key:
            logger.warning(
                "[webui] #8517: memory provider for session %s bound %r instead of its "
                "profile home %r; it is dropped for this turn and the agent is not cached.",
                session_id, bound_key, target_key,
            )
            _detach_memory(agent)
            if "skip_memory" in params:
                agent = _build_without_memory(agent_cls, agent_kwargs, params)
            return agent, False
        if state == "unverifiable":
            _log_once(
                "unverifiable",
                "[webui] #8517: this Hermes Agent does not report which profile home its "
                "memory provider bound; relying on the profile home override set for "
                "construction. Upgrade the Agent to verify it.",
            )
        return agent, True
    finally:
        restore()
