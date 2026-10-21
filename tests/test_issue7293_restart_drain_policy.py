"""#7293 — managed restart must drain or refuse active WebUI-owned runs.

Owner-approved policy (Option A):

* A WebUI-initiated update/restart REFUSES by default while WebUI-owned runs
  are active, reports the active count, and offers an explicit force override.
  Admission is fenced for the whole update so nothing new can start between the
  safety check and the re-exec; a forced restart settles the remaining runs as
  durably interrupted before the process image is replaced.
* An external SIGTERM/SIGINT gets ONE idempotent, bounded drain: admission
  closes, active runs may finish inside the budget, anything still running at
  the deadline is settled as interrupted durably, then the server stops.
* A repeated signal escalates to an immediate settled stop; it never starts a
  second drain.
* Gateway-backed runs are not WebUI-owned and are never waited on or settled.

These are the in-process contract tests. The real serving-process tests live in
``test_issue7293_restart_drain_real_process.py``.
"""

from __future__ import annotations

import pathlib
import threading
import time

import pytest

from api import config
from api import run_drain


REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_registries():
    run_drain._reset_for_tests()
    with config.STREAMS_LOCK:
        old_streams = dict(config.STREAMS)
        old_claims = dict(config.PRE_ADMISSION_CLAIMS)
        config.STREAMS.clear()
        config.PRE_ADMISSION_CLAIMS.clear()
    with config.ACTIVE_RUNS_LOCK:
        old_runs = dict(config.ACTIVE_RUNS)
        config.ACTIVE_RUNS.clear()
    try:
        yield
    finally:
        run_drain._reset_for_tests()
        with config.STREAMS_LOCK:
            config.STREAMS.clear()
            config.STREAMS.update(old_streams)
            config.PRE_ADMISSION_CLAIMS.clear()
            config.PRE_ADMISSION_CLAIMS.update(old_claims)
        with config.ACTIVE_RUNS_LOCK:
            config.ACTIVE_RUNS.clear()
            config.ACTIVE_RUNS.update(old_runs)


def _add_run(stream_id, *, backend="legacy", session_id="sess", phase="running"):
    with config.ACTIVE_RUNS_LOCK:
        config.ACTIVE_RUNS[stream_id] = {
            "stream_id": stream_id,
            "session_id": session_id,
            "backend": backend,
            "phase": phase,
            "started_at": time.time(),
        }


def _drop_run(stream_id):
    with config.ACTIVE_RUNS_LOCK:
        config.ACTIVE_RUNS.pop(stream_id, None)


# ── Admission fence ──────────────────────────────────────────────────────────


def test_admission_is_open_by_default():
    assert run_drain.run_admission_closed_response() is None
    with config.STREAMS_LOCK:
        run_drain.assert_run_admission_open_locked()  # must not raise
    assert run_drain.admission_status()["open"] is True


def test_update_hold_closes_admission_until_released():
    token = run_drain.hold_run_admission("update:webui")
    refusal = run_drain.run_admission_closed_response()
    assert refusal is not None
    assert refusal["_status"] == 503
    assert refusal["code"] == "server_draining"
    assert refusal["server_draining"] is True
    with config.STREAMS_LOCK:
        with pytest.raises(run_drain.RunAdmissionClosed) as exc_info:
            run_drain.assert_run_admission_open_locked()
    assert exc_info.value.status == 503
    assert exc_info.value.payload["code"] == "server_draining"

    run_drain.release_run_admission_hold(token)
    assert run_drain.run_admission_closed_response() is None


def test_restart_pending_keeps_admission_closed_after_hold_release():
    token = run_drain.hold_run_admission("update:agent")
    run_drain.mark_restart_pending()
    run_drain.release_run_admission_hold(token)
    assert run_drain.run_admission_closed_response() is not None
    assert run_drain.admission_status()["phase"] == "restart_pending"


def test_abandoned_restart_reopens_admission():
    token = run_drain.hold_run_admission("update:webui")
    epoch = run_drain.mark_restart_pending()
    run_drain.release_run_admission_hold(token)
    run_drain.abandon_restart_pending(epoch)
    assert run_drain.run_admission_closed_response() is None


def test_force_flag_follows_the_hold_into_restart_pending():
    token = run_drain.hold_run_admission("update:webui", force=True)
    run_drain.mark_restart_pending()
    run_drain.release_run_admission_hold(token)
    assert run_drain.restart_force_requested() is True


def test_force_flag_does_not_survive_a_failed_update():
    token = run_drain.hold_run_admission("update:webui", force=True)
    run_drain.release_run_admission_hold(token)
    # A later, non-forced restart must not inherit the earlier force decision.
    token2 = run_drain.hold_run_admission("update:webui")
    run_drain.mark_restart_pending()
    run_drain.release_run_admission_hold(token2)
    assert run_drain.restart_force_requested() is False


# ── WebUI-owned liveness (gateway exclusion) ────────────────────────────────


def test_webui_owned_runs_exclude_gateway_backed_runs():
    _add_run("legacy-1", backend="legacy")
    _add_run("gw-1", backend="gateway")
    _add_run("cancelling-1", backend="legacy", phase="cancelling")
    ids = sorted(r["stream_id"] for r in run_drain.webui_owned_active_runs())
    assert ids == ["cancelling-1", "legacy-1"]


def test_launching_streams_count_as_webui_owned_work():
    with config.STREAMS_LOCK:
        config.STREAMS["launching-1"] = object()
        config.PRE_ADMISSION_CLAIMS["launching-1"] = "tok"
    runs = run_drain.webui_owned_active_runs()
    assert [r["stream_id"] for r in runs] == ["launching-1"]
    assert runs[0]["launching"] is True


# ── Settlement ───────────────────────────────────────────────────────────────


def test_settle_cancels_only_webui_owned_runs_with_reason(monkeypatch):
    from api import streaming

    calls = []

    def fake_cancel(stream_id, reason=None):
        calls.append((stream_id, reason))
        _drop_run(stream_id)
        return True

    monkeypatch.setattr(streaming, "cancel_stream", fake_cancel)
    _add_run("legacy-a")
    _add_run("legacy-b")
    _add_run("gw-a", backend="gateway")

    result = run_drain.settle_webui_owned_runs("server stopping", timeout=2.0)

    assert sorted(c[0] for c in calls) == ["legacy-a", "legacy-b"]
    assert all(c[1] == "server stopping" for c in calls)
    assert sorted(result["settled"]) == ["legacy-a", "legacy-b"]
    assert result["timed_out"] is False
    with config.ACTIVE_RUNS_LOCK:
        assert "gw-a" in config.ACTIVE_RUNS


def test_settle_is_bounded_when_cancel_hangs(monkeypatch):
    from api import streaming

    release = threading.Event()

    def hanging_cancel(stream_id, reason=None):
        release.wait(10)
        return True

    monkeypatch.setattr(streaming, "cancel_stream", hanging_cancel)
    _add_run("legacy-stuck")
    started = time.monotonic()
    result = run_drain.settle_webui_owned_runs("server stopping", timeout=0.5)
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 2.0
    assert result["timed_out"] is True


def test_cancel_stream_records_the_shutdown_reason_in_the_durable_marker():
    from api import streaming

    content = streaming._cancelled_turn_content("The WebUI server stopped before this response finished")
    normalized = content.lower()
    assert "the webui server stopped before this response finished" in normalized
    # The marker must still be recognised as a cancel marker by every
    # dedupe / recovery path.
    assert any(p in normalized for p in streaming._CANCEL_MARKER_PATTERNS)
    import inspect

    assert "reason" in inspect.signature(streaming.cancel_stream).parameters


# ── Bounded shutdown drain (external SIGTERM) ───────────────────────────────


class _StopRecorder:
    def __init__(self):
        self.stopped = threading.Event()
        self.at = None

    def __call__(self):
        self.at = time.monotonic()
        self.stopped.set()


def test_zero_active_runs_stops_immediately():
    stop = _StopRecorder()
    drain = run_drain.ShutdownDrain(stop, budget_seconds=30, settle_seconds=1, poll_seconds=0.02)
    t0 = time.monotonic()
    drain.handle_signal(15, None)
    assert stop.stopped.wait(2.0)
    assert stop.at - t0 < 1.0
    assert drain.outcome["result"] == "drained"
    assert run_drain.run_admission_closed_response() is not None


def test_run_finishing_inside_budget_is_waited_for_not_settled(monkeypatch):
    from api import streaming

    settled = []
    monkeypatch.setattr(streaming, "cancel_stream", lambda sid, reason=None: settled.append(sid))
    _add_run("legacy-finishing")
    stop = _StopRecorder()
    drain = run_drain.ShutdownDrain(stop, budget_seconds=5, settle_seconds=1, poll_seconds=0.02)
    drain.handle_signal(15, None)
    time.sleep(0.3)
    assert not stop.stopped.is_set(), "drain must wait for the active run"
    _drop_run("legacy-finishing")
    assert stop.stopped.wait(2.0)
    assert settled == []
    assert drain.outcome["result"] == "drained"


def test_run_exceeding_budget_is_settled_then_server_stops(monkeypatch):
    from api import streaming

    settled = []

    def fake_cancel(stream_id, reason=None):
        settled.append((stream_id, reason))
        _drop_run(stream_id)
        return True

    monkeypatch.setattr(streaming, "cancel_stream", fake_cancel)
    _add_run("legacy-long")
    _add_run("gw-long", backend="gateway")
    stop = _StopRecorder()
    drain = run_drain.ShutdownDrain(stop, budget_seconds=0.4, settle_seconds=1, poll_seconds=0.02)
    t0 = time.monotonic()
    drain.handle_signal(15, None)
    assert stop.stopped.wait(3.0)
    assert stop.at - t0 >= 0.35
    assert [s[0] for s in settled] == ["legacy-long"]
    assert "shut" in (settled[0][1] or "").lower() or "stop" in (settled[0][1] or "").lower()
    assert drain.outcome["result"] == "deadline"
    assert drain.outcome["settled"] == ["legacy-long"]


def test_new_admission_is_refused_while_draining():
    _add_run("legacy-long")
    stop = _StopRecorder()
    drain = run_drain.ShutdownDrain(stop, budget_seconds=5, settle_seconds=1, poll_seconds=0.02)
    drain.handle_signal(15, None)
    time.sleep(0.1)
    refusal = run_drain.run_admission_closed_response()
    assert refusal is not None and refusal["_status"] == 503
    assert run_drain.admission_status()["phase"] == "draining"
    drain.escalate()
    stop.stopped.wait(3.0)


def test_second_signal_escalates_without_starting_a_second_drain(monkeypatch):
    from api import streaming

    settled = []

    def fake_cancel(stream_id, reason=None):
        settled.append(stream_id)
        _drop_run(stream_id)
        return True

    monkeypatch.setattr(streaming, "cancel_stream", fake_cancel)
    _add_run("legacy-long")
    stop = _StopRecorder()
    drain = run_drain.ShutdownDrain(stop, budget_seconds=60, settle_seconds=1, poll_seconds=0.02)
    drain.handle_signal(15, None)
    time.sleep(0.2)
    first_thread = drain._thread
    t0 = time.monotonic()
    drain.handle_signal(15, None)
    assert stop.stopped.wait(3.0)
    assert stop.at - t0 < 2.0, "a repeated signal must escalate to an immediate settled stop"
    assert drain._thread is first_thread, "a repeated signal must not start a second drain"
    assert settled == ["legacy-long"]
    assert drain.outcome["result"] == "escalated"
    # Further signals are inert.
    drain.handle_signal(15, None)
    drain.handle_signal(2, None)
    assert drain._thread is first_thread


def test_drain_budget_comes_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_SHUTDOWN_DRAIN_SECONDS", "7.5")
    assert run_drain.shutdown_drain_budget_seconds() == 7.5
    monkeypatch.setenv("HERMES_WEBUI_SHUTDOWN_DRAIN_SECONDS", "nonsense")
    assert run_drain.shutdown_drain_budget_seconds() == run_drain.DEFAULT_SHUTDOWN_DRAIN_SECONDS
    monkeypatch.setenv("HERMES_WEBUI_SHUTDOWN_DRAIN_SECONDS", "-4")
    assert run_drain.shutdown_drain_budget_seconds() == 0.0


# ── WebUI-initiated update/restart: refuse by default, explicit force ───────


def _git_dir(tmp_path, monkeypatch, upd):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(upd, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(upd, "_AGENT_DIR", tmp_path)


def test_apply_update_refusal_reports_count_and_offers_force(tmp_path, monkeypatch):
    import api.updates as upd

    _git_dir(tmp_path, monkeypatch, upd)
    monkeypatch.setattr(upd, "_apply_update_inner", lambda *a, **k: pytest.fail("must not update"))
    _add_run("legacy-1")
    result = upd.apply_update("webui")
    assert result["ok"] is False
    assert result["restart_blocked"] is True
    assert result["active_runs"] == 1
    assert result["force_available"] is True
    assert run_drain.run_admission_closed_response() is None, "a refusal must not leave admission fenced"


def test_forced_apply_update_proceeds_and_fences_admission_during_update(tmp_path, monkeypatch):
    import api.updates as upd

    _git_dir(tmp_path, monkeypatch, upd)
    seen = {}

    def inner(target, channel):
        seen["closed_during_update"] = run_drain.run_admission_closed_response() is not None
        return {"ok": False, "message": "pull failed"}

    monkeypatch.setattr(upd, "_apply_update_inner", inner)
    _add_run("legacy-1")
    result = upd.apply_update("webui", force_active_runs=True)
    assert result == {"ok": False, "message": "pull failed"}
    assert seen["closed_during_update"] is True
    # The update did not schedule a restart, so admission reopens.
    assert run_drain.run_admission_closed_response() is None


def test_apply_update_rechecks_after_fencing_admission(tmp_path, monkeypatch):
    """A run admitted between the first check and the fence must still refuse."""
    import api.updates as upd

    _git_dir(tmp_path, monkeypatch, upd)
    monkeypatch.setattr(upd, "_apply_update_inner", lambda *a, **k: pytest.fail("must not update"))
    real_hold = run_drain.hold_run_admission

    def hold_after_race(owner, *, force=False):
        _add_run("legacy-raced-in")  # admitted just before the fence closed
        return real_hold(owner, force=force)

    monkeypatch.setattr(run_drain, "hold_run_admission", hold_after_race)
    result = upd.apply_update("webui")
    assert result["ok"] is False
    assert result["restart_blocked"] is True
    assert run_drain.run_admission_closed_response() is None


def test_successful_update_keeps_admission_closed_until_restart(tmp_path, monkeypatch):
    import api.updates as upd

    _git_dir(tmp_path, monkeypatch, upd)
    monkeypatch.setattr(upd, "_wait_until_restart_safe", lambda *a, **k: {"restart_blocked": False})

    def inner(target, channel):
        upd._schedule_restart(delay=60)  # never fires during the test
        return {"ok": True, "restart_scheduled": True, "target": target}

    monkeypatch.setattr(upd, "_apply_update_inner", inner)
    result = upd.apply_update("webui")
    assert result["ok"] is True
    refusal = run_drain.run_admission_closed_response()
    assert refusal is not None and refusal["_status"] == 503


def test_force_and_clear_lock_paths_accept_force(tmp_path, monkeypatch):
    import inspect
    import api.updates as upd

    for fn in (upd.apply_update, upd.apply_force_update, upd.apply_clear_lock):
        assert "force_active_runs" in inspect.signature(fn).parameters, fn.__name__


def test_forced_restart_settles_webui_runs_before_execv_and_leaves_gateway_runs(monkeypatch):
    import os
    import sys
    import api.updates as upd
    from api import streaming

    events = []

    def fake_cancel(stream_id, reason=None):
        events.append(("cancel", stream_id, reason))
        _drop_run(stream_id)
        return True

    monkeypatch.setattr(streaming, "cancel_stream", fake_cancel)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(upd, "_purge_agent_pycache", lambda *_a: None)
    monkeypatch.setattr(os, "execv", lambda exe, args: events.append(("execv", exe)))
    _add_run("legacy-1")
    _add_run("gw-1", backend="gateway")

    token = run_drain.hold_run_admission("update:webui", force=True)
    upd._schedule_restart(delay=0.01)
    run_drain.release_run_admission_hold(token)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(e[0] == "execv" for e in events):
        time.sleep(0.02)

    kinds = [e[0] for e in events]
    assert "execv" in kinds
    assert kinds.index("cancel") < kinds.index("execv")
    assert [e[1] for e in events if e[0] == "cancel"] == ["legacy-1"]
    with config.ACTIVE_RUNS_LOCK:
        assert "gw-1" in config.ACTIVE_RUNS


def test_unforced_restart_wait_timeout_settles_instead_of_abandoning(monkeypatch):
    import os
    import sys
    import api.updates as upd
    from api import streaming

    events = []

    def fake_cancel(stream_id, reason=None):
        events.append(("cancel", stream_id))
        _drop_run(stream_id)
        return True

    monkeypatch.setattr(streaming, "cancel_stream", fake_cancel)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(upd, "_purge_agent_pycache", lambda *_a: None)
    monkeypatch.setattr(
        upd, "_wait_until_restart_safe",
        lambda *a, **k: {"restart_blocked": True, "wait_timed_out": True, "active_runs": 1},
    )
    monkeypatch.setattr(os, "execv", lambda exe, args: events.append(("execv", exe)))
    _add_run("legacy-stuck")

    upd._schedule_restart(delay=0.01)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(e[0] == "execv" for e in events):
        time.sleep(0.02)
    kinds = [e[0] for e in events]
    assert kinds == ["cancel", "execv"]


def test_wait_until_restart_safe_does_not_wait_when_forced(monkeypatch):
    import api.updates as upd

    _add_run("legacy-1")
    token = run_drain.hold_run_admission("update:webui", force=True)
    run_drain.mark_restart_pending()
    run_drain.release_run_admission_hold(token)
    sleeps = []
    monkeypatch.setattr(upd.time, "sleep", lambda s: sleeps.append(s))
    snapshot = upd._wait_until_restart_safe(poll_seconds=0.1, max_wait_seconds=30)
    assert sleeps == []
    assert snapshot.get("forced") is True


def test_restart_that_does_not_replace_the_process_reopens_admission(monkeypatch):
    import os
    import sys
    import api.updates as upd

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(upd, "_purge_agent_pycache", lambda *_a: None)
    monkeypatch.setattr(upd, "_wait_until_restart_safe", lambda *a, **k: {"restart_blocked": False})
    done = threading.Event()
    monkeypatch.setattr(os, "execv", lambda exe, args: done.set())
    upd._schedule_restart(delay=0.01)
    assert done.wait(5)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and run_drain.run_admission_closed_response() is not None:
        time.sleep(0.02)
    assert run_drain.run_admission_closed_response() is None


# ── Wiring guards ────────────────────────────────────────────────────────────


def test_server_routes_signals_through_the_bounded_drain():
    src = (REPO / "server.py").read_text(encoding="utf-8")
    assert "ShutdownDrain(" in src
    assert "signal.SIGTERM" in src and "signal.SIGINT" in src
    assert "httpd.shutdown" in src
    assert "RunAdmissionClosed" in src


def test_every_stream_registration_edge_checks_the_fence():
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
    registrations = src.count("STREAMS[stream_id] = stream")
    assert registrations >= 4
    assert src.count("assert_run_admission_open_locked()") >= registrations


def test_ctl_stop_waits_for_the_drain_budget():
    src = (REPO / "ctl.sh").read_text(encoding="utf-8")
    assert "HERMES_WEBUI_SHUTDOWN_DRAIN_SECONDS" in src
    assert "for i in {1..50}" not in src
