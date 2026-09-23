"""SPEC.md §9 test 6 — failure paths.

Nonzero exit mid-session -> `BackendError` with the stderr tail; a malformed
line or an unknown event type -> warning, not a crash; a silent shim that is fed
audio -> the feed-rate-aware silent-backend guard raises instead of hanging.
"""

from __future__ import annotations

import threading
import time
import warnings

import pytest
from support import drain, speech, wait_for

from apple_asr import BackendError, Ended, Partial

SILENT_TIMEOUT = 0.6


def _collect(st, sink: list) -> bool:
    """Poll once, appending to `sink`; True when something arrived."""
    got = st.poll(0.02)
    sink.extend(got)
    return bool(got)


def test_nonzero_exit_mid_session_raises_backend_error_with_stderr_tail(fake):
    fake.scenario(
        {
            "mode": "emulate",
            "emit_partials": False,
            "exit_after_frames": 8000,
            "exit_code": 3,
            "stderr": ["crash: exploding mid-session", "crash: boom"],
        }
    )
    st = fake.stream()
    st.push(speech(0.5))
    deadline = time.monotonic() + 5.0
    with pytest.raises(BackendError) as excinfo:
        while time.monotonic() < deadline:
            st.poll(0.05)
    message = str(excinfo.value)
    assert "status 3" in message
    assert "crash: exploding mid-session" in message, "the stderr tail must survive"
    assert "crash: boom" in message


def test_malformed_and_unknown_lines_warn_but_do_not_crash(fake):
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "frames",
                    "value": 1600,
                    "emit": {"type": "partial", "text": "alive", "range": [0.0, 0.2]},
                },
                {"on": "frames", "value": 3200, "raw": "this is not JSON at all"},
                {"on": "frames", "value": 4800, "raw": '{"type":"event_from_the_future"}'},
                {
                    "on": "frames",
                    "value": 6400,
                    "emit": {"type": "partial", "text": "still alive", "range": [0.0, 0.4]},
                },
            ],
        }
    )
    # simplefilter("always"): assert both warnings even if an earlier test
    # already tripped Python's per-location "default" dedup.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(RuntimeWarning) as record:
            st = fake.stream()
            events: list = []
            for _ in range(2):
                st.push(speech(0.2))
                assert wait_for(lambda: _collect(st, events), timeout=3.0), (
                    "a partial should have arrived"
                )
            st.close()
            events.extend(drain(st))

    messages = [str(w.message) for w in record]
    assert any("malformed JSON" in m for m in messages), messages
    assert any("unknown event type" in m for m in messages), messages
    texts = [e.text for e in events if isinstance(e, Partial)]
    assert texts == ["alive", "still alive"], "the session must keep working"


def test_silent_backend_guard_fires_when_fed_at_real_time(fake):
    fake.scenario({"mode": "silent"})
    st = fake.stream(silent_timeout=SILENT_TIMEOUT)

    stop = threading.Event()

    def feed() -> None:
        while not stop.is_set():
            try:
                st.push(speech(0.1))
            except Exception:
                return
            time.sleep(0.1)

    feeder = threading.Thread(target=feed, name="test-feeder", daemon=True)
    feeder.start()
    started = time.monotonic()
    try:
        with pytest.raises(BackendError) as excinfo:
            for _ in st.events():
                pass
    finally:
        stop.set()
        feeder.join(timeout=2.0)
        st.close()

    elapsed = time.monotonic() - started
    message = str(excinfo.value)
    assert elapsed < SILENT_TIMEOUT + 1.5, f"guard took {elapsed:.2f}s"
    assert "no output from the shim" in message
    assert "silent_timeout=0" in message, "the error must say how to disable the guard"


def test_guard_does_not_fire_on_a_fast_fed_burst(fake):
    """A burst is legitimate: 30 s of audio handed over at once stays quiet."""
    fake.scenario(
        {
            "mode": "emulate",
            "emit_partials": False,
            "no_endpointer": True,
            "partial_every_frames": 1_000_000,
        }
    )
    st = fake.stream(silent_timeout=SILENT_TIMEOUT)
    try:
        st.push(speech(30.0))
        deadline = time.monotonic() + SILENT_TIMEOUT + 1.0
        while time.monotonic() < deadline:
            assert st.poll(0.1) == [], "no events are expected from this scenario"
    finally:
        st.close()
        drain(st)


def test_guard_is_off_when_silent_timeout_is_zero(fake):
    fake.scenario({"mode": "silent"})
    st = fake.stream(silent_timeout=0.0)
    try:
        st.push(speech(0.2))
        time.sleep(SILENT_TIMEOUT + 0.3)
        assert st.poll(0.1) == [], "silent_timeout=0 disables the guard"
    finally:
        st.close()
        events = drain(st)
    assert any(isinstance(e, Ended) for e in events)
