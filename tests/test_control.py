"""SPEC.md §9 tests 4 and 5 — flush semantics and close/EOF."""

from __future__ import annotations

import pytest
from support import drain, speech, types

from apple_asr import Ended, Final, SessionClosed


def test_flush_through_emits_exactly_one_commit_and_flush_reason(fake):
    fake.scenario({"mode": "emulate", "no_endpointer": True, "partial_every_frames": 4000})
    st = fake.stream()
    try:
        st.push(speech(1.0))
        st.flush(through_s=0.5)
        st.flush(through_s=0.75)
    finally:
        st.close()
    events = drain(st)

    wire = fake.state()["emitted"]
    commits = [e for e in wire if e["type"] == "commit" and e.get("reason") == "flush"]
    assert [c["through"] for c in commits] == [0.5, 0.75], "one commit ack per finalize"
    finals = [e for e in events if isinstance(e, Final)]
    assert [f.reason for f in finals[:2]] == ["flush", "flush"]
    assert [round(f.end, 3) for f in finals[:2]] == [0.5, 0.75]


def test_flush_without_through_uses_the_cursor(fake):
    fake.scenario({"mode": "emulate", "no_endpointer": True, "partial_every_frames": 4000})
    st = fake.stream()
    try:
        st.push(speech(1.0))
        st.flush()
    finally:
        st.close()
    events = drain(st)

    finalizes = [c for c in fake.state()["commands"] if c.get("cmd") == "finalize"]
    assert finalizes and all("through" not in c for c in finalizes), (
        "flush() must not invent a through time"
    )
    final = next(e for e in events if isinstance(e, Final))
    assert (round(final.start, 3), round(final.end, 3)) == (0.0, 1.0)
    assert final.reason == "flush"


def test_close_is_idempotent_and_drains_pending_finals(fake):
    fake.scenario({"mode": "emulate", "no_endpointer": True, "partial_every_frames": 4000})
    st = fake.stream()
    st.push(speech(0.5))
    st.close()
    st.close()  # idempotent: no error, no second child, no second End
    events = drain(st)

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1, "close() must drain the pending final"
    assert finals[0].reason == "eof"
    assert events[-1] == Ended(reason="closed")
    assert types(events).count("Ended") == 1, "exactly one terminal event"
    assert st.stats.finals == 1


def test_input_after_close_raises_session_closed(fake):
    fake.scenario({"mode": "emulate", "no_endpointer": True})
    st = fake.stream()
    st.close()
    with pytest.raises(SessionClosed):
        st.push(speech(0.1))
    with pytest.raises(SessionClosed):
        st.prepare()
    drain(st)


def test_child_exit_without_ended_yields_eof(fake):
    """A shim that exits 0 after stdin EOF ends the session as `eof`."""
    fake.scenario(
        {
            "mode": "emulate",
            "emit_partials": False,
            "exit_after_frames": 8000,
            "exit_code": 0,
        }
    )
    st = fake.stream()
    st.push(speech(0.5))
    events = drain(st)
    ended = [e for e in events if isinstance(e, Ended)]
    assert ended and ended[0].reason == "eof"
