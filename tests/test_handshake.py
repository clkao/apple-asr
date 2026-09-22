"""SPEC.md §9 test 1 — handshake.

Runs anywhere: the fake shim is a Python script and the cache is a tmp dir.
"""

from __future__ import annotations

import pytest
from support import drain, speech

from apple_asr import PROTOCOL_VERSION, BackendError, ProtocolMismatch


def test_handshake_accepts_matching_protocol(fake):
    fake.scenario({"protocol": PROTOCOL_VERSION})
    with fake.stream() as st:
        assert st.capabilities == (
            "context",
            "flush",
            "pause_commit",
            "volatile",
            "word_runs",
        )
        st.push(speech(0.2))
        st.close()
        events = drain(st)
    assert any(type(e).__name__ == "Ended" for e in events)


def test_handshake_rejects_protocol_mismatch_naming_both_versions(fake):
    fake.scenario({"protocol": 999})
    with pytest.raises(ProtocolMismatch) as excinfo:
        fake.stream()
    message = str(excinfo.value)
    assert "999" in message and str(PROTOCOL_VERSION) in message
    assert "build" in message, "the error must name the next command to run"


def test_handshake_rejects_negotiated_format_mismatch(fake):
    fake.scenario(
        {"format": {"sample_rate": 8000, "channels": 1, "common_format": "int16"}}
    )
    with pytest.raises(BackendError) as excinfo:
        fake.stream()
    assert "8000" in str(excinfo.value)


def test_handshake_rejects_a_first_line_that_is_not_hello(fake):
    fake.scenario({"first_line": {"type": "partial", "text": "boo", "range": [0, 1]}})
    with pytest.raises(BackendError) as excinfo:
        fake.stream()
    assert "hello" in str(excinfo.value)


def test_capabilities_are_logged_not_assumed(fake):
    """A shim that advertises fewer capabilities is accepted, visibly."""
    fake.scenario({"capabilities": ["word_runs"], "reporting_options": []})
    st = fake.stream()
    try:
        assert st.capabilities == ("word_runs",)
    finally:
        st.close()
        drain(st)


def test_lines_emitted_with_hello_are_not_lost(fake):
    """A fast shim can emit events in the same pipe read as `hello`.

    Regression test for an order-1 transport bug: the bulk read that fetched
    `hello` discarded everything after the first newline.
    """
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {"on": "start", "emit": {"type": "partial", "text": "p0", "range": [0, 1]}},
                {"on": "start", "emit": {"type": "partial", "text": "p1", "range": [0, 2]}},
                {"on": "start", "emit": {"type": "partial", "text": "p2", "range": [0, 3]}},
            ],
        }
    )
    with fake.stream() as st:
        st.push(speech(0.3))
    events = drain(st)
    texts = [e.text for e in events if type(e).__name__ == "Partial"]
    assert texts == ["p0", "p1", "p2"]
