"""SPEC.md §9 test 2 — event mapping.

Partials supersede, a final's `runs` become `Word`s with the documented clock
mapping, and `Final.reason` is propagated from the `commit` acknowledgement.
"""

from __future__ import annotations

import warnings

import pytest
from support import drain, speech, types

from apple_asr import Final, Partial, Word


def test_partials_supersede_and_runs_become_words(fake):
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "start",
                    "emit": {"type": "partial", "text": "a", "range": [0.0, 0.5], "runs": []},
                },
                {
                    "on": "start",
                    "emit": {"type": "partial", "text": "ab", "range": [0.0, 1.0], "runs": []},
                },
                {
                    "on": "start",
                    "emit": {
                        "type": "final",
                        "text": "ab",
                        "range": [0.0, 1.5],
                        "runs": [
                            ["a", 0.0, 0.5, 0.75],
                            ["b", 0.5, 1.0],
                        ],
                        "reason": "pause",
                    },
                },
            ],
        }
    )
    with fake.stream() as st:
        st.push(speech(1.5))  # clock: 1.5 s consumed, no pauses -> mapping is identity
    events = drain(st)  # after close: nothing is lost, and Ended is terminal

    assert types(events) == ["Partial", "Partial", "Final", "Ended"]
    first, second = events[0], events[1]
    assert isinstance(first, Partial) and isinstance(second, Partial)
    # Later partials supersede: the consumer replaces, never appends.
    assert (first.start, first.end) == (0.0, 0.5)
    assert (second.start, second.end) == (0.0, 1.0)
    assert second.text == "ab"

    final = events[2]
    assert isinstance(final, Final)
    assert (final.start, final.end) == (0.0, 1.5)
    assert final.words == (
        Word(text="a", start=0.0, end=0.5, confidence=0.75),
        Word(text="b", start=0.5, end=1.0, confidence=None),
    )
    assert st.stats.partials == 2
    assert st.stats.finals == 1
    assert st.stats.words == 2


def test_final_reason_is_propagated_from_the_commit_ack(fake):
    """A final with no wire `reason` takes the reason of its `commit` ack."""
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {"on": "start", "emit": {"type": "commit", "through": 1.0, "reason": "interval"}},
                {
                    "on": "start",
                    "emit": {"type": "final", "text": "x", "range": [0.0, 1.0], "runs": []},
                },
                {"on": "start", "emit": {"type": "commit", "through": 2.0, "reason": "flush"}},
                {
                    "on": "start",
                    "emit": {
                        "type": "final",
                        "text": "y",
                        "range": [1.0, 2.0],
                        "runs": [],
                        "reason": "pause",
                    },
                },
            ],
        }
    )
    with fake.stream() as st:
        st.push(speech(2.0))
    events = drain(st)

    finals = [e for e in events if isinstance(e, Final)]
    assert [f.reason for f in finals] == ["interval", "pause"]
    # A wire reason always wins over the pending ack reason.
    assert [f.text for f in finals] == ["x", "y"]


def test_confidence_is_optional_and_reaches_the_shim(fake):
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "start",
                    "emit": {
                        "type": "final",
                        "text": "q",
                        "range": [0.0, 1.0],
                        "runs": [["q", 0.0, 1.0]],
                    },
                },
            ],
        }
    )
    with fake.stream(confidence=False) as st:
        st.push(speech(1.0))
    events = drain(st)

    final = next(e for e in events if isinstance(e, Final))
    assert final.words[0].confidence is None
    assert fake.state()["confidence"] is False, "--no-confidence must reach the shim"


def test_runs_are_optional_and_unknown_event_types_are_ignored(fake):
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {"on": "start", "emit": {"type": "future_event", "payload": 1}},
                {
                    "on": "start",
                    "emit": {"type": "partial", "text": "no runs key", "range": [0.0, 0.2]},
                },
            ],
        }
    )
    # simplefilter("always"): the same warning must be assertable even if an
    # earlier test already tripped Python's per-location "default" dedup.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(RuntimeWarning, match="unknown event type"):
            with fake.stream() as st:
                st.push(speech(0.2))
            events = drain(st)

    partial = next(e for e in events if isinstance(e, Partial))
    assert partial.words == ()
    assert partial.text == "no runs key"
