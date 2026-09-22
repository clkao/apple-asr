"""SPEC.md §9 test 7 — backpressure.

A burst that exceeds `queue_size` drops the *oldest* pending events, counts them
in `stats.dropped`, and warns exactly once (per Stream).
"""

from __future__ import annotations

import time

import pytest
from support import drain, speech

from apple_asr import Partial

BURST = 300


def test_burst_beyond_queue_size_drops_oldest_with_one_warning(fake):
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "start",
                    "count": BURST,
                    "emit": {"type": "partial", "text": "p", "range": [0.0, 0.1], "runs": []},
                },
            ],
        }
    )
    with pytest.warns(RuntimeWarning) as record:
        st = fake.stream(queue_size=8)
        st.push(speech(0.2))
        time.sleep(0.5)  # let the whole burst land unread
        st.close()
    events = drain(st)

    texts = [e.text for e in events if isinstance(e, Partial)]
    assert len(texts) <= 9, f"queue_size=8 must bound the queue, got {len(texts)}"
    assert f"p{BURST - 1}" in texts, "the newest events are kept (drop-oldest)"
    assert "p0" not in texts, "the oldest events are the ones dropped"
    assert st.stats.dropped == BURST - len(texts)
    assert st.stats.partials == BURST, "stats count what the shim sent, not what survived"

    drops = [w for w in record if "dropping oldest" in str(w.message)]
    assert len(drops) == 1, f"exactly one warning per Stream; got {len(drops)}"
    assert "queue_size" in str(drops[0].message), "the warning must say how to fix it"


def test_no_drop_when_the_caller_keeps_up(fake):
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "frames",
                    "value": 1600,
                    "count": 20,
                    "emit": {"type": "partial", "text": "q", "range": [0.0, 0.1], "runs": []},
                },
            ],
        }
    )
    with fake.stream(queue_size=64) as st:
        st.push(speech(0.3))
        events = []
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(
            [e for e in events if isinstance(e, Partial)]
        ) < 20:
            events.extend(st.poll(0.05))
        st.close()
        events.extend(drain(st))
    assert st.stats.dropped == 0
    assert len([e for e in events if isinstance(e, Partial)]) == 20
