"""SPEC.md amendment 4 — exact clock accounting, and why `Final.start` is clamped.

The client's session clock is the caller's timeline (pushed audio + the pause
duration the caller reports), while the shim's is its frame timeline; the two
differ by the synthesized silence while a pause is open. A final that lands
*during* a pause is mapped with the drift of its arrival instant, so two finals
can be mapped with different drifts — which is what the clamp in
`Transport._make_final` repairs. These two tests are the regression tests for
that (see ORDER2-REPORT.md decision D-4).
"""

from __future__ import annotations

import time

from support import SAMPLE_RATE, speech

from apple_asr import Final

PAUSE_COMMIT = 0.08


def _wait_final(st, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in st.poll(0.02):
            if isinstance(event, Final):
                return event
    return None


def _assert_no_overlap(finals: list[Final]) -> None:
    for previous, following in zip(finals, finals[1:], strict=False):
        assert following.start >= previous.end - 1e-9, (
            f"final ranges overlap: {previous.end} > {following.start}"
        )
    for final in finals:
        assert final.start <= final.end


def test_final_ranges_never_overlap_in_normal_use(fake):
    """A plain push/pause/pause_end/flush cycle keeps the §3.3 invariant."""
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    finals: list[Final] = []
    try:
        for _ in range(3):
            st.push(speech(0.5))
            st.pause_start()
            final = _wait_final(st)
            assert final is not None, "no commit landed during the pause"
            finals.append(final)
            st.pause_end(0.5)
        st.push(speech(0.5))
        st.flush()
        flushed = _wait_final(st)
        if flushed is not None:
            finals.append(flushed)
    finally:
        st.close()

    _assert_no_overlap(finals)
    wire_finals = [e for e in fake.state()["emitted"] if e["type"] == "final"]
    # Frame-exact shim ranges tile the timeline: [0, 8000], [8000, 16000], ...
    starts = [round(f["range"][0] * SAMPLE_RATE) for f in wire_finals]
    assert starts == sorted(starts), starts
    assert starts[0] == 0
    for previous, following in zip(wire_finals, wire_finals[1:], strict=False):
        assert following["range"][0] <= previous["range"][1] + 1
    # Mapped timestamps stay on the session clock: monotone and within it.
    assert all(f.end <= st.audio_time + 1e-6 for f in finals), [
        (f.start, f.end, st.audio_time) for f in finals
    ]
    duration = st.audio_time
    assert duration == 3 * (0.5 + 0.5) + 0.5, duration


def test_overlap_is_repaired_when_the_drift_diverges_between_finals(fake):
    """The documented case for the clamp: two finals mapped with different drift.

    Final 1 arrives right after the pause pre-roll (drift = the pre-roll), final 2
    only after the pause has been open ~0.25 s longer (drift = pre-roll + pumped
    silence). Without the clamp, final 2's start maps to `final1.end - 0.2`.
    """
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "frames",
                    "value": 9600,  # right after the 0.10 s pre-roll
                    "emit": {"type": "final", "text": "f1", "range": [0.0, 0.5], "runs": []},
                },
                {
                    "on": "frames",
                    "value": 32800,  # ~0.25 s of pump later: drift has moved
                    "emit": {"type": "final", "text": "f2", "range": [0.5, 1.0], "runs": []},
                },
            ],
        }
    )
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    finals: list[Final] = []
    try:
        st.push(speech(0.5))  # shim frame 8000
        st.pause_start()  # pre-roll -> 9600
        first = _wait_final(st)
        assert first is not None and first.text == "f1"
        finals.append(first)
        st.pause_end(1.0)

        st.push(speech(0.5))  # frame 24000 -> 28000
        st.pause_start()  # pre-roll -> 29600, then the pump
        second = _wait_final(st)
        assert second is not None and second.text == "f2"
        finals.append(second)
        st.pause_end(1.0)
    finally:
        st.close()

    _assert_no_overlap(finals)
    second = finals[1]
    assert abs(second.start - finals[0].end) < 1e-9, (
        "the second final was clamped up to the previous end (the documented "
        f"clamp), got {second.start} vs {finals[0].end}"
    )
    # The clamp is visible in the mapped timeline: without it, `second.start`
    # would be 0.2 (0.2 s of pumped silence of extra drift) and overlap.
