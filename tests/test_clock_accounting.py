"""SPEC.md amendment 4 — exact clock accounting (the `Final.start` clamp is gone).

The client's session clock is the caller's timeline (pushed audio + the pause
duration the caller reports), while the shim's is its frame timeline; the two
differ by the synthesized silence while a pause is open. `SessionClock` now maps
the two with a *boundary-anchored piecewise-linear* mapping refreshed at every
input boundary (`push`, `pause_start`, `pause_end`), so a final that lands
*during* a pause keeps the offset of its own audio instead of the stale one the
already-written silence implies: mapped finals are contiguous by construction and
`Transport._make_final` needs no clamp, hence no nudge of `Final.start` (order 2's
decision D-4 was "keep the clamp"; this is the fix that made it unnecessary).
"""

from __future__ import annotations

import time

import pytest
from support import SAMPLE_RATE, speech

from apple_asr import Final

PAUSE_COMMIT = 0.08
PREROLL_FRAMES = 1600  # _PAUSE_PREROLL_MIN_S = 0.10 s at 16 kHz


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


def _wire_finals(fake) -> list[dict]:
    return [e for e in fake.state()["emitted"] if e["type"] == "final"]


def _drain_finals(st, timeout: float = 3.0, quiet: float = 0.25) -> list[Final]:
    """Every `Final` still queued (the fake endpointer commits once per pause)."""
    out: list[Final] = []
    deadline = time.monotonic() + timeout
    quiet_until: float | None = None
    while time.monotonic() < deadline:
        got = [e for e in st.poll(0.05) if isinstance(e, Final)]
        if got:
            out.extend(got)
            quiet_until = time.monotonic() + quiet
        elif quiet_until is not None and time.monotonic() >= quiet_until:
            break
    return out


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
    wire_finals = _wire_finals(fake)
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


def test_a_final_that_lands_mid_pause_maps_to_the_shim_range_exactly(fake):
    """A final published during a pause is mapped with no nudge at all.

    The pre-roll (and, later, the pump) has already written synthesized silence
    that the caller reports only at `pause_end`, so the drift at arrival is
    already stale: under order 1's global drift this final's end mapped to
    0.4 s instead of 0.5 s, and the transport clamped the *start* of the next
    final to hide the resulting overlap. With the pause-onset anchor in force the
    mapped range equals the shim's own wire range, byte for byte.
    """
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "frames",
                    "value": PREROLL_FRAMES + 8000,  # 0.6 s: push 0.5 + pre-roll
                    "emit": {"type": "final", "text": "mid", "range": [0.0, 0.5], "runs": []},
                }
            ],
        }
    )
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    final: Final | None = None
    try:
        st.push(speech(0.5))
        st.pause_start()
        # Non-trivial: the shim only emits once it has consumed 9600 frames
        # (0.6 s) while the session clock still reads 0.5 s — the pre-roll is
        # unreported, so the arrival-instant drift is already off by 0.1 s.
        assert st.audio_time == 0.5
        final = _wait_final(st)
    finally:
        st.close()
    assert final is not None and final.text == "mid"
    wire = _wire_finals(fake)[0]
    assert [final.start, final.end] == list(wire["range"]), (final, wire)
    assert [final.start, final.end] == [0.0, 0.5]
    # ... and no clamp was involved: the width is the shim's own width.
    assert final.end - final.start == wire["range"][1] - wire["range"][0]


def test_consecutive_mid_pause_finals_are_exact_without_a_clamp(fake):
    """The engineered drift-divergence case: two finals, two pause offsets.

    Final 1 fires right after the first 0.10 s pre-roll (its own pause's offset);
    final 2 fires after a whole pause has since advanced the session clock, so
    the arrival-instant drifts differ. Under order 1's global drift final 2's
    start mapped to 0.2 (a 0.2 s overlap) and the clamp pulled it up to
    `final1.end`; anchored piecewise the two map contiguously and exactly.
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
                    "value": 32800,  # after pause_end(1.0) + push + pre-roll
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

        st.push(speech(0.5))
        st.pause_start()  # pre-roll -> 33600, then the pump
        second = _wait_final(st)
        assert second is not None and second.text == "f2"
        finals.append(second)
        st.pause_end(1.0)
    finally:
        st.close()

    _assert_no_overlap(finals)
    wire = _wire_finals(fake)
    assert [[f.start, f.end] for f in finals] == [list(w["range"]) for w in wire]
    assert [[f.start, f.end] for f in finals] == [[0.0, 0.5], [0.5, 1.0]]
    # The clamp's old job — making these two touch — now holds exactly, and the
    # second final's start is the shim's own range start (0.5), not a nudge.
    assert second.start == first.end == wire[1]["range"][0] == 0.5


def test_a_push_with_a_pause_open_keeps_the_pause_onset_anchor(fake):
    """Audio arriving behind unreported silence must not move the anchor.

    A caller whose VAD resumed speech without calling `pause_end` pushes while the
    pause's synthesized silence is already on the shim's clock but not yet on the
    session clock. Moving the anchor to that point would put the mapping on the
    stale offset `session - written` and pull every timestamp back by the pumped
    silence; the pause-onset anchor stays in force instead, so the audio the
    commit covers maps where it really was.
    """
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "frames",
                    "value": 20_000,  # after the push + pre-roll + pump + push
                    "emit": {"type": "final", "text": "resumed", "range": [0.0, 0.5], "runs": []},
                }
            ],
        }
    )
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    final: Final | None = None
    try:
        st.push(speech(0.5))  # shim frame 8000
        st.pause_start()  # pre-roll -> 9600, then the pump
        time.sleep(0.3)  # unreported synthesized silence on the shim clock
        st.push(speech(0.5))  # speech resumed: this closes the pause bookkeeping
        final = _wait_final(st)
        st.pause_end(0.5)  # the caller reports the pause after all
        audio_time = st.audio_time
    finally:
        st.close()
    assert final is not None and final.text == "resumed"
    assert [final.start, final.end] == [0.0, 0.5], final
    # The reported pause still lands exactly on the session clock (1.0 s pushed
    # + 0.5 s reported), which is what the anchor rejoins at `pause_end`.
    assert audio_time == 1.5


def test_an_over_delivered_pause_is_attributed_to_the_session_clock(fake):
    """A pause held open past its reported duration keeps the clocks 1:1.

    The pump has written more silence than the caller later reports (`P > d`).
    That excess is real elapsed audio the shim consumed, so the session clock —
    which advances with consumed audio (SPEC.md §3.3) — gains it too, and every
    final maps 1:1 to the shim's own range. Shearing the excess off the timeline
    (advancing the session clock by `d` only) would instead leave the two clocks
    apart for the rest of the session, and the first final after the pause would
    map back over the one the pause already published. This is the regression
    test for how the excess is attributed.
    """
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    d = 0.5
    finals: list[Final] = []
    try:
        st.push(speech(0.5))
        st.pause_start()
        first = _wait_final(st)
        assert first is not None
        finals.append(first)
        time.sleep(1.2)  # hold the pause open: the pump over-delivers silence
        st.pause_end(d)
        # The endpointer latches per quiet run, so the long pause committed
        # exactly once and the drain after `pause_end` has nothing to collect.
        finals.extend(_drain_finals(st))
        st.push(speech(0.5))
        st.flush()
        finals.extend(_drain_finals(st))
        # Exactly the pause commit and the flush commit: one per pause.
        assert len(finals) == 2, [f.reason for f in finals]
        assert [f.reason for f in finals] == ["pause", "flush"], [f.reason for f in finals]
        audio_time = st.audio_time
    finally:
        st.close()

    state = fake.state()
    assert len(state["quiet_commits"]) == 1, state["quiet_commits"]
    consumed_s = state["frames"] / SAMPLE_RATE
    assert consumed_s > 1.5, (
        f"the pause did not over-deliver ({consumed_s} shim seconds vs the 1.5 s "
        f"of pushed audio plus the reported {d} s); the test would be vacuous"
    )
    # The session clock followed the consumed audio, not the reported duration.
    assert audio_time == pytest.approx(consumed_s, abs=1e-9)
    assert audio_time > 1.5
    assert first.end >= 0.5  # a commit really did land during the pause
    # Every final maps to the shim's own wire range, exactly — no nudge, no
    # compression (the wire ranges are chunk-granular in emulate mode).
    wire_ranges = [tuple(w["range"]) for w in _wire_finals(fake)]
    for final in finals:
        assert any(
            abs(final.start - w[0]) <= 1e-9 and abs(final.end - w[1]) <= 1e-9
            for w in wire_ranges
        ), (final, wire_ranges)
    assert all(f.end <= audio_time + 1e-9 for f in finals), [
        (f.start, f.end, audio_time) for f in finals
    ]
    _assert_no_overlap(finals)
