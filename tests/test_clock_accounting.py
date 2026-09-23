"""SPEC.md amendment 4/5 — the caller's declared timeline, and the clamp it needs.

The session clock is the **caller's declared timeline**: pushed audio plus the
pause durations reported at `pause_end(d)`. The shim's frame timeline is what the
client actually fed it, which is longer whenever a pause is held open past its
reported `d` (the pre-roll and the pump keep writing while a caller waits for the
transcriber's final — WhisperLiveKit holds every pause up to ~0.3 s). That excess
is compressed out of the session timeline at `pause_end`, so later timestamps are
not ahead by the accumulated over-delivery.

`SessionClock` maps shim time onto it with a *boundary-anchored piecewise-linear*
mapping: the anchor is refreshed at every input boundary, but only `push()` and
`pause_end(d)` (which re-anchors unconditionally) may move it — while a pause is
open the *pause-onset* anchor stays in force, so a final that lands during a pause
keeps the offset of its own audio instead of the stale one the already-written
silence implies. Compression is retroactive (a shim time already published during
the pause can map earlier on the post-pause line), so `Transport._make_final`
clamps `Final.start` up to the previous final's end — bounded by the
over-delivery of that one pause — and the published ranges stay monotone. The
over-held-pause test below is the regression test for that clamp.
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


def test_consecutive_mid_pause_finals_stay_exact_and_contiguous(fake):
    """A pause that is *not* over-held leaves the mapping exactly where it was.

    Final 1 fires right after the first 0.10 s pre-roll (its own pause's offset);
    final 2 fires after a whole pause has since advanced the session clock. Both
    pauses report more than the pump wrote, so the top-up makes the written
    silence exactly the reported duration and `pause_end` re-anchors onto the same
    1:1 line: the two map contiguously and exactly, with no clamp nudge.
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
    # Neither pause over-delivered, so nothing was compressed and nothing needed
    # clamping: the second final's start is the shim's own range start.
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


def test_a_pause_held_past_its_reported_duration_keeps_the_declared_clock(fake):
    """The reviewed case: push 0.4 s, hold the pause open, report 0.1 s.

    A live caller (WhisperLiveKit holds every pause up to ~0.3 s while it waits
    for the transcriber's final) has already had the pre-roll and several pump
    ticks written when `pause_end(0.1)` lands: the shim consumed ~0.7 s while the
    caller's own cursor is 0.5 s (0.4 pushed + 0.1 reported). The session clock
    must read the caller's 0.5 s — not the 0.7 s the old
    `max(synthesized, declared)` rule advanced it by — and the finals must stay
    monotone, non-overlapping and on that declared timeline.
    """
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    finals: list[Final] = []
    try:
        st.push(speech(0.4))
        st.pause_start()
        time.sleep(0.5)  # hold the pause open: the pump over-delivers silence
        st.pause_end(0.1)
        after_pause = st.audio_time
        assert after_pause == pytest.approx(0.5, abs=1e-9), after_pause
        st.push(speech(0.2))
        st.flush()
        finals.extend(_drain_finals(st))
        audio_time = st.audio_time
    finally:
        st.close()

    state = fake.state()
    consumed_s = state["frames"] / SAMPLE_RATE
    assert consumed_s > 0.75, (
        f"the pause did not over-deliver ({consumed_s} shim seconds vs the 0.5 s "
        f"of pushed audio plus the reported 0.1 s); the test would be vacuous"
    )
    # The declared timeline: 0.4 pushed + 0.1 reported + 0.2 pushed.
    assert audio_time == pytest.approx(0.7, abs=1e-9), audio_time
    assert audio_time < consumed_s
    assert finals, "no finals were collected"
    _assert_no_overlap(finals)
    assert all(f.end <= audio_time + 1e-6 for f in finals), [
        (f.start, f.end, audio_time) for f in finals
    ]
    assert any(f.text.startswith("final-") for f in finals)


def test_repeated_short_pauses_do_not_accumulate_over_delivery(fake):
    """5 x (push 0.2 s, pause reported 0.05 s) == 1.25 s exactly, no accumulation.

    The declared 0.05 s is shorter than the 0.10 s pre-roll, so every one of the
    five pauses over-delivers at least 0.05 s of silence; if the session clock
    counted what was written instead of what was declared, the error would
    accumulate across the pauses (1.25 s declared vs 1.75 s consumed, measured) —
    the bug a live caller with many short pauses sees.
    """
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    finals: list[Final] = []
    try:
        for _ in range(5):
            st.push(speech(0.2))
            st.pause_start()
            time.sleep(0.1)  # a pause is held open on the way to the next chunk
            st.pause_end(0.05)
        declared = st.audio_time
        st.flush()
        finals.extend(_drain_finals(st))
    finally:
        st.close()

    consumed_s = fake.state()["frames"] / SAMPLE_RATE
    assert declared == pytest.approx(5 * (0.2 + 0.05), abs=1e-9), declared
    assert consumed_s > declared + 0.4, (consumed_s, declared)
    _assert_no_overlap(finals)
    assert all(f.end <= declared + 1e-6 for f in finals), [
        (f.start, f.end, declared) for f in finals
    ]


def test_an_over_held_pause_with_a_mid_pause_final_stays_monotone(fake):
    """The engineered case: the `Final.start` clamp is what keeps ranges ordered.

    One final is published *during* the pause (mapped on the pause-onset line,
    1:1) and one *after* it (mapped on the post-pause line, shifted back by the
    over-delivered 0.05 s+). The shim's ranges tile its timeline, so the second
    final's start is a shim time that compression now maps *before* the first
    final's end: without the clamp the published ranges overlap. With it the
    second final starts exactly where the first ended — the bounded nudge the
    clamp exists for (bounded by that pause's over-delivery) — and its end stays
    on the caller's declared timeline.
    """
    fake.scenario(
        {
            "mode": "scripted",
            "steps": [
                {
                    "on": "frames",
                    "value": PREROLL_FRAMES + 6400,  # push 0.4 + the 0.10 s pre-roll
                    "emit": {"type": "final", "text": "mid", "range": [0.0, 0.4], "runs": []},
                },
                {
                    "on": "frames",
                    "value": 20_000,  # after pause_end(0.05) + the 0.6 s push
                    "emit": {"type": "final", "text": "after", "range": [0.4, 0.8], "runs": []},
                },
            ],
        }
    )
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    finals: list[Final] = []
    try:
        st.push(speech(0.4))
        st.pause_start()
        mid = _wait_final(st)
        assert mid is not None and mid.text == "mid", mid
        finals.append(mid)
        time.sleep(0.4)  # hold it open past the 0.05 s the caller reports
        st.pause_end(0.05)
        st.push(speech(0.6))
        after = _wait_final(st)
        assert after is not None and after.text == "after", after
        finals.append(after)
        audio_time = st.audio_time
    finally:
        st.close()

    consumed_s = fake.state()["frames"] / SAMPLE_RATE
    declared = 0.4 + 0.05 + 0.6
    assert audio_time == pytest.approx(declared, abs=1e-9), audio_time
    assert consumed_s > declared + 0.05, (
        f"the pause did not over-deliver ({consumed_s} shim seconds vs the {declared} s "
        f"declared); the test would be vacuous"
    )
    # The mid-pause final is untouched: mapped on the pause-onset line, 1:1.
    assert [mid.start, mid.end] == [0.0, 0.4], mid
    _assert_no_overlap(finals)
    # What makes that hold: the clamp pinned the second final's start to the end
    # already published, instead of letting compression map it back behind it.
    assert after.start == mid.end, (after, mid)
    assert after.end <= audio_time + 1e-6, (after, audio_time)
    assert after.end > after.start, after


def test_push_only_sessions_are_unchanged(fake):
    """Requirement 3: no pauses means no divergence, so no timestamps move.

    A caller that feeds continuous audio (silence included) never over-delivers
    anything: the session clock is exactly the audio pushed, and every reported
    range is the shim's own wire range, byte for byte (the mapping is identity).
    """
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    finals: list[Final] = []
    try:
        st.push(speech(0.5))
        st.push(speech(0.5))
        st.flush()
        finals.extend(_drain_finals(st))
        audio_time = st.audio_time
    finally:
        st.close()

    assert audio_time == pytest.approx(1.0, abs=1e-9), audio_time
    assert finals, "no finals were collected"
    # The shim emits one more zero-width final at `close` (after this drain), so
    # compare against the wire's prefix.
    wire_ranges = [list(w["range"]) for w in _wire_finals(fake)]
    assert [[f.start, f.end] for f in finals] == wire_ranges[: len(finals)], (
        finals,
        wire_ranges,
    )
    assert all(f.end <= audio_time + 1e-6 for f in finals), [
        (f.start, f.end, audio_time) for f in finals
    ]
