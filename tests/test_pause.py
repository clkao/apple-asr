"""SPEC.md §9 test 3 — pause synthesis and the commit-latency regression.

The fake endpointer emulates the measured behaviour: a lump of silence at the
pause onset does not publish a final — more audio must keep arriving — which is
why the client's silence pump exists (order-1 report §2.3: a pre-roll alone gave
>1.2 s; the pump gives 0.05-0.15 s).
"""

from __future__ import annotations

import time

from support import SAMPLE_RATE, speech

from apple_asr import Final

PAUSE_COMMIT = 0.08
#: Generous, because it must hold on a loaded CI runner: the pump delivers 0.05 s
#: of audio every 0.0625 s, so the real latency is one or two ticks.
EPSILON = 0.25


def _open_pause_and_wait_for_final(st, timeout: float = 2.0):
    """`pause_start()` then observe until a Final lands; returns (latency, final)."""
    start = time.monotonic()
    st.pause_start()
    while time.monotonic() - start < timeout:
        for event in st.poll(0.02):
            if isinstance(event, Final):
                return time.monotonic() - start, event
    return None, None


def test_pause_commit_lands_within_pause_commit_plus_epsilon(fake):
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    try:
        st.push(speech(0.5))
        latency, final = _open_pause_and_wait_for_final(st)
        assert final is not None, "no Final landed during the pause"
        assert latency <= PAUSE_COMMIT + EPSILON, f"pause->final latency {latency:.3f}s"
        assert final.reason == "pause"
        st.pause_end(0.5)
    finally:
        st.close()

    state = fake.state()
    assert state["pause_commit"] == PAUSE_COMMIT
    # The quiet run starts at the end of the pushed speech (frame 8000), quantized
    # up to the shim's read-chunk boundary (it reads 4096 B = 1024 frames at a time).
    assert len(state["quiet_commits"]) == 1
    quiet_frame = state["quiet_commits"][0]["frame"]
    assert 8000 <= quiet_frame <= 8000 + 2048, quiet_frame


def test_a_pause_writes_exactly_its_reported_duration_as_silence(fake):
    """`pause_end(d)` delivers exactly `d` seconds of synthesized silence."""
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    try:
        st.push(speech(0.5))
        before = st.stats.bytes
        _, final = _open_pause_and_wait_for_final(st)
        assert final is not None
        time.sleep(0.05)
        st.pause_end(0.5)
        written_frames = (st.stats.bytes - before) // 4
    finally:
        st.close()

    assert written_frames == round(0.5 * SAMPLE_RATE), "pause silence must be frame-exact"
    state = fake.state()
    # The shim agrees: every frame the client wrote is a frame it consumed.
    assert state["frames"] == st.stats.bytes // 4


def test_the_pump_delivers_silence_while_the_pause_is_open(fake):
    """Silence keeps arriving during the pause (not just a lump at the onset)."""
    fake.scenario({"mode": "emulate", "partial_every_frames": 4000})
    st = fake.stream(pause_commit=PAUSE_COMMIT)
    try:
        st.push(speech(0.5))
        before = st.stats.bytes  # BEFORE pause_start, so the pre-roll counts too
        st.pause_start()
        time.sleep(0.5)
        pumped_s = (st.stats.bytes - before) / 4 / SAMPLE_RATE
        st.pause_end(1.0)  # the caller reports the true duration; the top-up is exact
        total_s = (st.stats.bytes - before) / 4 / SAMPLE_RATE
        st.close()
    finally:
        if not st._closed:
            st.close()

    # 0.10 s pre-roll + ~0.8x for 0.5 s of wall time.
    assert 0.15 <= pumped_s <= 0.85, f"pause delivered {pumped_s:.3f}s in 0.5s"
    assert total_s == 1.0, "pause_end(d) must top up to exactly d seconds"
