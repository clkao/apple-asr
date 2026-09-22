"""The replay driver (`apple_asr.replay`).

The real-time, real-shim version lives in the integration tests; here the same
driver is exercised against the scripted fake shim, so it is covered on any OS.
"""

from __future__ import annotations

import numpy as np
import pytest
from support import AUDIO_ZH_LONG, silence, speech

from apple_asr.replay import ReplayResult, load_wav, replay, segment


def synth() -> np.ndarray:
    """Speech, a pause, then speech (the live VAD pattern)."""
    return np.concatenate([speech(2.0), silence(0.4), speech(2.0)])


def test_segment_finds_the_pause():
    spans = segment(synth())
    assert [k for k, _, _ in spans] == ["speech", "silence", "speech"]
    speech_span, pause, tail = spans
    assert 1.9 <= speech_span[1] + (speech_span[2] - speech_span[1]) <= 2.1
    assert 0.3 <= pause[2] - pause[1] <= 0.5
    assert tail[2] == pytest.approx(4.4, abs=0.1)


def test_segment_merges_runs_shorter_than_the_minimums():
    audio = np.concatenate([speech(1.0), silence(0.02), speech(1.0)])
    spans = segment(audio, min_silence_s=0.15)
    assert [k for k, _, _ in spans] == ["speech"]


def test_replay_drives_the_pause_pattern_and_measures_latency(fake):
    fake.scenario({"mode": "emulate", "partial_every_frames": 16_000})
    st = fake.stream(pause_commit=0.08)
    try:
        result = replay(synth(), pace=0.0, stream=st, locale="zh-TW")
    finally:
        st.close()

    assert isinstance(result, ReplayResult)
    assert len(result.pause_commits) == 1
    commit = result.pause_commits[0]
    assert commit.audio_t == pytest.approx(2.0, abs=0.2)
    # pace=0 observes the pause for 0.3 s; the fake endpointer commits well inside.
    assert commit.latency_s is not None and commit.latency_s <= 0.3
    assert not result.misses
    assert result.finals, "the replay must end with a flushed final"
    assert result.partials, "streaming mode emits volatile partials"


def test_replay_result_summary_reports_the_measurements(fake):
    fake.scenario({"mode": "emulate"})
    st = fake.stream(pause_commit=0.08)
    try:
        result = replay(synth(), pace=0.0, stream=st)
    finally:
        st.close()
    summary = result.summary()
    assert "finals=" in summary and "max_pause_latency=" in summary
    assert result.max_pause_latency is not None
    assert result.ended_reason is None, "the session was not closed by the driver"


def test_load_wav_reads_the_bundled_clip():
    audio, rate = load_wav(str(AUDIO_ZH_LONG))
    assert rate == 16_000
    assert audio.dtype == np.float32
    assert len(audio) / rate == pytest.approx(31.55, abs=0.01)
