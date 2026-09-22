"""SPEC.md §9 test 11 — macOS 26 integration: live-pattern replay.

Drives the bundled clip through the real shim at real time exactly as a live
pipeline does (active audio by `push`, pauses by `pause_start`/`pause_end`), via
`apple_asr.replay.replay` — the port of the prototype's `drive_live.py`. This is
the end-to-end version of test 3: every pause's commit must land within ~0.3 s of
its pause start.

It also compares cadence against the WhisperLiveKit golden fixture for the same
audio (`tests/golden/zh_long_ideal.jsonl`): 12 streaming finals, first commit
2.99 s, mean gap 2.97 s (MEASUREMENTS.md). Skipped outside macOS 26+.
"""

from __future__ import annotations

import json

import pytest
from support import GOLDEN_ZH_LONG, macos26_available

from apple_asr.replay import load_wav, replay

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not macos26_available(),
        reason="needs macOS 26+ and the Speech framework",
    ),
]

LOCALE = "zh-TW"
MAX_PAUSE_LATENCY_S = 0.3
#: How far our cadence may sit from the golden before the comparison is a lie.
FIRST_COMMIT_TOLERANCE_S = 1.5
FINALS_TOLERANCE = 4


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_ZH_LONG.exists():
        pytest.skip(f"golden fixture missing: {GOLDEN_ZH_LONG}")
    rows = [json.loads(line) for line in GOLDEN_ZH_LONG.read_text().splitlines() if line]
    finals = [r for r in rows if r["type"] == "transcription_final"]
    return {
        "finals": len(finals),
        "first_commit": finals[0]["audio_t"],
        "audio_t": finals[-1]["audio_t"],
    }


@pytest.fixture(scope="module")
def replay_result(zh_long_audio):
    # No `shim=` override and no fake cache here: this must resolve (or build) the
    # real shim. A failure is a real failure on macOS 26, not a skip.
    return replay(zh_long_audio, sample_rate=16_000, locale=LOCALE, pace=1.0)


def test_every_pause_commit_lands_within_a_third_of_a_second(replay_result):
    assert replay_result.pause_commits, "the VAD found no pauses in the bundled clip"
    assert not replay_result.misses, (
        "these pauses never committed: "
        f"{[(c.index, c.audio_t, c.pause_s) for c in replay_result.misses]}"
    )
    worst = replay_result.max_pause_latency
    assert worst is not None and worst <= MAX_PAUSE_LATENCY_S, (
        f"max pause->commit latency {worst:.3f}s exceeds {MAX_PAUSE_LATENCY_S}s; "
        f"latencies={[round(x, 3) for x in replay_result.commit_latencies]}"
    )


def test_replay_produces_a_full_transcript_with_word_runs(replay_result):
    finals = replay_result.finals
    assert len(finals) >= 8, f"only {len(finals)} finals: {replay_result.summary()}"
    assert all(f.words for f in finals), "every final must carry word runs"
    assert all(f.start <= f.end for f in finals)
    assert all(
        f.reason in ("pause", "flush", "eof") for f in finals
    ), [f.reason for f in finals]


def test_cadence_matches_the_wlk_golden_fixture(replay_result, golden):
    assert replay_result.ended_reason in ("closed", "eof")
    first = replay_result.first_commit_t
    assert first is not None
    assert abs(first - golden["first_commit"]) <= FIRST_COMMIT_TOLERANCE_S, (
        f"first commit {first:.2f}s vs golden {golden['first_commit']:.2f}s"
    )
    assert abs(len(replay_result.finals) - golden["finals"]) <= FINALS_TOLERANCE, (
        f"{len(replay_result.finals)} finals vs golden {golden['finals']}: "
        f"{replay_result.summary()}"
    )
    print(
        f"\ncadence: ours={replay_result.summary()} "
        f"golden(finals={golden['finals']}, first={golden['first_commit']:.2f}s)"
    )


def test_load_wav_matches_the_session_contract(zh_long_audio):
    audio, rate = load_wav(str(GOLDEN_ZH_LONG.parent.parent / "fixtures" / "audio" / "zh_long.wav"))
    assert rate == 16_000
    assert len(audio) == len(zh_long_audio)
    assert audio.dtype.name == "float32"
