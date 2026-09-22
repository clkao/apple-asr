"""SPEC.md §9 test 10 — macOS 26 integration: `--file` on the bundled clip.

Skipped outside macOS 26+ (needs the real shim and the Speech framework); it must
NOT skip on a macOS 26 machine — a missing shim there is a failure. Excluded from
the fast suite by the `integration` marker.

Cadence note: `--file` is batch mode, so the analyzer commits at file boundaries
(1-3 finals for this clip) instead of once per pause. The golden fixture
(`tests/golden/zh_long_ideal.jsonl`, 12 streaming finals) is therefore compared
in `test_integration_replay.py`, which drives the same audio through the live
push/pause pattern — the apples-to-apples path (MEASUREMENTS.md).
"""

from __future__ import annotations

import json
import subprocess

import pytest
from support import AUDIO_ZH_LONG, macos26_available

from apple_asr import shim_info

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not macos26_available(),
        reason="needs macOS 26+ and the Speech framework (skipped on CI's ubuntu job)",
    ),
]

LOCALE = "zh-TW"


@pytest.fixture(scope="module")
def file_finals():
    """Run the real shim in `--file` mode over the bundled clip."""
    path = shim_info().path  # resolves/builds, and fails loudly if it cannot
    assert AUDIO_ZH_LONG.exists(), f"bundled audio fixture missing: {AUDIO_ZH_LONG}"
    proc = subprocess.run(
        [path, "--file", str(AUDIO_ZH_LONG), "--locale", LOCALE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"shim --file failed:\n{proc.stderr[-2000:]}"
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    hello = [e for e in events if e["type"] == "hello"]
    assert hello, "the first stdout line must be hello"
    return [e for e in events if e["type"] == "final"], proc.stderr


def test_file_mode_commits_with_word_runs(file_finals):
    finals, stderr = file_finals
    assert len(finals) >= 1, f"no finals from --file; stderr:\n{stderr[-2000:]}"
    assert all(f["runs"] for f in finals), "every final must carry word runs"
    words = [word for final in finals for word in final["runs"]]
    assert len(words) >= 50, f"expected a full transcript, got {len(words)} word runs"
    assert all(len(word) == 4 for word in words), "confidence is the 4th run element"


def test_file_mode_timestamps_are_monotonic_and_never_overlap(file_finals):
    finals, _ = file_finals
    for final in finals:
        start, end = final["range"]
        assert start <= end, final
        assert final["reason"] in ("pause", "interval", "flush", "eof"), final
    for previous, following in zip(finals, finals[1:], strict=False):
        assert following["range"][0] >= previous["range"][1] - 1e-6, (previous, following)
    words = [word for final in finals for word in final["runs"]]
    for previous, following in zip(words, words[1:], strict=False):
        assert following[1] >= previous[2] - 1e-6, (previous, following)
        assert 0.0 <= following[3] <= 1.0, following
