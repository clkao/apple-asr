"""pytest fixtures for the apple-asr suite.

The whole non-integration suite runs against the scripted fake shim
(`tests/fixtures/fake_shim.py`) in a tmp cache directory, so it needs no macOS
26, no Speech framework, no mic, and no writable ``~/.cache``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/support.py

from support import AUDIO_ZH_LONG, FakeBackend, macos26_available  # noqa: E402


@pytest.fixture
def fake(tmp_path, monkeypatch) -> FakeBackend:
    """A fake shim installed in a tmp ``APPLE_ASR_CACHE``, scenario-driven."""
    return FakeBackend(tmp_path, monkeypatch)


def pytest_collection_modifyitems(config, items):
    """Integration tests are skipped outside macOS 26+ (they need the real shim).

    They must NOT skip on a macOS 26 machine: a missing shim there is a real
    failure, not a reason to pass silently.
    """
    reason = f"macOS 26+ with the Speech framework required (this is {sys.platform})"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "integration" in item.keywords and not macos26_available():
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs macOS 26 + the real shim (skipped elsewhere)"
    )


@pytest.fixture(scope="session")
def zh_long_audio():
    """The bundled zh clip used by the integration tests (16 kHz mono float32)."""
    import wave

    import numpy as np

    if not AUDIO_ZH_LONG.exists():
        pytest.skip(f"bundled audio fixture missing: {AUDIO_ZH_LONG}")
    with wave.open(str(AUDIO_ZH_LONG), "rb") as w:
        assert w.getframerate() == 16_000 and w.getnchannels() == 1, (
            "the bundled fixture must be 16 kHz mono"
        )
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
