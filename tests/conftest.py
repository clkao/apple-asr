"""pytest fixtures for the apple-asr suite.

The whole non-integration suite runs against the scripted fake shim
(`tests/fixtures/fake_shim.py`) in a tmp cache directory, so it needs no macOS
26, no Speech framework, no mic, and no writable ``~/.cache`` — the §8 platform
gate is neutralized for those tests (see `neutralized_platform_gate`).

Two test-only controls (read here and in `support.forced_platform`, never by
``apple_asr``) make the Linux CI path reproducible locally:

* ``APPLE_ASR_TEST_PLATFORM=linux|macos25|macos26`` forces what the gate and the
  integration-skip decision see for the whole session.
* ``APPLE_ASR_TEST_NEUTRALIZE_GATE=0`` turns the neutralization off, leaving the
  real gate in force — combine with the first knob to reproduce the un-fixed
  failure faithfully.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/support.py

from support import (  # noqa: E402
    AUDIO_ZH_LONG,
    FakeBackend,
    forced_platform,
    macos26_available,
)


@pytest.fixture
def fake(tmp_path, monkeypatch) -> FakeBackend:
    """A fake shim installed in a tmp ``APPLE_ASR_CACHE``, scenario-driven."""
    return FakeBackend(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def neutralized_platform_gate(request, monkeypatch):
    """Stub the §8 gate so the fake-shim transport/client tests run on any OS.

    The gate itself is exercised for real by ``tests/test_platform.py``, which is
    marked ``real_platform_gate`` and excluded here. The gate symbols are
    patched where they are *imported* (``stream``/``shim`` bind them at module
    import), not just in ``apple_asr.platform``. Set
    ``APPLE_ASR_TEST_NEUTRALIZE_GATE=0`` to keep the real gate everywhere.
    """
    if os.environ.get("APPLE_ASR_TEST_NEUTRALIZE_GATE", "1") == "0":
        return
    if request.node.get_closest_marker("real_platform_gate"):
        return
    for target in (
        "apple_asr.platform.require_supported",
        "apple_asr.stream.require_supported",
        "apple_asr.shim.require_supported",
    ):
        monkeypatch.setattr(target, lambda: None)


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
    config.addinivalue_line(
        "markers", "real_platform_gate: exercises the real §8 gate (never neutralized)"
    )
    _force_simulated_platform(config)


def pytest_unconfigure(config):
    restore = getattr(config, "_apple_asr_platform_restore", None)
    if restore is not None:
        restore()


def _force_simulated_platform(config) -> None:
    """Test-only: make the §8 gate see the forced platform for the session.

    Done at configure time (before collection) so the integration-skip decision
    and the gate agree. Restored in :func:`pytest_unconfigure`.
    """
    try:
        forced = forced_platform()
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None
    if forced is None:
        return

    from apple_asr import platform as apple_platform

    sim_platform, major = forced
    original_platform = sys.platform
    original_macos_major = apple_platform.macos_major
    sys.platform = sim_platform
    apple_platform.macos_major = lambda: major

    def restore() -> None:
        sys.platform = original_platform
        apple_platform.macos_major = original_macos_major

    config._apple_asr_platform_restore = restore


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
