"""SPEC.md §9 test 9 — platform gating.

`require_supported()` is the single choke point; the simulated platform is
monkeypatched here so the gate is tested on linux CI too. Marked
``real_platform_gate`` so conftest's neutralization (which unblocks the fake-shim
tests on any OS) leaves this module exercising the real gate.
"""

from __future__ import annotations

import sys

import pytest

from apple_asr import Stream, UnsupportedPlatform, list_locales, shim_info

pytestmark = pytest.mark.real_platform_gate


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A cache dir with no shim in it: gating must fire before resolution."""
    monkeypatch.setenv("APPLE_ASR_CACHE", str(tmp_path / "empty"))
    monkeypatch.delenv("APPLE_ASR_SHIM", raising=False)
    monkeypatch.delenv("APPLE_ASR_FAKE_SCENARIO", raising=False)
    return tmp_path


def test_non_darwin_raises_unsupported_platform(fake_cache, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(UnsupportedPlatform) as excinfo:
        Stream(locale="en-US")
    message = str(excinfo.value)
    assert "macOS 26" in message
    assert "linux" in message


def test_macos_25_raises_unsupported_platform(fake_cache, monkeypatch):
    monkeypatch.setattr("apple_asr.platform.macos_major", lambda: 25)
    with pytest.raises(UnsupportedPlatform) as excinfo:
        Stream(locale="en-US")
    assert "26" in str(excinfo.value)


def test_gate_applies_to_the_module_helpers(fake_cache, monkeypatch):
    monkeypatch.setattr("apple_asr.platform.macos_major", lambda: 25)
    with pytest.raises(UnsupportedPlatform):
        list_locales()
    with pytest.raises(UnsupportedPlatform):
        shim_info()


def test_supported_platform_is_a_noop(monkeypatch):
    """On a supported machine the gate returns None and nothing is logged."""
    if sys.platform != "darwin":
        pytest.skip("this machine is not macOS")
    from apple_asr.platform import macos_major, require_supported

    if macos_major() < 26:
        pytest.skip(f"this machine reports macOS {macos_major()}")
    assert require_supported() is None
