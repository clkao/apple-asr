"""Platform gating (SPEC.md §8).

`require_supported()` is the single choke point; it is a module-level function
so the order-2 test suite can monkeypatch it to simulate non-darwin / macOS 25
on an ubuntu runner.
"""

from __future__ import annotations

import platform
import sys

from .errors import UnsupportedPlatform

__all__ = ["require_supported", "macos_major"]

MIN_MACOS_MAJOR = 26


def macos_major() -> int:
    """The running macOS major version, or 0 when it cannot be determined."""
    ver = platform.mac_ver()[0]
    if not ver:
        return 0
    try:
        return int(ver.split(".")[0])
    except ValueError:
        return 0


def require_supported() -> None:
    """Raise :class:`UnsupportedPlatform` unless this is macOS 26+."""
    if sys.platform != "darwin":
        raise UnsupportedPlatform(
            f"apple-asr requires macOS 26+ (Apple SpeechAnalyzer); "
            f"this platform is {sys.platform!r}. Nothing to install here."
        )
    major = macos_major()
    if major < MIN_MACOS_MAJOR:
        reported = platform.mac_ver()[0] or "unknown"
        raise UnsupportedPlatform(
            f"apple-asr requires macOS {MIN_MACOS_MAJOR}+ (Apple SpeechAnalyzer); "
            f"this system reports macOS {reported}."
        )
