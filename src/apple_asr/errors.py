"""Exception hierarchy for :mod:`apple_asr`.

Every error is actionable: it names what failed (path, version, locale) and the
next command to run (see SPEC.md §8).
"""

from __future__ import annotations

__all__ = [
    "AppleAsrError",
    "UnsupportedPlatform",
    "ShimUnavailable",
    "AssetUnavailable",
    "ProtocolMismatch",
    "SessionClosed",
    "BackendError",
]


class AppleAsrError(Exception):
    """Base class for every error raised by :mod:`apple_asr`."""


class UnsupportedPlatform(AppleAsrError):
    """Raised on non-macOS or macOS < 26 (Apple SpeechAnalyzer is 26+)."""


class ShimUnavailable(AppleAsrError):
    """Raised when the shim executable cannot be resolved or built."""


class AssetUnavailable(AppleAsrError):
    """Raised when a locale asset cannot be installed or reserved."""


class ProtocolMismatch(AppleAsrError):
    """Raised when the shim's wire protocol version is not ours."""


class SessionClosed(AppleAsrError):
    """Raised when input is pushed to a finished session."""


class BackendError(AppleAsrError):
    """Raised when the shim misbehaves (nonzero exit, malformed hello, ...)."""
