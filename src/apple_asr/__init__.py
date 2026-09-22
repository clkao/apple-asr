"""apple-asr — streaming, on-device speech-to-text for macOS 26+ from Python.

Public API (SPEC.md §3). No Apple type names appear here; JSONL is an internal
transport detail. Two entry points to know:

* :class:`~apple_asr.stream.Stream` — the session. ``mode="streaming"`` (default)
  gives volatile partials (~13.2 mean zh CER); ``mode="accurate"`` gives finals
  only (~11.2). See MEASUREMENTS.md for the evidence.
* :mod:`apple_asr.replay` — the real-time replay driver behind the
  ``apple-asr-replay`` console script.
"""

from __future__ import annotations

from .errors import (
    AppleAsrError,
    AssetUnavailable,
    BackendError,
    ProtocolMismatch,
    SessionClosed,
    ShimUnavailable,
    UnsupportedPlatform,
)
from .events import (
    Ended,
    Error,
    Final,
    Locales,
    Partial,
    ShimInfo,
    Stats,
    Word,
)
from .events import Event as Event
from .protocol import PROTOCOL_VERSION, SHIM_VERSION
from .shim import build_shim, ensure_installed, list_locales, shim_info
from .stream import Stream

__version__ = SHIM_VERSION

__all__ = [
    # session
    "Stream",
    # events
    "Partial",
    "Final",
    "Word",
    "Ended",
    "Error",
    "Event",
    # values
    "Locales",
    "ShimInfo",
    "Stats",
    # helpers
    "list_locales",
    "ensure_installed",
    "shim_info",
    "build_shim",
    "PROTOCOL_VERSION",
    "SHIM_VERSION",
    "__version__",
    # exceptions
    "AppleAsrError",
    "UnsupportedPlatform",
    "ShimUnavailable",
    "AssetUnavailable",
    "ProtocolMismatch",
    "SessionClosed",
    "BackendError",
]
