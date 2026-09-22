"""Public event and value types (SPEC.md §3.3).

Frozen dataclasses; no Apple type names leak into this surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Word",
    "Partial",
    "Final",
    "Ended",
    "Error",
    "Event",
    "Stats",
    "Locales",
    "ShimInfo",
]

#: `Final.reason` — why a commit landed (observability that must survive).
FinalReason = Literal["pause", "interval", "flush", "eof"]

#: `Ended.reason`. SPEC.md §3.3 lists closed/shim_exit/protocol_error; §9 also
#: requires "eof" for stdin EOF, so "eof" is included here.
EndedReason = Literal["closed", "shim_exit", "protocol_error", "eof"]


@dataclass(frozen=True)
class Word:
    """A single word with its session-clock span."""

    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True)
class Partial:
    """Volatile text: may be revised or replaced. Consumers replace, never append."""

    text: str
    start: float | None
    end: float | None
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Final:
    """Committed text: never revised for its range; ranges never overlap."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...]
    reason: FinalReason = "pause"


@dataclass(frozen=True)
class Error:
    """The shim reported an error; the session may still be alive."""

    message: str
    detail: str = ""


@dataclass(frozen=True)
class Ended:
    """Terminal event: the session is over."""

    reason: EndedReason


Event = Partial | Final | Ended | Error


@dataclass(frozen=True)
class Stats:
    """Session counters (SPEC.md §3.2 observability)."""

    partials: int = 0
    finals: int = 0
    words: int = 0
    dropped: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class Locales:
    """`list_locales()` result: what is installed vs what is supported."""

    installed: tuple[str, ...]
    supported: tuple[str, ...]


@dataclass(frozen=True)
class ShimInfo:
    """`shim_info()` result."""

    path: str
    version: str
    protocol: int
    build: str = ""
