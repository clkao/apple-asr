"""Wire-protocol constants (SPEC.md §5). Internal: not public API."""

from __future__ import annotations

#: Must match `protocolVersion` in the Swift shim.
PROTOCOL_VERSION = 1

#: Must match `shimVersion` in the Swift shim; participates in the cache path.
SHIM_VERSION = "0.1.0"

#: The stdin audio contract: float32 LE, mono, 16 kHz.
SAMPLE_RATE = 16_000
CHANNELS = 1

#: The negotiated format the shim must report in `hello`.
NEGOTIATED_FORMAT = {"sample_rate": SAMPLE_RATE, "channels": CHANNELS}

#: Event types this client understands; anything else is warned and ignored.
KNOWN_EVENT_TYPES = frozenset({"hello", "partial", "final", "commit", "error", "ended"})

#: Commands the client may write on the control channel (fd 3).
KNOWN_COMMANDS = frozenset({"prepare", "finalize", "context", "close"})

#: Capabilities the shim may advertise in `hello`.
KNOWN_CAPABILITIES = frozenset({"volatile", "word_runs", "pause_commit", "flush", "context"})
