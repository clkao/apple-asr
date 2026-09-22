"""The public session object (SPEC.md §3).

Input modes
-----------
Both are first-class:

* a caller with **continuous audio** (silence included) uses :meth:`Stream.push`
  alone — pauses are real, and the framework handles them;
* a caller whose upstream VAD **strips silence** (WhisperLiveKit does) pushes
  active audio and announces pauses with :meth:`Stream.pause_start` /
  :meth:`Stream.pause_end`. The client synthesizes that silence for the
  framework at the audio rate — a declared capability, not a hidden hack. That
  is what `pause_commit` is for.

Clock
-----
Session timestamps are seconds on a clock that starts at 0 and advances with
consumed audio, including synthesized pauses. The shim owns the audio timeline
(frames written == its clock); the client maps shim time -> session time with a
drift term while it synthesizes pauses (:class:`~apple_asr.transport.SessionClock`).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Literal

import numpy as np

from .errors import SessionClosed
from .events import Event, Stats
from .platform import require_supported
from .protocol import SAMPLE_RATE
from .shim import resolve_shim
from .transport import (
    COMMIT_TIMEOUT_S,
    EOF_SENTINEL,
    ExitError,
    SessionClock,
    Transport,
    build_argv,
)

__all__ = ["Stream", "PRESETS"]

PRESETS = ("progressive", "transcription", "timeIndexedProgressive")

#: Minimum synthesized silence at `pause_start()` so the shim's PauseCommitter
#: (pause_commit seconds of quiet) fires DURING the pause rather than at resume.
_PAUSE_PREROLL_MIN_S = 0.10

#: While a pause is open, deliver silence at (slightly under) the audio rate in
#: these steps — a lump at the pause onset is not enough: the transcriber
#: publishes the commit only once more audio keeps arriving (same finding as the
#: reference driver). Under 1.0 so `pause_end(d)` always tops up to exactly `d`
#: instead of over-delivering, which would shift the drift and let consecutive
#: final ranges overlap.
_SILENCE_TICK_S = 0.05
_SILENCE_PUMP_RATE = 0.8


def _as_float32(pcm: np.ndarray) -> np.ndarray:
    """Canonicalize PCM to contiguous 1-D float32 (clamp-and-scale, no dither)."""
    arr = np.asarray(pcm)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"pcm must be 1-D or (n, 1); got shape {arr.shape}")
    if arr.dtype == np.float32:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    raise ValueError(
        f"pcm must be float32 (canonical), float64, or int16; got {arr.dtype}"
    )


class Stream:
    """A streaming SpeechAnalyzer session. Use as a context manager."""

    def __init__(
        self,
        locale: str = "en-US",
        *,
        preset: Literal["progressive", "transcription", "timeIndexedProgressive"] = "progressive",
        context: Sequence[str] = (),
        pause_commit: float = 0.08,
        commit_interval: float = 0.0,
        confidence: bool = True,
        shim: str | None = None,
        stderr: Literal["capture", "inherit", "null"] = "capture",
        queue_size: int = 256,
    ) -> None:
        require_supported()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS!r}; got {preset!r}")
        if stderr not in ("capture", "inherit", "null"):
            raise ValueError(f"stderr must be capture|inherit|null; got {stderr!r}")
        if pause_commit < 0 or commit_interval < 0:
            raise ValueError("pause_commit and commit_interval must be >= 0")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")

        self.locale = locale
        self.preset = preset
        self.context = tuple(context)
        self.pause_commit = float(pause_commit)
        self.commit_interval = float(commit_interval)
        self.confidence = bool(confidence)

        self._shim_path = resolve_shim(shim, exclude=sys.argv[0] if sys.argv else None)
        argv = build_argv(
            self._shim_path,
            locale,
            preset,
            self.pause_commit,
            self.commit_interval,
            self.context,
            self.confidence,
        )
        self._clock = SessionClock()
        self._transport = Transport(
            argv, clock=self._clock, stderr_mode=stderr, queue_size=queue_size
        )
        self._closed = False
        self._pause_open = False
        self._pause_written = 0
        self._preroll_frames = int(
            round(max(_PAUSE_PREROLL_MIN_S, self.pause_commit + 0.02) * SAMPLE_RATE)
        )
        self._consumer: str | None = None
        self._lock = threading.Lock()

        # Silence pump: writes zeros at the audio rate while a pause is open so
        # the transcriber sees a real pause and commits during it.
        self._pause_cv = threading.Condition()
        self._pump_idle = threading.Event()
        self._pump_idle.set()
        self._closing = False
        self._pump = threading.Thread(
            target=self._silence_pump, name="apple-asr-pause", daemon=True
        )
        self._pump.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def shim_path(self) -> str:
        return self._shim_path

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(self._transport.capabilities)

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def prepare(self) -> None:
        """Warm the model (`prepareToAnalyze`)."""
        self._require_open()
        self._transport.send_command({"cmd": "prepare"})

    def close(self) -> None:
        """Graceful shutdown: finalize + end of input + terminate. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._pause_cv:
            self._closing = True
            self._pause_open = False
            self._pause_cv.notify_all()
        self._transport.shutdown()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def push(self, pcm: np.ndarray) -> None:
        """Push float32 (or int16) mono 16 kHz audio. Caller resamples."""
        self._require_open()
        frames = _as_float32(pcm)
        n = int(frames.shape[0])
        if n == 0:
            return
        self._transport.write_audio(frames.tobytes())
        self._clock.add_written(n, SAMPLE_RATE)
        self._clock.add_session(n / SAMPLE_RATE)
        if self._pause_open:
            # Speech resumed without an explicit pause_end: close the pause bookkeeping.
            with self._pause_cv:
                self._pause_open = False
                self._pause_cv.notify_all()

    def pause_start(self) -> None:
        """Announce a VAD pause onset (silence was stripped upstream)."""
        self._require_open()
        with self._pause_cv:
            if self._pause_open:
                return
            self._pause_open = True
            self._pause_written = 0
            self._pump_idle.clear()
            self._pause_cv.notify_all()
        # Pre-roll so the shim's PauseCommitter fires immediately; the pump then
        # keeps delivering silence for as long as the pause is open.
        self._write_silence(self._preroll_frames)
        with self._pause_cv:
            self._pause_written += self._preroll_frames

    def pause_end(self, duration_s: float) -> None:
        """Announce a VAD pause end, with its true duration in seconds."""
        self._require_open()
        if duration_s < 0:
            raise ValueError("pause duration must be >= 0")
        with self._pause_cv:
            self._pause_open = False
            self._pause_cv.notify_all()
        # Wait for any in-flight pump write so the top-up below is exact.
        self._pump_idle.wait(1.0)
        target = int(round(float(duration_s) * SAMPLE_RATE))
        remainder = target - self._pause_written
        if remainder > 0:
            self._write_silence(remainder)
        self._clock.add_session(float(duration_s))
        with self._pause_cv:
            self._pause_written = 0

    def _silence_pump(self) -> None:
        tick_frames = int(round(_SILENCE_TICK_S * SAMPLE_RATE))
        while True:
            with self._pause_cv:
                while not self._pause_open and not self._closing:
                    self._pump_idle.set()
                    self._pause_cv.wait(0.2)
                if self._closing:
                    return
                self._pump_idle.clear()
            time.sleep(_SILENCE_TICK_S / _SILENCE_PUMP_RATE)
            with self._pause_cv:
                if not self._pause_open:
                    continue
            self._write_silence(tick_frames)
            with self._pause_cv:
                self._pause_written += tick_frames

    def _write_silence(self, frames: int) -> None:
        if frames <= 0:
            return
        zeros = np.zeros(frames, dtype=np.float32)
        self._transport.write_audio(zeros.tobytes())
        self._clock.add_written(frames, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def flush(self, through_s: float | None = None) -> None:
        """Issue `finalize(through:)` and block for its `commit` acknowledgement."""
        self._require_open()
        cmd: dict[str, object] = {"cmd": "finalize"}
        if through_s is not None:
            cmd["through"] = float(through_s)
        self._transport.send_command(cmd)
        self._transport.wait_commit(timeout=COMMIT_TIMEOUT_S)

    @property
    def audio_time(self) -> float:
        """Seconds consumed on the session clock."""
        return self._clock.session_s

    @property
    def stats(self) -> Stats:
        counts = self._transport.counts
        return Stats(
            partials=counts["partials"],
            finals=counts["finals"],
            words=counts["words"],
            dropped=self._transport.dropped,
            bytes=counts["bytes"],
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def events(self) -> Iterator[Event]:
        """Blocking iterator over typed events, ending after `Ended`."""
        self._claim_consumer("events")
        while True:
            event = self._next_event(None)
            if event is None:
                return
            yield event

    async def aevents(self) -> AsyncIterator[Event]:
        """asyncio bridge over the same queue."""
        self._claim_consumer("aevents")
        loop = asyncio.get_running_loop()
        while True:
            event = await loop.run_in_executor(None, self._next_event, None)
            if event is None:
                return
            yield event

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        """Non-blocking bulk drain; with `timeout_s` wait that long for the first."""
        self._claim_consumer("poll")
        out: list[Event] = []
        item = self._transport.get_event(max(0.0, float(timeout_s)))
        while item is not None and item is not EOF_SENTINEL:
            if isinstance(item, ExitError):
                raise item.exc
            out.append(item)
            item = self._transport.get_event(0.0)
        return out

    def _next_event(self, timeout: float | None) -> Event | None:
        item = self._transport.get_event(timeout)
        if item is None or item is EOF_SENTINEL:
            return None
        if isinstance(item, ExitError):
            raise item.exc
        return item  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosed("the session is closed; create a new Stream")

    def _claim_consumer(self, name: str) -> None:
        with self._lock:
            if self._consumer is None:
                self._consumer = name
            elif self._consumer != name:
                raise RuntimeError(
                    f"Stream consumers must not be mixed: already iterating via "
                    f"{self._consumer!r}, now {name!r}"
                )
