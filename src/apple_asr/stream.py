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
boundary-anchored anchor refreshed at every input boundary, which keeps a final
that lands mid-pause on its own audio's timeline
(:class:`~apple_asr.transport.SessionClock`).

Modes
-----
`mode="streaming"` (default) asks for volatile partials at ~13.2 mean zh CER;
`mode="accurate"` gives finals only at ~11.2 (MEASUREMENTS.md). `mode` is a
convenience over `preset`/`reporting_option`.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Literal

import numpy as np

from .errors import BackendError, SessionClosed
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

__all__ = ["Stream", "PRESETS", "MODES", "REPORTING_OPTIONS"]

#: SpeechTranscriber presets exposed by the shim (SPEC.md §3.2).
PRESETS = ("progressive", "transcription", "timeIndexedProgressive")

#: `Stream(mode=)` values (SPEC.md amendment, MEASUREMENTS.md).
MODES = ("streaming", "accurate")

#: Values `Stream(reporting_option=)` accepts; they name the framework
#: reporting option the session asks for on top of the preset. `fastResults`
#: implies `volatileResults` (the framework's options are cumulative).
REPORTING_OPTIONS = ("volatileResults", "fastResults")

#: What each mode means, measured on the zh board (MEASUREMENTS.md):
#: `streaming` = progressive preset + fastResults -> volatile partials, ~13.2 zh
#: CER; `accurate` = transcription preset, no fastResults -> finals only, ~11.2.
_MODE_SETTINGS: dict[str, tuple[str, str | None]] = {
    "streaming": ("progressive", "fastResults"),
    "accurate": ("transcription", None),
}

#: The silent-backend guard stands down while the caller is burst-feeding
#: rather than streaming: a fast-fed burst may legitimately produce no output for
#: seconds (the transcriber is chewing whole-file audio).
_SILENT_FEED_RATE_MAX = 1.05
#: ...but a pipeline hands over audio in chunks (0.1-0.5 s at a time), and the
#: chunk's whole duration is credited at push time, so the rate comparison
#: carries this much absolute grace.
_SILENT_CHUNK_GRACE_S = 1.0

#: How long a single bounded wait may be before the guard re-checks.
_GUARD_TICK_S = 0.2

#: Minimum synthesized silence at `pause_start()` so the shim's PauseCommitter
#: (pause_commit seconds of quiet) fires DURING the pause rather than at resume.
_PAUSE_PREROLL_MIN_S = 0.10

#: While a pause is open, deliver silence at (slightly under) the audio rate in
#: these steps — a lump at the pause onset is not enough: the transcriber
#: publishes the commit only once more audio keeps arriving (same finding as the
#: reference driver). Under 1.0 so `pause_end(d)` always tops up to exactly `d`
#: instead of over-delivering silence the caller never reported; if a caller does
#: hold the pause past `d`, the clock counts that excess as the consumed audio it
#: is, so the two timelines stay 1:1 (`SessionClock`).
_SILENCE_TICK_S = 0.05
_SILENCE_PUMP_RATE = 0.8


def _resolve_config(
    mode: str | None,
    preset: str | None,
    reporting_option: str | None,
) -> tuple[str, str, str | None]:
    """Fold `mode`/`preset`/`reporting_option` into one config (amendment 1).

    `mode` is a convenience over the two explicit knobs. An explicitly passed
    `mode` is cross-checked against explicitly passed knobs (a contradiction is
    a `ValueError`, never a silent override); when `mode` is not passed, the
    explicit knobs are authoritative, so every order-1 spelling keeps working.
    Returns `(mode_label, preset, reporting_option)`.
    """
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES!r}; got {mode!r}")
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS!r}; got {preset!r}")
    if reporting_option is not None and reporting_option not in REPORTING_OPTIONS:
        raise ValueError(
            f"reporting_option must be one of {REPORTING_OPTIONS!r}; "
            f"got {reporting_option!r}"
        )
    implied_preset, implied_reporting = _MODE_SETTINGS[mode or "streaming"]
    if mode is not None:
        if preset is not None and preset != implied_preset:
            raise ValueError(
                f"mode={mode!r} implies preset={implied_preset!r}, but preset="
                f"{preset!r} was passed. Drop preset= and use mode={mode!r}, or "
                f"set mode={_mode_for(preset, reporting_option)!r}."
            )
        if reporting_option is not None and reporting_option != implied_reporting:
            raise ValueError(
                f"mode={mode!r} implies reporting_option={implied_reporting!r}, "
                f"but reporting_option={reporting_option!r} was passed. Drop "
                f"reporting_option= and use mode={mode!r}, or drop mode= to mix the "
                f"knobs directly."
            )
    effective_preset = preset or implied_preset
    effective_reporting = (
        reporting_option if reporting_option is not None else implied_reporting
    )
    if mode is not None:
        label = mode
    else:
        label = _mode_for(effective_preset, effective_reporting)
    return label, effective_preset, effective_reporting


def _mode_for(preset: str, reporting_option: str | None) -> str:
    """The mode label matching a resolved (preset, reporting_option) pair."""
    return (
        "streaming"
        if (preset, reporting_option) == _MODE_SETTINGS["streaming"]
        else "accurate"
    )


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
    """A streaming SpeechAnalyzer session. Use as a context manager.

    Parameters
    ----------
    locale:
        BCP-47 identifier; fixed for the session (a second locale is a second
        `Stream`).
    mode:
        ``"streaming"`` (default) = progressive preset + the framework's
        ``fastResults`` reporting option: volatile partials as they form.
        Measured ~13.2 mean zh CER on the project's FLEURS board.
        ``"accurate"`` = transcription preset without ``fastResults``: finals
        only, ~11.2 zh CER (equal to the batch/file path). Pass ``mode``
        explicitly and it is cross-checked against the explicit knobs below;
        leave it out and the knobs are authoritative.
    preset:
        The ``SpeechTranscriber`` preset. ``None`` (default) means "whatever
        `mode` implies": ``progressive`` for streaming, ``transcription`` for
        accurate. Passing it together with a contradicting ``mode`` raises.
    reporting_option:
        ``"fastResults"``, ``"volatileResults"`` or ``None`` (from `mode`).
        ``fastResults`` trades ~2 CER points for emission speed; the cost is
        measured, not folklore (MEASUREMENTS.md).
    context:
        ``AnalysisContext.contextualStrings`` — hotword strings.
    pause_commit:
        Seconds of quiet before a pause commits. Only meaningful in the
        synthesized-pause input mode (see the module docstring); measured
        default 0.08.
    commit_interval:
        Commit-latency ceiling in seconds; measured default 0.0 (pause-only)
        because an interval ceiling cut mid-phrase and corrupted text.
    confidence:
        Request per-run ``transcriptionConfidence`` when the framework offers it.
    silent_timeout:
        Seconds without any shim output, *while audio is being fed at real time
        or slower*, before a `BackendError` is raised instead of hanging (a
        fast-fed burst may legitimately be quiet for seconds, so a faster feed
        disables the guard). ``0`` disables it.
    shim:
        Explicit shim path; otherwise the resolution order of SPEC.md §6.
    stderr:
        ``capture`` (default, tail kept for error messages), ``inherit``,
        ``null``.
    queue_size:
        Bounded pending-event queue; on overflow the oldest events are dropped
        and counted in ``stats.dropped``.
    """

    def __init__(
        self,
        locale: str = "en-US",
        *,
        mode: Literal["streaming", "accurate"] | None = None,
        preset: Literal["progressive", "transcription", "timeIndexedProgressive"]
        | None = None,
        reporting_option: Literal["volatileResults", "fastResults"] | None = None,
        context: Sequence[str] = (),
        pause_commit: float = 0.08,
        commit_interval: float = 0.0,
        confidence: bool = True,
        shim: str | None = None,
        stderr: Literal["capture", "inherit", "null"] = "capture",
        queue_size: int = 256,
        silent_timeout: float = 30.0,
    ) -> None:
        require_supported()
        label, effective_preset, effective_reporting = _resolve_config(
            mode, preset, reporting_option
        )
        if stderr not in ("capture", "inherit", "null"):
            raise ValueError(f"stderr must be capture|inherit|null; got {stderr!r}")
        if pause_commit < 0 or commit_interval < 0:
            raise ValueError("pause_commit and commit_interval must be >= 0")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if silent_timeout < 0:
            raise ValueError("silent_timeout must be >= 0 (0 disables the guard)")

        self.locale = locale
        self.mode = label
        self.reporting_option = effective_reporting
        self.preset = effective_preset
        self.silent_timeout = float(silent_timeout)
        self.context = tuple(context)
        self.pause_commit = float(pause_commit)
        self.commit_interval = float(commit_interval)
        self.confidence = bool(confidence)

        self._shim_path = resolve_shim(shim, exclude=sys.argv[0] if sys.argv else None)
        argv = build_argv(
            self._shim_path,
            locale,
            effective_preset,
            self.pause_commit,
            self.commit_interval,
            self.context,
            self.confidence,
            fast_results=effective_reporting == "fastResults",
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
        """What the shim advertised in `hello` (sorted); degrades explicitly."""
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
        """Push audio: float32 (canonical) or int16, 1-D or ``(n, 1)``, mono, 16 kHz.

        Your caller resamples; the framework's analyzer format is Int16, so int16
        input is converted by clamp-and-scale (no dithering). Raises `ValueError`
        for any other dtype/shape and `SessionClosed` after `close()`. An empty
        array is a no-op.
        """
        self._require_open()
        frames = _as_float32(pcm)
        n = int(frames.shape[0])
        if n == 0:
            return
        self._transport.write_audio(frames.tobytes())
        self._clock.add_written(n, SAMPLE_RATE)
        self._clock.add_session(n / SAMPLE_RATE)
        self._clock.anchor()
        if self._pause_open:
            # Speech resumed without an explicit pause_end: close the pause bookkeeping.
            with self._pause_cv:
                self._pause_open = False
                self._pause_cv.notify_all()

    def pause_start(self) -> None:
        """Announce a VAD pause onset (the silence was stripped upstream).

        Writes a short pre-roll immediately (so the shim's pause detector fires
        during the pause) and starts the silence pump, which keeps delivering
        silence until `pause_end()`. Repeated calls while a pause is open are
        ignored.
        """
        self._require_open()
        with self._pause_cv:
            if self._pause_open:
                return
            self._pause_open = True
            self._pause_written = 0
            self._pump_idle.clear()
            self._pause_cv.notify_all()
        # Anchor the mapping at the pause onset *before* the pre-roll: the
        # pre-roll and the pump are synthesized silence the caller has not
        # reported yet (it reports `d` at `pause_end`), so the session clock must
        # not follow them (SessionClock).
        self._clock.anchor()
        # Pre-roll so the shim's PauseCommitter fires immediately; the pump then
        # keeps delivering silence for as long as the pause is open.
        self._write_silence(self._preroll_frames)
        with self._pause_cv:
            self._pause_written += self._preroll_frames

    def pause_end(self, duration_s: float) -> None:
        """Announce a VAD pause end, with its true duration in seconds.

        The synthesized silence for this pause totals `duration_s` (pre-roll +
        pump + top-up) whenever you do not hold the pause open past it, and the
        session clock — which advances with consumed audio (SPEC.md §3.3) —
        advances by exactly that silence. Holding the pause open longer makes the
        pump over-deliver (`P > duration_s`); that excess is real elapsed audio
        the shim consumed, so it is counted here too, which is what keeps the
        shim and session timelines 1:1 (and therefore keeps mapped final ranges
        contiguous). See :class:`~apple_asr.transport.SessionClock`.
        """
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
        self._clock.add_session(max(self._pause_written, target) / SAMPLE_RATE)
        self._clock.anchor()
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
        """Issue `finalize(through:)` and block for its `commit` acknowledgement.

        `through_s` may name a session-clock time; `None` means "through the
        current cursor". Blocks up to `COMMIT_TIMEOUT_S` for the shim's `commit`
        ack and raises `BackendError` on timeout, so it is usable as a barrier.
        """
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
        """Counters for this session: partials, finals, words, dropped, bytes."""
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
        """Blocking iterator over typed events, ending after `Ended`.

        Waits indefinitely between events (the silent-backend guard still
        applies unless `silent_timeout=0`). Do not mix with `poll()`/`aevents()`
        on one instance.
        """
        self._claim_consumer("events")
        while True:
            event = self._next_event(None)
            if event is None:
                return
            yield event

    async def aevents(self) -> AsyncIterator[Event]:
        """asyncio bridge over the same queue (one worker thread per event)."""
        self._claim_consumer("aevents")
        loop = asyncio.get_running_loop()
        while True:
            event = await loop.run_in_executor(None, self._next_event, None)
            if event is None:
                return
            yield event

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        """Non-blocking bulk drain; with `timeout_s` wait that long for the first.

        Raises `BackendError` when the silent-backend guard trips
        (`silent_timeout`), or the stored `BackendError` when the child died
        badly.
        """
        self._claim_consumer("poll")
        out: list[Event] = []
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            item = self._transport.get_event(0.0)
            if item is None:
                self._check_silent_backend()
                if time.monotonic() >= deadline:
                    return out
                item = self._transport.get_event(self._wait_slice(deadline))
                if item is None:
                    continue
            if item is EOF_SENTINEL:
                return out
            if isinstance(item, ExitError):
                raise item.exc
            out.append(item)

    def _next_event(self, timeout: float | None) -> Event | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            item = self._transport.get_event(0.0)
            if item is None:
                self._check_silent_backend()
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                item = self._transport.get_event(self._wait_slice(deadline))
                if item is None:
                    continue
            if item is EOF_SENTINEL:
                return None
            if isinstance(item, ExitError):
                raise item.exc
            return item  # type: ignore[return-value]

    def _wait_slice(self, deadline: float | None) -> float | None:
        """One bounded wait for the transport, respecting the guard cadence."""
        if self.silent_timeout > 0:
            if deadline is None:
                return _GUARD_TICK_S
            return min(_GUARD_TICK_S, max(0.0, deadline - time.monotonic()))
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _check_silent_backend(self) -> None:
        """Raise `BackendError` when a live shim has gone quiet (SPEC.md §9 test 6).

        Feed-rate aware on purpose: the transcriber may legitimately emit nothing
        for seconds after a fast-fed burst (offline/whole-file feeds hand it 30 s
        of audio in 50 ms). The guard therefore only fires when audio has been
        fed at no faster than real time *and* nothing has come back for
        `silent_timeout` seconds. Set `silent_timeout=0` to disable it.
        """
        timeout = self.silent_timeout
        if timeout <= 0:
            return
        transport = self._transport
        if transport.exited or transport.first_audio_monotonic is None:
            return
        audio_bytes = transport.counts["bytes"]
        if audio_bytes <= 0:
            return
        now = time.monotonic()
        fed_s = now - transport.first_audio_monotonic
        if fed_s <= 0.0:
            return
        audio_s = audio_bytes / (4.0 * SAMPLE_RATE)
        if audio_s > fed_s * _SILENT_FEED_RATE_MAX + _SILENT_CHUNK_GRACE_S:
            return  # burst-fed: quiet is expected, not a hang
        silent_s = now - transport.last_event_monotonic
        if silent_s >= timeout:
            raise BackendError(
                f"no output from the shim for {silent_s:.1f}s while {audio_s:.1f}s of "
                f"audio was fed over {fed_s:.1f}s (slower than real time): the "
                f"backend looks hung. shim={self._shim_path}; stderr tail:\n"
                f"{transport.stderr_tail()}\n"
                f"Set silent_timeout=0 to disable this guard."
            )

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
