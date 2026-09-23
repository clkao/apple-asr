"""Subprocess transport for the shim (SPEC.md §5). Internal, not public API.

Owns the child process, the stdout reader thread, the bounded event queue, the
fd-3 command channel, the stderr capture, and the shim-clock -> session-clock
mapping. Everything Apple-flavoured about the wire is confined here.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
import time
import warnings
from collections import deque
from typing import Any

from .errors import BackendError, ProtocolMismatch, SessionClosed
from .events import Ended, Error, Final, Partial, Word
from .protocol import KNOWN_EVENT_TYPES, NEGOTIATED_FORMAT, PROTOCOL_VERSION

__all__ = ["Transport", "SessionClock", "read_hello", "build_argv", "EOF_SENTINEL", "ExitError"]

_EOF = object()

#: Public aliases for the queue sentinels.
EOF_SENTINEL = _EOF

#: How long `flush()` waits for its `commit` acknowledgement.
COMMIT_TIMEOUT_S = 15.0

#: How long `close()` waits for the child to exit before killing it.
EXIT_TIMEOUT_S = 20.0

#: How many stderr lines to keep for a BackendError tail.
STDERR_TAIL_LINES = 60


class SessionClock:
    """Maps the shim's frame timeline onto the session clock (SPEC.md §5).

    The shim owns the audio timeline (frames written == its clock), and SPEC.md
    §3.3 defines the session clock as advancing with **consumed audio, including
    synthesized pauses**. The two timelines are therefore 1:1, and the client's
    job is only to keep them 1:1 while a pause is *open*: the pre-roll and the
    pump write silence whose duration the caller reports at `pause_end`, so the
    session clock has not counted that audio yet and neither may the mapping.

    The mapping is boundary-anchored: an anchor pair `(shim_frame, session_s)` is
    refreshed at every input boundary — each `push()`, `pause_start()` and
    `pause_end(d)` (:meth:`anchor`) — and a shim time `t` maps as
    `session_anchor + (t * rate - shim_anchor) / rate`, the open segment running
    1:1 from the anchor. The anchor is moved only when the new point sits on the
    same 1:1 line (`_same_offset`), i.e. when that boundary did not leave the two
    timelines apart:

    * `push()` and `pause_end(d)` advance both timelines by the same audio, so
      they move the anchor;
    * while a pause is open the pump keeps writing silence the session clock has
      not counted, so the *pause-onset* anchor stays in force and a shim time
      inside the pause maps 1:1 from the pause onset — where the audio the final
      covers really was — instead of being pulled back by the already-written
      synthesized silence (the old global `drift = session - written`, which is
      transiently stale mid-pause).

    Because every boundary leaves the two timelines 1:1 (`Stream.pause_end`
    advances the session clock by the silence actually written, which is exactly
    the reported `d` whenever the caller does not hold the pause open past it),
    the mapping is monotonically non-decreasing and 1:1. The shim's final ranges
    tile its timeline, so the mapped ranges tile the session timeline:
    consecutive finals cannot overlap and :class:`Transport` needs no clamp.
    """

    #: The negotiated sample rate (the shim timeline unit; SPEC.md §5).
    _RATE = int(NEGOTIATED_FORMAT["sample_rate"])

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._written_frames = 0
        self._session_s = 0.0
        #: The anchor pair the open segment extends from, `(shim_frame, session_s)`.
        self._anchor: tuple[int, float] = (0, 0.0)

    def add_written(self, frames: int, rate: int) -> None:
        with self._lock:
            self._written_frames += frames

    def add_session(self, seconds: float) -> None:
        with self._lock:
            self._session_s += seconds

    def anchor(self) -> None:
        """Refresh the anchor at an input boundary.

        Called at the end of every `push()`, and around the synthesized silence at
        `pause_start()` (before the pre-roll) and `pause_end(d)` (after the
        session clock has been advanced by that silence). A point that is not on
        the anchor's 1:1 line — a `push()` during an open pause, whose pumped
        silence the session clock has not counted — leaves the pause-onset anchor
        in force.
        """
        with self._lock:
            point = (self._written_frames, self._session_s)
            if point != self._anchor and self._same_offset(point, self._anchor):
                self._anchor = point

    @staticmethod
    def _same_offset(a: tuple[int, float], b: tuple[int, float]) -> bool:
        """True when two points sit on the same 1:1 shim/session line."""
        return (
            abs((a[1] - a[0] / SessionClock._RATE) - (b[1] - b[0] / SessionClock._RATE))
            <= 1e-9
        )

    @property
    def session_s(self) -> float:
        with self._lock:
            return self._session_s

    @property
    def drift(self) -> float:
        """The offset in force at the current point (`session - shim` seconds)."""
        with self._lock:
            return self._session_s - (self._written_frames / self._RATE)

    def map(self, shim_s: float) -> float:
        """Shim seconds -> session seconds (never negative)."""
        with self._lock:
            anchor_f, anchor_s = self._anchor
            mapped = anchor_s + (shim_s * self._RATE - anchor_f) / self._RATE
        return mapped if mapped > 0.0 else 0.0


class _EventQueue:
    """Bounded FIFO with drop-oldest and terminal items that bypass capacity."""

    def __init__(self, maxsize: int) -> None:
        self._max = max(1, int(maxsize))
        self._dq: deque[Any] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, item: Any) -> bool:
        """Append, dropping the oldest item if full. Returns True if dropped."""
        with self._cv:
            if len(self._dq) >= self._max:
                self._dq.popleft()
                self.dropped += 1
                self._cv.notify()
                self._dq.append(item)
                return True
            self._dq.append(item)
            self._cv.notify()
            return False

    def put_terminal(self, item: Any) -> None:
        with self._cv:
            self._dq.append(item)
            self._closed = True
            self._cv.notify_all()

    def get(self, timeout: float | None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while not self._dq:
                if self._closed:
                    return _EOF
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            return self._dq.popleft()

    def drain(self) -> list[Any]:
        with self._cv:
            out = list(self._dq)
            self._dq.clear()
            return out


class ExitError:
    """Terminal queue item carrying the BackendError for a bad child exit."""

    def __init__(self, exc: BackendError) -> None:
        self.exc = exc


def build_argv(
    shim_path: str,
    locale: str,
    preset: str,
    pause_commit: float,
    commit_interval: float,
    context: tuple[str, ...] = (),
    confidence: bool = True,
    fast_results: bool = True,
) -> list[str]:
    """The shim argv for one session (SPEC.md §4 flags).

    `fast_results=False` adds `--no-fast`, which drops the framework's
    `.fastResults` reporting option (the ``accurate`` half of ``Stream(mode=)``).
    The shim's own ``--fast`` flag is a deprecated no-op kept for CLI
    compatibility, so it is never passed.
    """
    argv = [
        shim_path,
        "--stdin",
        "--locale",
        locale,
        "--preset",
        preset,
        "--pause-commit",
        repr(float(pause_commit)),
        "--commit-interval",
        repr(float(commit_interval)),
    ]
    if context:
        argv += ["--context", ",".join(context)]
    if not confidence:
        argv.append("--no-confidence")
    if not fast_results:
        argv.append("--no-fast")
    return argv


def read_hello(path: str, timeout: float = 60.0) -> dict[str, Any]:
    """Spawn `path --stdin` with an empty stdin and return its parsed `hello`.

    Used both by session startup and by the build helper's validation. The
    process is killed after `hello` is read; the shim emits `hello` before any
    asset work, so this is fast even on a cold locale.
    """
    proc = subprocess.Popen(
        [path, "--stdin"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        line, _leftover = _readline_with_timeout(proc, timeout)
        if line is None:
            code = proc.poll()
            raise BackendError(
                f"shim {path!r} did not emit a hello line within {timeout:.0f}s "
                f"(exit status {code!r}); run it manually: {path} --stdin"
            )
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"shim {path!r} emitted a non-JSON hello line: {line[:200]!r}"
            ) from exc
        if obj.get("type") != "hello":
            raise BackendError(
                f"shim {path!r} emitted {obj.get('type')!r} as its first line, not 'hello'"
            )
        return obj
    finally:
        _kill(proc)


def _readline_with_timeout(proc: subprocess.Popen, timeout: float) -> tuple[str | None, bytes]:
    """Read one line from `proc.stdout` within `timeout`.

    Returns `(line, leftover)`: `leftover` is whatever else the single bulk read
    pulled off the pipe (a fast shim can emit more lines in the same read — the
    caller must not drop them).
    """
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    buf = b""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, buf
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            return None, buf
        chunk = os.read(fd, 65536)
        if not chunk:
            return buf.decode("utf-8", "replace").strip() or None, b""
        buf += chunk
        nl = buf.find(b"\n")
        if nl >= 0:
            return buf[:nl].decode("utf-8", "replace").strip() or None, buf[nl + 1 :]


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        pass


class Transport:
    """One shim subprocess plus its reader thread and event queue."""

    def __init__(
        self,
        argv: list[str],
        *,
        clock: SessionClock | None = None,
        stderr_mode: str = "capture",
        queue_size: int = 256,
        hello_timeout: float = 60.0,
    ) -> None:
        self.clock = clock or SessionClock()
        self._queue = _EventQueue(queue_size)
        self._argv = argv
        self._write_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._commands_closed = False
        self._close_requested = False

        self._stderr_lines: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._exit_error: BackendError | None = None
        self._exited = threading.Event()
        self._exited_code: int | None = None
        self._drop_warned = False

        # Liveness bookkeeping for `Stream.silent_timeout` (SPEC.md §9 test 6):
        # when the shim last said anything, and when the caller started feeding
        # audio (so the guard can tell a fast-fed burst - legitimately quiet for
        # seconds - from a hung backend fed at real time or slower).
        self.last_event_monotonic = time.monotonic()
        self.first_audio_monotonic: float | None = None

        # Commit acknowledgements: count + reason FIFO, paired with finals.
        self._commit_cv = threading.Condition()
        self._commit_count = 0
        self._pending_reasons: deque[str] = deque()

        # Bytes of stdout read while waiting for `hello`; the reader thread must
        # consume them first (see `_readline_with_timeout`).
        self._hello_leftover = b""

        self._counts = {"partials": 0, "finals": 0, "words": 0, "bytes": 0}

        self._ctl_r, self._ctl_w = os.pipe()
        env = dict(os.environ)
        env["APPLE_ASR_CTL_FD"] = str(self._ctl_r)
        stderr = {
            "capture": subprocess.PIPE,
            "inherit": None,
            "null": subprocess.DEVNULL,
        }[stderr_mode]

        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=env,
                pass_fds=(self._ctl_r,),
            )
        except OSError as exc:
            os.close(self._ctl_r)
            os.close(self._ctl_w)
            raise BackendError(f"could not start shim {argv[0]!r}: {exc}") from exc
        # The child holds the read end now.
        os.close(self._ctl_r)

        self._stderr_thread: threading.Thread | None = None
        if stderr_mode == "capture":
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, name="apple-asr-stderr", daemon=True
            )
            self._stderr_thread.start()

        self.hello = self._read_and_validate_hello(hello_timeout)

        self._reader = threading.Thread(
            target=self._read_loop, name="apple-asr-reader", daemon=True
        )
        self._reader.start()

    # ------------------------------------------------------------------
    # Hello
    # ------------------------------------------------------------------
    def _read_and_validate_hello(self, timeout: float) -> dict[str, Any]:
        assert self._proc.stdout is not None
        line, leftover = _readline_with_timeout(self._proc, timeout)
        self._hello_leftover = leftover
        if line is None:
            tail = self.stderr_tail()
            code = self._proc.poll()
            raise BackendError(
                f"shim emitted no hello within {timeout:.0f}s (exit status {code!r}). "
                f"stderr tail:\n{tail}"
            )
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"shim's first stdout line is not JSON: {line[:200]!r}"
            ) from exc
        if obj.get("type") != "hello":
            raise BackendError(
                f"shim's first stdout line is type {obj.get('type')!r}, expected 'hello'"
            )
        theirs = obj.get("protocol")
        if theirs != PROTOCOL_VERSION:
            raise ProtocolMismatch(
                f"shim protocol {theirs!r} != client protocol {PROTOCOL_VERSION}; "
                f"rebuild the shim: python -m apple_asr.build --force"
            )
        fmt = obj.get("format") or {}
        for key, want in NEGOTIATED_FORMAT.items():
            got = fmt.get(key)
            if got != want:
                raise BackendError(
                    f"shim negotiated format {key}={got!r}, client needs {want!r} "
                    f"(hello format={fmt!r})"
                )
        caps = set(obj.get("capabilities") or [])
        self.capabilities = tuple(sorted(caps))
        return obj

    # ------------------------------------------------------------------
    # stderr
    # ------------------------------------------------------------------
    def _stderr_loop(self) -> None:
        assert self._proc.stderr is not None
        try:
            for raw in self._proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                self._stderr_lines.append(line)
        except Exception:  # pragma: no cover - defensive
            pass

    def stderr_tail(self, lines: int = 20) -> str:
        tail = list(self._stderr_lines)[-lines:]
        return "\n".join(tail)

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------
    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        buf = self._hello_leftover
        saw_ended = False
        try:
            while True:
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    raw, buf = buf[:nl], buf[nl + 1 :]
                    if self._handle_line(raw):
                        saw_ended = True
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
        except Exception as exc:  # pragma: no cover - defensive
            self._warn(f"reader thread failed: {exc!r}")
        finally:
            self._finish(saw_ended)

    def _handle_line(self, raw: bytes) -> bool:
        """Map one stdout line onto the queue. Returns True when it was `ended`.

        Any line (even a malformed one) proves the shim is alive: it feeds the
        liveness clock behind `Stream.silent_timeout`.
        """
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            return False
        self.last_event_monotonic = time.monotonic()
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            self._warn(f"ignoring malformed JSON line from shim: {line[:160]!r}")
            return False
        if not isinstance(obj, dict):
            self._warn(f"ignoring non-object JSON event: {line[:160]!r}")
            return False
        self._dispatch(obj)
        return obj.get("type") == "ended"

    def _dispatch(self, obj: dict[str, Any]) -> None:
        kind = obj.get("type")
        if kind == "partial":
            self._counts["partials"] += 1
            self._put(self._make_partial(obj))
        elif kind == "final":
            self._counts["finals"] += 1
            self._put(self._make_final(obj))
        elif kind == "commit":
            with self._commit_cv:
                self._commit_count += 1
                self._pending_reasons.append(str(obj.get("reason") or "pause"))
                self._commit_cv.notify_all()
        elif kind == "ended":
            self._put(Ended(reason=_map_ended_reason(obj.get("reason"))))
        elif kind == "error":
            self._put(Error(str(obj.get("message") or ""), str(obj.get("detail") or "")))
        elif kind in KNOWN_EVENT_TYPES:
            # 'hello' repeated on the wire: informational only.
            pass
        else:
            self._warn(f"ignoring unknown event type {kind!r}")

    def _make_partial(self, obj: dict[str, Any]) -> Partial:
        rng = obj.get("range") or [None, None]
        return Partial(
            text=str(obj.get("text") or ""),
            start=None if rng[0] is None else self.clock.map(float(rng[0])),
            end=None if len(rng) < 2 or rng[1] is None else self.clock.map(float(rng[1])),
            words=self._make_words(obj),
        )

    def _make_final(self, obj: dict[str, Any]) -> Final:
        # No clamp on `Final.start`: `SessionClock` maps the shim timeline 1:1 and
        # monotonically (its anchor keeps that true while a pause's already-written
        # silence is still unreported), and the shim's final ranges tile its
        # timeline, so the mapped ranges are contiguous by construction (SPEC.md
        # §3.3). Order 1 needed the clamp because a mid-pause final used to be
        # mapped with the global drift, which the synthesized silence had already
        # made stale (measured 0.05 s, up to the 0.10 s pre-roll) and which nudged
        # the timestamp. tests/test_clock_accounting.py is the regression test for
        # the exactness that makes the clamp unnecessary.
        rng = obj.get("range") or [0.0, 0.0]
        raw_start = float(rng[0] if rng and rng[0] is not None else 0.0)
        raw_end = float(rng[1] if len(rng) > 1 and rng[1] is not None else raw_start)
        start = self.clock.map(raw_start)
        end = self.clock.map(raw_end)
        if end < start:  # malformed wire range only; a monotone map preserves order
            end = start
        wire_reason = obj.get("reason")
        with self._commit_cv:
            pending = self._pending_reasons.popleft() if self._pending_reasons else None
        reason = wire_reason or pending or "pause"
        return Final(text=str(obj.get("text") or ""), start=start, end=end,
                     words=self._make_words(obj), reason=reason)

    def _make_words(self, obj: dict[str, Any]) -> tuple[Word, ...]:
        runs = obj.get("runs") or []
        words: list[Word] = []
        for run in runs:
            if not run:
                continue
            text = str(run[0])
            if not text:
                continue
            conf = None
            if len(run) > 3 and run[3] is not None:
                conf = float(run[3])
            words.append(
                Word(
                    text=text,
                    start=self.clock.map(float(run[1])),
                    end=self.clock.map(float(run[2])),
                    confidence=conf,
                )
            )
        self._counts["words"] += len(words)
        return tuple(words)

    def _finish(self, saw_ended: bool) -> None:
        code: int | None = None
        try:
            code = self._proc.wait(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            _kill(self._proc)
            code = self._proc.returncode
        self._exited_code = code
        if code not in (0, None) and not saw_ended:
            tail = self.stderr_tail()
            exc = BackendError(
                f"shim exited with status {code} while the session was open; "
                f"stderr tail:\n{tail}"
            )
            self._exit_error = exc
            self._queue.put_terminal(ExitError(exc))
        elif not saw_ended:
            reason = "closed" if self._close_requested else "eof"
            self._queue.put_terminal(Ended(reason=reason))
        self._queue.put_terminal(_EOF)
        with self._commit_cv:
            self._commit_cv.notify_all()
        self._exited.set()

    def _put(self, item: Any) -> None:
        if self._queue.put(item) and self.note_drop():
            self._warn(
                "event queue full; dropping oldest events (raise queue_size to keep up)"
            )

    def _warn(self, message: str) -> None:
        warnings.warn(message, RuntimeWarning, stacklevel=1)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def write_audio(self, data: bytes) -> None:
        with self._write_lock:
            stdin = self._proc.stdin
            if stdin is None:
                raise SessionClosed("shim stdin is closed")
            try:
                stdin.write(data)
                stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                raise BackendError(
                    f"shim stdin is gone ({exc}); stderr tail:\n{self.stderr_tail()}"
                ) from exc
            self._counts["bytes"] += len(data)
            if self.first_audio_monotonic is None:
                self.first_audio_monotonic = time.monotonic()

    def send_command(self, obj: dict[str, Any]) -> None:
        with self._command_lock:
            if self._commands_closed:
                raise SessionClosed("control channel is closed (session already closing)")
            line = (json.dumps(obj) + "\n").encode("utf-8")
            try:
                os.write(self._ctl_w, line)
            except (BrokenPipeError, OSError) as exc:
                raise BackendError(f"shim control channel is gone ({exc})") from exc

    def close_input(self) -> None:
        """Send `close` on fd 3 and EOF stdin; idempotent."""
        with self._command_lock:
            if self._commands_closed:
                return
            self._commands_closed = True
        self._close_requested = True
        try:
            os.write(self._ctl_w, (json.dumps({"cmd": "close"}) + "\n").encode("utf-8"))
        except (BrokenPipeError, OSError):
            pass
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def terminate(self) -> None:
        try:
            os.close(self._ctl_w)
        except OSError:
            pass
        _kill(self._proc)

    # ------------------------------------------------------------------
    # Commits / exit
    # ------------------------------------------------------------------
    def commit_count(self) -> int:
        with self._commit_cv:
            return self._commit_count

    def wait_commit(self, timeout: float = COMMIT_TIMEOUT_S) -> None:
        """Block until a new `commit` acknowledgement arrives after this call."""
        with self._commit_cv:
            start = self._commit_count
            deadline = time.monotonic() + timeout
            while self._commit_count <= start:
                if self._exit_error is not None:
                    raise self._exit_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendError(
                        f"shim did not acknowledge finalize within {timeout:.1f}s; "
                        f"stderr tail:\n{self.stderr_tail()}"
                    )
                self._commit_cv.wait(remaining)

    def wait_exited(self, timeout: float = EXIT_TIMEOUT_S) -> bool:
        return self._exited.wait(timeout)

    @property
    def exit_error(self) -> BackendError | None:
        return self._exit_error

    @property
    def exited(self) -> bool:
        return self._exited.is_set()

    # ------------------------------------------------------------------
    # Event queue
    # ------------------------------------------------------------------
    def get_event(self, timeout: float | None) -> Any:
        """Internal: returns an event, `None` on timeout, `_EOF` at end.

        Raises the stored BackendError when the child died badly.
        """
        return self._queue.get(timeout)

    def drain_events(self) -> list[Any]:
        """Internal: everything queued right now (may include `_EOF`)."""
        return self._queue.drain()

    @property
    def dropped(self) -> int:
        return self._queue.dropped

    def note_drop(self) -> bool:
        """Returns True the first time a drop happens (for the one warning)."""
        if self._drop_warned:
            return False
        self._drop_warned = True
        return True

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def shutdown(self) -> None:
        """Best-effort teardown: close, wait, kill. Never raises."""
        try:
            self.close_input()
        except Exception:
            pass
        self.wait_exited(EXIT_TIMEOUT_S)
        self.terminate()


def _map_ended_reason(reason: Any) -> str:
    r = str(reason or "eof")
    if r in ("closed", "shim_exit", "protocol_error", "eof"):
        return r
    return "shim_exit"
