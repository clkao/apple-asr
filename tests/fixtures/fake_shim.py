#!/usr/bin/env python3
"""A scripted fake `apple-asr-shim` for the order-2 test suite (SPEC.md §9).

This is **test scaffolding**, not part of the package: it speaks the §5 wire
protocol without any Apple framework, so tests 1-9 of the spec run on any OS.

It is driven by a *scenario file* (JSON) whose path comes from the
``APPLE_ASR_FAKE_SCENARIO`` environment variable; when
``APPLE_ASR_FAKE_STATE`` is set it dumps an observability JSON on exit (frames
consumed, commands seen, every wire event emitted, the argv the client passed).
That dump is how tests assert on the wire rather than on their own bookkeeping.

Scenario modes
--------------
``emulate`` (default)
    Behaves like the real shim: reads float32 PCM from stdin, emits a volatile
    ``partial`` every ``partial_every_frames``, and runs an *endpointer* that
    commits a ``final`` + ``commit`` once it has seen ``pause_commit`` seconds
    of consecutive silence. Frame-exact: a final's range tiles the shim timeline
    (``range = [utterance_start, quiet_start]``), which is the worst case for
    the client's clock mapping.
``scripted``
    Emits exactly the steps listed in ``steps``, each fired at most once when
    its trigger becomes true. Nothing is emitted unless a step says so.
``silent``
    Consumes stdin and never emits anything after ``hello`` — the input to
    §9 test 6's silent-backend guard.

Faithfulness notes (why the emulation looks like this)
------------------------------------------------------
* A lump of silence at the pause onset alone does NOT publish a final: the real
  transcriber needs more audio to keep arriving. The endpointer therefore
  *arms* when the quiet run crosses ``pause_commit`` and fires on the next
  chunk. Order 1 measured this: a pre-roll alone gave >1.2 s (and only landed at
  resume), the silence pump gives 0.05-0.15 s.
* Finals tile the timeline with no gaps (the real transcriber emitted
  ``3.25 -> 3.25`` boundaries in the order-1 run), so a client that maps the
  two ends with different drift terms will produce an overlap.
"""

from __future__ import annotations

import json
import os
import select
import struct
import sys
import threading
import time

SAMPLE_RATE = 16_000
PROTOCOL_VERSION = 1
SHIM_VERSION = "0.1.0"
FRAME_BYTES = 4  # float32 mono

#: Absolute safety net: never let a broken scenario hang the test suite.
MAX_WALL_S = float(os.environ.get("APPLE_ASR_FAKE_MAX_WALL_S", "60"))
#: How often the main loop polls state (also the floor on commit latency).
TICK_S = 0.005
#: A float32 sample below this magnitude counts as silence.
SILENCE_EPS = 1e-4
#: Read granularity from stdin.
READ_BYTES = 4096


class FakeShim:
    def __init__(self) -> None:
        path = os.environ.get("APPLE_ASR_FAKE_SCENARIO")
        if not path:
            self._fatal("APPLE_ASR_FAKE_SCENARIO is not set")
        with open(path, encoding="utf-8") as fh:
            self.scn: dict = json.load(fh)

        self.argv = list(sys.argv[1:])
        self.opt = _parse_args(self.argv)
        self.mode = str(self.scn.get("mode") or "emulate")
        self.lock = threading.Lock()
        self.t0 = time.monotonic()

        # Timeline state (shim frames == audio timeline, SPEC.md §5).
        self.frames = 0
        self.bytes_read = 0
        self.quiet_run = 0
        self.quiet_start: int | None = None
        self.armed: bool = False
        self.chunks_since_arm = 0
        self.utt_start = 0
        self.utterance_open = False

        self.commands: list[dict] = []
        self.emitted: list[dict] = []
        self.quiet_commits: list[dict] = []
        self.exit_code = int(self.scn.get("exit_code") or 0)
        self._stop = threading.Event()
        self._exited = False
        self._finished = False

        pause_commit = self.scn.get("pause_commit")
        self.pause_commit = float(
            self.opt["pause_commit"] if pause_commit is None else pause_commit
        )
        self.pause_commit_frames = max(1, int(round(self.pause_commit * SAMPLE_RATE)))
        self.exit_after_frames = self.scn.get("exit_after_frames")
        self.partial_every_frames = int(self.scn.get("partial_every_frames") or 16_000)
        self.emit_partials = bool(self.scn.get("emit_partials", True))
        self.ctl_fd = int(os.environ.get("APPLE_ASR_CTL_FD") or -1)

        # Scripted-mode bookkeeping.
        self._step_done = [False] * len(self.scn.get("steps") or [])
        self._pending: list[dict] = []
        self._last_partial_frame = 0

    # ------------------------------------------------------------------
    # Wire
    # ------------------------------------------------------------------
    def emit(self, obj: dict) -> None:
        with self.lock:
            self.emitted.append(obj)
        line = json.dumps(obj, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def emit_hello(self) -> None:
        fmt = self.scn.get("format") or {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "common_format": "int16",
        }
        hello = {
            "type": "hello",
            "protocol": self.scn.get("protocol", PROTOCOL_VERSION),
            "shim_version": str(self.scn.get("shim_version") or SHIM_VERSION),
            "locale": self.scn.get("locale") or self.opt["locale"],
            "preset": self.scn.get("preset") or self.opt["preset"],
            "format": fmt,
            "capabilities": self.scn.get(
                "capabilities",
                ["volatile", "word_runs", "pause_commit", "flush", "context"],
            ),
            "reporting_options": self.scn.get(
                "reporting_options", ["volatileResults", "fastResults"]
            ),
        }
        if "hello_extra" in self.scn:
            hello.update(self.scn["hello_extra"])
        delay = float(self.scn.get("hello_delay_s") or 0.0)
        if delay:
            time.sleep(delay)
        if self.scn.get("first_line") is not None:
            self.emit(self.scn["first_line"])
            return
        self.emit(hello)

    def runs_for(self, start_frame: int, end_frame: int) -> list[list]:
        """Deterministic word runs covering [start, end) at 0.5 s each."""
        runs: list[list] = []
        span = max(0, end_frame - start_frame)
        if not span:
            return runs
        step = int(0.5 * SAMPLE_RATE)
        i = 0
        while i < span:
            a = start_frame + i
            b = min(end_frame, a + step)
            run: list = [f"w{len(runs)}", a / SAMPLE_RATE, b / SAMPLE_RATE]
            if self.opt["confidence"]:
                run.append(0.9)
            runs.append(run)
            i += step
        return runs

    def emit_final(self, start_frame: int, end_frame: int, reason: str) -> None:
        end_frame = max(start_frame, end_frame)
        self.emit(
            {
                "type": "final",
                "text": self.scn.get("final_text") or f"final-{start_frame}-{end_frame}",
                "range": [start_frame / SAMPLE_RATE, end_frame / SAMPLE_RATE],
                "runs": self.runs_for(start_frame, end_frame),
                "reason": reason,
            }
        )

    def emit_commit(self, through_frame: int, reason: str) -> None:
        self.emit(
            {
                "type": "commit",
                "through": through_frame / SAMPLE_RATE,
                "reason": reason,
                "wall": time.monotonic() - self.t0,
            }
        )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def read_loop(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            try:
                chunk = os.read(fd, READ_BYTES)
            except OSError:
                break
            if not chunk:
                with self.lock:
                    self.stdin_eof = True
                break
            self._consume(chunk)

    def _consume(self, chunk: bytes) -> None:
        n = len(chunk)
        frames = n // FRAME_BYTES
        silent = 0
        for (sample,) in struct.iter_unpack("<f", chunk[: frames * FRAME_BYTES]):
            if -SILENCE_EPS < sample < SILENCE_EPS:
                silent += 1
            else:
                break
        with self.lock:
            self.bytes_read += n
            self.frames += frames
            if silent:
                if self.quiet_run == 0:
                    self.quiet_start = self.frames - frames
                self.quiet_run += silent
            if silent < frames:
                # Speech resumed: reset the quiet run and the endpointer arming.
                self.quiet_run = 0
                self.quiet_start = None
                self.armed = False
                self.utterance_open = True
            if self.armed:
                self.chunks_since_arm += 1
            elif (
                self.quiet_run >= self.pause_commit_frames
                and self.quiet_start is not None
                and not self.scn.get("no_endpointer")
            ):
                self.armed = True
                self.chunks_since_arm = 0

    # ------------------------------------------------------------------
    # Commands (fd 3)
    # ------------------------------------------------------------------
    def poll_commands(self) -> None:
        if self.ctl_fd < 0:
            return
        while True:
            try:
                ready, _, _ = select.select([self.ctl_fd], [], [], 0)
            except (OSError, ValueError):
                return
            if not ready:
                return
            try:
                data = os.read(self.ctl_fd, 65536)
            except OSError:
                return
            if not data:
                return
            self._cmd_buf = getattr(self, "_cmd_buf", b"") + data
            while b"\n" in self._cmd_buf:
                line, self._cmd_buf = self._cmd_buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self.lock:
                    self.commands.append(obj)
                self.handle_command(obj)

    def handle_command(self, obj: dict) -> None:
        cmd = obj.get("cmd")
        if cmd == "finalize":
            if self.mode != "emulate":
                return
            through = obj.get("through")
            end = self.frames if through is None else int(round(float(through) * SAMPLE_RATE))
            end = max(self.utt_start, min(self.frames, end))
            self.emit_final(self.utt_start, end, "flush")
            self.emit_commit(end, "flush")
            self.utt_start = end
        elif cmd == "close":
            self.finish("closed")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        self.emit_hello()
        for raw in self.scn.get("raw_lines") or []:
            sys.stdout.write(raw + "\n")
            sys.stdout.flush()
        for line in self.scn.get("stderr") or []:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()

        self.stdin_eof = False
        reader = threading.Thread(target=self.read_loop, name="fake-shim-reader", daemon=True)
        reader.start()

        exit_code = None
        while True:
            self.poll_commands()
            if self._finished:
                exit_code = self.exit_code
                break
            if self.mode == "scripted":
                self.run_steps()
            elif self.mode == "emulate":
                self.run_emulate()
            # else: silent — consume and emit nothing.
            if self._finished:
                exit_code = self.exit_code
                break

            with self.lock:
                frames, eof = self.frames, self.stdin_eof
            if self.exit_after_frames is not None and frames >= int(self.exit_after_frames):
                exit_code = self.exit_code
                break
            if eof and not self.scn.get("ignore_eof"):
                exit_code = self.finish("eof")
                break
            if time.monotonic() - self.t0 > MAX_WALL_S:
                sys.stderr.write("fake-shim: max wall time exceeded\n")
                exit_code = 97
                break
            time.sleep(TICK_S)
        self._stop.set()
        return int(exit_code if exit_code is not None else self.exit_code)

    def finish(self, why: str) -> int:
        """Finalize the open utterance, emit `ended`, and stop the loop.

        Idempotent: only the first caller emits (a `close` command followed by
        stdin EOF must not report two endings).
        """
        if self._finished:
            return self.exit_code
        self._finished = True
        if self.mode == "emulate" and not self.scn.get("skip_final_finalize"):
            end = self.frames
            if end > self.utt_start or self.utterance_open:
                self.emit_final(self.utt_start, end, "eof")
            self.emit_commit(end, "eof")
        self.emit({"type": "ended", "reason": why})
        return self.exit_code

    def run_emulate(self) -> None:
        with self.lock:
            armed = self.armed
            chunks = self.chunks_since_arm
            frames = self.frames
            quiet_start = self.quiet_start
        if armed and chunks >= 1:
            end = self.quiet_start if quiet_start is None else quiet_start
            self.emit_final(self.utt_start, end, "pause")
            self.emit_commit(end, "pause")
            with self.lock:
                self.quiet_commits.append({"frame": end, "wall": time.monotonic() - self.t0})
                self.utt_start = end
                self.armed = False
                self.chunks_since_arm = 0
                self.utterance_open = False
                self._last_partial_frame = frames
            return
        if self.emit_partials and frames - self._last_partial_frame >= self.partial_every_frames:
            with self.lock:
                self._last_partial_frame = frames
            self.emit(
                {
                    "type": "partial",
                    "text": f"partial-{frames}",
                    "range": [self.utt_start / SAMPLE_RATE, frames / SAMPLE_RATE],
                    "runs": [],
                }
            )

    # ------------------------------------------------------------------
    # Scripted mode
    # ------------------------------------------------------------------
    def trigger_fires(self, step: dict) -> bool:
        """True when a scripted step's trigger condition is met right now."""
        on = step.get("on", "start")
        with self.lock:
            frames, cmds, eof = self.frames, list(self.commands), self.stdin_eof
        if on == "start":
            return True
        if on == "frames":
            return frames >= int(step["value"])
        if on == "bytes":
            return frames * FRAME_BYTES >= int(step["value"])
        if on == "cmd":
            return any(c.get("cmd") == step["value"] for c in cmds)
        if on == "eof":
            return eof
        raise ValueError(f"unknown trigger {on!r}")

    def run_steps(self) -> None:
        for i, step in enumerate(self.scn.get("steps") or []):
            if self._step_done[i] or not self.trigger_fires(step):
                continue
            self._step_done[i] = True
            if (sleep := step.get("sleep")) is not None:
                time.sleep(float(sleep))
            if "emit" in step and not step.get("count"):
                repeat = int(step.get("repeat") or 1)
                for _ in range(repeat):
                    self.emit(step["emit"])
            if "raw" in step:
                repeat = int(step.get("repeat") or 1)
                for _ in range(repeat):
                    sys.stdout.write(str(step["raw"]) + "\n")
                    sys.stdout.flush()
            # `count` copies of `emit`, with the index appended to `text`.
            for i in range(int(step.get("count") or 0)):
                obj = dict(step["emit"])
                obj["text"] = str(step["emit"].get("text", "")) + str(i)
                self.emit(obj)
            if step.get("exit") is not None:
                raise SystemExit(int(step["exit"]))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def dump_state(self, status: str) -> None:
        path = os.environ.get("APPLE_ASR_FAKE_STATE")
        if not path:
            return
        with self.lock:
            state = {
                "status": status,
                "frames": self.frames,
                "bytes": self.bytes_read,
                "commands": self.commands,
                "emitted": self.emitted,
                "quiet_commits": self.quiet_commits,
                "argv": self.argv,
                "locale": self.opt["locale"],
                "preset": self.opt["preset"],
                "confidence": self.opt["confidence"],
                "fast_results": self.opt["fast_results"],
                "volatile": self.opt["volatile"],
                "context": self.opt["context"],
                "pause_commit": self.pause_commit,
                "mode": self.mode,
            }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False)
        except OSError:
            pass

    def _fatal(self, message: str) -> None:
        sys.stderr.write(f"fake-shim: {message}\n")
        raise SystemExit(2)

    def main(self) -> int:
        status = "ok"
        try:
            code = self.run()
        except SystemExit as exc:  # scripted `exit` step / parse failures
            code = int(exc.code or 0)
            status = "scripted_exit"
        finally:
            self.dump_state(status)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except OSError:
            pass
        return int(code)


def _parse_args(argv: list[str]) -> dict:
    opt: dict = {
        "locale": "en-US",
        "preset": "progressive",
        "confidence": True,
        "fast_results": True,
        "volatile": True,
        "pause_commit": 0.08,
        "commit_interval": 0.0,
        "context": [],
        "mode": None,
    }
    args = list(argv)
    while args:
        a = args.pop(0)
        if a in ("--locale", "--preset", "--pause-commit", "--commit-interval",
                 "--context", "--vad-sensitivity", "--file"):
            if not args:
                raise SystemExit(2)
            v = args.pop(0)
            if a == "--locale":
                opt["locale"] = v
            elif a == "--preset":
                opt["preset"] = v
            elif a == "--pause-commit":
                opt["pause_commit"] = float(v)
            elif a == "--commit-interval":
                opt["commit_interval"] = float(v)
            elif a == "--context":
                opt["context"] = [s for s in (x.strip() for x in v.split(",")) if s]
        elif a == "--stdin":
            opt["mode"] = "stdin"
        elif a == "--mic":
            opt["mode"] = "mic"
        elif a == "--no-confidence":
            opt["confidence"] = False
        elif a == "--no-fast":
            opt["fast_results"] = False
        elif a == "--fast":
            opt["fast_results"] = True
        elif a == "--no-volatile":
            opt["volatile"] = False
        elif a in ("--help", "-h", "--list-locales", "--ensure-installed"):
            opt.setdefault("special", []).append(a)
            if a == "--ensure-installed" and args:
                opt["ensure_installed_locale"] = args.pop(0)
        else:
            sys.stderr.write(f"fake-shim: unknown arg: {a}\n")
            raise SystemExit(2)
    return opt


def main() -> int:
    shim = FakeShim()
    special = shim.opt.get("special") or []
    if "--help" in special or "-h" in special:
        print("usage: apple-asr-shim {--stdin|--file PATH|--mic} [options]")
        return 0
    if "--list-locales" in special:
        shim.emit(
            {
                "type": "locales",
                "supported": shim.scn.get("supported_locales", ["en-US", "zh-TW"]),
                "installed": shim.scn.get("installed_locales", ["en-US"]),
            }
        )
        shim.dump_state("locales")
        return 0
    if "--ensure-installed" in special:
        if shim.scn.get("ensure_installed_fails"):
            sys.stderr.write(
                f"locale {shim.opt.get('ensure_installed_locale')} could not be installed\n"
            )
            shim.dump_state("ensure_failed")
            return 1
        shim.dump_state("ensure_ok")
        return 0
    return shim.main()


if __name__ == "__main__":
    raise SystemExit(main())
