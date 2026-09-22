"""Real-time replay driver (SPEC.md §11 order 1 item 5, §9 tests 10-11).

Feeds a 16 kHz mono wav through :class:`~apple_asr.Stream` the way a live
pipeline does — active audio via ``push()``, pauses announced with
``pause_start()``/``pause_end(d)`` (the caller's VAD strips silence, so the
client synthesizes it) — and measures how long each pause's commit takes to
land. This is the tool behind the measured cadence in MEASUREMENTS.md, ported
from the prototype's ``_work/sa-spike/drive_live.py``.

Programmatic use::

    from apple_asr.replay import load_wav, replay

    audio, rate = load_wav("clip.wav")
    result = replay(audio, sample_rate=rate, locale="zh-TW")
    print(result.summary())
    assert result.max_pause_latency < 0.3

Console script::

    apple-asr-replay clip.wav --locale zh-TW --max-latency 0.3
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .events import Ended, Event, Final, Partial
from .stream import Stream

__all__ = [
    "PauseCommit",
    "ReplayResult",
    "load_wav",
    "segment",
    "replay",
    "main",
]

#: A 20 ms frame is above this RMS counts as speech (a crude, dependency-free VAD).
DEFAULT_RMS_SPEECH = 0.004
DEFAULT_MIN_SILENCE_S = 0.15
DEFAULT_MIN_SPEECH_S = 0.10
DEFAULT_WINDOW_S = 0.02
DEFAULT_CHUNK_S = 0.1
#: At `pace=0` a pause has no real duration to wait out; observe this long.
UNPACED_PAUSE_WINDOW_S = 0.3


@dataclass(frozen=True)
class PauseCommit:
    """One detected pause and the commit that answered it."""

    index: int
    audio_t: float
    pause_s: float
    latency_s: float | None  # wall seconds from pause_start() to its Final
    reason: str | None
    text: str


@dataclass
class ReplayResult:
    """Everything one replay observed."""

    spans: list[tuple[str, float, float]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    pause_commits: list[PauseCommit] = field(default_factory=list)
    wall_s: float = 0.0
    audio_s: float = 0.0

    @property
    def finals(self) -> list[Final]:
        return [e for e in self.events if isinstance(e, Final)]

    @property
    def partials(self) -> list[Partial]:
        return [e for e in self.events if isinstance(e, Partial)]

    @property
    def ended_reason(self) -> str | None:
        for event in reversed(self.events):
            if isinstance(event, Ended):
                return event.reason
        return None

    @property
    def misses(self) -> list[PauseCommit]:
        """Pauses whose commit never landed (latency is None)."""
        return [c for c in self.pause_commits if c.latency_s is None]

    @property
    def commit_latencies(self) -> list[float]:
        return [c.latency_s for c in self.pause_commits if c.latency_s is not None]

    @property
    def max_pause_latency(self) -> float | None:
        latencies = self.commit_latencies
        return max(latencies) if latencies else None

    @property
    def first_commit_t(self) -> float | None:
        return self.finals[0].end if self.finals else None

    @property
    def mean_commit_gap(self) -> float | None:
        ends = [f.end for f in self.finals]
        if len(ends) < 2:
            return None
        gaps = [b - a for a, b in zip(ends, ends[1:], strict=False)]
        return sum(gaps) / len(gaps)

    def summary(self) -> str:
        parts = [
            f"spans={len(self.spans)}",
            f"events={len(self.events)}",
            f"partials={len(self.partials)}",
            f"finals={len(self.finals)}",
            f"wall={self.wall_s:.1f}s",
            f"audio={self.audio_s:.1f}s",
            f"ended={self.ended_reason}",
            f"pauses={len(self.pause_commits)}",
            f"missed={len(self.misses)}",
        ]
        if (latency := self.max_pause_latency) is not None:
            parts.append(f"max_pause_latency={latency:.3f}s")
        if (first := self.first_commit_t) is not None:
            parts.append(f"first_commit={first:.2f}s")
        if (gap := self.mean_commit_gap) is not None:
            parts.append(f"mean_commit_gap={gap:.2f}s")
        return " ".join(parts)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16 kHz mono PCM wav as float32 (no resampling: caller's job)."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    if channels != 1:
        raise ValueError(f"{path}: expected mono, got {channels} channels")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(audio), rate


def segment(
    audio: np.ndarray,
    *,
    sample_rate: int = 16_000,
    rms_speech: float = DEFAULT_RMS_SPEECH,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    min_speech_s: float = DEFAULT_MIN_SPEECH_S,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[tuple[str, float, float]]:
    """Crude VAD: a list of ``("speech"|"silence", start_s, end_s)`` spans.

    Runs shorter than `min_speech_s` / `min_silence_s` are merged into their
    neighbour, so a live pipeline's pause announcements are realistic.
    """
    window = max(1, int(window_s * sample_rate))
    spans: list[tuple[str, float, float]] = []
    kind: str | None = None
    start = 0.0
    for i in range(0, len(audio), window):
        block = audio[i : i + window]
        rms = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0
        current = "speech" if rms >= rms_speech else "silence"
        t = i / sample_rate
        if kind is None:
            kind, start = current, 0.0
        elif current != kind:
            spans.append((kind, start, t))
            kind, start = current, t
    spans.append((kind or "speech", start, len(audio) / sample_rate))

    merged: list[tuple[str, float, float]] = []
    for span_kind, span_start, span_end in spans:
        floor = min_silence_s if span_kind == "silence" else min_speech_s
        if merged and merged[-1][0] == span_kind:
            # Adjacent runs of the same kind are one span.
            merged[-1] = (span_kind, merged[-1][1], span_end)
            continue
        if span_end - span_start < floor and merged:
            previous_kind, previous_start, _ = merged[-1]
            merged[-1] = (previous_kind, previous_start, span_end)
            continue
        merged.append((span_kind, span_start, span_end))
    return merged


def replay(
    audio: np.ndarray,
    *,
    sample_rate: int = 16_000,
    locale: str = "en-US",
    pace: float = 1.0,
    chunk_s: float = DEFAULT_CHUNK_S,
    rms_speech: float = DEFAULT_RMS_SPEECH,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    min_speech_s: float = DEFAULT_MIN_SPEECH_S,
    stream: Stream | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    on_event: Callable[[Event], None] | None = None,
    on_pause: Callable[[PauseCommit], None] | None = None,
) -> ReplayResult:
    """Drive `audio` through a `Stream` at `pace`x real time and measure commits.

    `pace=1.0` is real time (the live pattern); `pace=0` means "as fast as the
    caller can feed" (a burst — commits then depend on the shim's own
    endpointer, so pause latencies are not meaningful). Pass `stream` to reuse a
    session (tests drive a fake shim that way), or `stream_kwargs` to build one.
    """
    result = ReplayResult(spans=segment(
        audio,
        sample_rate=sample_rate,
        rms_speech=rms_speech,
        min_silence_s=min_silence_s,
        min_speech_s=min_speech_s,
    ))
    owns_stream = stream is None
    if stream is None:
        kwargs = dict(stream_kwargs or {})
        kwargs.setdefault("locale", locale)
        stream = Stream(**kwargs)
    session = stream

    chunk_frames = max(1, int(chunk_s * sample_rate))
    started = time.monotonic()

    def pump() -> list[Event]:
        events = session.poll(0.0)
        result.events.extend(events)
        if on_event is not None:
            for event in events:
                on_event(event)
        return events

    try:
        for index, (kind, span_start, span_end) in enumerate(result.spans):
            if kind == "speech":
                first, last = int(span_start * sample_rate), int(span_end * sample_rate)
                for i in range(first, last, chunk_frames):
                    session.push(audio[i : i + chunk_frames])
                    pump()
                    if pace > 0:
                        time.sleep(min(chunk_frames, last - i) / sample_rate / pace)
            else:
                audio_t = session.audio_time
                pause_s = span_end - span_start
                pause_started = time.monotonic()
                session.pause_start()
                latency: float | None = None
                reason: str | None = None
                text = ""

                # Keep the pause open for its real duration (the live pattern),
                # polling as a live pipeline would. At `pace=0` (a burst feed)
                # there is no real duration to wait out, so observe briefly.
                window = pause_s / pace if pace > 0 else UNPACED_PAUSE_WINDOW_S
                deadline = pause_started + window
                while time.monotonic() < deadline:
                    for event in pump():
                        if isinstance(event, Final) and latency is None:
                            latency = time.monotonic() - pause_started
                            reason = event.reason
                            text = event.text
                    if latency is not None:
                        break
                    time.sleep(0.01)
                for event in pump():  # one last look as the pause closes
                    if isinstance(event, Final) and latency is None:
                        latency = time.monotonic() - pause_started
                        reason = event.reason
                        text = event.text
                session.pause_end(pause_s)
                commit = PauseCommit(
                    index=index,
                    audio_t=audio_t,
                    pause_s=pause_s,
                    latency_s=latency,
                    reason=reason,
                    text=text,
                )
                result.pause_commits.append(commit)
                if on_pause is not None:
                    on_pause(commit)
        session.flush()
        time.sleep(0.05)
        pump()
    finally:
        result.audio_s = session.audio_time
        if owns_stream:
            session.close()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not pump():
                    break
                time.sleep(0.01)
        result.wall_s = time.monotonic() - started
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apple-asr-replay",
        description=(
            "Replay a 16 kHz mono wav through apple_asr at real time, printing "
            "each pause's commit latency (the live-pipeline pattern)."
        ),
    )
    parser.add_argument("audio", help="16 kHz mono 16-bit PCM wav")
    parser.add_argument("--locale", default="en-US", help="BCP-47 locale (default en-US)")
    parser.add_argument(
        "--mode",
        choices=["streaming", "accurate"],
        default=None,
        help="streaming (default: progressive + fastResults) or accurate",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="1.0 = real time, 0 = as fast as possible (default 1.0)",
    )
    parser.add_argument(
        "--pause-commit", type=float, default=None, help="quiet seconds before a commit"
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="exit 1 if any pause commit lands later than this many seconds",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for `apple-asr-replay`."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    import json

    audio, rate = load_wav(args.audio)
    if rate != 16_000:
        print(
            f"apple-asr-replay: {args.audio} is {rate} Hz; resample to 16000 Hz first",
            file=sys.stderr,
        )
        return 2

    def show(commit: PauseCommit) -> None:
        if args.quiet:
            return
        latency = "no commit" if commit.latency_s is None else f"+{commit.latency_s:.3f}s"
        label = f"[{commit.audio_t:6.2f}s] pause {commit.pause_s:.2f}s -> {latency}"
        if commit.text:
            label += f"  FINAL[{commit.reason}] {commit.text[:60]!r}"
        print(label)

    kwargs: dict[str, Any] = {"locale": args.locale}
    if args.mode is not None:
        kwargs["mode"] = args.mode
    if args.pause_commit is not None:
        kwargs["pause_commit"] = args.pause_commit
    result = replay(
        audio,
        sample_rate=rate,
        pace=args.pace,
        stream_kwargs=kwargs,
        on_pause=show,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "summary": result.summary(),
                    "spans": result.spans,
                    "audio_s": result.audio_s,
                    "wall_s": result.wall_s,
                    "finals": [f.end for f in result.finals],
                    "pause_latencies": result.commit_latencies,
                    "missed_pauses": len(result.misses),
                    "ended": result.ended_reason,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(result.summary())

    if (latency := result.max_pause_latency) is not None and args.max_latency is not None:
        if latency > args.max_latency:
            print(
                f"apple-asr-replay: max pause latency {latency:.3f}s exceeds "
                f"--max-latency {args.max_latency:.3f}s",
                file=sys.stderr,
            )
            return 1
    if args.max_latency is not None and result.misses:
        print(
            f"apple-asr-replay: {len(result.misses)} pause(s) never committed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
