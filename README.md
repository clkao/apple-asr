# apple-asr

Streaming, on-device speech-to-text for macOS 26+ from Python, on Apple's
SpeechAnalyzer stack (`SpeechTranscriber`): live partials, finalized segments
with per-word timings, push-based audio, and pause-aware commits.

Status: **order 1** — package skeleton, transport, public API, packaging. The
test suite, the replay/CER console scripts, and the README's full API docs land
in order 2. The frozen interface contract is
[`_work/apple-asr-package/SPEC.md`](../../WhisperLiveKit/_work/apple-asr-package/SPEC.md).

## Install

```bash
pip install apple-asr          # or: uv pip install apple-asr
python -m apple_asr.build      # compiles the bundled Swift shim into ~/.cache/apple_asr
```

Requires macOS 26+ and, for the build step, a Swift toolchain (`swiftc`).
Everywhere else the package raises `UnsupportedPlatform` / `ShimUnavailable`
with the exact command to run.

## Quickstart

```python
import numpy as np
from apple_asr import Stream, Final, Partial, Ended

with Stream(locale="zh-TW", preset="progressive") as st:
    st.push(np.zeros(16000, dtype=np.float32))   # caller resamples to 16 kHz mono float32
    st.pause_start()
    st.pause_end(0.4)
    st.flush()                                    # one commit acknowledgement
    for ev in st.events():
        if isinstance(ev, Final):
            print(ev.text, [(w.text, round(w.start, 2)) for w in ev.words])
        elif isinstance(ev, Ended):
            break
```

## The clock (the subtle part)

The shim owns the audio timeline: frames written to it *are* its clock. When a
caller's VAD strips silence (WhisperLiveKit does), the client synthesizes the
pause back as silence at the audio rate and maps shim time to **session time**
with a drift term. Session timestamps start at 0 and advance with consumed
audio, including synthesized pauses; they are monotonic, in seconds. Any future
transport (C ABI, shared memory) must preserve those semantics, not the wire
format.

## Public API

`Stream`, `Partial`, `Final`, `Word`, `Ended`, `Error`, `list_locales`,
`ensure_installed`, `shim_info`, `PROTOCOL_VERSION`, and the exception
hierarchy. No Apple type names appear in the public surface; the JSONL wire
protocol is an internal transport detail.

## Known limitations

macOS 26+ only; the synthesized-pause mode simulates audio the caller's VAD
discarded; commits depend on the framework's endpointer accepting a pause as an
utterance boundary (`flush()` is the escape hatch, not a mid-phrase force-cut);
`SpeechDetector` VAD sensitivity is exposed but inert in the prototype.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
ruff check .
```
