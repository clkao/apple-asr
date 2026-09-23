# apple-asr

Streaming, on-device speech-to-text for **macOS 26+** from Python, on Apple's
SpeechAnalyzer stack (`SpeechTranscriber`).

* **live** operation: volatile partials as they form, plus finalized segments
* **per-word timings** on finals (caption/MT pipelines need them)
* **push-based audio**: your pipeline owns capture and pacing
* **pause-aware commits** for callers whose VAD strips silence
* no Apple type names in the public API — JSONL is an internal detail

Measured accuracy on the project's FLEURS board (mean per-sample zh CER, n=10;
full evidence in `_work/apple-asr-package/MEASUREMENTS.md`):

| mode | configuration | zh CER |
|---|---|---|
| `mode="accurate"` | `preset=transcription`, no `fastResults` (finals only) | **11.23** |
| `mode="streaming"` (default) | `preset=progressive` + `fastResults` (volatile partials) | **13.19** |

Both beat every local MLX backend on that board (best: qwen3-1.7B, 18.20). The
~2 CER points are the measured price of volatile partials plus emission speed.

## Install

**From a platform wheel — no Swift toolchain needed.** The `wheel` workflow
(`.github/workflows/wheel.yml`, run on a `macos-26` runner) compiles the shim
into the package and attaches the resulting wheel to the GitHub release for the
tag:

```bash
pip install ./apple_asr-0.1.1-py3-none-macosx_26_0_arm64.whl   # or the release URL
python -c "import apple_asr; print(apple_asr.shim_info())"      # uses the bundled shim
```

No Xcode CLT, no `swiftc`, no build step: the first `Stream(...)` runs the shim
that shipped inside the wheel. The wheel is **platform-tagged**
(`macosx_26_0_arm64`, not `py3-none-any`) so pip refuses it on Intel Macs and on
macOS < 26 — where the bundled arm64 binary could not run at all.

**From source — needs a Swift toolchain.**

```bash
pip install apple-asr              # or: uv pip install apple-asr
python -m apple_asr.build          # compiles the bundled Swift shim into the cache
```

Requires **macOS 26+** either way; elsewhere the package raises
`UnsupportedPlatform` with the requirement in the message. The source build needs
a Swift toolchain (`swiftc`); if it is missing you get `ShimUnavailable` naming
the exact command. The first session for a locale may download that locale's
on-device asset (`ensure_installed("zh-TW")` does it explicitly).

The shim is resolved in this order (first hit wins): `Stream(shim=...)` →
`$APPLE_ASR_SHIM` → the package cache
`~/.cache/apple_asr/<version>/apple-asr-shim` → `apple-asr-shim` on `PATH` →
**the shim bundled in the wheel**, `apple_asr/shim/apple-asr-shim` (absent from a
source install) → build-on-demand → `ShimUnavailable`.

**Executable bit.** The executable bit of wheel package data is not something
to rely on. The zip member carries a mode (hatchling records 0755, and pip/uv on
POSIX honour it today), but honouring it is installer courtesy, not a wheel
contract — unpack the same wheel with an extractor that does not restore modes
(`python -m zipfile -e`, an artifact-zip hop, a Windows-side unzip) and the shim
lands at `0644`. A `0644` shim then fails at `execv` with `EACCES`, which reads
as "no shim" rather than "shim not executable". The resolver therefore
`chmod +x`es the shim on first use, for the two locations the package owns (the
version cache and the bundled copy). A path you supply yourself
(`Stream(shim=...)`, `$APPLE_ASR_SHIM`, a `PATH` hit) is left alone. See
`apple_asr.shim.ensure_executable`.

**Cache escape hatch.** The cache root is `$APPLE_ASR_CACHE` (falling back to
`$XDG_CACHE_HOME`, then `~/.cache`). Set it when `~/.cache` is not writable —
sandboxes, containers, CI:

```bash
export APPLE_ASR_CACHE="$PWD/.cache"
```

## Quickstart — streaming (live captions)

```python
import numpy as np
from apple_asr import Final, Partial, Stream

with Stream(locale="en-US", mode="streaming") as st:   # progressive + fastResults
    st.push(np.zeros(16_000, dtype=np.float32))         # 16 kHz mono float32, caller resampled
    st.pause_start()
    st.pause_end(0.4)                                  # the VAD pause you stripped, in seconds
    st.flush()                                         # one `commit` acknowledgement
    for event in st.poll(1.0):                         # non-blocking; `events()` blocks instead
        if isinstance(event, Partial):
            print("volatile:", event.text)             # supersede, never append
        elif isinstance(event, Final):
            print("final:", round(event.start, 2), round(event.end, 2), event.text)
            print("words:", [(w.text, round(w.start, 2)) for w in event.words])
    print("session clock:", round(st.audio_time, 2))
```

## Quickstart — accurate (file/offline work, finals only)

```python
import numpy as np
from apple_asr import Final, Stream

with Stream(locale="zh-TW", mode="accurate") as st:    # transcription, no fastResults
    st.push(np.zeros(16_000, dtype=np.float32))
    st.flush()
    for event in st.poll(1.0):
        if isinstance(event, Final):
            print(event.reason, round(event.end, 2), event.text)
```

`mode` is a convenience over two explicit knobs, which keep working:

| `mode` | `preset` | `reporting_option` |
|---|---|---|
| `"streaming"` (default) | `"progressive"` | `"fastResults"` |
| `"accurate"` | `"transcription"` | `None` |

`Stream(preset=..., reporting_option=...)` without `mode=` is authoritative (so
every earlier spelling keeps working). Passing `mode=` **and** a contradicting
knob raises `ValueError` naming both values rather than silently picking one.

### Replay/instrumentation

```bash
apple-asr-replay clip.wav --locale zh-TW --max-latency 0.3
```

Feeds a 16 kHz mono wav at real time through the live push/pause pattern and
prints each pause's commit latency; exits 1 if any commit exceeds
`--max-latency`. `--mode`, `--pace`, `--pause-commit` and `--json` are also
available; `apple_asr.replay.replay()` returns the same measurements as data.

## The clock (the subtle part)

**The shim owns the audio timeline**: frames written to it *are* its clock. When
your VAD strips silence, this client synthesizes that silence back (a pre-roll at
`pause_start()`, then silence at ~0.8× the audio rate until `pause_end(d)`, which
tops up to exactly `d` seconds — asserted frame-exactly in the test suite).

Session timestamps start at 0, advance with consumed audio **including
synthesized pauses** (SPEC §3.3), and are monotonic seconds. The client maps shim
time → session time with a boundary-anchored pair: an anchor `(shim_frame,
session_s)` is refreshed at every input boundary — each `push()`,
`pause_start()` and `pause_end(d)` — and a shim time `t` maps as
`anchor_session + (t·rate − anchor_shim_frame)/rate`, 1:1 from the anchor. The
anchor is what keeps the two timelines aligned where it matters: the pre-roll and
the pump write synthesized silence the caller only reports at `pause_end`, so
while a pause is open the **pause-onset anchor stays in force** and shim times
inside the pause map 1:1 from the pause onset — a commit the framework publishes a
few milliseconds into that silence maps back onto the pause onset instead of being
pulled back by silence you had already fed but not yet reported (the old global
`drift = session_consumed − frames_written`, which is transiently stale mid-pause).
Once a future transport carries the true audio, no anchor is ever needed and the
mapping is identity; any future transport (C ABI, shared memory) must preserve
these semantics, not the JSONL wire format.

This is what keeps §"ranges never overlap" true *by construction* — the mapping is
monotonically non-decreasing and 1:1, and the shim's final ranges tile its
timeline, so consecutive mapped ranges are contiguous with no clamp and no nudged
timestamp (`Final.start` is the shim's own range start, mapped). If you hold a
pause open longer than the `d` you report, the pump has already written more
silence (`P > d`); that excess is real elapsed audio the shim consumed, so the
session clock gains it too (it advances with consumed audio) and the two timelines
stay 1:1 — `Stream.audio_time` follows the consumed audio, not the sum of the
durations you reported. Counting only `d` would leave the clocks apart for the rest
of the session, and the first final after the pause would map back over the one the
pause already published. See `tests/test_clock_accounting.py` for the regression
tests (mid-pause exactness, the engineered drift-divergence case, and the
over-delivered pause).

## Input modes (both first-class)

| your upstream | what to call |
|---|---|
| continuous audio (silence included) | `push()` only — pauses are real, the framework handles them |
| VAD that strips silence (e.g. WhisperLiveKit) | `push()` for active audio + `pause_start()` / `pause_end(d)` |

`pause_commit` (default **0.08 s**) is the pause-mode knob: quiet seconds before a
pause commits. It is a declared capability, not a hack hidden in the client.
`commit_interval` defaults to **0** (pause-only) because a commit ceiling was
measured *worse*: it cut mid-phrase and corrupted characters (越 → 月), poisoning
translations.

## Public API

```python
from apple_asr import (
    Stream,                 # the session object
    Partial, Final, Word,   # event types (frozen dataclasses)
    Ended, Error,           # terminal / error events
    list_locales, ensure_installed, shim_info,
    PROTOCOL_VERSION,
    AppleAsrError, UnsupportedPlatform, ShimUnavailable, AssetUnavailable,
    ProtocolMismatch, SessionClosed, BackendError,
)
```

`Stream`: `prepare()`, `close()`, context manager, `push(pcm)`,
`pause_start()`, `pause_end(d)`, `flush(through_s=None)`, `audio_time`,
`events()` / `aevents()` / `poll(timeout_s)`, `stats`, plus the resolved
`mode` / `preset` / `reporting_option` / `locale` / `pause_commit` /
`commit_interval` / `silent_timeout` attributes. Do not mix `events()`,
`aevents()` and `poll()` on one instance (asserted).

**Silent-backend guard.** `silent_timeout` (default **30 s**) raises
`BackendError` instead of hanging when a live shim produces *nothing at all*
while audio is being fed at real time or slower. It is feed-rate aware on
purpose: a fast-fed burst (whole-file/offline feeds hand the shim 30 s of audio
in 50 ms) may legitimately be quiet for seconds, so a burst stands the guard
down. `silent_timeout=0` disables it.

## Wire protocol (internal — not public API)

The shim is a separate executable. `hello` is always the first stdout line, then
JSONL events; stdin carries float32 LE mono 16 kHz PCM; fd 3 (overridable with
`$APPLE_ASR_CTL_FD`) carries JSONL commands (`prepare`, `finalize`,
`context`, `close`). The client refuses a protocol or format mismatch (naming
both sides) and ignores unknown event types with a warning.

```json
{"type":"hello","protocol":1,"shim_version":"0.1.0","locale":"zh-TW","preset":"progressive",
 "format":{"sample_rate":16000,"channels":1,"common_format":"int16"},
 "capabilities":["volatile","word_runs","pause_commit","flush","context"],
 "reporting_options":["volatileResults","fastResults"]}
{"type":"partial","text":"我們今天來","range":[0.0,2.94],"runs":[]}
{"type":"final","text":"我們今天來討論鐳射在醫學上的應用","range":[0.0,3.23],
 "runs":[["我們",0.0,0.5],["今天",0.5,1.1]],"reason":"pause"}
{"type":"commit","through":3.23,"reason":"pause","wall":1.42}
{"type":"error","message":"...","detail":"..."}
{"type":"ended","reason":"eof"}
```

* `runs` is `[[word, start, end], ...]` in seconds on the session clock; `[]` is
  legal. **Extension:** with `confidence=True` (the default) the shim emits a 4th
  element, `[word, start, end, confidence]`, giving `Word.confidence`; the client
  accepts both 3- and 4-element runs and maps 3-element runs to
  `confidence=None`.
* `commit` is the shim's acknowledgement that it acted on a finalize decision; it
  is what `flush()` waits for and where a final without its own `reason` gets it.
* `Final.reason` (`pause` | `interval` | `flush` | `eof`) says *why* the commit
  landed — the observability that made live latency debuggable.

## Known limitations (declared, not to be fixed in v1)

* **macOS 26+ only**; the framework does not exist earlier.
* The synthesized-pause mode *simulates* audio your VAD discarded; the analyzer's
  timeline then carries that silence. Preferred upstream fix (a WhisperLiveKit-side
  capability flag so silence is fed through) is a follow-up, not v1.
* Test-suite flakes: see *Known issues* under Development.
* The first commit is pause-bound; a speaker who never pauses commits at
  `commit_interval` — which is off by default because a ceiling cut mid-phrase and
  corrupted text.
* Commits depend on the framework's own endpointer accepting a pause as an
  utterance boundary. `flush()` is the escape hatch, not a mid-phrase force-cut.
* `--vad-sensitivity` (`SpeechDetector`) is exposed but was **inert** in the
  prototype.
* `--file`/`--mic` are the shim's own modes; `--mic` needs microphone TCC (run
  from Terminal.app if your terminal cannot get it).
* `apple-asr-shim --fast` is a **deprecated no-op** kept for CLI compatibility
  (`fastResults` is on unless `--no-fast`); slated for removal in a future major.
* The CER/WER harness is not ported into this package yet (the numbers above come
  from the workspace's `bench_cer.py`).

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest -q -m "not integration"     # anywhere: fake shim, no macOS/Speech/cache
ruff check .
```

The non-integration suite drives the real transport through a scripted **fake
shim** (`tests/fixtures/fake_shim.py`) installed into a tmp `APPLE_ASR_CACHE`, so
it runs on Linux/CI and never touches `~/.cache`, the Speech framework, or a
microphone.

The macOS 26 integration tests (`-m integration`) use the real shim on the
bundled clip (`tests/fixtures/audio/zh_long.wav`) and compare cadence with the
golden fixture (`tests/golden/zh_long_ideal.jsonl`). On macOS 26 they **fail**
rather than skip if the shim cannot be resolved or built:

```bash
python -m apple_asr.build
export APPLE_ASR_CACHE="$PWD/.cache"   # only if ~/.cache is not writable
pytest -q -m integration
```

### Building a platform wheel yourself

CI and a local build run the same script (`bash scripts/build_wheel.sh`; the
workflow only adds checkout/UV setup, artifact upload and the release asset):

```bash
bash scripts/build_wheel.sh        # -> dist/apple_asr-0.1.1-py3-none-macosx_26_0_arm64.whl
```

It compiles the shim (`swiftc -O -parse-as-library`) into
`src/apple_asr/shim/apple-asr-shim`, builds the ordinary `py3-none-any` wheel with
`uv build --wheel`, then retags it with `wheel tags --platform-tag
macosx_26_0_arm64 --remove`. The retag is deliberate: the package is pure Python
with one bundled binary, so a single `py3-none-<platform>` wheel should serve
CPython 3.10-3.13 on macOS 26 arm64. The alternative — a setuptools
`BinaryDistribution.has_ext_modules()` override — would emit an
interpreter-specific `cp313-cp313-macosx_...` wheel per Python version (and mean
leaving hatchling) for no benefit. The script refuses to finish unless the binary
is arm64, the tag is platform-specific, `apple_asr/shim/apple-asr-shim` is inside
the zip, and the `any`-tagged original is gone. `PLATFORM_TAG=...` overrides the
tag. The binary is a build product (`.gitignore`) that pyproject's wheel
`artifacts` re-includes in the zip, and the sdist never carries it.

### Fixture attribution

`tests/fixtures/audio/zh_long.wav` is a 31.55 s / 16 kHz mono clip derived from
the `google/fleurs` corpus, which is licensed **CC-BY-4.0**; see
`tests/fixtures/audio/README.md` for the full notice. It is a cadence fixture for
the golden test (that golden fixture was produced from exactly this audio), not a
general-purpose audio sample.

### Known issues

* **Unresolved flake in the full suite:**
  `tests/test_control.py::test_flush_without_through_uses_the_cursor` failed once
  in a full-suite run — it expected an exact shipped range `(0.0, 1.0)` — but
  never in isolation (3/3 passes) or in the fake-shim subset (4/4). Suspected
  cause: the fake shim's main loop polls its control fd *before* it reads stdin,
  so a `finalize` command that arrives while the last pushed frames are still
  in flight commits `through` the shim's frozen frame count, and the final lands
  short. Draining stdin before handling a command would be the fix, but it is
  racy by construction (the frames may not have left the writer yet); the race
  is **unresolved** and the fake shim is left as-is.

## License

MIT.
