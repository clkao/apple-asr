"""Console script `apple-asr-shim`: resolve (building if needed) then exec.

The real engine is a native binary, so the console script is a thin resolver +
`execv`: `apple-asr-shim --stdin ...` from any shell behaves exactly like the
compiled shim, and resolution step 4 of SPEC.md §6 (PATH) is satisfied.

`--help` (and no arguments at all) is special-cased: it prints the flag list
below and exits **without** resolving or building anything, so asking for help
on a machine with no shim never triggers a Swift build (order-1 report §4 Q5).
"""

from __future__ import annotations

import os
import sys

from .errors import AppleAsrError
from .protocol import SHIM_VERSION
from .shim import BUILD_COMMAND, resolve_shim

HELP_FLAGS = ("--help", "-h")

USAGE = f"""apple-asr-shim - Apple SpeechAnalyzer streaming shim (console script)

usage: apple-asr-shim {{--stdin|--file PATH|--mic}} [options]
       apple-asr-shim --list-locales
       apple-asr-shim --ensure-installed LOCALE
       apple-asr-shim --help

modes (mutually exclusive)
  --stdin                    read float32 LE mono 16 kHz PCM from stdin (what
                             the apple_asr package uses)
  --file PATH                batch: transcribe a file the shim reads itself
  --mic                      capture the default input device (needs mic TCC)

options
  --locale L                 BCP-47 locale (default en-US)
  --preset P                 progressive | transcription | timeIndexedProgressive
  --context a,b,c            AnalysisContext.contextualStrings (hotwords)
  --pause-commit S           quiet seconds before a pause commits (default 0.08)
  --commit-interval S        commit-latency ceiling; 0 = pause-only (default 0)
  --vad-sensitivity LEVEL    off | low | medium | high (inert in this build)
  --no-fast                  drop the .fastResults reporting option (finals are
                             more accurate, ~2 CER points on zh)
  --no-volatile              disable volatile (partial) results
  --no-confidence            omit per-run transcriptionConfidence

deprecated
  --fast                     accepted for CLI compatibility and IGNORED: the
                             analyzer already consumes audio as fast as it can,
                             and fastResults is on unless --no-fast is given.
                             (Kept because SPEC.md §4 lists it. Slated for
                             removal in a future major version.)

`--help` never builds the shim. To compile it:

    {BUILD_COMMAND}

(shim version {SHIM_VERSION}; rebuild with: {BUILD_COMMAND} --force)
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or any(a in HELP_FLAGS for a in argv):
        # Never resolve/build just to print help (order-1 report §4 Q5).
        sys.stdout.write(USAGE)
        return 0
    try:
        path = resolve_shim(exclude=sys.argv[0])
    except AppleAsrError as exc:
        print(f"apple-asr-shim: {exc}", file=sys.stderr)
        return 1
    os.execv(path, [path, *argv])
    return 0  # unreachable
