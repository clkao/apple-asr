"""Console script `apple-asr-shim`: resolve (building if needed) then exec.

The real engine is a native binary, so the console script is a thin resolver +
`execv`: `apple-asr-shim --stdin ...` from any shell behaves exactly like the
compiled shim, and resolution step 4 of SPEC.md §6 (PATH) is satisfied.
"""

from __future__ import annotations

import os
import sys

from .errors import AppleAsrError
from .protocol import SHIM_VERSION
from .shim import resolve_shim

USAGE = f"""apple-asr-shim - Apple SpeechAnalyzer streaming shim

usage: apple-asr-shim {{--stdin|--file PATH|--mic}} [options]
       apple-asr-shim --list-locales
       apple-asr-shim --ensure-installed LOCALE
       apple-asr-shim --help

Run `apple-asr-shim --help` for the full flag list.
(shim version {SHIM_VERSION}; rebuild with: python -m apple_asr.build --force)
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stdout.write(USAGE)
        return 0
    try:
        path = resolve_shim(exclude=sys.argv[0])
    except AppleAsrError as exc:
        print(f"apple-asr-shim: {exc}", file=sys.stderr)
        return 1
    os.execv(path, [path, *argv])
    return 0  # unreachable
