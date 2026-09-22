"""`python -m apple_asr.build` — compile the shim and validate its `hello`.

Resolves the bundled Swift source, compiles it with
`swiftc -O -parse-as-library` into `~/.cache/apple_asr/<version>/apple-asr-shim`,
validates the result by parsing the built shim's `hello` line, and prints the
resolved path.
"""

from __future__ import annotations

import argparse
import sys

from .errors import AppleAsrError
from .shim import build_shim, cache_path, source_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apple_asr.build",
        description="Build and validate the apple-asr shim.",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is valid")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(argv)

    if not args.quiet:
        print(f"source: {source_path()}")
        print(f"target: {cache_path()}")
    try:
        path = build_shim(force=args.force, quiet=args.quiet)
    except AppleAsrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
