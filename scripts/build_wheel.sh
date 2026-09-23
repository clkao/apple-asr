#!/usr/bin/env bash
# Build a macOS 26 arm64 wheel for apple-asr with the Swift shim prebuilt inside.
#
# This is the wheel-shipping path: .github/workflows/wheel.yml calls exactly this
# script, so a local run and the CI run produce the same artifact. It needs
# `swiftc` and `uv` on PATH and nothing else — no Xcode project, no Speech
# framework, no signing.
#
#   1. swiftc -O -parse-as-library -> src/apple_asr/shim/apple-asr-shim
#      (package data; .gitignore excludes it from git, pyproject's wheel
#      `artifacts` puts it in the zip)
#   2. uv build --wheel            -> apple_asr-<version>-py3-none-any.whl
#   3. wheel tags --remove         -> apple_asr-<version>-py3-none-macosx_26_0_arm64.whl
#   4. assertions: the shim is inside the zip, the binary is arm64, and the
#      surviving wheel's tag is platform-specific
#
# Why retag with `wheel tags` instead of a setuptools
# `BinaryDistribution.has_ext_modules()` override: this package is pure Python
# with one bundled binary, so the artifact should install on any CPython
# 3.10-3.13 running macOS 26 arm64. has_ext_modules() makes bdist_wheel emit an
# interpreter-specific tag (cp313-cp313-macosx_...), i.e. one wheel per Python
# version, for no benefit — and it would mean switching the build backend off
# hatchling. Retagging yields `py3-none-macosx_26_0_arm64`, which is exactly what
# the artifact is: pure Python that only runs where the bundled arm64 binary
# runs. The tag matters: a `py3-none-any` wheel installs happily on Intel Macs
# and on macOS < 26, where the binary cannot run at all.
#
# Override the tag with PLATFORM_TAG=... (e.g. for a future macosx_27_0_arm64).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PLATFORM_TAG="${PLATFORM_TAG:-macosx_26_0_arm64}"
SRC="$ROOT/src/apple_asr/shim/speechanalyzer.swift"
SHIM="$ROOT/src/apple_asr/shim/apple-asr-shim"

if ! command -v swiftc >/dev/null 2>&1; then
    echo "error: swiftc not found; install Xcode or the Swift toolchain" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found; see https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "== swiftc -O -parse-as-library $SRC -o $SHIM"
swiftc -O -parse-as-library "$SRC" -o "$SHIM"
chmod +x "$SHIM"

# The tag below claims arm64; a cross-arch or Intel build must not get it.
# (`file ... | grep -q` would be racy under `pipefail`: grep exits at the first
# match and SIGPIPEs the writer. Match on captured output instead.)
shim_desc="$(file -b "$SHIM")"
case "$shim_desc" in
    *arm64*) ;;
    *)
        echo "error: $SHIM is not an arm64 binary ($shim_desc), refusing to tag it $PLATFORM_TAG" >&2
        exit 1
        ;;
esac

echo "== uv build --wheel"
rm -rf "$ROOT/dist"
uv build --wheel

built="$(ls "$ROOT"/dist/*-py3-none-any.whl)"
echo "== wheel tags --platform-tag $PLATFORM_TAG (--remove)"
# `--remove` deletes the py3-none-any original: it must never be shipped, since
# it would install on machines that cannot run the binary. `wheel tags` prints
# the new *filename*, so put the dist dir back in front of it.
tagged="$(uvx --quiet --from wheel python -m wheel tags --remove \
    --platform-tag "$PLATFORM_TAG" "$built")"
wheel="$ROOT/dist/$(basename "$tagged")"

case "$wheel" in
    *"$PLATFORM_TAG"*) ;;
    *)
        echo "error: the retagged wheel is not platform-tagged: $wheel" >&2
        exit 1
        ;;
esac
# Same pipefail hazard as above: capture the listing and match on it.
listing="$(unzip -l "$wheel")"
case "$listing" in
    *apple_asr/shim/apple-asr-shim*) ;;
    *)
        echo "error: $wheel does not contain the prebuilt shim" >&2
        exit 1
        ;;
esac
shipped=("$ROOT"/dist/*.whl)
if [ "${#shipped[@]}" != "1" ]; then
    echo "error: dist/ must hold only the platform wheel; it holds: ${shipped[*]}" >&2
    exit 1
fi

echo "wheel: $wheel"
