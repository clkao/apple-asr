"""Shim resolution, build, and the locale/info helpers (SPEC.md §6).

Resolution order (first hit wins):
1. an explicit path (`Stream(shim=...)`)
2. `$APPLE_ASR_SHIM`
3. the package cache `~/.cache/apple_asr/<shim_version>/apple-asr-shim`
4. `apple-asr-shim` on PATH (skipping this package's own console script)
5. build-on-demand into the cache, when a Swift toolchain is present
6. otherwise :class:`ShimUnavailable`, naming the exact build command.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AssetUnavailable, BackendError, ShimUnavailable
from .events import Locales, ShimInfo
from .platform import require_supported
from .protocol import PROTOCOL_VERSION, SHIM_VERSION
from .transport import read_hello

__all__ = [
    "cache_root",
    "cache_path",
    "source_path",
    "resolve_shim",
    "build_shim",
    "shim_info",
    "list_locales",
    "ensure_installed",
    "BUILD_COMMAND",
]

BUILD_COMMAND = "python -m apple_asr.build"


def cache_root() -> Path:
    base = os.environ.get("APPLE_ASR_CACHE")
    if not base:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "apple_asr"


def cache_path() -> Path:
    """The version-scoped binary path so upgrades rebuild cleanly."""
    return cache_root() / SHIM_VERSION / "apple-asr-shim"


def source_path() -> Path:
    return Path(__file__).resolve().parent / "shim" / "speechanalyzer.swift"


def _is_our_console_script(path: str) -> bool:
    """True when `path` is this package's own `apple-asr-shim` entry point."""
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")[:2048]
    except OSError:
        return False
    return "apple_asr.cli" in head


def _explicit(path: str, what: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise ShimUnavailable(
            f"{what} points at {path!r}, which does not exist. "
            f"Build the shim with: {BUILD_COMMAND}"
        )
    return str(p.resolve())


def resolve_shim(
    explicit: str | None = None,
    *,
    exclude: str | None = None,
    build_on_demand: bool = True,
) -> str:
    """Resolve the shim executable per the §6 order."""
    if explicit:
        return _explicit(explicit, "Stream(shim=...)")

    env = os.environ.get("APPLE_ASR_SHIM")
    if env:
        return _explicit(env, "$APPLE_ASR_SHIM")

    cached = cache_path()
    if cached.exists():
        return str(cached)

    found = shutil.which("apple-asr-shim")
    if found:
        resolved = str(Path(found).resolve())
        skipped = resolved == (exclude and str(Path(exclude).resolve()))
        if not skipped and not _is_our_console_script(resolved):
            return resolved

    if build_on_demand and shutil.which("swiftc"):
        return build_shim()

    raise ShimUnavailable(
        "no apple-asr-shim found. Resolution tried $APPLE_ASR_SHIM, "
        f"{cached}, and PATH. Build one with: {BUILD_COMMAND}"
    )


def build_shim(*, force: bool = False, quiet: bool = False) -> str:
    """Compile the bundled Swift source into the cache and validate `hello`."""
    require_supported()
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise ShimUnavailable(
            f"swiftc not found on PATH; install Xcode or the Swift toolchain, "
            f"then run: {BUILD_COMMAND}"
        )
    src = source_path()
    if not src.exists():
        raise ShimUnavailable(
            f"bundled shim source missing at {src}; reinstall the package "
            f"(apple-asr {SHIM_VERSION})"
        )

    out = cache_path()
    if out.exists() and not force:
        try:
            hello = read_hello(str(out), timeout=30.0)
            fresh = (
                hello.get("shim_version") == SHIM_VERSION
                and hello.get("protocol") == PROTOCOL_VERSION
            )
            if fresh:
                return str(out)
        except BackendError:
            pass  # stale/broken cache entry: rebuild below

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    cmd = [swiftc, "-O", "-parse-as-library", str(src), "-o", str(tmp)]
    if not quiet:
        print(f"building shim: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise BackendError(
            f"swiftc failed (status {proc.returncode}):\n{proc.stderr.strip()}"
        )

    try:
        hello = read_hello(str(tmp), timeout=60.0)
    except BackendError:
        tmp.unlink(missing_ok=True)
        raise
    if hello.get("protocol") != PROTOCOL_VERSION or hello.get("shim_version") != SHIM_VERSION:
        tmp.unlink(missing_ok=True)
        raise BackendError(
            f"built shim reports protocol={hello.get('protocol')!r} "
            f"version={hello.get('shim_version')!r}, package expects "
            f"protocol={PROTOCOL_VERSION} version={SHIM_VERSION!r}"
        )
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, out)
    if not quiet:
        print(f"shim ready: {out}")
    return str(out)


def _first_json_line(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise BackendError(f"shim produced no JSON line; output was:\n{text[:500]}")


def list_locales() -> Locales:
    """Ask the shim which locales are installed and supported."""
    require_supported()
    path = resolve_shim()
    proc = subprocess.run([path, "--list-locales"], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise BackendError(
            f"shim --list-locales failed (status {proc.returncode}):\n{proc.stderr.strip()}"
        )
    obj = _first_json_line(proc.stdout)
    return Locales(
        installed=tuple(obj.get("installed") or ()),
        supported=tuple(obj.get("supported") or ()),
    )


def ensure_installed(locale: str) -> None:
    """Install/reserve the locale asset; raise :class:`AssetUnavailable` on failure."""
    require_supported()
    path = resolve_shim()
    proc = subprocess.run(
        [path, "--ensure-installed", locale], capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0:
        raise AssetUnavailable(
            f"locale {locale!r} could not be installed (shim status {proc.returncode}). "
            f"stderr tail:\n{proc.stderr.strip()[-800:]}"
        )


def shim_info() -> ShimInfo:
    """Resolve the shim and report the `hello`-advertised identity."""
    require_supported()
    path = resolve_shim()
    hello = read_hello(path)
    try:
        mtime = os.path.getmtime(path)
        build = datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        build = ""
    return ShimInfo(
        path=path,
        version=str(hello.get("shim_version") or ""),
        protocol=int(hello.get("protocol") or 0),
        build=build,
    )

