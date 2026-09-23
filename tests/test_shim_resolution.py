"""SPEC.md §6 — resolution order, plus the shipped-wheel path.

Step 5 of the order (the shim prebuilt into a platform wheel as package data) is
what makes `pip install <wheel>` toolchain-free: with an empty cache, no
`$APPLE_ASR_SHIM` and no `apple-asr-shim` on PATH, resolution must hand back the
package's own `apple_asr/shim/apple-asr-shim` and never build anything.

The bundled binary is not in a source checkout (it is a git-ignored build
product), so these tests point `bundled_path()` at a stand-in file: the resolution
*order* and the exec-bit repair are what is under test, not the binary itself.
"""

from __future__ import annotations

import os
import stat

import pytest

from apple_asr import ShimUnavailable
from apple_asr import shim as shim_mod


@pytest.fixture
def bare_machine(tmp_path, monkeypatch):
    """No cache, no `$APPLE_ASR_SHIM`, nothing named apple-asr-shim on PATH."""
    monkeypatch.setenv("APPLE_ASR_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("APPLE_ASR_SHIM", raising=False)
    monkeypatch.setattr(shim_mod.shutil, "which", lambda name: None)
    return tmp_path


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    """A stand-in for the wheel's package data, installed *without* the exec bit.

    That is a state any extractor that does not restore zip modes leaves it in
    (``python -m zipfile -e <wheel>`` does exactly this), so every test here
    starts from it.
    """
    path = tmp_path / "site-packages" / "apple_asr" / "shim" / "apple-asr-shim"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)  # 0644
    assert not os.access(path, os.X_OK)
    monkeypatch.setattr(shim_mod, "bundled_path", lambda: path)
    return path


def test_resolution_uses_the_bundled_shim_without_building(bare_machine, bundled, monkeypatch):
    def refuse(*a, **kw):  # pragma: no cover - only runs on regression
        raise AssertionError("build-on-demand must not run when a bundled shim is present")

    monkeypatch.setattr(shim_mod, "build_shim", refuse)
    assert shim_mod.resolve_shim() == str(bundled)


def test_the_bundled_shim_is_chmodded_on_first_use(bare_machine, bundled):
    """pip does not reliably preserve the exec bit of package data."""
    assert shim_mod.resolve_shim() == str(bundled)
    assert os.access(bundled, os.X_OK), "resolution must repair the exec bit"
    assert bundled.stat().st_mode & stat.S_IXOTH, "all exec bits are set, like build_shim"


def test_an_already_executable_bundled_shim_is_left_alone(bare_machine, bundled):
    bundled.chmod(0o755)
    before = bundled.stat().st_mode
    assert shim_mod.resolve_shim() == str(bundled)
    assert bundled.stat().st_mode == before


def test_the_version_cache_still_wins_over_the_bundle(bare_machine, bundled):
    """A shim the user built into the cache keeps priority (documented order)."""
    cached = shim_mod.cache_path()
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"#!/bin/sh\nexit 0\n")
    cached.chmod(0o755)
    assert shim_mod.resolve_shim() == str(cached)


def test_build_on_demand_is_only_the_step_after_the_bundle(bare_machine, monkeypatch, tmp_path):
    """No bundle -> the old behaviour (build when swiftc exists) is intact."""
    monkeypatch.setattr(shim_mod, "bundled_path", lambda: tmp_path / "absent" / "apple-asr-shim")
    monkeypatch.setattr(
        shim_mod.shutil,
        "which",
        lambda name: "/usr/bin/swiftc" if name == "swiftc" else None,
    )
    monkeypatch.setattr(shim_mod, "build_shim", lambda **kw: "/built/apple-asr-shim")
    assert shim_mod.resolve_shim() == "/built/apple-asr-shim"


def test_an_empty_machine_names_the_bundle_and_the_build_command(bare_machine, tmp_path):
    monkeypatch_path = tmp_path / "absent" / "apple-asr-shim"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim_mod, "bundled_path", lambda: monkeypatch_path)
        with pytest.raises(ShimUnavailable) as excinfo:
            shim_mod.resolve_shim()
    message = str(excinfo.value)
    assert "apple-asr-shim" in message
    assert shim_mod.BUILD_COMMAND in message
