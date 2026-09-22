"""SPEC.md §4/§6 — the `apple-asr-shim` console script.

`--help` (and no arguments) must print the flag list **without** resolving or
building a shim: asking for help on a machine with no toolchain must not start a
Swift build (order-1 report §4 Q5).
"""

from __future__ import annotations

import pytest

from apple_asr.cli import main as shim_main


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("APPLE_ASR_CACHE", str(tmp_path / "never-created"))
    monkeypatch.delenv("APPLE_ASR_SHIM", raising=False)
    return tmp_path / "never-created"


def test_help_prints_the_flags_without_building(capsys, empty_cache):
    assert shim_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "--stdin" in out and "--no-fast" in out and "--list-locales" in out
    assert "deprecated" in out and "--fast" in out, "the --fast no-op must be documented"
    assert not empty_cache.exists(), "help must not resolve or build the shim"


def test_no_arguments_prints_usage_without_building(capsys, empty_cache):
    assert shim_main([]) == 0
    assert "usage: apple-asr-shim" in capsys.readouterr().out
    assert not empty_cache.exists()


def test_help_never_touches_a_broken_cache(capsys, empty_cache):
    empty_cache.mkdir(parents=True, exist_ok=True)
    (empty_cache / "apple_asr" / "0.1.0").mkdir(parents=True)
    assert shim_main(["-h"]) == 0
    assert list(empty_cache.rglob("apple-asr-shim")) == []


def test_replay_console_script_has_help():
    from apple_asr.replay import build_parser

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
