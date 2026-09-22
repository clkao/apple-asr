"""Shared test scaffolding: the fake-shim harness and small async helpers.

Everything here is fake-shim plumbing (SPEC.md §9 tests 1-9). It never touches
the real shim, the real ``~/.cache``, or the Speech framework, so these tests
run on any OS and any Python 3.10-3.13.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
import time
from pathlib import Path

import numpy as np

from apple_asr import Ended, Stream
from apple_asr.protocol import SAMPLE_RATE, SHIM_VERSION

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_SHIM_SRC = FIXTURES / "fake_shim.py"
AUDIO_ZH_LONG = FIXTURES / "audio" / "zh_long.wav"
GOLDEN_ZH_LONG = Path(__file__).resolve().parent / "golden" / "zh_long_ideal.jsonl"

#: Scenario defaults shared by every test; override per test via `scenario()`.
DEFAULT_SCENARIO: dict = {
    "mode": "emulate",
    "protocol": 1,
    "partial_every_frames": 16_000,
}

_LAUNCHER = """\
#!{python}
import runpy, sys
sys.argv[0] = {src!r}
runpy.run_path({src!r}, run_name="__main__")
"""


class FakeBackend:
    """Installs the scripted fake shim into a tmp cache and drives it.

    Mirrors SPEC.md §6 resolution order 3 (the package cache) so the tests
    exercise the same path a real install uses: the launcher lives at
    ``$APPLE_ASR_CACHE/apple_asr/<shim_version>/apple-asr-shim``.
    """

    def __init__(self, root: Path, monkeypatch) -> None:
        self.root = root
        self.cache = root / "cache"
        self.scenario_path = root / "scenario.json"
        self.state_path = root / "shim-state.json"
        self.bin_dir = self.cache / "apple_asr" / SHIM_VERSION
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.shim_path = self.bin_dir / "apple-asr-shim"
        _write_launcher(self.shim_path)

        monkeypatch.setenv("APPLE_ASR_CACHE", str(self.cache))
        monkeypatch.setenv("APPLE_ASR_FAKE_SCENARIO", str(self.scenario_path))
        monkeypatch.setenv("APPLE_ASR_FAKE_STATE", str(self.state_path))
        monkeypatch.setenv("APPLE_ASR_FAKE_MAX_WALL_S", "30")
        monkeypatch.delenv("APPLE_ASR_SHIM", raising=False)
        self.scenario({})

    # ------------------------------------------------------------------
    # Scenario + state
    # ------------------------------------------------------------------
    def scenario(self, scn: dict) -> Path:
        """Merge `scn` over the defaults and write the scenario file."""
        merged = {**DEFAULT_SCENARIO, **scn}
        self.scenario_path.write_text(json.dumps(merged), encoding="utf-8")
        return self.scenario_path

    def state(self, timeout: float = 10.0) -> dict:
        """The fake shim's post-mortem dump (waits for the file to land)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.state_path.exists():
                text = self.state_path.read_text(encoding="utf-8")
                if text.strip():
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
            time.sleep(0.02)
        raise AssertionError(f"fake shim wrote no state to {self.state_path}")

    @property
    def wire(self) -> list[dict]:
        """Every JSON event the fake shim emitted, in order."""
        return list(self.state()["emitted"])

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def stream(self, **kw) -> Stream:
        """A `Stream` resolving the fake shim through the tmp cache."""
        kw.setdefault("locale", "zh-TW")
        return Stream(**kw)


def macos26_available() -> bool:
    """True on macOS 26+ (where the real shim and the Speech framework exist)."""
    if sys.platform != "darwin":
        return False
    try:
        return int(platform.mac_ver()[0].split(".")[0]) >= 26
    except (ValueError, IndexError):
        return False


def _write_launcher(path: Path) -> None:
    path.write_text(
        _LAUNCHER.format(python=sys.executable, src=str(FAKE_SHIM_SRC)),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def speech(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    """Non-silent float32 audio (the fake endpointer treats it as speech)."""
    return np.full(int(round(seconds * SAMPLE_RATE)), amplitude, dtype=np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)


def drain(stream: Stream, timeout: float = 10.0, stop_on_ended: bool = True) -> list:
    """Collect events until `Ended` (or `timeout`), never blocking forever."""
    out: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = stream.poll(0.05)
        out.extend(events)
        if stop_on_ended and any(isinstance(e, Ended) for e in events):
            break
    return out


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def types(events: list) -> list[str]:
    return [type(e).__name__ for e in events]


def env_cache_dir() -> str:
    return os.environ.get("APPLE_ASR_CACHE", "")
