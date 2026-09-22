"""SPEC.md §9 test 8 — `list_locales()` / `ensure_installed()` parsing.

These helpers resolve the shim through the package cache (the fixture installs
the fake shim into a tmp `APPLE_ASR_CACHE`), so no real build or asset download
happens. The platform gate is stubbed because these tests run on any OS.
"""

from __future__ import annotations

import pytest

from apple_asr import AssetUnavailable, ensure_installed, list_locales, shim_info
from apple_asr.protocol import PROTOCOL_VERSION, SHIM_VERSION


@pytest.fixture
def ungate(monkeypatch):
    """Neutralize the §8 platform gate for non-macOS runners."""
    monkeypatch.setattr("apple_asr.shim.require_supported", lambda: None)


def test_list_locales_parses_shim_output(fake, ungate):
    fake.scenario(
        {
            "supported_locales": ["en-US", "zh-TW", "ja-JP"],
            "installed_locales": ["en-US", "zh-TW"],
        }
    )
    locales = list_locales()
    assert locales.installed == ("en-US", "zh-TW")
    assert locales.supported == ("en-US", "zh-TW", "ja-JP")


def test_list_locales_tolerates_an_empty_install_set(fake, ungate):
    fake.scenario({"supported_locales": ["en-US"], "installed_locales": []})
    locales = list_locales()
    assert locales.installed == ()
    assert locales.supported == ("en-US",)


def test_ensure_installed_succeeds(fake, ungate):
    fake.scenario({})
    ensure_installed("zh-TW")
    assert fake.state()["status"] == "ensure_ok"


def test_ensure_installed_raises_asset_unavailable_with_stderr(fake, ungate):
    fake.scenario({"ensure_installed_fails": True})
    with pytest.raises(AssetUnavailable) as excinfo:
        ensure_installed("zh-TW")
    message = str(excinfo.value)
    assert "zh-TW" in message
    assert "could not be installed" in message


def test_shim_info_reports_the_hello_identity(fake, ungate):
    fake.scenario({})
    info = shim_info()
    assert info.version == SHIM_VERSION
    assert info.protocol == PROTOCOL_VERSION
    assert info.path == str(fake.shim_path.resolve())
    assert info.build, "build must be populated (binary mtime, ISO-8601)"
