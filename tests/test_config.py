"""SPEC.md amendment 1 — `mode` ("streaming" | "accurate") and the explicit knobs.

The measured numbers behind the two modes (MEASUREMENTS.md): streaming =
progressive + fastResults, ~13.2 mean zh CER; accurate = transcription, no
fastResults, ~11.2 mean zh CER. `mode` is a convenience over `preset` and
`reporting_option`; both explicit knobs keep working, and a contradicting
combination is rejected rather than silently resolved.
"""

from __future__ import annotations

import pytest
from support import drain, speech

from apple_asr import Stream


def _shim_options(fake) -> dict:
    state = fake.state()
    return {
        "preset": state["preset"],
        "fast_results": state["fast_results"],
        "argv": state["argv"],
    }


def _run(fake, **kw) -> dict:
    st = fake.stream(**kw)
    try:
        st.push(speech(0.1))
    finally:
        st.close()
        drain(st)
    return _shim_options(fake)


def test_default_mode_is_streaming(fake):
    st = fake.stream()
    try:
        assert st.mode == "streaming"
        assert st.preset == "progressive"
        assert st.reporting_option == "fastResults"
    finally:
        st.close()
        drain(st)
    assert _shim_options(fake)["preset"] == "progressive"
    assert _shim_options(fake)["fast_results"] is True


def test_accurate_mode_uses_the_transcription_preset_without_fast_results(fake):
    st = fake.stream(mode="accurate")
    try:
        assert st.mode == "accurate"
        assert st.preset == "transcription"
        assert st.reporting_option is None
    finally:
        st.close()
        drain(st)
    options = _shim_options(fake)
    assert options["preset"] == "transcription"
    assert options["fast_results"] is False
    assert "--no-fast" in options["argv"]


def test_streaming_mode_does_not_pass_no_fast(fake):
    _run(fake, mode="streaming")
    assert "--no-fast" not in _shim_options(fake)["argv"]


def test_explicit_knobs_still_work_without_mode(fake):
    """The order-1 spellings must keep working (no mode passed)."""
    st = fake.stream(preset="transcription")
    try:
        assert st.preset == "transcription"
        assert st.mode == "accurate", "the mode label follows the resolved knobs"
    finally:
        st.close()
        drain(st)
    assert _shim_options(fake)["preset"] == "transcription"

    _run(fake, preset="progressive", reporting_option="fastResults")
    assert _shim_options(fake)["fast_results"] is True

    _run(fake, reporting_option="volatileResults")
    assert _shim_options(fake)["fast_results"] is False
    assert _shim_options(fake)["preset"] == "progressive"


def test_explicit_knobs_win_when_mode_is_not_given(fake):
    _run(fake, preset="timeIndexedProgressive")
    assert _shim_options(fake)["preset"] == "timeIndexedProgressive"


def test_conflicting_mode_and_preset_is_rejected(fake):
    with pytest.raises(ValueError) as excinfo:
        fake.stream(mode="accurate", preset="progressive")
    message = str(excinfo.value)
    assert "accurate" in message and "progressive" in message
    assert "transcription" in message, "the error must name the implied value"


def test_conflicting_mode_and_reporting_option_is_rejected(fake):
    with pytest.raises(ValueError) as excinfo:
        fake.stream(mode="accurate", reporting_option="fastResults")
    assert "fastResults" in str(excinfo.value)


def test_invalid_values_are_rejected(fake):
    with pytest.raises(ValueError):
        fake.stream(mode="fast")
    with pytest.raises(ValueError):
        fake.stream(preset="nope")
    with pytest.raises(ValueError):
        fake.stream(reporting_option="bogus")
    with pytest.raises(ValueError):
        Stream(locale="en-US", silent_timeout=-1)


def test_measured_defaults_are_not_improved(fake):
    """pause_commit 0.08 / commit_interval 0.0 stay put (MEASUREMENTS.md)."""
    st = fake.stream()
    try:
        assert st.pause_commit == 0.08
        assert st.commit_interval == 0.0
        assert st.silent_timeout == 30.0
    finally:
        st.close()
        drain(st)
    state = fake.state()
    assert state["pause_commit"] == 0.08
    argv = state["argv"]
    assert argv[argv.index("--commit-interval") + 1] == "0.0"


def test_context_and_locale_reach_the_shim(fake):
    st = fake.stream(locale="zh-TW", context=["鐳射", "Earendil"])
    try:
        st.push(speech(0.1))
    finally:
        st.close()
        drain(st)
    state = fake.state()
    assert state["locale"] == "zh-TW"
    assert state["context"] == ["鐳射", "Earendil"]
