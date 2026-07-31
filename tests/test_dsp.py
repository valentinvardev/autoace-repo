"""Synthetic-audio tests for every deterministic detector.

Ground truth is known by construction: silences of exact duration, noise at
exact SNR, clipping of known length, clicks at known times. A detector that
cannot recover these has no business labeling production audio.
"""

import numpy as np
import pytest

from autoace_pipeline import dsp
from autoace_pipeline.fields import CAL, decide_audio_quality, decide_long_silence
from autoace_pipeline.schema import Features

SR = 16000
RNG = np.random.default_rng(42)


def noise(dur_s: float, level_db: float, sr: int = SR) -> np.ndarray:
    amp = 10 ** (level_db / 20)
    return RNG.normal(0, amp, int(dur_s * sr)).astype(np.float32)


def speech_like(dur_s: float, level_db: float, sr: int = SR) -> np.ndarray:
    """Amplitude-modulated tone bursts approximating speech energy patterns."""
    t = np.arange(int(dur_s * sr)) / sr
    carrier = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t)
    envelope = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)
    amp = 10 ** (level_db / 20)
    return (amp * carrier * envelope / 1.5).astype(np.float32)


# ---------------------------------------------------------------- true_runs

@pytest.mark.parametrize("mask,expect", [
    ([], []),
    ([0, 0], []),
    ([1, 1], [(0, 2)]),
    ([1, 1, 0, 1], [(0, 2), (3, 4)]),
    ([0, 1, 1, 0], [(1, 3)]),
    ([1], [(0, 1)]),
    ([0, 1], [(1, 2)]),
])
def test_true_runs_edges(mask, expect):
    assert dsp.true_runs(np.array(mask, bool)) == expect


# ----------------------------------------------------------------- silence

@pytest.mark.parametrize("gap_s", [2.0, 7.34, 12.0])
def test_silence_gap_duration_recovered(gap_s):
    x = np.concatenate([
        speech_like(5, -14), np.zeros(int(gap_s * SR), np.float32), speech_like(5, -14),
    ]) + noise(10 + gap_s, -65)
    db = dsp.frame_db(x, SR)
    gaps = dsp.silence_gaps(db, 0.010, dsp.adaptive_silence_threshold(db), min_dur_s=1.0)
    assert gaps, "no gap found"
    assert abs(gaps[0][1] - gap_s) < 0.15, f"gap {gap_s}s detected as {gaps[0][1]:.2f}s"


def test_long_silence_threshold():
    f = Features(duration_s=60, max_gap_s=7.3)
    assert decide_long_silence(f) is False  # labeled false on call_003
    f = Features(duration_s=60, max_gap_s=CAL["long_silence_s"] + 0.5)
    assert decide_long_silence(f) is True


def test_gap_at_file_start_and_end():
    x = np.concatenate([
        np.zeros(3 * SR, np.float32), speech_like(4, -14), np.zeros(2 * SR, np.float32),
    ]) + noise(9, -70)
    db = dsp.frame_db(x, SR)
    gaps = dsp.silence_gaps(db, 0.010, dsp.adaptive_silence_threshold(db), min_dur_s=1.0)
    durs = sorted(g[1] for g in gaps)
    assert abs(durs[-1] - 3.0) < 0.2 and abs(durs[-2] - 2.0) < 0.2


# --------------------------------------------------------------------- SNR

@pytest.mark.parametrize("snr_target", [40.0, 20.0, 10.0])
def test_snr_estimate(snr_target):
    x = speech_like(20, -14) + noise(20, -14 - snr_target)
    lv = dsp.levels(dsp.frame_db(x, SR))
    assert abs(lv["snr_db"] - snr_target) < 6.0


# ---------------------------------------------------------------- clipping

def test_real_clipping_flagged():
    x = np.clip(speech_like(10, -3) * 8, -1, 1)
    c = dsp.clipping(x, SR)
    assert c["clip_runs_ge3"] > 10
    assert c["clip_max_run"] >= 3


def test_intersample_overs_not_flagged():
    """Isolated 1-2 sample overs (Opus decoder artifacts) must not count."""
    x = speech_like(10, -14)
    idx = RNG.integers(0, len(x), 200)
    x[idx] = 1.0
    c = dsp.clipping(x, SR)
    assert c["clip_runs_ge3"] <= 2


# ------------------------------------------------------------------ clicks

def test_clicks_recovered_and_gated():
    x = noise(20, -60)
    click_at = [2.0, 5.0, 8.0, 15.0]
    for t in click_at:
        x[int(t * SR)] += 0.5
    times = dsp.click_times(x, SR)
    assert len(times) == len(click_at)
    assert all(abs(t - c) < 0.01 for t, c in zip(sorted(times), click_at))

    pauses = [(0.0, 10.0)]
    inside = dsp.gate_times(times, pauses, inside=True)
    assert len(inside) == 3  # 15.0 s excluded


# --------------------------------------------------------------- HF bursts

def test_hf_bursts_detected():
    sr = 48000
    x = noise(20, -60, sr)
    for t in (4.0, 9.0, 14.0):
        i = int(t * sr)
        burst = (RNG.normal(0, 0.2, 2400)).astype(np.float32)  # 50 ms wideband
        x[i:i + 2400] += burst
    times, _ = dsp.hf_burst_times(x, sr)
    assert len(times) >= 3


# ---------------------------------------------------------------- dropouts

def test_dropouts_recovered():
    x = speech_like(30, -14) + noise(30, -60)
    for i in range(20):  # 20 holes of 60 ms in 0.5 min -> 40/min
        s = int((1 + i * 1.4) * SR)
        x[s:s + int(0.06 * SR)] = 0.0
    rate = dsp.dropouts(x, SR, [(0.0, 30.0)])
    assert 30 <= rate <= 50


def test_no_dropouts_on_clean():
    x = speech_like(30, -14) + noise(30, -60)
    assert dsp.dropouts(x, SR, [(0.0, 30.0)]) == 0.0


# ------------------------------------------------------------ audio quality

def test_quality_clear_on_clean():
    f = Features(duration_s=60, snr_db=55, clip_runs_per_min=4, rolloff95_hz=2100)
    assert decide_audio_quality(f) == "clear"


def test_quality_degrades_with_snr():
    f = Features(duration_s=60, snr_db=20, clip_runs_per_min=0, rolloff95_hz=2100)
    assert decide_audio_quality(f) == "slightly_impaired"
    f = Features(duration_s=60, snr_db=8, clip_runs_per_min=0, rolloff95_hz=2100)
    assert decide_audio_quality(f) == "severely_impaired"


def test_quality_muffled_bandwidth():
    f = Features(duration_s=60, snr_db=50, clip_runs_per_min=0, rolloff95_hz=500)
    assert decide_audio_quality(f) == "severely_impaired"
