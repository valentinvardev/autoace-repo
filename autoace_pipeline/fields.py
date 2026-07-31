"""Feature -> field decisions for the locally-determined outputs.

All thresholds live in CAL, in one place, so the synthetic-calibration stage
adjusts them with data rather than scattering magic numbers. Values marked
`provisional` are initial settings from the three labeled calls and the
brief's definitions; they are re-fit against the synthetic validation set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import AudioQuality, Features, Severity

CAL = {
    # A verified 7.3 s mid-call gap is labeled long_silence=false: normal
    # hold while an agent checks something does not count. Bar sits above it.
    "long_silence_s": 10.0,

    # pause-floor bands for noise severity (dBFS), provisional
    "noise_floor_low": -55.0,
    "noise_floor_medium": -45.0,
    "noise_floor_high": -35.0,

    # impulsive static (pause-gated clicks/min), provisional
    "clicks_low": 10.0,
    "clicks_medium": 40.0,

    # audio quality, provisional; all three labeled calls are `clear`
    # at SNR 50-57 dB with clip runs up to 10 samples
    "quality_snr_slight": 25.0,
    "quality_snr_severe": 12.0,
    "quality_cliprate_slight": 30.0,   # runs>=3 per minute
    "quality_cliprate_severe": 200.0,
    "quality_rolloff_slight": 1000,    # Hz; telephony floor is ~
    "quality_rolloff_severe": 600,
}


@dataclass
class LocalDecisions:
    """Fields decidable without the LLM, plus fusion inputs for the rest."""

    long_silence_present: bool
    audio_quality: AudioQuality
    noise_present_local: bool
    noise_severity_local: Severity
    static_suspected: bool


def decide_long_silence(f: Features) -> bool:
    return f.max_gap_s >= CAL["long_silence_s"]


def decide_audio_quality(f: Features) -> AudioQuality:
    severe = (
        f.snr_db < CAL["quality_snr_severe"]
        or f.clip_runs_per_min > CAL["quality_cliprate_severe"]
        or (f.rolloff95_hz and f.rolloff95_hz < CAL["quality_rolloff_severe"])
    )
    if severe:
        return "severely_impaired"
    slight = (
        f.snr_db < CAL["quality_snr_slight"]
        or f.clip_runs_per_min > CAL["quality_cliprate_slight"]
        or (f.rolloff95_hz and f.rolloff95_hz < CAL["quality_rolloff_slight"])
    )
    return "slightly_impaired" if slight else "clear"


def decide_noise_local(f: Features) -> tuple[bool, Severity, bool]:
    """Local view of background noise from pause-gated evidence.

    Steady noise raises the pause floor; impulsive static raises the
    pause-gated click rate without moving the floor. Speech-like background
    (TV, chatter) can hide from both — the LLM's perceptual vote covers that
    side in fusion.
    """
    floor = f.pause_floor_db if f.pause_floor_db is not None else f.noise_floor_db

    if floor > CAL["noise_floor_high"]:
        steady: Severity = "high"
    elif floor > CAL["noise_floor_medium"]:
        steady = "medium"
    elif floor > CAL["noise_floor_low"]:
        steady = "low"
    else:
        steady = "none"

    if f.clicks_per_min_pause > CAL["clicks_medium"]:
        impulsive: Severity = "medium"
    elif f.clicks_per_min_pause > CAL["clicks_low"]:
        impulsive = "low"
    else:
        impulsive = "none"

    order = ["none", "low", "medium", "high"]
    severity: Severity = max(steady, impulsive, key=order.index)  # type: ignore[assignment]
    static_suspected = impulsive != "none" or f.hf_energy_ratio > 0.003
    return severity != "none", severity, static_suspected


def decide_local(f: Features) -> LocalDecisions:
    present, severity, static = decide_noise_local(f)
    return LocalDecisions(
        long_silence_present=decide_long_silence(f),
        audio_quality=decide_audio_quality(f),
        noise_present_local=present,
        noise_severity_local=severity,
        static_suspected=static,
    )
