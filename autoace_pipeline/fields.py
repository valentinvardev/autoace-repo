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

    # pause-SNR bands for steady-noise severity (dB between speech level and
    # pause floor); robust to absolute-level differences between clips.
    # Fitted against the synthetic corpus (see analysis/synthetic_validation.py)
    "noise_snr_none": 40.0,    # cleaner than this: no meaningful noise
    "noise_snr_low": 28.0,     # audible, not interfering
    "noise_snr_medium": 15.0,  # occasionally interferes; below: high

    # impulsive static (pause-gated clicks/min). Set above the natural
    # transient chatter of the clean calls (4-14/min); genuine static also
    # registers through the HF-ratio path.
    "clicks_low": 20.0,
    "clicks_medium": 100.0,
    "static_hf_ratio": 0.003,

    # audio quality; validated against the synthetic corpus
    "quality_snr_slight": 25.0,
    "quality_snr_severe": 12.0,
    "quality_clipfrac_slight": 0.002,  # fraction of samples in runs >= 3
    "quality_clipfrac_severe": 0.02,
    "quality_dropout_slight": 10.0,    # zero-holes per speech-minute
    "quality_dropout_severe": 100.0,
    "quality_rolloff_slight": 1000,    # Hz
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
        or f.clip_frac > CAL["quality_clipfrac_severe"]
        or f.dropouts_per_min > CAL["quality_dropout_severe"]
        or (f.rolloff95_hz and f.rolloff95_hz < CAL["quality_rolloff_severe"])
    )
    if severe:
        return "severely_impaired"
    slight = (
        f.snr_db < CAL["quality_snr_slight"]
        or f.clip_frac > CAL["quality_clipfrac_slight"]
        or f.dropouts_per_min > CAL["quality_dropout_slight"]
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
    snr = f.pause_snr_db if f.pause_snr_db is not None else f.snr_db

    if snr < CAL["noise_snr_medium"]:
        steady: Severity = "high"
    elif snr < CAL["noise_snr_low"]:
        steady = "medium"
    elif snr < CAL["noise_snr_none"]:
        steady = "low"
    else:
        steady = "none"

    # packet-loss boundary transients are quality damage, not background noise
    clicks = 0.0 if f.dropouts_per_min > CAL["quality_dropout_slight"] else f.clicks_per_min_pause
    if clicks > CAL["clicks_medium"]:
        impulsive: Severity = "medium"
    elif clicks > CAL["clicks_low"]:
        impulsive = "low"
    else:
        impulsive = "none"

    order = ["none", "low", "medium", "high"]
    severity: Severity = max(steady, impulsive, key=order.index)  # type: ignore[assignment]
    # static is impulsive by nature: pause-gated clicks, or elevated HF energy
    # WITHOUT steady broadband noise (which would explain the HF on its own)
    static_suspected = impulsive != "none" or (
        f.hf_energy_ratio > CAL["static_hf_ratio"] and steady == "none")
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
