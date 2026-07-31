"""Required output schema (verbatim from the trial brief) plus internal feature report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EmotionalTone = Literal["neutral", "satisfied", "frustrated", "upset", "distressed"]
Intensity = Literal["low", "medium", "high"]
Severity = Literal["none", "low", "medium", "high"]
AudioQuality = Literal["clear", "slightly_impaired", "severely_impaired"]


class CallResult(BaseModel):
    """The nine fields the trial requires, exactly as specified."""

    emotional_tone: EmotionalTone
    emotional_intensity: Intensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Severity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)


class Features(BaseModel):
    """Raw acoustic measurements every field decision is derived from.

    Kept alongside each prediction for debugging, calibration, and the
    dashboard's per-file detail view.
    """

    duration_s: float
    codec: str = ""
    source_sample_rate: int = 0
    source_channels: int = 0
    is_dual_mono: bool | None = None

    # levels
    noise_floor_db: float = 0.0
    speech_db: float = 0.0
    snr_db: float = 0.0
    pause_floor_db: float | None = None
    pause_snr_db: float | None = None

    # silence
    max_gap_s: float = 0.0
    max_gap_at_s: float | None = None
    silence_frac: float = 0.0
    speech_frac: float = 0.0

    # clipping / peaks
    peak: float = 0.0
    clip_runs_ge3: int = 0
    clip_max_run: int = 0
    clip_runs_per_min: float = 0.0

    # impulses and HF content (static / crackle evidence)
    clicks_per_min_all: float = 0.0
    clicks_per_min_pause: float = 0.0
    hf_bursts_per_min_pause: float = 0.0
    hf_energy_ratio: float = 0.0

    # spectrum
    rolloff95_hz: int = 0
    rolloff99_hz: int = 0
    spectral_flatness_speech: float | None = None

    # optional model outputs, filled by later stages
    vad_backend: str = ""
    noise_tags: list[tuple[str, float]] = []
    overlap_ratio: float | None = None
