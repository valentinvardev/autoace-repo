"""Per-file feature extraction and local field decisions."""

from __future__ import annotations

import numpy as np

from . import dsp
from .fields import LocalDecisions, decide_local
from .ingest import DecodedAudio, load
from .schema import Features


def extract_features(a: DecodedAudio, use_vad: bool = True) -> Features:
    db = dsp.frame_db(a.mono16k, 16000)
    hop_s = 0.010
    lv = dsp.levels(db)

    th = dsp.adaptive_silence_threshold(db)
    gaps = dsp.silence_gaps(db, hop_s, th)
    max_gap, max_gap_at = (gaps[0][1], gaps[0][0]) if gaps else (0.0, None)
    silence_frac = float(np.mean(db < th))

    clip = dsp.clipping(a.native, a.native_sr)
    clicks_all = dsp.click_times(a.native, a.native_sr)
    hf_times, hf_ratio = dsp.hf_burst_times(a.native, a.native_sr)
    shape = dsp.spectral_shape(a.native, a.native_sr)

    minutes = max(a.duration_s / 60, 1e-6)

    speech: list[tuple[float, float]] = []
    pauses: list[tuple[float, float]] = []
    vad_backend = ""
    if use_vad:
        from . import vad
        speech = vad.speech_spans(a.mono16k, a.duration_s)
        pauses = vad.pause_spans(speech, a.duration_s)
        vad_backend = "silero"

    pause_floor = pause_snr = None
    clicks_pause_rate = hf_pause_rate = 0.0
    if pauses:
        sel = np.zeros(len(db), bool)
        for s, e in pauses:
            sel[int(s / hop_s): int(e / hop_s)] = True
        if sel.any():
            pause_floor = float(np.median(db[sel]))
            pause_snr = lv["speech_db"] - pause_floor
        pause_total_min = sum(e - s for s, e in pauses) / 60
        if pause_total_min > 0:
            clicks_pause_rate = len(dsp.gate_times(clicks_all, pauses, inside=True)) / pause_total_min
            hf_pause_rate = len(dsp.gate_times(hf_times, pauses, inside=True)) / pause_total_min

    flatness = None
    if speech:
        seg = np.concatenate([
            a.mono16k[int(s * 16000): int(e * 16000)] for s, e in speech
        ]) if speech else a.mono16k
        flatness = dsp.spectral_flatness(seg, 16000)

    speech_frac = float(sum(e - s for s, e in speech) / a.duration_s) if speech else 0.0

    return Features(
        duration_s=round(a.duration_s, 2),
        codec=a.codec,
        source_sample_rate=a.source_sample_rate,
        source_channels=a.source_channels,
        is_dual_mono=a.is_dual_mono,
        **{k: round(v, 1) for k, v in lv.items()},
        pause_floor_db=round(pause_floor, 1) if pause_floor is not None else None,
        pause_snr_db=round(pause_snr, 1) if pause_snr is not None else None,
        max_gap_s=round(max_gap, 2),
        max_gap_at_s=round(max_gap_at, 1) if max_gap_at is not None else None,
        silence_frac=round(silence_frac, 3),
        speech_frac=round(speech_frac, 3),
        peak=round(clip["peak"], 4),
        clip_runs_ge3=clip["clip_runs_ge3"],
        clip_max_run=clip["clip_max_run"],
        clip_runs_per_min=round(clip["clip_runs_per_min"], 1),
        clicks_per_min_all=round(len(clicks_all) / minutes, 1),
        clicks_per_min_pause=round(clicks_pause_rate, 1),
        hf_bursts_per_min_pause=round(hf_pause_rate, 1),
        hf_energy_ratio=round(hf_ratio, 5),
        rolloff95_hz=shape["rolloff95_hz"],
        rolloff99_hz=shape["rolloff99_hz"],
        spectral_flatness_speech=round(flatness, 4) if flatness is not None else None,
        vad_backend=vad_backend,
    )


def analyze_local(path: str, use_vad: bool = True) -> tuple[Features, LocalDecisions]:
    audio = load(path)
    feats = extract_features(audio, use_vad=use_vad)
    return feats, decide_local(feats)


# Overlap counts when pyannote finds at least this much simultaneous speech.
# Calibrated on the labeled calls (overlap: false, true, true).
OVERLAP_MIN_S = 1.0


def analyze_full(path: str) -> dict:
    """Local features + pyannote overlap + Gemini vote, fused to a CallResult.

    Degrades gracefully: if overlap or the LLM is unavailable the result is
    still produced, with the gap recorded in the fusion trace and confidence.
    """
    from .fusion import fuse
    from .llm import GeminiAnalyzer

    audio = load(path)
    feats = extract_features(audio, use_vad=True)
    local = decide_local(feats)

    overlap_present = None
    overlap_info: dict = {}
    try:
        from .overlap import detect_overlap
        speech_s = feats.speech_frac * feats.duration_s
        overlap_info = detect_overlap(audio.mono16k, audio.duration_s, speech_s)
        overlap_present = overlap_info["overlap_total_s"] >= OVERLAP_MIN_S
        feats.overlap_ratio = overlap_info["overlap_ratio"]
    except Exception as e:  # noqa: BLE001 - unavailable model must not kill the file
        overlap_info = {"error": str(e)[:200]}

    llm = GeminiAnalyzer().analyze(path)
    result, trace = fuse(feats, local, llm, overlap_present,
                         overlap_info.get("overlap_ratio"))

    return {
        "result": result,
        "features": feats,
        "local": local,
        "llm": llm,
        "overlap": overlap_info,
        "trace": trace,
    }
