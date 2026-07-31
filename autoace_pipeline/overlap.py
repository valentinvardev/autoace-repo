"""Overlapped-speech detection via pyannote/segmentation-3.0 (gated weights).

All provided calls are dual-mono, so channel energy cannot separate
speakers; a learned segmentation model is the only reliable overlap signal.
LLM answers on this field flipped between identical runs in testing, so it
stays fully local.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np


class OverlapUnavailable(Exception):
    """HF token missing or gated-model terms not accepted."""


@lru_cache(maxsize=1)
def _pipeline():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise OverlapUnavailable("HF_TOKEN not set")
    try:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import OverlappedSpeechDetection
        model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
        pipe = OverlappedSpeechDetection(segmentation=model)
        pipe.instantiate({"min_duration_on": 0.15, "min_duration_off": 0.1})
        return pipe
    except Exception as e:  # gated download refused, network, etc.
        raise OverlapUnavailable(str(e)) from e


def detect_overlap(mono16k: np.ndarray, duration_s: float,
                   speech_total_s: float | None = None) -> dict:
    """Returns overlap spans, total seconds, and ratio vs speech time."""
    import torch

    pipe = _pipeline()
    wave = torch.from_numpy(np.ascontiguousarray(mono16k)).unsqueeze(0)
    ann = pipe({"waveform": wave, "sample_rate": 16000})
    spans = [(float(s.start), float(s.end)) for s in ann.get_timeline().support()]
    total = sum(e - s for s, e in spans)
    base = speech_total_s if speech_total_s and speech_total_s > 0 else duration_s
    return {
        "overlap_spans": [(round(s, 1), round(e, 1)) for s, e in spans],
        "overlap_total_s": round(total, 2),
        "overlap_ratio": round(total / base, 4),
    }
