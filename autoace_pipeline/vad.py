"""Speech/pause segmentation via Silero VAD (torch, CPU).

Isolated in its own module so dsp.py stays importable without torch and the
test suite for the deterministic features runs in milliseconds.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def speech_spans(mono16k: np.ndarray, duration_s: float) -> list[tuple[float, float]]:
    """(start_s, end_s) speech segments over the 16 kHz mono signal."""
    import torch
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(mono16k)),
        _model(),
        sampling_rate=16000,
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def pause_spans(speech: list[tuple[float, float]], duration_s: float,
                min_dur_s: float = 0.25, edge_margin_s: float = 0.5) -> list[tuple[float, float]]:
    """Complement of speech within the clip, excluding the file edges
    (leading/trailing silence says nothing about background noise)."""
    out = []
    prev = 0.0
    for a, b in sorted(speech):
        if a - prev >= min_dur_s and a > edge_margin_s and prev < duration_s - edge_margin_s:
            out.append((max(prev, edge_margin_s), min(a, duration_s - edge_margin_s)))
        prev = max(prev, b)
    if duration_s - prev >= min_dur_s and prev < duration_s - edge_margin_s:
        out.append((prev, duration_s - edge_margin_s))
    return [(a, b) for a, b in out if b - a >= min_dur_s]
