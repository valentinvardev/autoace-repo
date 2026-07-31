"""Deterministic acoustic features. Pure numpy/scipy — no ML imports here.

Every function is unit-tested against synthetic audio with known ground
truth (tests/test_dsp.py) before being trusted on production calls: the
original silence detector for this project shipped with a run-length edge
bug that silently reported a 7.3 s gap as 1.6 s.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

EPS = 1e-12


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) index pairs for each run of True."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return []
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size]
    return list(zip(starts.tolist(), ends.tolist()))


def frame_db(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    """Per-frame RMS level in dBFS."""
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20 * np.log10(rms + EPS)


def levels(db: np.ndarray) -> dict:
    """Global level statistics from a frame-dB envelope."""
    floor = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 95))
    return {"noise_floor_db": floor, "speech_db": speech, "snr_db": speech - floor}


def silence_gaps(db: np.ndarray, hop_s: float, thresh_db: float,
                 min_dur_s: float = 0.0) -> list[tuple[float, float]]:
    """(start_s, dur_s) of below-threshold runs, longest first."""
    out = [(a * hop_s, (b - a) * hop_s) for a, b in true_runs(db < thresh_db)]
    out = [g for g in out if g[1] >= min_dur_s]
    return sorted(out, key=lambda g: -g[1])


def adaptive_silence_threshold(db: np.ndarray) -> float:
    """Floor + 10 dB, capped so loud-throughout audio still yields a sane bar."""
    return min(float(np.percentile(db, 10)) + 10.0, -40.0)


def clipping(x: np.ndarray, sr: int, ceiling: float = 0.999) -> dict:
    """Clipping evidence via consecutive samples at full scale.

    Isolated overs are intersample artifacts of lossy decoders, not real
    saturation; only runs of >= 3 samples count toward the quality decision.
    """
    over = np.abs(x) >= ceiling
    runs = [b - a for a, b in true_runs(over)]
    qual = [n for n in runs if n >= 3]
    minutes = len(x) / sr / 60
    return {
        "peak": float(np.abs(x).max()) if len(x) else 0.0,
        "clip_max_run": int(max(runs, default=0)),
        "clip_runs_ge3": len(qual),
        "clip_runs_per_min": float(len(qual) / minutes) if minutes > 0 else 0.0,
        "clip_frac": float(sum(qual) / max(len(x), 1)),
    }


def dropouts(x: np.ndarray, sr: int, speech: list[tuple[float, float]],
             min_ms: float = 20.0, max_ms: float = 250.0) -> float:
    """Digital-zero holes inside speech (packet loss), per minute of speech."""
    if not speech:
        return 0.0
    hits, total_s = 0, 0.0
    for a, b in speech:
        seg = x[int(a * sr): int(b * sr)]
        total_s += b - a
        for s, e in true_runs(np.abs(seg) < 1e-5):
            if min_ms <= (e - s) / sr * 1000 <= max_ms:
                hits += 1
    return hits / max(total_s / 60, 1e-6)


def click_times(x: np.ndarray, sr: int, z_thresh: float = 60.0,
                merge_ms: float = 10.0) -> np.ndarray:
    """Times (s) of impulsive transients via robust z-score of the first
    difference. Sharp static / crackle lives here; steady noise does not."""
    d = np.abs(np.diff(x))
    # baseline from non-silent samples only: long digital-zero stretches
    # (inserted gaps, packet loss) would shrink the MAD and inflate z-scores
    active = d[np.abs(x[:-1]) > 1e-5]
    base = np.median(active) if active.size else np.median(d)
    z = d / (1.4826 * base + EPS)
    idx = np.flatnonzero(z > z_thresh)
    if idx.size == 0:
        return np.empty(0)
    keep = idx[np.r_[True, np.diff(idx) > merge_ms / 1000 * sr]]
    return keep / sr


def hf_burst_times(x: np.ndarray, sr: int, band_hz: float = 6000.0,
                   rel_db: float = 20.0) -> tuple[np.ndarray, float]:
    """(times of high-frequency energy bursts, global HF energy ratio)."""
    nper = 1024 if sr >= 32000 else 512
    f, t, Z = sps.stft(x, sr, nperseg=nper, noverlap=nper // 2)
    mag = np.abs(Z)
    hf = 20 * np.log10(mag[f >= band_hz].mean(axis=0) + EPS)
    burst = hf > np.median(hf) + rel_db
    times = np.array([t[a] for a, _ in true_runs(burst)])
    total = float((mag ** 2).sum())
    ratio = float((mag[f >= 4000] ** 2).sum() / (total + EPS))
    return times, ratio


def spectral_shape(x: np.ndarray, sr: int) -> dict:
    f, p = sps.welch(x, sr, nperseg=4096 if sr >= 32000 else 2048)
    c = np.cumsum(p) / (p.sum() + EPS)
    return {
        "rolloff95_hz": int(f[np.searchsorted(c, 0.95)]),
        "rolloff99_hz": int(f[np.searchsorted(c, 0.99)]),
    }


def spectral_flatness(x: np.ndarray, sr: int) -> float:
    """Geometric/arithmetic PSD mean ratio; higher = noisier/whiter."""
    _, p = sps.welch(x, sr, nperseg=2048)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def gate_times(times: np.ndarray, spans: list[tuple[float, float]],
               inside: bool) -> np.ndarray:
    """Keep event times inside (or outside) a list of (start_s, end_s) spans."""
    if times.size == 0 or not spans:
        return times if not inside else np.empty(0)
    mask = np.zeros(times.shape, bool)
    for a, b in spans:
        mask |= (times >= a) & (times < b)
    return times[mask] if inside else times[~mask]
