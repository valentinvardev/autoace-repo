"""Audit of production call audio vs provided labels.

Decodes each file twice (native 48k stereo for channel/peak/click analysis,
16k mono for envelope/silence/SNR), measures the acoustic facts each schema
field depends on, and prints them next to the provided label.

Usage:
    python analysis/audit.py resources/audio/*.ogg --labels resources/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sps

EPS = 1e-12


def decode(path: str, sr: int | None = None, channels: int | None = None) -> np.ndarray:
    """Decode with ffmpeg to float32 PCM. Returns (n,) mono or (n, ch)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if channels:
        cmd += ["-ac", str(channels)]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    if channels and channels > 1:
        x = x.reshape(-1, channels)
    return x


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


def selftest() -> None:
    """The plan documents a run-length edge bug that silently broke silence
    detection. Never trust the detector without these."""
    assert true_runs(np.array([])) == []
    assert true_runs(np.array([1, 1, 0, 1], bool)) == [(0, 2), (3, 4)]
    assert true_runs(np.array([0, 0], bool)) == []
    assert true_runs(np.array([1, 1], bool)) == [(0, 2)]
    assert true_runs(np.array([0, 1, 1, 0], bool)) == [(1, 3)]
    # synthetic 3.00 s silence inside noise must be recovered within one frame
    sr, hop = 16000, 160
    x = np.random.default_rng(0).normal(0, 0.1, sr * 10).astype(np.float32)
    x[int(4.0 * sr) : int(7.0 * sr)] = 0.0
    db = frame_db(x, sr, frame=400, hop=hop)
    gaps = silence_gaps(db, hop_s=hop / sr, thresh_db=-50.0)
    longest = max(g[2] for g in gaps)
    assert abs(longest - 3.0) < 0.05, f"synthetic gap detected as {longest:.3f}s"


def frame_db(x: np.ndarray, sr: int, frame: int = 400, hop: int = 160) -> np.ndarray:
    """Per-frame RMS in dBFS (25 ms frame / 10 ms hop at 16 kHz)."""
    n = 1 + max(0, (len(x) - frame)) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20 * np.log10(rms + EPS)


def silence_gaps(db: np.ndarray, hop_s: float, thresh_db: float) -> list[tuple[float, float, float]]:
    """(start_s, end_s, dur_s) of every below-threshold run."""
    out = []
    for a, b in true_runs(db < thresh_db):
        out.append((a * hop_s, b * hop_s, (b - a) * hop_s))
    return out


def analyze(path: Path) -> dict:
    st = decode(str(path), channels=2)  # native rate, both channels
    sr_native = 48000
    L, R = st[:, 0].astype(np.float64), st[:, 1].astype(np.float64)
    dur = len(L) / sr_native
    mono48 = ((L + R) / 2).astype(np.float32)
    m16 = decode(str(path), sr=16000, channels=1)
    sr16 = 16000

    r: dict = {"file": path.name, "duration_s": round(dur, 2)}

    # --- channel identity (is the stereo real?) ---
    r["chan_pearson_r"] = float(np.corrcoef(L, R)[0, 1])
    r["chan_rms_diff"] = float(np.sqrt(np.mean((L - R) ** 2)))
    win = 5 * sr_native
    seg_r = []
    for a in range(0, len(L) - win, win // 2):
        l, rr = L[a : a + win], R[a : a + win]
        if l.std() > 1e-6 and rr.std() > 1e-6:
            seg_r.append((float(np.corrcoef(l, rr)[0, 1]), a / sr_native))
    if seg_r:
        mn = min(seg_r)
        r["chan_min_seg_r"], r["chan_min_seg_at_s"] = round(mn[0], 6), round(mn[1], 1)

    # --- peaks / clipping (require consecutive runs, not lone samples) ---
    ax = np.abs(mono48)
    r["peak"] = float(ax.max())
    over = ax >= 0.999
    r["over_count"] = int(over.sum())
    runs = [b - a for a, b in true_runs(over)]
    r["over_max_run"] = int(max(runs, default=0))
    r["over_runs_ge3"] = int(sum(1 for n in runs if n >= 3))

    # --- envelope, floor, speech level, SNR (16 k mono) ---
    db = frame_db(m16, sr16)
    hop_s = 160 / sr16
    floor = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 95))
    r["noise_floor_db"] = round(floor, 1)
    r["speech_db"] = round(speech, 1)
    r["snr_db"] = round(speech - floor, 1)

    # --- silence gaps (threshold: floor + 10 dB, capped at -40 dBFS) ---
    th = min(floor + 10.0, -40.0)
    gaps = [g for g in silence_gaps(db, hop_s, th) if g[2] >= 1.5]
    gaps.sort(key=lambda g: -g[2])
    r["silence_thresh_db"] = round(th, 1)
    r["gaps_ge_1p5s"] = [(round(a, 1), round(d, 2)) for a, _, d in gaps[:6]]
    r["silence_frac"] = round(float(np.mean(db < th)), 3)

    # --- noise inside speech pauses only (>=0.25 s, excludes file edges) ---
    pause = [g for g in silence_gaps(db, hop_s, th) if g[2] >= 0.25
             and g[0] > 0.5 and g[1] < dur - 0.5]
    if pause:
        sel = np.zeros(len(db), bool)
        for a, b, _ in pause:
            sel[int(a / hop_s) : int(b / hop_s)] = True
        r["pause_floor_db"] = round(float(np.median(db[sel])), 1)
        pf = np.concatenate([m16[int(a * sr16) : int(b * sr16)] for a, b, _ in pause])
        r["pause_zero_frac"] = round(float(np.mean(pf == 0.0)), 4)

    # --- impulsive clicks ("sharp static" would live here) ---
    d = np.diff(mono48)
    z = np.abs(d) / (1.4826 * np.median(np.abs(d)) + EPS)
    for th_z in (30, 60, 100):
        idx = np.flatnonzero(z > th_z)
        # merge hits closer than 10 ms into one click event
        events = 1 + int(np.sum(np.diff(idx) > 0.010 * sr_native)) if idx.size else 0
        r[f"clicks_z{th_z}"] = events
    if (idx := np.flatnonzero(z > 60)).size:
        merged = idx[np.r_[True, np.diff(idx) > 0.010 * sr_native]]
        r["click_times_z60"] = [round(i / sr_native, 1) for i in merged[:8]]

    # --- high-frequency bursts (>=6 kHz energy spikes) ---
    f, t, Z = sps.stft(mono48, sr_native, nperseg=1024, noverlap=512)
    hf = 20 * np.log10(np.abs(Z[f >= 6000]).mean(axis=0) + EPS)
    burst = hf > np.median(hf) + 20
    ev = true_runs(burst)
    r["hf_burst_events"] = len(ev)
    r["hf_burst_times"] = [round(t[a], 1) for a, _ in ev[:8]]

    # --- spectral shape ---
    f2, p = sps.welch(mono48, sr_native, nperseg=4096)
    c = np.cumsum(p) / (p.sum() + EPS)
    r["rolloff95_hz"] = int(f2[np.searchsorted(c, 0.95)])
    r["rolloff99_hz"] = int(f2[np.searchsorted(c, 0.99)])
    r["energy_over_4k"] = round(float(p[f2 > 4000].sum() / (p.sum() + EPS)), 5)

    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    selftest()
    print("[selftest ok: run-length + synthetic 3.0s gap recovered]\n")

    labels = {}
    if args.labels:
        with open(args.labels, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                labels[row["name"]] = json.loads(row["result_json"])

    results = []
    for fp in args.files:
        p = Path(fp)
        r = analyze(p)
        results.append(r)
        print(f"=== {r['file']}  ({r['duration_s']}s) ===")
        for k, v in r.items():
            if k not in ("file", "duration_s"):
                print(f"  {k:>18}: {v}")
        if lab := labels.get(p.name):
            print("  --- provided label ---")
            for k in ("emotional_tone", "emotional_intensity", "background_noise_present",
                      "background_noise_type", "background_noise_severity", "audio_quality",
                      "speaker_overlap_present", "long_silence_present"):
                print(f"  {k:>18}: {lab.get(k)}")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
