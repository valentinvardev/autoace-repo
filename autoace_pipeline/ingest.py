"""Decode and normalize arbitrary input audio via ffmpeg.

Every downstream stage consumes the canonical forms produced here:
- 16 kHz mono float32 (VAD, envelope, silence, SNR)
- native-rate mono float32, clamped (clipping, clicks, HF, spectrum)

Clamping matters: Opus decodes with intersample overs (|x| > 1.0); converting
without a clamp overflows int16 and manufactures distortion that is not in
the call. Channel correlation is measured before downmix so dual-mono
sources (all three provided calls) are detected rather than assumed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

import numpy as np


class IngestError(Exception):
    """Raised when a file cannot be probed or decoded. One bad file must
    fail alone, never the batch."""


@dataclass
class DecodedAudio:
    path: str
    duration_s: float
    codec: str
    source_sample_rate: int
    source_channels: int
    mono16k: np.ndarray          # float32, 16 kHz
    native: np.ndarray           # float32 mono at native rate, clamped
    native_sr: int
    is_dual_mono: bool | None = None
    chan_min_corr: float | None = field(default=None)


def _run(cmd: list[str]) -> bytes:
    try:
        p = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise IngestError(f"ffmpeg/ffprobe not found: {e}") from e
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown error"]
        raise IngestError(tail[0]) from e
    return p.stdout


def probe(path: str) -> dict:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", path])
    info = json.loads(out)
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise IngestError("no audio stream found")
    s = audio[0]
    return {
        "codec": s.get("codec_name", "?"),
        "sample_rate": int(s.get("sample_rate", 0) or 0),
        "channels": int(s.get("channels", 0) or 0),
        "duration_s": float(info.get("format", {}).get("duration", 0) or 0),
    }


def _decode(path: str, sr: int | None, channels: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", str(channels)]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-"]
    x = np.frombuffer(_run(cmd), np.float32)
    if channels > 1:
        x = x.reshape(-1, channels)
    if x.size == 0:
        raise IngestError("decoded zero samples")
    return x


def load(path: str) -> DecodedAudio:
    meta = probe(path)
    native_sr = meta["sample_rate"] or 48000

    is_dual_mono = None
    min_corr = None
    if meta["channels"] >= 2:
        st = _decode(path, None, 2).astype(np.float64)
        L, R = st[:, 0], st[:, 1]
        win = 5 * native_sr
        corrs = []
        for a in range(0, max(1, len(L) - win), max(1, win // 2)):
            l, r = L[a:a + win], R[a:a + win]
            if l.std() > 1e-6 and r.std() > 1e-6:
                corrs.append(float(np.corrcoef(l, r)[0, 1]))
        min_corr = min(corrs) if corrs else 1.0
        is_dual_mono = min_corr > 0.98
        native = ((L + R) / 2).astype(np.float32)
    else:
        native = _decode(path, None, 1)

    native = np.clip(native, -1.0, 1.0)
    mono16k = np.clip(_decode(path, 16000, 1), -1.0, 1.0)

    dur = meta["duration_s"] or len(native) / native_sr
    return DecodedAudio(
        path=path, duration_s=dur, codec=meta["codec"],
        source_sample_rate=meta["sample_rate"], source_channels=meta["channels"],
        mono16k=mono16k, native=native, native_sr=native_sr,
        is_dual_mono=is_dual_mono, chan_min_corr=min_corr,
    )
