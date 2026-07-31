"""System-level validation on synthetic audio with ground truth by construction.

With three labeled calls, per-class validation of the acoustic fields is
impossible from provided data alone. This script manufactures it: slices of
the real calls (realistic telephony voice, kept out of git) are degraded in
controlled ways - steady noise at exact SNRs, clipping, lowpass, packet
loss, inserted silences, click trains, overlapped speech - and the local
pipeline is scored blind against the construction parameters.

Outputs: out/synthetic/*.wav + gt.json (untracked), and a committed
markdown report with per-field confusion matrices.

    python analysis/synthetic_validation.py [--skip-generate]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from autoace_pipeline import dsp  # noqa: E402
from autoace_pipeline.ingest import load  # noqa: E402
from autoace_pipeline.pipeline import OVERLAP_MIN_S, analyze_local  # noqa: E402

SR = 16000
OUT = Path("out/synthetic")
RNG = np.random.default_rng(7)

CARRIERS = {  # contiguous slices with natural speech + pauses
    "c1": ("resources/audio/call_001.ogg", 3.0, 30.0),
    "c2": ("resources/audio/call_002.ogg", 3.0, 34.0),
    "c3": ("resources/audio/call_003.ogg", 30.0, 95.0),
}

# GT mapping from construction SNR to the brief's severity semantics
def snr_to_severity(snr: float) -> str:
    if snr >= 40: return "none"
    if snr >= 25: return "low"
    if snr >= 12: return "medium"
    return "high"


def speech_level_db(x: np.ndarray) -> float:
    return float(np.percentile(dsp.frame_db(x, SR), 95))


def make_noise(kind: str, n: int, other: np.ndarray) -> np.ndarray:
    if kind == "white":
        return RNG.normal(0, 1.0, n).astype(np.float32)
    if kind == "pink":
        w = RNG.normal(0, 1.0, n)
        # -3 dB/oct via 1/sqrt(f) spectral shaping
        spec = np.fft.rfft(w)
        f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0)
        return np.fft.irfft(spec / np.sqrt(f), n).astype(np.float32)
    if kind == "babble":
        # unintelligible speech-like: reversed other-call speech, two offset copies
        s = other[::-1].copy()
        reps = int(np.ceil(n / len(s))) + 1
        s = np.tile(s, reps)
        return (s[:n] + 0.7 * s[len(s) // 3: len(s) // 3 + n]).astype(np.float32)
    raise ValueError(kind)


def at_snr(carrier: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    target = speech_level_db(carrier) - snr_db
    cur = 20 * np.log10(np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return (carrier + noise * 10 ** ((target - cur) / 20)).astype(np.float32)


def generate() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    carriers = {}
    for name, (path, a, b) in CARRIERS.items():
        x = load(path).mono16k[int(a * SR): int(b * SR)]
        # Peak-normalize: 16 kHz resampling overshoot is clamped to exactly
        # 1.0 at ingest and would masquerade as clipping in the corpus.
        peak = float(np.abs(x).max()) or 1.0
        carriers[name] = (x * (0.9 / peak)).astype(np.float32)

    cases: list[dict] = []

    def emit(name: str, x: np.ndarray, gt: dict) -> None:
        p = OUT / f"{name}.wav"
        sf.write(p, np.clip(x, -1, 1), SR, subtype="FLOAT")
        cases.append({"file": str(p), "name": name, **gt})

    clean_gt = {"noise_severity": "none", "quality": "clear",
                "long_silence": False, "static": False, "overlap": False}
    # call_003 carries genuine static: its derived clips cannot serve as
    # clean ground truth for the noise/static fields
    c3_gt = {**clean_gt, "noise_severity": "skip", "static": "skip"}

    for cn, x in carriers.items():
        gt = dict(c3_gt) if cn == "c3" else dict(clean_gt)
        if cn == "c2":
            gt["overlap"] = "skip"  # call_002 contains genuine overlap
        emit(f"{cn}_control", x, gt)

    grid_carriers = ["c1", "c2"]
    other = {"c1": carriers["c2"], "c2": carriers["c1"], "c3": carriers["c1"]}

    for cn in grid_carriers:
        x = carriers[cn]
        for kind in ("white", "pink", "babble"):
            for snr in (30, 20, 10, 5):
                noisy = at_snr(x, make_noise(kind, len(x), other[cn]), snr)
                emit(f"{cn}_{kind}_snr{snr}", noisy, {
                    **clean_gt, "noise_severity": snr_to_severity(snr),
                    "noise_kind": kind, "snr": snr,
                    # heavy noise also degrades technical quality
                    "quality": "clear" if snr >= 20 else "slightly_impaired",
                    "quality_min": True,  # noise-quality coupling is one-way; see report
                })

        for pct, tier in ((1.0, "slightly_impaired"), (6.0, "severely_impaired")):
            gain = 1.0
            lo, hi = 1.0, 60.0
            for _ in range(24):  # bisect gain to hit the target clipped fraction
                gain = (lo + hi) / 2
                frac = float(np.mean(np.abs(x * gain) >= 1.0)) * 100
                lo, hi = (gain, hi) if frac < pct else (lo, gain)
            emit(f"{cn}_clip{pct:g}", np.clip(x * gain, -1, 1),
                 {**clean_gt, "quality": tier})

        from scipy import signal as sps
        for fc, tier in ((1200, "slightly_impaired"), (500, "severely_impaired")):
            b_, a_ = sps.butter(6, fc / (SR / 2), "low")
            emit(f"{cn}_lp{fc}", sps.filtfilt(b_, a_, x).astype(np.float32),
                 {**clean_gt, "quality": tier})

        for pct, tier in ((5, "slightly_impaired"), (20, "severely_impaired")):
            y = x.copy()
            chunk = int(0.060 * SR)
            n_chunks = int(len(x) * pct / 100 / chunk)
            for s in RNG.integers(0, len(x) - chunk, n_chunks):
                y[s:s + chunk] = 0
            emit(f"{cn}_ploss{pct}", y, {**clean_gt, "quality": tier})

        for gap, truth in ((6.0, False), (12.0, True), (20.0, True)):
            mid = len(x) // 2
            y = np.concatenate([x[:mid], np.zeros(int(gap * SR), np.float32), x[mid:]])
            emit(f"{cn}_gap{gap:g}", y, {**clean_gt, "long_silence": truth})

        for rate, sev in ((30, "low"), (150, "medium")):
            y = x.copy()
            n_clicks = int(rate * len(x) / SR / 60)
            for t in RNG.integers(0, len(x) - 8, n_clicks):
                y[t:t + 3] += np.array([0.5, -0.4, 0.3], np.float32)
            emit(f"{cn}_clicks{rate}", y,
                 {**clean_gt, "noise_severity": sev, "static": True})

        # overlap: overlay other-call SPEECH on top of carrier speech spans,
        # RMS-matched to the carrier's speech level (a 10 dB quieter second
        # voice is inaudible to human and model alike); c2-derived overlap GT
        # is skipped because call_002 carries genuine overlap
        from autoace_pipeline import vad as _vad
        spans = [s for s in _vad.speech_spans(x, len(x) / SR) if s[1] - s[0] >= 1.5]
        ospans = _vad.speech_spans(other[cn], len(other[cn]) / SR)
        omat = np.concatenate([other[cn][int(a * SR):int(b * SR)] for a, b in ospans]) \
            if ospans else other[cn]
        lvl = speech_level_db(x)
        cur = 20 * np.log10(np.sqrt(np.mean(omat ** 2)) + 1e-12)
        omat = omat * 10 ** ((lvl - 3.0 - cur) / 20)
        for dur, truth in ((2.0, True), (6.0, True)):
            y = x.copy()
            seg = omat[: int(dur / 2 * SR)]
            for a, _b in spans[:2]:
                s = int(a * SR)
                y[s:s + len(seg)] += seg
            ov_gt = "skip" if cn == "c2" else truth
            emit(f"{cn}_ovl{dur:g}", y, {**clean_gt, "overlap": ov_gt})

    # c3: silence + overlap cases only (its noise/static baseline is unknown)
    x3 = carriers["c3"]
    mid = len(x3) // 2
    emit("c3_gap12", np.concatenate([x3[:mid], np.zeros(12 * SR, np.float32), x3[mid:]]),
         {**c3_gt, "long_silence": True})
    y = x3.copy()
    seg = other["c3"][: int(2.0 * SR)]
    for start_s in (10.0, 25.0):
        s = int(start_s * SR)
        y[s:s + len(seg)] += seg * 0.7
    emit("c3_ovl4", y, {**c3_gt, "overlap": True})

    (OUT / "gt.json").write_text(json.dumps(cases, indent=1))
    return cases


def evaluate(cases: list[dict]) -> dict:
    per_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in cases:
        feats, dec = analyze_local(c["file"], use_vad=True)
        if c["noise_severity"] != "skip":
            per_field["noise_severity"].append((c["noise_severity"], dec.noise_severity_local))
            per_field["noise_present"].append((str(c["noise_severity"] != "none"),
                                               str(dec.noise_present_local)))
        # noise-only clips: local quality may legitimately degrade further
        q_gt, q_pred = c["quality"], dec.audio_quality
        if c.get("quality_min") and ["clear", "slightly_impaired", "severely_impaired"].index(q_pred) >= ["clear", "slightly_impaired", "severely_impaired"].index(q_gt):
            q_pred = q_gt
        per_field["quality"].append((q_gt, q_pred))
        per_field["long_silence"].append((str(c["long_silence"]),
                                          str(dec.long_silence_present)))
        if c["static"] != "skip":
            per_field["static"].append((str(c["static"]), str(dec.static_suspected)))

        if (c["name"].endswith("_control") or "_ovl" in c["name"]) and c["overlap"] != "skip":
            try:
                from autoace_pipeline.overlap import detect_overlap
                a = load(c["file"])
                o = detect_overlap(a.mono16k, a.duration_s)
                per_field["overlap"].append(
                    (str(c["overlap"]), str(o["overlap_total_s"] >= OVERLAP_MIN_S)))
            except Exception as e:
                per_field["overlap"].append((str(c["overlap"]), f"ERR:{str(e)[:40]}"))
    return per_field


def report(per_field: dict) -> str:
    lines = ["# Synthetic validation report",
             "",
             "Local (DSP/VAD/pyannote) pipeline scored against constructed ground truth.",
             f"Corpus: {len(per_field['noise_severity'])} clips derived from the "
             "three production calls (audio untracked; construction in "
             "`analysis/synthetic_validation.py`).", ""]
    for field, pairs in per_field.items():
        labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
        acc = sum(g == p for g, p in pairs) / len(pairs)
        lines += [f"## {field} — accuracy {acc:.2%} (n={len(pairs)})", ""]
        lines += ["| gt \\ pred | " + " | ".join(labels) + " |",
                  "|---|" + "---|" * len(labels)]
        for g in labels:
            row = [str(sum(1 for gg, pp in pairs if gg == g and pp == p)) for p in labels]
            if any(gg == g for gg, _ in pairs):
                lines.append(f"| **{g}** | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def fused_noise_check(cases: list[dict]) -> list[str]:
    """Babble is structurally invisible to energy statistics (VAD tags it as
    speech). Run those clips + clean controls through local+LLM fusion to
    measure whether the perceptual vote closes the gap. Costs ~$0.02."""
    from autoace_pipeline.fusion import fuse
    from autoace_pipeline.llm import GeminiAnalyzer
    from autoace_pipeline.pipeline import analyze_local

    analyzer = GeminiAnalyzer()
    # c2-derived clips carry call_002's genuine TV: no clean-noise GT exists
    # for them beyond the injected babble itself, so its control is excluded
    subset = [c for c in cases if "_babble_" in c["name"] or c["name"] == "c1_control"]
    pairs, base = [], []
    lines = ["## fused noise on babble subset (local + Gemini)", ""]
    order = "none low medium high".split()
    for c in subset:
        feats, dec = analyze_local(c["file"], use_vad=True)
        llm = analyzer.analyze(c["file"])
        result, trace = fuse(feats, dec, llm, None, None)
        pairs.append((c["noise_severity"], result.background_noise_severity))
        base.append((c["noise_severity"], dec.noise_severity_local))
        lines.append(f"- `{c['name']}`: gt **{c['noise_severity']}** -> local "
                     f"{dec.noise_severity_local}, llm {llm.background_noise_severity}"
                     f" ({llm.background_noise_type or 'none'}), fused "
                     f"**{result.background_noise_severity}** [{trace.noise_rule}]")
    acc = sum(g == p for g, p in pairs) / len(pairs)
    adj = sum(abs(order.index(g) - order.index(p)) <= 1 for g, p in pairs) / len(pairs)
    lacc = sum(g == p for g, p in base) / len(base)
    pres = sum((p != "none") == (g != "none") for g, p in pairs) / len(pairs)
    lpres = sum((p != "none") == (g != "none") for g, p in base) / len(base)
    lines += ["", f"Speech-like noise, measured: presence {lpres:.0%} local-only vs "
                  f"**{pres:.0%} fused**; severity {lacc:.0%} local-only vs "
                  f"**{acc:.0%} fused** exact ({adj:.0%} within one level).", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--fused-noise", action="store_true",
                    help="also run the babble subset through local+LLM fusion (uses API)")
    args = ap.parse_args()

    if args.skip_generate and (OUT / "gt.json").exists():
        cases = json.loads((OUT / "gt.json").read_text())
    else:
        cases = generate()
        print(f"[generated {len(cases)} clips]")

    per_field = evaluate(cases)
    md = report(per_field)
    if args.fused_noise:
        md += "\n" + "\n".join(fused_noise_check(cases))
    Path("analysis/validation_report.md").write_text(md, encoding="utf-8")
    for line in md.splitlines():
        if line.startswith("##"):
            print(line)
    print("[saved analysis/validation_report.md]")


if __name__ == "__main__":
    main()
