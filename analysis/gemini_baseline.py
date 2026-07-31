"""Pure-Gemini baseline: all 9 schema fields from one audio prompt.

Serves three purposes: (a) the "simple baseline" the brief asks for,
(b) a perceptual ear on the labels our DSP can't explain (TV, sharp static),
(c) live cost/latency numbers for the memo.

Audio is transcoded to 16 kHz mono FLAC before upload: Opus-in-Ogg support
is inconsistent across Gemini services, and tokens are billed per second,
so transcoding is free.

Usage:
    python analysis/gemini_baseline.py resources/audio/*.ogg --labels resources/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]

# $/1M tokens (standard tier), per model, from ai.google.dev pricing.
PRICES = {
    "gemini-3.5-flash-lite": {"audio_in": 0.30, "text_in": 0.10, "text_out": 0.40},
    "gemini-3.1-flash-lite": {"audio_in": 0.50, "text_in": 0.10, "text_out": 0.40},
    "gemini-2.5-flash-lite": {"audio_in": 0.30, "text_in": 0.10, "text_out": 0.40},
    "gemini-3-flash-preview": {"audio_in": 1.00, "text_in": 0.30, "text_out": 2.50},
}

PROMPT = """\
You are analyzing one recorded phone call between a car-dealership agent and a customer.
Evaluate the CUSTOMER's emotional state, background noise, and technical audio quality.
Return every field of the JSON schema.

Field definitions (apply them exactly):
- emotional_tone: the primary emotional tone expressed by the CUSTOMER (not the agent).
  neutral = no clear positive or negative emotion. satisfied = pleased, relieved,
  appreciative, or clearly positive. frustrated = annoyed, impatient, or dissatisfied
  without strong anger or distress. upset = clearly angry, agitated, or strongly
  dissatisfied. distressed = highly emotional, overwhelmed, panicked, crying, or
  otherwise emotionally escalated.
- emotional_intensity: strength of the detected tone. low = subtle or mild.
  medium = clear and sustained. high = strong, escalated, or likely to require attention.
- background_noise_present: whether meaningful non-speech sound is audible in the
  background. Barely perceptible artifacts do not count.
- background_noise_type: concise description of the dominant background noise
  (e.g. office chatter, music, road noise, television, keyboard typing, wind,
  mechanical noise). Empty string when no noise is present.
- background_noise_severity: none = no meaningful noise. low = audible but does not
  interfere. medium = occasionally interferes with understanding. high = materially
  impairs the conversation.
- audio_quality: technical quality only, independent of emotion: distortion, clipping,
  echo, static, low volume, muffled speech, robotic audio, packet loss.
  clear | slightly_impaired | severely_impaired.
- speaker_overlap_present: two or more speakers talking at the same time enough to
  affect understanding or analysis.
- long_silence_present: an unusually long silence or dead air suggesting a call-flow
  or audio problem. Normal conversational pauses or brief hold while the agent checks
  something do not count.
- confidence: your confidence in the overall result, 0.0-1.0.

Cautions: do NOT infer frustration or distress solely from loudness. Do NOT infer
background noise solely from poor audio quality. Judge emotion from BOTH what the
customer says and how they say it: attend to pitch, speaking rate, volume dynamics,
sighs, sharp exhales, clipped or curt phrasing, sarcasm, and interruptions - not only
the literal words. Listen carefully for faint background sounds (television, music,
chatter, static, hiss, crackle) especially during pauses between words.

Also fill the two evidence fields: tone_evidence = short quote/paraphrase of what the
customer said or did that drove your tone decision, with approximate timestamp.
noise_evidence = what you heard, where in the call, or "none".
"""

SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "emotional_tone": types.Schema(type=types.Type.STRING, enum=["neutral", "satisfied", "frustrated", "upset", "distressed"]),
        "emotional_intensity": types.Schema(type=types.Type.STRING, enum=["low", "medium", "high"]),
        "background_noise_present": types.Schema(type=types.Type.BOOLEAN),
        "background_noise_type": types.Schema(type=types.Type.STRING),
        "background_noise_severity": types.Schema(type=types.Type.STRING, enum=["none", "low", "medium", "high"]),
        "audio_quality": types.Schema(type=types.Type.STRING, enum=["clear", "slightly_impaired", "severely_impaired"]),
        "speaker_overlap_present": types.Schema(type=types.Type.BOOLEAN),
        "long_silence_present": types.Schema(type=types.Type.BOOLEAN),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "tone_evidence": types.Schema(type=types.Type.STRING),
        "noise_evidence": types.Schema(type=types.Type.STRING),
    },
    required=["emotional_tone", "emotional_intensity", "background_noise_present",
              "background_noise_type", "background_noise_severity", "audio_quality",
              "speaker_overlap_present", "long_silence_present", "confidence",
              "tone_evidence", "noise_evidence"],
)


def to_flac(path: str, sr: int) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a.flac"
        subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                        str(out)], check=True, capture_output=True)
        return out.read_bytes()


def run_one(client: genai.Client, model: str, path: Path, sr: int = 16000) -> dict:
    audio = to_flac(str(path), sr)
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=audio, mime_type="audio/flac"), PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=SCHEMA,
        ),
    )
    latency = time.perf_counter() - t0
    u = resp.usage_metadata
    modal = {m.modality.name.lower(): m.token_count for m in (u.prompt_tokens_details or [])}
    audio_tok = modal.get("audio", 0)
    text_tok = (u.prompt_token_count or 0) - audio_tok
    out_tok = u.candidates_token_count or 0
    price = PRICES.get(model, max(PRICES.values(), key=lambda p: p["audio_in"]))
    cost = (audio_tok * price["audio_in"] + text_tok * price["text_in"]
            + out_tok * price["text_out"]) / 1e6
    dur_s = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())
    return {
        "file": path.name,
        "prediction": json.loads(resp.text),
        "latency_s": round(latency, 2),
        "tokens": {"audio_in": audio_tok, "text_in": text_tok, "out": out_tok},
        "cost_usd": round(cost, 6),
        "cost_per_min_usd": round(cost / (dur_s / 60), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--model")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out", default="out/gemini_baseline_results.json")
    args = ap.parse_args()

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    model = args.model
    if not model:
        for cand in MODELS:
            try:
                client.models.generate_content(model=cand, contents="ping")
                model = cand
                break
            except Exception as e:
                print(f"[{cand} unavailable: {str(e)[:90]}]")
    print(f"[model: {model}]\n")

    labels = {}
    if args.labels:
        with open(args.labels, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                labels[row["name"]] = json.loads(row["result_json"])

    results = []
    for fp in args.files:
        p = Path(fp)
        r = run_one(client, model, p, sr=args.sr)
        r["model"], r["sr"] = model, args.sr
        results.append(r)
        pred, lab = r["prediction"], labels.get(p.name, {})
        print(f"=== {p.name}  ({r['latency_s']}s, ${r['cost_per_min_usd']}/min) ===")
        keys = ["emotional_tone", "emotional_intensity", "background_noise_present",
                "background_noise_type", "background_noise_severity", "audio_quality",
                "speaker_overlap_present", "long_silence_present"]
        for k in keys:
            mark = ""
            if lab:
                mark = "  == label" if pred.get(k) == lab.get(k) else f"  != label: {lab.get(k)}"
            print(f"  {k:>26}: {pred.get(k)}{mark}")
        print(f"  {'confidence':>26}: {pred.get('confidence')}")
        print(f"  {'tone_evidence':>26}: {pred.get('tone_evidence')}")
        print(f"  {'noise_evidence':>26}: {pred.get('noise_evidence')}\n")

    Path("out").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[saved {args.out}]")


if __name__ == "__main__":
    main()
