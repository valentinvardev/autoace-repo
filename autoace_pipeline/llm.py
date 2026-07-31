"""Gemini perceptual analysis: primary for emotion, advisory vote for noise.

The prompt encodes three rules learned from the labeled calls:
- judge the dominant tone across the whole call, not the worst moment
- brief or casual profanity alone does not imply upset
- never attribute background-media speech (TV, radio) to the customer

Audio is sent as 16 kHz mono FLAC: Opus support is inconsistent across
Gemini services, tokens are billed per second so transcoding is free, and
48 kHz input showed no accuracy gain in controlled comparison.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from .schema import CallResult, EmotionalTone, Intensity, Severity

MODEL_CHAIN = ["gemini-3-flash-preview", "gemini-2.5-flash"]

# $/1M tokens, standard tier (ai.google.dev/gemini-api/docs/pricing).
# Batch API halves these; the cost memo reports both.
PRICES = {
    "gemini-3-flash-preview": {"audio_in": 1.00, "text_in": 0.30, "text_out": 2.50},
    "gemini-2.5-flash": {"audio_in": 1.00, "text_in": 0.30, "text_out": 2.50},
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
  (e.g. office chatter, music, road noise, TV, keyboard typing, wind, static,
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
- confidence: your confidence in the overall result, 0.0-1.0. Reflect genuine
  uncertainty; ambiguous tone or hard-to-hear audio should lower it.

Judgment rules:
- Judge the DOMINANT tone across the entire call. A brief flash of irritation or an
  isolated profanity in an otherwise normal call does not override the dominant tone.
- Casual or joking profanity alone does not imply upset; weigh how it was delivered.
- If a television, radio, or other media is audible, do NOT attribute its speech to
  the customer.
- Do NOT infer frustration or distress solely from loudness. Do NOT infer background
  noise solely from poor audio quality.
- Attend to prosody as well as words: pitch, speaking rate, volume dynamics, sighs,
  clipped phrasing, sarcasm, interruptions.
- satisfied vs neutral: a customer who is appreciative, thanks the agent, sounds
  pleased or relieved, or happily completes their goal is satisfied, not neutral.
  Reserve neutral for calls with genuinely no positive or negative signal.
- frustrated vs upset: escalating impatience, repeatedly demanding answers or a human
  agent, or a raised voice indicates upset; mild annoyance without escalation is
  frustrated.
- emotional_intensity: medium when the tone is clear and sustained; reserve low for
  barely-there emotion and high for strong escalation needing attention.

Also fill: tone_evidence = short quote/paraphrase with approximate timestamp that
drove your tone decision. noise_evidence = what you heard and where, or "none".
"""

_SCHEMA = types.Schema(
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


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    emotional_tone: EmotionalTone
    emotional_intensity: Intensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Severity
    audio_quality: str
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float
    tone_evidence: str
    noise_evidence: str
    model: str = ""
    latency_s: float = 0.0
    cost_usd: float = 0.0
    audio_tokens: int = 0


def _to_flac_16k(path: str) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a.flac"
        subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                        str(out)], check=True, capture_output=True)
        return out.read_bytes()


class GeminiAnalyzer:
    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.models = models or MODEL_CHAIN

    def analyze(self, path: str, max_attempts: int = 3) -> LLMResult:
        audio = _to_flac_16k(path)
        last: Exception | None = None
        for model in self.models:
            for attempt in range(max_attempts):
                try:
                    return self._call(model, audio)
                except Exception as e:  # noqa: BLE001 - any API failure retries
                    last = e
                    time.sleep(min(2 ** attempt * 2.0, 15.0))
        raise LLMError(f"all models failed: {last}") from last

    def _call(self, model: str, audio: bytes) -> LLMResult:
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=audio, mime_type="audio/flac"), PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_SCHEMA,
            ),
        )
        latency = time.perf_counter() - t0
        data = json.loads(resp.text)

        u = resp.usage_metadata
        modal = {m.modality.name.lower(): m.token_count for m in (u.prompt_tokens_details or [])}
        audio_tok = modal.get("audio", 0)
        text_tok = (u.prompt_token_count or 0) - audio_tok
        out_tok = u.candidates_token_count or 0
        price = PRICES.get(model, PRICES[MODEL_CHAIN[0]])
        cost = (audio_tok * price["audio_in"] + text_tok * price["text_in"]
                + out_tok * price["text_out"]) / 1e6

        return LLMResult(**data, model=model, latency_s=round(latency, 2),
                         cost_usd=round(cost, 6), audio_tokens=audio_tok)
