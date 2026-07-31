"""Speech-emotion comparator: SenseVoiceSmall per speech segment.

Role: the second "materially different approach" the brief requires, and a
temporal-evidence source the LLM lacks. The LLM hears the whole call and
names a tone; SenseVoice tags each speech segment independently, so the
duration-weighted class fractions quantify how *sustained* an emotion is —
which is what separates a brief flash of anger from a genuinely upset call.

Limitation (documented in the memo): without diarization the fractions mix
customer and agent speech. The agent is a steady synthetic voice that tags
neutral, so it dilutes but does not flip the customer's signal; thresholds
are set with that dilution in mind. SenseVoice also tags BGM (background
music), a free corroboration signal for TV/media noise.

License: SenseVoiceSmall official weights permit commercial use under the
FunASR Model License; code is MIT.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_EMO = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
_TAG = re.compile(r"<\|([A-Z_]+)\|>")


class SERUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def _model():
    try:
        from funasr import AutoModel
        return AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf",
                         disable_update=True, device="cpu")
    except Exception as e:
        raise SERUnavailable(str(e)) from e


def analyze_segments(mono16k: np.ndarray,
                     speech: list[tuple[float, float]]) -> dict:
    """Per-segment emotion tags -> duration-weighted fractions.

    Returns {"fractions": {...}, "dominant": str, "bgm_s": float,
             "segments": [(start, end, emotion), ...]}
    """
    model = _model()
    weights: dict[str, float] = {}
    segments = []
    bgm_s = 0.0
    for a, b in speech:
        seg = mono16k[int(a * 16000): int(b * 16000)]
        if len(seg) < 1600:  # <0.1 s tells us nothing
            continue
        res = model.generate(input=seg, language="auto", use_itn=False)
        text = res[0]["text"] if res else ""
        tags = set(_TAG.findall(text))
        emo = next((t for t in tags if t in _EMO), "NEUTRAL")
        if emo == "UNKNOWN":
            emo = "NEUTRAL"
        dur = b - a
        weights[emo] = weights.get(emo, 0.0) + dur
        if "BGM" in tags:
            bgm_s += dur
        segments.append((round(a, 1), round(b, 1), emo))

    total = sum(weights.values()) or 1.0
    fractions = {k: round(v / total, 3) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}
    dominant = next(iter(fractions), "NEUTRAL")
    return {"fractions": fractions, "dominant": dominant,
            "bgm_s": round(bgm_s, 1), "segments": segments}


def negative_fraction(fractions: dict[str, float]) -> float:
    return sum(fractions.get(k, 0.0) for k in ("ANGRY", "SAD", "FEARFUL", "DISGUSTED"))


def positive_fraction(fractions: dict[str, float]) -> float:
    return fractions.get("HAPPY", 0.0)
