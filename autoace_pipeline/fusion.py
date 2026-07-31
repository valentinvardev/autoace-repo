"""Per-field merge of local measurements and the LLM's perceptual vote.

Routing rationale (each choice validated on the labeled calls):
- emotion: LLM. Content + prosody beat acoustics-only SER on this taxonomy.
- noise: fusion. Steady noise and impulsive static are measured locally;
  speech-like background (TV, chatter) is only audible to the LLM. A faint
  LLM-only report without physical corroboration is treated as the brief's
  "barely perceptible artifacts" and does not count.
- audio_quality, long_silence: local only. Deterministic, and the LLM
  conflates static with impairment / flow gaps with dead air.
- overlap: pyannote only. LLM answers flipped between identical runs.
- confidence: starts from the LLM's own value, penalized by cross-source
  disagreement, floored/capped. Calibration documented in the memo.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fields import LocalDecisions
from .llm import LLMResult
from .schema import CallResult, Features, Severity

_SEV_ORDER = ["none", "low", "medium", "high"]


def _bump(sev: Severity) -> Severity:
    i = min(_SEV_ORDER.index(sev) + 1, len(_SEV_ORDER) - 1)
    return _SEV_ORDER[i]  # type: ignore[return-value]


@dataclass
class FusionTrace:
    """Why each contested field ended where it did (dashboard detail view)."""
    noise_rule: str = ""
    quality_disagreement: bool = False
    silence_disagreement: bool = False
    overlap_source: str = ""
    confidence_penalties: list[str] = None  # type: ignore[assignment]


def fuse(f: Features, local: LocalDecisions, llm: LLMResult,
         overlap_present: bool | None, overlap_ratio: float | None) -> tuple[CallResult, FusionTrace]:
    trace = FusionTrace(confidence_penalties=[])

    # --- background noise ---
    corroborated = local.noise_present_local or local.static_suspected
    llm_sev: Severity = llm.background_noise_severity if llm.background_noise_present else "none"

    if local.noise_present_local and not llm.background_noise_present:
        # local steady/impulsive evidence stands on its own
        present, severity, ntype = True, local.noise_severity_local, "background noise"
        trace.noise_rule = "local-only evidence"
        trace.confidence_penalties.append("llm missed measured noise")
    elif llm.background_noise_present and _SEV_ORDER.index(llm_sev) >= 2:
        # LLM hears interfering noise; trust type, severity from LLM
        present, severity, ntype = True, llm_sev, llm.background_noise_type
        trace.noise_rule = "llm >= medium"
    elif llm.background_noise_present and corroborated:
        # faint to the LLM but physically corroborated: count it, escalate one step
        present, ntype = True, llm.background_noise_type
        severity = _bump(llm_sev) if local.static_suspected else llm_sev
        trace.noise_rule = "llm low + local corroboration"
    elif llm.background_noise_present:
        # faint and uncorroborated = barely perceptible artifacts: do not count
        present, severity, ntype = False, "none", ""
        trace.noise_rule = "llm low, uncorroborated -> none"
    else:
        present, severity, ntype = False, "none", ""
        trace.noise_rule = "both clean"

    # --- audio quality: local decision; LLM disagreement only costs confidence ---
    quality = local.audio_quality
    if llm.audio_quality != quality:
        trace.quality_disagreement = True
        trace.confidence_penalties.append("quality disagreement")

    # --- long silence: local; disagreement noted ---
    silence = local.long_silence_present
    if llm.long_silence_present != silence:
        trace.silence_disagreement = True
        trace.confidence_penalties.append("silence disagreement")

    # --- overlap ---
    if overlap_present is not None:
        ov = overlap_present
        trace.overlap_source = "pyannote"
    else:
        ov = llm.speaker_overlap_present
        trace.overlap_source = "llm-fallback"
        trace.confidence_penalties.append("overlap from llm fallback")

    # --- confidence ---
    conf = min(max(llm.confidence, 0.0), 1.0)
    conf = min(conf, 0.95)  # the LLM's flat 0.95 is a ceiling, not information
    conf -= 0.07 * len(trace.confidence_penalties)
    if f.snr_db < 15:
        conf -= 0.1
    conf = round(min(max(conf, 0.3), 0.95), 2)

    result = CallResult(
        emotional_tone=llm.emotional_tone,
        emotional_intensity=llm.emotional_intensity,
        background_noise_present=present,
        background_noise_type=ntype if present else "",
        background_noise_severity=severity if present else "none",
        audio_quality=quality,
        speaker_overlap_present=ov,
        long_silence_present=silence,
        confidence=conf,
    )
    return result, trace
