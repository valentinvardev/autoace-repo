# Technical Memo — Voice Tone & Background Noise Analysis

**System:** hybrid per-field pipeline — deterministic signal processing for the
technical fields, an audio-LLM for the perceptual fields, fused with
corroboration rules and agreement-based confidence.

**Headline numbers:** $0.0019–0.0025 per audio minute (standard tier;
~$0.001 with batch pricing) against the $0.003 ceiling · 12–26 s per clip
warm (~9 s per audio minute on long clips) · synthetic-corpus validation:
silence 100%, static 96%, audio quality 91%, noise presence 90% · speech-like
noise detection 44% → 100% when the LLM vote joins the local evidence.

---

## 1. Approaches tested and why the final one was selected

The brief asks for a comparison of materially different approaches. Four were
built and measured on the three labeled calls plus a synthetic corpus:

**A. Single audio-LLM, small (Gemini 3.5 Flash-Lite), all 9 fields.**
The simple baseline. It transcribes well but is functionally deaf to
paralinguistics: on the labeled calls it heard no TV, no static, no overlap,
read an audibly impatient caller as neutral, inverted `long_silence` relative
to the measured audio, and pinned confidence at 0.95 everywhere. Rejected.

**B. Single audio-LLM, larger (Gemini 3 Flash), all 9 fields.**
A different animal: it correctly heard the TV ("loud movie dialogue
throughout"), the static ("persistent hiss and clicking" — confirmed by our
own HF-energy measurements), the overlap, and the caller's impatience. But it
remains unreliable on temporal/acoustic booleans: `speaker_overlap` flipped
between two identical temperature-0 runs, `background_noise_severity` flipped
medium/high between runs, and `long_silence` contradicted the measured
envelope. An LLM's word on measurable facts is not evidence. Rejected as a
*sole* engine; retained where it is genuinely superior.

**C. Fully local (DSP + Silero VAD + SenseVoice SER + pyannote).**
The technical fields work very well locally (validation below). Two measured
failures: (1) SenseVoiceSmall reads this telephone audio as almost entirely
neutral — it missed both the impatient caller (94.8% neutral, zero anger) and
an isolated profanity — consistent with published results of specialized SER
degrading outside acted corpora; (2) speech-like background noise (TV,
chatter, babble) is structurally invisible to energy statistics, because VAD
absorbs it as speech: on a controlled babble corpus, local-only presence
detection was 44%. Rejected as a sole engine.

**D. Hybrid per-field routing (final).** Each field goes to the instrument
that measures it best, and contested fields are fused with explicit rules.
Selected because every routing decision is backed by a measured failure of
the alternative (above).

A supporting experiment: sending 48 kHz audio instead of 16 kHz to the LLM
gave no systematic gain (2 of 3 noise/quality fields slightly worse), so the
API path uses 16 kHz mono FLAC — transcoding is free since Gemini bills audio
per second, not per byte.

## 2. Final architecture

```
upload → ffmpeg normalize (clamped decode, dual-mono detection)
       ├─ local: envelope/SNR, silence gaps, clipping runs, pause-gated
       │         clicks & HF bursts, dropouts, Silero VAD, SenseVoice SER,
       │         pyannote overlapped-speech detection
       ├─ Gemini 3 Flash (concurrent with local): 9-field structured output,
       │         verbatim label definitions + judgment rules in the prompt
       └─ fusion → 9-field result + confidence + decision trace
```

| Field | Decided by | Rationale (each backed by a measured failure) |
|---|---|---|
| `emotional_tone`, `intensity` | LLM, adjusted by SER | Content + prosody beat acoustics-only SER on this taxonomy; SER's duration-weighted class fractions quantify how *sustained* an emotion is — zero sustained anger demotes an LLM "upset" to "frustrated" (an isolated flash of irritation does not define a call). `distressed` is never demoted: missing escalation is the costliest production error. |
| `background_noise_*` | Fusion | LLM hears speech-like noise the DSP cannot; local click/HF evidence corroborates static and escalates severity; an LLM-only faint report without physical corroboration is treated as the brief's "barely perceptible artifacts" and dropped; an LLM "high" claim is capped to medium when measured pauses are clean (< −55 dBFS) — you cannot "materially impair" a call whose pauses are silent. |
| `audio_quality` | Local only | Gemini downsamples audio to 16 kbps mono before the model sees it (per Google's own docs), destroying exactly the evidence this field needs; the LLM also conflates background static with technical impairment, which the label definitions explicitly separate. |
| `long_silence` | Local only | Deterministic from the envelope; the LLM contradicted measured gaps in both directions. Threshold 10 s: a verified 7.3 s mid-call hold is labeled *false* in the provided data — normal agent-checks-something silence does not count. |
| `speaker_overlap` | pyannote only | LLM answers flipped between identical runs; all provided audio is dual-mono (channel correlation 0.999999 in every 5 s window), so channel energy cannot help. Threshold: ≥0.5 s of detected overlap — the overlap-free call measures exactly 0 s, the overlap-labeled calls 0.88 s and 2.06 s. |
| `confidence` | Fusion | Starts from the LLM's self-estimate capped at 0.95 (its flat 0.95 is a ceiling, not information), −0.07 per cross-source disagreement, −0.1 under low SNR, floored at 0.3. Every penalty is recorded in a per-file decision trace visible in the dashboard. |

Prompt engineering was deliberately principled rather than label-fitted: the
rules added after error analysis (judge the dominant tone across the whole
call; brief or casual profanity alone does not imply upset; never attribute
background-media speech to the customer; the loudness and noise-vs-quality
cautions from the brief) are rules any human rater would accept, not
patches for individual clips.

## 3. Validation

With n=3 labeled calls, honest per-class validation from provided data alone
is impossible — so ground truth was manufactured. **Synthetic corpus:** 55
clips built from slices of the production calls with degradations whose
answer is known by construction: white/pink/babble noise mixed at exact SNRs
(30/20/10/5 dB), clipping driven to known clipped-sample fractions, lowpass
filtering, packet loss, inserted silences of known duration, click trains at
known rates, level-matched overlapped speech. Full confusion matrices in
[`analysis/validation_report.md`](analysis/validation_report.md); summary:

| Field (local path) | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| `long_silence` | 100% (55/55) | 1.000 | gaps recovered within one frame |
| static suspicion | 96% | 0.889 | |
| `audio_quality` | 91% | 0.899 | misses concentrate at tier boundaries |
| noise presence | 90% | 0.904 | all misses are speech-like noise (see below) |
| noise severity | 85% | 0.814 | per-class F1 0.75–0.91 across four levels |
| overlap | 80% | 0.800 | misses only overlap events ≤1 s per occurrence |

Macro F1 is reported alongside accuracy because accuracy alone rewards
predicting the dominant class; the report also gives per-class precision,
recall, F1 and support for every field.

**The fusion experiment:** the babble subset (speech-like noise, the local
path's known blind spot) run through local+LLM fusion moved presence
detection from 44% to **100%** and exact severity from 11% to 56% (100%
within one level). This is the quantified case for the hybrid.

**Provided labeled calls (calibration fit, not an accuracy claim):** the
final system matches 19 of 24 scoreable field decisions, including 18/18 on
the six technical fields. All 5 misses are emotion tone/intensity, and all
are adjacent-class (frustrated vs upset; frustrated vs neutral on a call with
an isolated profanity; neutral vs satisfied). Per the brief's own warning,
we do not report training-set accuracy as expected performance; with n=3 the
honest statement is leave-one-out behavior: the routing and thresholds were
chosen from measured evidence, and the emotion boundary cases are exactly
where more labeled data is needed (next steps).

**On macro F1.** It is the right metric for this problem and it is reported
for every field where it is computable (table above, per-class breakdowns in
the report): unlike accuracy it cannot be inflated by a dominant class, so a
model that always answers the majority label scores badly — on a 90/10
corpus, always predicting the majority yields 90% accuracy but macro F1 0.47.
The one field where we do **not** report it is `emotional_tone`, and not by
omission: three labeled calls cover three of five classes with exactly one
example each, so per-class F1 can only come out 0.0 or 1.0 and averaging
those produces noise, not a measurement. With a labeled set of roughly 200
calls, macro F1 with confidence intervals over grouped (per-caller)
cross-validation folds is what we would report and tune against.

**Leakage disclosure:** the synthetic carriers derive from the same three
calls used for calibration; thresholds fitted on that corpus are therefore
not fully independent of the calibration audio. Degradations are synthetic
and label-free, which limits but does not eliminate the coupling. Stated
here deliberately.

## 4. Cost analysis

Gemini bills audio at 32 tokens/second (1,920 tokens per audio minute).
Measured on real calls with `gemini-3-flash-preview` (audio in $1.00/M,
text in $0.30/M, output $2.50/M):

| Scenario | Cost per audio minute |
|---|---|
| 3-minute call, standard tier | **$0.0017–0.0019** |
| 30-second clip, standard tier (worst case: fixed prompt+output amortize badly) | **$0.0025** |
| Batch API (50% discount), production path | **~$0.001** |
| Local fields | $0 marginal; compute amortized below |

Assumptions: prompt ≈ 700 text tokens; output ≈ 150–250 tokens (evidence
fields included — dropping them in production trims ~30% of output cost);
prices as of July 2026. Local compute amortization: a $5–20/month container
processes ~6–10 audio minutes per wall-clock minute per worker (measured RTF
0.1–0.15 warm), i.e. compute adds ≈$0.0001–0.0005 per audio minute at modest
utilization. **Ceiling compliance holds in the worst case measured
($0.0025/min < $0.003/min) and by 3x on the production path.**

## 5. Latency analysis

Measured warm (models resident; one-time prewarm 60–90 s at boot):

| Stage | Time |
|---|---|
| Local analysis (DSP + VAD + SER + overlap), 3-min call | ~26 s (SER is 15 s of it) |
| Gemini call | 5–30 s per clip, length-insensitive |
| **End-to-end per clip** (LLM runs concurrently with local) | **12–26 s** |
| Per audio minute (long calls) | ~9 s |

Single-worker throughput: 2–4 clips/minute; a 50-clip evaluation batch
completes in ~15–25 minutes with per-file progress visible. The pipeline is
stateless per file, so throughput scales linearly with workers/vCPUs; the
dashboard ships with one worker (concurrent model *loading* is unsafe on
some platforms; loads are pre-warmed serially at boot) and `WORKERS` is a
one-variable scale-up after a stability pass on the target host.

## 6. External API disclosure & data handling

- **API:** Google Gemini, `gemini-3-flash-preview` (pinned fallback:
  `gemini-2.5-flash`), paid tier only. Pricing assumptions in §4.
- **Retention:** on the paid tier Google does not use prompts or responses
  to train its models; content is logged temporarily solely for
  abuse-monitoring, and a Zero-Data-Retention program exists for approved
  projects. The free tier *does* train on content and is never used.
- **Does audio leave controlled infrastructure?** Yes, to exactly one
  processor: the Gemini API receives a 16 kHz mono FLAC of each clip. No
  other third party receives audio. All other processing (including both
  local models) runs on the deployment host. If the LLM is unreachable or
  the key absent, the system degrades to local-only fields rather than
  failing.
- **License diligence:** the technically most convenient SER model
  (audeering wav2vec2 dimensional, with a clean arousal/valence mapping to
  this taxonomy) is CC-BY-NC — unusable in a commercial product — and was
  rejected for that reason. SenseVoiceSmall's official weights permit
  commercial use (FunASR model license; MIT code). pyannote segmentation-3.0
  weights are gated but MIT-licensed toolkit; terms accepted on our account.

## 7. Failure modes, limitations, next steps

**Known failure modes, from observation rather than speculation:**
- *LLM run-to-run variance at temperature 0* (severity and overlap flips).
  Mitigated by routing measurable fields locally and capping severity with
  physical evidence; residual variance remains on emotional tone.
- *A hung API call once blocked the worker queue* during field testing; the
  fix (120 s per-attempt timeout, 300 s per-file deadline, file fails alone,
  batch continues) is in place, plus orphan re-queue on restart — a killed
  process resumes its batch on boot, which was verified live.
- *SER near-uniform neutrality on telephone audio* — it is a guard rail and
  comparison, not a decider; its failure direction (under-detecting emotion)
  is why `distressed` is exempt from demotion.
- *Overlap events shorter than ~1 s* are below pyannote's reliable
  sensitivity here (measured); borderline for the "enough to affect
  understanding" definition, but a real limitation.
- *Label noise*: the provided labels carry the PDF example's confidence 0.82
  verbatim on all three rows, and one call labeled `neutral` contains an
  isolated profanity. The labeler's rubric (dominant whole-call tone,
  perceptual noise judgment) was reverse-engineered and encoded, but a
  hidden set labeled by a different person may shift boundaries.
- *Adversarial audio* (speech that addresses the analysis system) is
  constrained by strict enum-schema output but not fully tested.

**Limitations by construction:** dual-mono source audio forecloses
channel-based speaker separation; intensity for `neutral` calls is
under-defined by the label semantics (the provided data suggests a `medium`
prior); the `-preview` model can be deprecated (fallback pinned and the cost
model holds at the same price for the fallback).

**Next steps, in value order:**
1. ~200 labeled calls would turn emotion boundary heuristics into fitted
   calibration curves (and per-class confusion matrices with real power).
2. Multi-worker throughput pass on the production host (linear scaling).
3. Batch API tier for the production path (halves LLM cost; enables N=2
   self-consistency voting on tone within the same budget).
4. Capture telephony legs separately upstream (agent/customer channel split
   would make speaker attribution and overlap near-trivial).
5. Distil the fusion into a single calibrated classifier over local features
   + LLM logits once enough labeled volume exists.

---

*Reproducibility: `README.md` covers setup, environment variables, and
deploy; `requirements.lock.txt` pins the working dependency set; `pytest`
runs 24 synthetic-audio tests in seconds; `analysis/` contains the audit,
model-selection experiments, and the synthetic-validation generator that
rebuilds every number in §3.*
