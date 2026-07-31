# AutoAce — Voice Tone & Background Noise Analysis

Hybrid system for classifying emotional tone and detecting background noise
in production call audio, built for the AutoAce technical trial. A hosted
dashboard accepts batch uploads (ZIP + CSV manifest), processes every clip
through a local-DSP + audio-LLM fusion pipeline, and returns the required
9-field JSON per clip with downloadable results.

## Architecture in one paragraph

Six technical fields are measured **locally** on the original audio —
silence gaps, SNR/noise floor, clipping by consecutive-run analysis,
pause-gated click/HF-burst rates (static), dropout detection, and
pyannote-3.0 overlapped-speech detection — because Gemini transcodes audio
to 16 kbps mono before the model sees it, which destroys exactly the
evidence those fields need. The two emotion fields come from **Gemini 3
Flash** prompted with the trial's verbatim label definitions (content +
prosody beat acoustics-only models on this taxonomy), cross-checked by a
local SenseVoice SER pass whose duration-weighted emotion fractions damp
one-off flashes and feed `confidence`. Noise fields are **fused**: the LLM
hears speech-like noise (TV, chatter) that energy statistics structurally
miss — measured 44% → 100% presence detection on a synthetic babble corpus
— while local click/HF evidence corroborates static and caps implausible
severity claims. See `analysis/validation_report.md` for per-field
confusion matrices on the 55-clip synthetic ground-truth corpus.

## Repository layout

| Path | What it is |
|---|---|
| `autoace_pipeline/` | Analysis pipeline (ingest, DSP, VAD, SER, overlap, LLM, fusion) |
| `app/` | FastAPI dashboard: auth, batch upload, worker, results, downloads |
| `analysis/` | Audit, model-selection experiments, synthetic validation |
| `tests/` | Synthetic-audio unit tests for every deterministic detector |

## Run locally

Requirements: Python 3.12, `ffmpeg` on PATH.

```bash
pip install -e .
cp .env.example .env       # fill in the values below
uvicorn app.main:app --port 8000
```

Open http://localhost:8000, sign in, drop a ZIP containing audio files at
the root plus one `labels.csv` (`name,result_json`; `result_json` may be
empty). CLI alternative without the dashboard:

```bash
python -m autoace_pipeline.cli --full path/to/*.ogg --json results.json
```

Tests: `pytest` (runs in seconds; no models or network needed).

## Environment variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Paid-tier Gemini key (paid tier is required: the free tier trains on submitted content, which the trial's confidentiality constraint forbids) |
| `HF_TOKEN` | Hugging Face token with access to the gated `pyannote/segmentation-3.0` |
| `DASH_USERNAME` / `DASH_PASSWORD` | Dashboard login |
| `SESSION_SECRET` | Cookie-signing secret (any long random string) |
| `DATA_DIR` | Where uploads and the SQLite DB live (default `data/`) |
| `WORKERS` | Analysis worker threads (default 1) |
| `COOKIE_SECURE` | Set to `1` behind HTTPS |

## Deploy (Railway / Render)

The included `Dockerfile` builds a CPU image with ffmpeg and the public
models baked in (silero, SenseVoice); pyannote's small gated weights
download at first boot using `HF_TOKEN`.

1. Create a service from this repo (both platforms auto-detect the Dockerfile).
2. Set the environment variables above.
3. Attach a persistent volume at `/data` (batch files + results DB).
4. First boot pre-warms models (~1–2 min) before the worker accepts jobs;
   the dashboard is available immediately.

## Behavior guarantees

- A malformed or unsupported file fails **alone**: its row records the
  reason; the batch continues. Manifest mismatches (rows without files,
  files without rows, bad JSON) surface as warnings, never batch failures.
- If the process restarts mid-batch, queued/processing files are re-queued
  automatically on boot.
- Results download as CSV (`name,result_json`, same shape as the input
  manifest) or JSON, preserving original filenames.

## Data handling

Audio is processed in-memory / on the service's own volume and is sent to
exactly one external service: the Gemini API (paid tier — Google does not
train on paid-tier content; prompts are retained temporarily only for
abuse monitoring, with a Zero-Data-Retention program available). No other
third party receives audio. A `--local-only` variant (skip the LLM, keep
local fields) degrades gracefully if the API key is absent.
