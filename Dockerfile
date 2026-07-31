FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU wheels for the torch stack (the default Linux wheels bundle CUDA)
RUN pip install --no-cache-dir torch==2.8.* torchaudio==2.8.* \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY autoace_pipeline ./autoace_pipeline
COPY app ./app
RUN pip install --no-cache-dir .

# Bake the public models into the image so cold starts don't re-download
# ~1 GB (build with --build-arg SKIP_BAKE=1 to skip and let them download at
# first boot into HF_HOME instead — point HF_HOME at a persistent volume
# then). pyannote weights are gated and small; they always fetch at runtime
# using HF_TOKEN.
ARG SKIP_BAKE=0
ENV HF_HOME=/models/hf TORCH_HOME=/models/torch
RUN if [ "$SKIP_BAKE" != "1" ]; then \
      python -c "from silero_vad import load_silero_vad; load_silero_vad()" \
      && python -c "from funasr import AutoModel; AutoModel(model='FunAudioLLM/SenseVoiceSmall', hub='hf', disable_update=True, device='cpu')" \
      || true; \
    fi

# Runtime state (uploads + results DB). On Railway, attach a Volume mounted
# at /data instead of a Docker VOLUME declaration (unsupported there).
ENV DATA_DIR=/data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
