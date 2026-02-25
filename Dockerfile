# ============================================================
# PodcastCleaner — Multi-stage Docker Build
# ============================================================
# Build:  docker build -t podcast-cleaner .
# Run:    docker run --gpus all -v ./output:/app/output podcast-cleaner run --url "..."
# CPU:    docker run -v ./output:/app/output podcast-cleaner run --url "..."
# ============================================================

# --- Stage 1: Builder ---
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    ffmpeg libsndfile1 libopenblas-dev git \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
WORKDIR /app
COPY pyproject.toml .
COPY podcast_cleaner/ podcast_cleaner/
RUN pip install --no-cache-dir -e '.[all]'

# Pre-download ML models (baked into image for faster startup)
RUN python3.11 -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"
RUN python3.11 -c "from df.enhance import init_df; init_df()"

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    ffmpeg libsndfile1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv and app from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Copy model caches
COPY --from=builder /root/.cache/torch /root/.cache/torch
COPY --from=builder /root/.cache/DeepFilterNet /root/.cache/DeepFilterNet

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Default config — user can mount their own
COPY config.example.yaml config.yaml

ENTRYPOINT ["podcast-cleaner"]
CMD ["--help"]
