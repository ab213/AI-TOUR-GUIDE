# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for GPS, TTS, audio, and MLC-LLM
# MLC-LLM requires build tools and system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GPS and location services
    gpsd \
    gpsd-clients \
    # TTS and audio
    espeak \
    espeak-data \
    ffmpeg \
    alsa-utils \
    libasound2-dev \
    # MLC-LLM build dependencies
    cmake \
    git \
    wget \
    curl \
    # MLC-LLM runtime dependencies
    libstdc++6 \
    # Additional dependencies for MLC-LLM
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --upgrade pip setuptools wheel && \
    # Install MLC-LLM from official mlc.ai wheels (CPU version)
    # Following mlc-ai documentation: https://llm.mlc.ai/docs/install/mlc_llm.html
    python -m pip install --pre -U -f https://mlc.ai/wheels \
        mlc-llm-nightly-cpu \
        mlc-ai-nightly-cpu && \
    # Install other Python dependencies
    python -m pip install -r requirements.txt && \
    # Verify MLC-LLM installation
    python -c "import mlc_llm; print(f'MLC-LLM installed at: {mlc_llm.__path__}')"

# Copy the source code into the container.
COPY --chown=appuser:appuser . .

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on (if needed for future web interface).
EXPOSE 8000

# Run the application.
CMD ["python3", "core/main.py"]
