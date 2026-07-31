# GSV-TTS-Lite MultiSpeaker — CPU image (API server)
FROM python:3.11-slim

WORKDIR /app

# System deps: libsndfile (soundfile), ffmpeg (av), git (for -e .)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the repo (including WebUI/API/tests)
COPY . .

# CPU torch + package + API deps (API/requirements.txt installs local gsv_tts via -e ..)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r API/requirements.txt

# Pretrained model cache (mount a volume here to avoid re-downloading)
VOLUME /root/.cache/gsv

EXPOSE 9880

# Health check: the API is ready once this returns
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9880/', timeout=3)" || exit 1

CMD ["python", "API/personal_api.py", "--port", "9880"]
