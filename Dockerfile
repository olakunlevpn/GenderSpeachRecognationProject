# Multi-stage lightweight Dockerfile optimized for free cloud hosting (Railway / Render / Hugging Face Spaces)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies (ffmpeg & libsndfile1 for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for rapid dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition
COPY pyproject.toml .

# Install dependencies (CPU-only PyTorch to minimize container size to < 500MB)
RUN uv pip install --system --no-cache \
    "numpy<2" \
    scipy \
    soundfile \
    librosa \
    scikit-learn \
    joblib \
    pandas \
    matplotlib \
    streamlit \
    tabulate \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy application source code and models
COPY src/ /app/src/
COPY app/ /app/app/
COPY saved_models/ /app/saved_models/

# Expose Railway dynamic port
EXPOSE 8501

# Run Streamlit server bound to 0.0.0.0 and dynamic $PORT
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
