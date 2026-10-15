# Exoplanet Atlas: container image for the Streamlit app.
#
# Build:  docker build -t exoplanet-atlas .
# Run:    docker run -p 8501:8501 exoplanet-atlas      (then open http://localhost:8501)
#
# The trained model (models/) and the six precomputed demo reports
# (data/examples/) are copied into the image, so the demo works offline.
# "Re-run live analysis" downloads TESS data from MAST at run time.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# libgomp is needed by XGBoost; curl is used by the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install the pinned dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code, trained model, demo reports, UI assets and Streamlit config.
COPY . .

# Run as a non-root user; the cache directory for downloaded light curves must be writable.
RUN useradd --create-home appuser \
    && mkdir -p /app/data/cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
