# syntax=docker/dockerfile:1
# Build a target:   docker build -f docker/Dockerfile --target api -t viabsa-api .
#                   docker build -f docker/Dockerfile --target app -t viabsa-app .
# EXTRAS controls what is installed. Default serves baseline + ONNX models on CPU (no PyTorch).
# Use --build-arg EXTRAS=serve,train to serve a PyTorch checkpoint (explanations need PyTorch).

FROM python:3.11-slim AS builder
ARG EXTRAS="serve,onnx"
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu ".[${EXTRAS}]"

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app reports ./reports
USER app

FROM runtime AS api
ENV VIABSA_MODEL_PATH=/app/models/baseline
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "viabsa.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS app
COPY --chown=app:app app ./app
ENV VIABSA_API_URL=http://api:8000
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
