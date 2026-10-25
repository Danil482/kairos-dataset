ARG KAIROS_DEPENDENCIES_IMAGE=dependencies
FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 p7zip-full ca-certificates \
    && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home runner \
    && mkdir -p /app /opt/venv && chown -R 10001:10001 /app /opt/venv
USER 10001:10001
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
WORKDIR /app
RUN python -m venv /opt/venv
COPY --chown=10001:10001 requirements/ requirements/
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes \
    -r requirements/cpu.lock -r requirements/gpu.lock
FROM ${KAIROS_DEPENDENCIES_IMAGE} AS runtime
USER 10001:10001
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
RUN find /app -mindepth 1 -maxdepth 1 -exec rm -rf {} +
COPY --chown=10001:10001 kairos/ kairos/
COPY --chown=10001:10001 kairos_lab/ kairos_lab/
COPY --chown=10001:10001 solution/ solution/
COPY --chown=10001:10001 baseline.py evaluate.py ./
COPY --chown=10001:10001 requirements/ requirements/
COPY --chown=10001:10001 configs/night.json configs/night.json
COPY --chown=10001:10001 scripts/ scripts/
COPY --chown=10001:10001 tests/ tests/
RUN chmod -R a+rX /app
ENTRYPOINT ["python", "-m", "solution"]
