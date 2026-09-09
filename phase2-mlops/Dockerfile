# syntax=docker/dockerfile:1
#
# Multi-stage build for the iris inference service.
#
# Build from the REPOSITORY ROOT, because the shared feature contract
# (../shared) must be inside the build context -- serving imports the same
# preprocessing code as training instead of duplicating it:
#
#   docker build -f phase2-mlops/Dockerfile -t iris-inference:phase2 .
#
# Layer-by-layer rationale is in the comments below; see README section
# "镜像体积" for the measured effect of each optimization.

# =============================================================================
# Stage 1 (builder): resolve and install dependencies into a private venv.
# Nothing from this stage ships except the venv directory itself.
# =============================================================================
FROM python:3.10-slim AS builder

# [builder 1/5] Build-time interpreter settings. No .pyc/pip cache is worth
# keeping in a stage that gets discarded, and unbuffered output makes build
# logs readable/auditable in CI.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# [builder 2/5] A standalone virtualenv at a fixed path. Installing into a venv
# (instead of the system interpreter) means the runtime stage can take the
# dependency tree as ONE self-contained directory, leaving pip, setuptools
# caches and any build tooling behind.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# [builder 3/5] Copy only the dependency inputs first. This layer is cached and
# re-used across builds; editing app/ code below never re-runs pip.
WORKDIR /build
COPY phase2-mlops/requirements.txt phase2-mlops/requirements.txt
COPY shared/ shared/

# [builder 4/5] Install runtime dependencies. Executed from phase2-mlops/ so the
# relative `../shared` entry in requirements.txt resolves to the copied shared
# package -- the same path that works for local development.
# requirements-dev.txt is intentionally NOT installed: no pytest/httpx in the image.
WORKDIR /build/phase2-mlops
RUN pip install --no-cache-dir -r requirements.txt

# [builder 5/5] Strip bytecode caches from site-packages (~108 MB measured, see
# README). Safe because .pyc files are regenerated on demand, and
# PYTHONDONTWRITEBYTECODE keeps them from coming back at runtime.
# Note: bundled `tests/` directories are deliberately NOT deleted -- numpy
# imports `numpy._core.tests` while deserializing the model, so removing them
# breaks model loading (verified).
RUN find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# =============================================================================
# Stage 2 (runtime): the shipped image. Gets the venv + app code, nothing else.
# =============================================================================
FROM python:3.10-slim AS runtime

# [runtime 1/8] Runtime interpreter settings plus the venv on PATH, so `uvicorn`
# and `python` resolve to the pinned dependency set without activating anything.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/srv \
    MODEL_ARTIFACT_NAME=model
# MODEL_URI is deliberately NOT defaulted: the model to serve is supplied at
# `docker run` time, so the image contains no reference to a specific run.

# [runtime 2/8] Unprivileged user (uid/gid 10001). The service never runs as
# root, and a fixed numeric id keeps file ownership predictable when host
# volumes with models are mounted in.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

# [runtime 3/8] The only artifact taken from the builder: the installed venv.
# Build toolchain, pip caches and the shared package source stay behind.
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# [runtime 4/8] Application code only -- no tests, no README, no Dockerfile
# (also excluded from the build context by .dockerignore).
WORKDIR /srv
COPY --chown=appuser:appuser phase2-mlops/app/ ./app/

# [runtime 5/8] Writable mount point for models supplied as a local directory
# (MODEL_URI=/models/...), owned by the unprivileged user.
RUN mkdir -p /models && chown appuser:appuser /models

# [runtime 6/8] Drop privileges before anything is served.
USER appuser

# [runtime 7/8] Documented port + liveness probe. python:3.10-slim ships no
# curl/wget, so the check uses the interpreter that is already present. A
# degraded service (no model loaded) answers /health with 503 and is reported
# unhealthy.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# [runtime 8/8] Single uvicorn worker: the model is loaded once per process, so
# scale out with replicas (or --workers) rather than reloading per request.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
