"""FastAPI application serving the phase-1 iris model.

Endpoints:
    GET  /health   service state + identity of the loaded model
    POST /predict  batch prediction with per-class probabilities
    GET  /         service banner with links to the OpenAPI docs

Operational contract:

* The model is loaded **once**, during application startup, from ``$MODEL_URI``.
  Requests never touch the tracking store.
* If the model cannot be loaded the process still starts but reports
  ``degraded`` on /health with HTTP 503, so an orchestrator's readiness probe
  keeps it out of rotation instead of the container crash-looping silently.
* Errors are returned as a uniform ``{"error", "detail"}`` envelope. Raw
  exceptions and tracebacks are logged server-side, never sent to the client.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.model_loader import (
    FeatureValidationError,
    ModelBundle,
    ModelLoadError,
    TrackingError,
    load_model_bundle,
    predict_batch,
)
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelIdentity,
    Prediction,
    PredictRequest,
    PredictResponse,
)

SERVICE_NAME = "iris-inference"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; keep the process up if that fails."""
    app.state.bundle = None
    app.state.model_error = None
    try:
        app.state.bundle = load_model_bundle()
    except (ModelLoadError, TrackingError) as exc:
        # Expected, actionable misconfiguration: stay up and report 503 so the
        # operator sees *why* on /health instead of only in a crash log.
        app.state.model_error = str(exc)
        LOGGER.error("Startup model load failed: %s", exc)
    except Exception as exc:  # noqa: BLE001 - never let startup raise raw
        app.state.model_error = f"Unexpected error while loading the model: {type(exc).__name__}."
        LOGGER.exception("Unexpected startup model load failure")
    yield
    app.state.bundle = None


app = FastAPI(
    title="Iris inference service",
    description=(
        "Serves the RandomForest model trained in phase 1. Preprocessing is imported "
        "from the shared feature contract used at training time (`mlops_shared.features`), "
        "so training and serving cannot drift apart."
    ),
    version=__version__,
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


# --------------------------------------------------------------------------- #
# Error handling: uniform envelope, no raw exception text to clients
# --------------------------------------------------------------------------- #
def _error(status_code: int, code: str, detail: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(ErrorResponse(error=code, detail=detail)),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Schema violations (missing/extra fields, wrong types, bad ranges) -> 422."""
    detail = [
        {
            "location": list(error.get("loc", ())),
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", detail)


@app.exception_handler(FeatureValidationError)
async def handle_feature_error(_: Request, exc: FeatureValidationError) -> JSONResponse:
    """Feature-contract violations that survive the schema -> 422, not 500."""
    return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "feature_contract_error", str(exc))


@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    """Reshape framework errors into the same envelope as everything else."""
    code = "model_unavailable" if exc.status_code == 503 else "http_error"
    return _error(exc.status_code, code, exc.detail)


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """Anything unforeseen: log the detail, return a generic message."""
    LOGGER.exception("Unhandled error while serving request", exc_info=exc)
    return _error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "Internal server error. See service logs for details.",
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _require_bundle(request: Request) -> ModelBundle:
    """Return the loaded model or fail with 503 (never load per request)."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "model_error", None) or "No model is loaded.",
        )
    return bundle


def _identity(bundle: ModelBundle) -> ModelIdentity:
    return ModelIdentity(**bundle.identity())


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "version": __version__, "docs": "/docs", "health": "/health"}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service state and loaded model identity",
)
async def health(request: Request) -> JSONResponse:
    """200 with the model identity once loaded; 503 while degraded."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        payload = HealthResponse(
            status="degraded",
            service=SERVICE_NAME,
            version=__version__,
            model=None,
            detail=getattr(request.app.state, "model_error", None) or "No model is loaded.",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=jsonable_encoder(payload),
        )

    payload = HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=__version__,
        model=_identity(bundle),
        detail=None,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(payload))


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict iris species for a batch of samples",
)
async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """Score a batch with the already-loaded model.

    Preprocessing runs through the shared feature contract, so the result is
    identical to calling the same model offline with the same rows.
    """
    bundle = _require_bundle(request)
    results = predict_batch(bundle, payload.to_feature_mappings())
    return PredictResponse(
        model=_identity(bundle),
        predictions=[
            Prediction(
                label=result.label,
                label_name=result.label_name,
                confidence=result.confidence,
                probabilities=result.probabilities,
            )
            for result in results
        ],
    )
