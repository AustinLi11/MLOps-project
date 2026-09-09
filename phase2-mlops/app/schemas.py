"""Pydantic request/response models for the inference API.

Field names come from :data:`mlops_shared.features.API_FIELD_TO_COLUMN`, the
same mapping ``model_loader`` uses to build model input, so the HTTP contract
and the model's feature contract cannot disagree. Anything that fails these
schemas is rejected with HTTP 422 before the model is ever touched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from mlops_shared.features import API_FIELD_TO_COLUMN, feature_example

#: Upper bound on a single request, so one client cannot exhaust the worker.
MAX_BATCH_SIZE: int = 1000

#: Measurements are centimetres: non-positive or absurd values are rejected.
_MIN_CM: float = 0.0
_MAX_CM: float = 1000.0


class FeatureRecord(BaseModel):
    """One iris sample, keyed by API field name."""

    model_config = ConfigDict(
        extra="forbid",  # unknown/extra features -> 422 instead of silent drop
        json_schema_extra={"example": feature_example()},
    )

    sepal_length: float = Field(gt=_MIN_CM, le=_MAX_CM, description="Sepal length in cm.")
    sepal_width: float = Field(gt=_MIN_CM, le=_MAX_CM, description="Sepal width in cm.")
    petal_length: float = Field(gt=_MIN_CM, le=_MAX_CM, description="Petal length in cm.")
    petal_width: float = Field(gt=_MIN_CM, le=_MAX_CM, description="Petal width in cm.")

    def to_feature_mapping(self) -> dict[str, float]:
        """Record keyed by model feature column, via the shared field mapping."""
        values = self.model_dump()
        return {API_FIELD_TO_COLUMN[field]: values[field] for field in API_FIELD_TO_COLUMN}


class PredictRequest(BaseModel):
    """A batch of samples to score."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"instances": [feature_example()]}},
    )

    instances: list[FeatureRecord] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Between 1 and {MAX_BATCH_SIZE} samples.",
    )

    def to_feature_mappings(self) -> list[dict[str, float]]:
        return [instance.to_feature_mapping() for instance in self.instances]


class ModelIdentity(BaseModel):
    """Which model answered -- enough to trace a response back to a run."""

    requested_uri: str = Field(description="Raw MODEL_URI value the service was started with.")
    resolved_uri: str = Field(description="URI actually loaded (run id expanded to runs:/...).")
    run_id: Optional[str] = Field(default=None, description="MLflow run id, if applicable.")
    model_uuid: Optional[str] = Field(default=None, description="MLflow model identifier.")
    tracking_uri: Optional[str] = Field(default=None, description="Tracking store consulted.")
    loaded_at: datetime = Field(description="When the process loaded this model (UTC).")
    supports_confidence: bool = Field(description="Whether real probabilities are available.")
    git_commit: Optional[str] = Field(
        default=None, description="git_commit tag of the training run, when recorded."
    )
    training_params: dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters logged with the training run."
    )


class Prediction(BaseModel):
    """One prediction with its confidence."""

    label: int = Field(description="Predicted class index.")
    label_name: str = Field(description="Predicted class name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted class.")
    probabilities: dict[str, float] = Field(description="Probability per class name.")


class PredictResponse(BaseModel):
    """Predictions plus the identity of the model that produced them."""

    model: ModelIdentity
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Service state and the identity of the loaded model."""

    status: Literal["ok", "degraded"] = Field(description="'ok' once a model is loaded.")
    service: str = Field(description="Service name.")
    version: str = Field(description="Service version.")
    model: Optional[ModelIdentity] = Field(
        default=None, description="Null only when no model could be loaded."
    )
    detail: Optional[str] = Field(default=None, description="Why the service is degraded.")


class ErrorResponse(BaseModel):
    """Uniform error envelope: a stable code plus a sanitized message."""

    error: str = Field(description="Machine-readable error code.")
    detail: Any = Field(description="Human-readable explanation or field-level errors.")


__all__ = [
    "ErrorResponse",
    "FeatureRecord",
    "HealthResponse",
    "MAX_BATCH_SIZE",
    "ModelIdentity",
    "Prediction",
    "PredictRequest",
    "PredictResponse",
]
