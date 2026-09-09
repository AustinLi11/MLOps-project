"""Model loading and preprocessing for the inference service.

Two rules drive this module:

1. **No hardcoded model.** The model to serve comes from the ``MODEL_URI``
   environment variable and may be a bare MLflow ``run_id``, a
   ``runs:/<run_id>/model`` / ``models:/<name>/<version>`` URI, or a local
   model directory. Swapping models is a restart with a different env var, not
   a code change.
2. **No second preprocessing implementation.** Request features are turned into
   model input by :func:`mlops_shared.features.build_feature_frame` -- the very
   function ``phase1-mlops/train.py`` applies before fitting. There is no
   serving-side copy of that logic to drift out of sync.

The model is loaded once (at application startup, see ``app.main``) and reused
for every request.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from mlops_shared.errors import FeatureValidationError, ModelLoadError, TrackingError
from mlops_shared.features import build_feature_frame, decode_label
from mlops_shared.tracking import (
    MODEL_ARTIFACT_NAME,
    normalize_model_uri,
    prepare_tracking_backend,
    requires_tracking_store,
    resolve_tracking_uri,
    run_id_from_uri,
)

LOGGER = logging.getLogger(__name__)

#: Environment variable holding the model reference. Never defaulted to a
#: specific run id -- an unset value is a configuration error.
MODEL_URI_ENV: str = "MODEL_URI"
ARTIFACT_NAME_ENV: str = "MODEL_ARTIFACT_NAME"

#: Run tags/params surfaced through /health so a response can be traced back to
#: the code and hyperparameters that produced the model.
TRACEABLE_TAGS: tuple[str, ...] = ("git_commit", "git_branch", "git_dirty")
TRACEABLE_PARAMS: tuple[str, ...] = ("n_estimators", "max_depth", "random_state", "test_size")


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model plus everything needed to identify it in a response."""

    model: Any
    model_uri: str
    requested_uri: str
    loaded_at: datetime
    supports_confidence: bool
    classes: tuple[int, ...]
    tracking_uri: Optional[str] = None
    run_id: Optional[str] = None
    model_uuid: Optional[str] = None
    run_tags: Mapping[str, str] = field(default_factory=dict)
    run_params: Mapping[str, str] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        """Serializable model identity for the /health and /predict responses."""
        return {
            "requested_uri": self.requested_uri,
            "resolved_uri": self.model_uri,
            "run_id": self.run_id,
            "model_uuid": self.model_uuid,
            "tracking_uri": self.tracking_uri,
            "loaded_at": self.loaded_at,
            "supports_confidence": self.supports_confidence,
            "git_commit": self.run_tags.get("git_commit"),
            "training_params": dict(self.run_params),
        }


@dataclass(frozen=True)
class PredictionResult:
    """One row's prediction."""

    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def configured_model_uri(env: Optional[Mapping[str, str]] = None) -> str:
    """Read the raw ``MODEL_URI`` value from the environment.

    Raises:
        ModelLoadError: The variable is unset or empty.
    """
    environ = os.environ if env is None else env
    value = (environ.get(MODEL_URI_ENV) or "").strip()
    if not value:
        raise ModelLoadError(
            f"{MODEL_URI_ENV} is not set. Point it at a model: a 32-character MLflow "
            f"run id, 'runs:/<run_id>/{MODEL_ARTIFACT_NAME}', "
            "'models:/<name>/<version>', or a local model directory."
        )
    return value


def configured_artifact_name(env: Optional[Mapping[str, str]] = None) -> str:
    """Artifact name used when ``MODEL_URI`` is a bare run id."""
    environ = os.environ if env is None else env
    return (environ.get(ARTIFACT_NAME_ENV) or "").strip() or MODEL_ARTIFACT_NAME


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _collect_run_metadata(
    run_id: str, tracking_uri: Optional[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Best-effort tags/params of the training run; never fails the load."""
    try:
        from mlflow.tracking import MlflowClient

        run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    except Exception as exc:  # noqa: BLE001 - metadata is a nice-to-have
        LOGGER.warning("Could not read metadata for run %s: %s", run_id, exc)
        return {}, {}

    tags = {key: run.data.tags[key] for key in TRACEABLE_TAGS if key in run.data.tags}
    params = {key: run.data.params[key] for key in TRACEABLE_PARAMS if key in run.data.params}
    return tags, params


def _model_uuid(model_uri: str) -> Optional[str]:
    """Best-effort model identifier, for /health to report *which* model is live.

    For a local model directory the ``MLmodel`` file is read directly: asking
    the registry about a model that was merely copied onto disk would fail, and
    a mounted model directory is exactly how the container is meant to run.
    """
    mlmodel = Path(model_uri) / "MLmodel"
    if mlmodel.is_file():
        try:
            import yaml

            metadata = yaml.safe_load(mlmodel.read_text(encoding="utf-8")) or {}
            return metadata.get("model_uuid") or metadata.get("model_id")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            LOGGER.warning("Could not read %s: %s", mlmodel, exc)
            return None

    try:
        import mlflow.models

        info = mlflow.models.get_model_info(model_uri)
        return getattr(info, "model_uuid", None) or getattr(info, "model_id", None)
    except Exception as exc:  # noqa: BLE001 - identity is a nice-to-have
        LOGGER.warning("Could not read model metadata for %s: %s", model_uri, exc)
        return None


def load_model_bundle(
    requested_uri: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> ModelBundle:
    """Resolve and load the configured model exactly once.

    Args:
        requested_uri: Overrides ``$MODEL_URI`` (used by tests).
        tracking_uri: Overrides ``$MLFLOW_TRACKING_URI``.

    Raises:
        ModelLoadError: The reference is unusable or the model cannot be loaded.
        TrackingError: A ``runs:``/``models:`` reference was given but the
            tracking store is missing or unreadable.
    """
    import mlflow
    import mlflow.sklearn

    raw_uri = requested_uri if requested_uri is not None else configured_model_uri()
    resolved_uri = normalize_model_uri(raw_uri, artifact_name=configured_artifact_name())

    store_uri: Optional[str] = None
    if requires_tracking_store(resolved_uri):
        store_uri = resolve_tracking_uri(tracking_uri)
        if not store_uri:
            raise TrackingError(
                f"{MODEL_URI_ENV}='{raw_uri}' needs an MLflow tracking store to resolve, "
                "but MLFLOW_TRACKING_URI is not set. Either set it (e.g. "
                "'sqlite:////data/mlflow.db') or pass a local model directory instead."
            )
        prepare_tracking_backend(store_uri, must_exist=True)
        mlflow.set_tracking_uri(store_uri)

    LOGGER.info("Loading model from %s (tracking store: %s)", resolved_uri, store_uri or "n/a")
    try:
        # The sklearn flavor (rather than pyfunc) is used deliberately: it keeps
        # predict_proba available so /predict can report a confidence.
        model = mlflow.sklearn.load_model(resolved_uri)
    except Exception as exc:  # noqa: BLE001 - mlflow raises many unrelated types
        raise ModelLoadError(
            f"Could not load the model from '{resolved_uri}': {type(exc).__name__}: {exc}. "
            f"Check {MODEL_URI_ENV}, the tracking store, and that the artifact still exists."
        ) from exc

    run_id = run_id_from_uri(resolved_uri)
    tags, params = _collect_run_metadata(run_id, store_uri) if run_id else ({}, {})
    # `classes_` is a numpy array: test for None explicitly, never truthiness.
    raw_classes = getattr(model, "classes_", None)
    classes = tuple(int(label) for label in raw_classes) if raw_classes is not None else ()

    bundle = ModelBundle(
        model=model,
        model_uri=resolved_uri,
        requested_uri=raw_uri,
        loaded_at=datetime.now(timezone.utc),
        supports_confidence=hasattr(model, "predict_proba"),
        classes=classes,
        tracking_uri=store_uri,
        run_id=run_id,
        model_uuid=_model_uuid(resolved_uri),
        run_tags=tags,
        run_params=params,
    )
    LOGGER.info(
        "Model ready: run_id=%s model_uuid=%s classes=%s confidence=%s",
        bundle.run_id,
        bundle.model_uuid,
        list(bundle.classes),
        bundle.supports_confidence,
    )
    return bundle


# --------------------------------------------------------------------------- #
# Preprocessing + inference (shared with training via mlops_shared.features)
# --------------------------------------------------------------------------- #
def prepare_features(records: Sequence[Mapping[str, float]]) -> pd.DataFrame:
    """Turn validated API records into model input using the shared contract.

    Raises:
        FeatureValidationError: The records violate the feature contract.
    """
    return build_feature_frame(records)


def predict_batch(
    bundle: ModelBundle, records: Sequence[Mapping[str, float]]
) -> list[PredictionResult]:
    """Predict a batch of API records, with per-class probabilities.

    Raises:
        FeatureValidationError: The features violate the shared contract.
        ModelLoadError: The loaded model failed to produce a prediction.
    """
    frame = prepare_features(records)

    try:
        labels = [int(value) for value in bundle.model.predict(frame)]
    except Exception as exc:  # noqa: BLE001 - estimator-specific failures
        raise ModelLoadError(
            f"The loaded model failed to predict: {type(exc).__name__}: {exc}."
        ) from exc

    if not bundle.supports_confidence:
        # Degenerate but honest: without predict_proba the only defensible
        # confidence is 1.0 for the predicted class.
        return [
            PredictionResult(
                label=label,
                label_name=decode_label(label),
                confidence=1.0,
                probabilities={decode_label(label): 1.0},
            )
            for label in labels
        ]

    try:
        proba = bundle.model.predict_proba(frame)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"The loaded model failed to produce probabilities: {type(exc).__name__}: {exc}."
        ) from exc

    class_order = bundle.classes or tuple(range(proba.shape[1]))
    results: list[PredictionResult] = []
    for label, row in zip(labels, proba):
        probabilities = {
            decode_label(int(class_label)): float(value)
            for class_label, value in zip(class_order, row)
        }
        results.append(
            PredictionResult(
                label=label,
                label_name=decode_label(label),
                confidence=probabilities.get(decode_label(label), float(max(row))),
                probabilities=probabilities,
            )
        )
    return results


__all__ = [
    "FeatureValidationError",
    "ModelBundle",
    "ModelLoadError",
    "PredictionResult",
    "TrackingError",
    "configured_model_uri",
    "load_model_bundle",
    "predict_batch",
    "prepare_features",
]
