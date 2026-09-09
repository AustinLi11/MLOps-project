"""Code shared by every phase of the MLOps project.

Importing the feature contract from here -- rather than re-implementing it --
is what keeps training and serving in lockstep. See :mod:`mlops_shared.features`.
"""

from __future__ import annotations

from .errors import FeatureValidationError, MlopsError, ModelLoadError, TrackingError
from .features import (
    API_FIELD_TO_COLUMN,
    COLUMN_TO_API_FIELD,
    FEATURE_COLUMNS,
    FEATURE_DTYPE,
    TARGET_NAMES,
    build_feature_frame,
    decode_label,
    decode_labels,
    feature_example,
    n_features,
)
from .tracking import (
    MODEL_ARTIFACT_NAME,
    default_artifact_location,
    default_local_tracking_uri,
    is_run_id,
    normalize_model_uri,
    prepare_tracking_backend,
    requires_tracking_store,
    resolve_tracking_uri,
    run_id_from_uri,
)

__version__ = "0.2.0"

__all__ = [
    "API_FIELD_TO_COLUMN",
    "COLUMN_TO_API_FIELD",
    "FEATURE_COLUMNS",
    "FEATURE_DTYPE",
    "FeatureValidationError",
    "MODEL_ARTIFACT_NAME",
    "MlopsError",
    "ModelLoadError",
    "TARGET_NAMES",
    "TrackingError",
    "build_feature_frame",
    "decode_label",
    "decode_labels",
    "default_artifact_location",
    "default_local_tracking_uri",
    "feature_example",
    "is_run_id",
    "n_features",
    "normalize_model_uri",
    "prepare_tracking_backend",
    "requires_tracking_store",
    "resolve_tracking_uri",
    "run_id_from_uri",
    "__version__",
]
