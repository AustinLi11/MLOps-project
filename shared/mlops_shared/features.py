"""The feature contract: the single source of truth against training-serving skew.

Both the trainer (``phase1-mlops/train.py``) and the inference service
(``phase2-mlops/app/model_loader.py``) build their model input by calling
:func:`build_feature_frame`. Column names, column order, dtype and label
decoding therefore cannot drift apart between training and serving -- there is
only one implementation of them.

The iris features carry spaces and units in their scikit-learn names
("sepal length (cm)"), which are awkward as JSON keys. :data:`API_FIELD_TO_COLUMN`
holds that translation once, so the HTTP schema and the model input never
disagree about which value is which feature.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import pandas as pd

from .errors import FeatureValidationError

#: Model input columns, in the exact order the model was trained on.
FEATURE_COLUMNS: tuple[str, ...] = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)

#: Class index -> human readable label, as ordered by ``sklearn.datasets.load_iris``.
TARGET_NAMES: tuple[str, ...] = ("setosa", "versicolor", "virginica")

#: All model input is coerced to this dtype before hitting the estimator.
FEATURE_DTYPE: str = "float64"

#: JSON/API field name -> model feature column.
API_FIELD_TO_COLUMN: dict[str, str] = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}

#: Model feature column -> JSON/API field name.
COLUMN_TO_API_FIELD: dict[str, str] = {
    column: field for field, column in API_FIELD_TO_COLUMN.items()
}

FeatureRecords = Union[pd.DataFrame, Sequence[Mapping[str, Any]]]


def n_features() -> int:
    """Number of features the model expects."""
    return len(FEATURE_COLUMNS)


def feature_example() -> dict[str, float]:
    """A valid API-shaped record, used in docs, tests and the OpenAPI schema."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename API field names to model column names; leave canonical names alone."""
    renames = {
        column: API_FIELD_TO_COLUMN[column]
        for column in frame.columns
        if column in API_FIELD_TO_COLUMN
    }
    return frame.rename(columns=renames) if renames else frame


def build_feature_frame(records: FeatureRecords) -> pd.DataFrame:
    """Normalize raw records into the exact model input the estimator was fit on.

    Accepts a :class:`pandas.DataFrame` or a sequence of mappings, keyed either
    by API field name (``sepal_length``) or by model column name
    (``"sepal length (cm)"``). The result always has :data:`FEATURE_COLUMNS` in
    order, cast to :data:`FEATURE_DTYPE`, with the input index preserved.

    Raises:
        FeatureValidationError: Empty input, missing or unexpected features,
            non-numeric values, or NaN/infinite values.
    """
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise FeatureValidationError(
                "Expected a DataFrame or a list of feature records, got "
                f"{type(records).__name__}."
            )
        if any(not isinstance(record, Mapping) for record in records):
            raise FeatureValidationError("Every feature record must be a mapping of name -> value.")
        frame = pd.DataFrame(list(records))

    if frame.empty:
        raise FeatureValidationError("No feature rows to process.")

    frame = _canonicalize_columns(frame)

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    unexpected = [str(column) for column in frame.columns if column not in FEATURE_COLUMNS]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise FeatureValidationError(
            f"Feature contract violated: {'; '.join(details)}. "
            f"Expected exactly {len(FEATURE_COLUMNS)} features: {list(FEATURE_COLUMNS)} "
            f"(API field names: {list(API_FIELD_TO_COLUMN)})."
        )

    frame = frame[list(FEATURE_COLUMNS)]

    try:
        frame = frame.astype(FEATURE_DTYPE)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(
            f"All features must be numeric and castable to {FEATURE_DTYPE}: {exc}."
        ) from exc

    if not frame.notna().all().all():
        offending = [column for column in FEATURE_COLUMNS if frame[column].isna().any()]
        raise FeatureValidationError(f"Features contain missing values: {offending}.")
    if not frame.apply(lambda column: column.abs().lt(float("inf")).all()).all():
        raise FeatureValidationError("Features contain infinite values.")

    return frame


def decode_label(label: int) -> str:
    """Class index -> class name, falling back to the raw index if out of range."""
    return TARGET_NAMES[label] if 0 <= label < len(TARGET_NAMES) else str(label)


def decode_labels(labels: Sequence[int]) -> list[str]:
    """Vectorized :func:`decode_label`."""
    return [decode_label(int(label)) for label in labels]
