"""Load a model logged by ``train.py`` from its MLflow ``run_id`` and predict.

This is the reference offline loader. It shares the feature contract with both
the trainer and the phase-2 inference service by importing
:mod:`mlops_shared.features`, so a prediction made here and the same prediction
made over HTTP go through identical preprocessing::

    from load_and_predict import load_model, predict

    model = load_model(run_id="a1b2c3...")      # pyfunc: framework-agnostic
    labels = predict(model, features_dataframe)

Example (CLI)::

    python load_and_predict.py --run-id a1b2c3...
    python load_and_predict.py --run-id a1b2c3... --input-csv samples.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.datasets import load_iris

from mlops_shared.errors import FeatureValidationError, ModelLoadError
from mlops_shared.features import (
    API_FIELD_TO_COLUMN,
    build_feature_frame,
    decode_labels,
    n_features,
)
from mlops_shared.tracking import (
    MODEL_ARTIFACT_NAME,
    normalize_model_uri,
    prepare_tracking_backend,
)
from train import TrainingError, resolve_tracking_uri

__all__ = [
    "InputDataError",
    "ModelLoadError",
    "PredictionError",
    "load_model",
    "predict",
    "run",
]


class InputDataError(TrainingError):
    """The features supplied for prediction are missing or malformed."""


class PredictionError(TrainingError):
    """The model rejected the supplied features."""


@dataclass(frozen=True)
class PredictionOutput:
    """Predictions alongside the features they were produced from."""

    features: pd.DataFrame
    labels: list[int]
    label_names: list[str]
    true_names: Optional[list[str]] = None


def model_uri(run_id: str, artifact_name: str = MODEL_ARTIFACT_NAME) -> str:
    """Build the stable ``runs:/<run_id>/<artifact>`` URI used by every phase."""
    return normalize_model_uri(run_id, artifact_name=artifact_name)


def load_model(
    run_id: str,
    tracking_uri: Optional[str] = None,
    artifact_name: str = MODEL_ARTIFACT_NAME,
) -> mlflow.pyfunc.PyFuncModel:
    """Load the model logged under ``run_id`` as a generic pyfunc model.

    Args:
        run_id: MLflow run id printed by ``train.py`` (a full model URI also works).
        tracking_uri: Overrides the resolved default (CLI > env > local store).
        artifact_name: Artifact name used at logging time.

    Returns:
        A pyfunc model exposing ``.predict(DataFrame)``. Use
        ``mlflow.sklearn.load_model(uri)`` instead when you need the native
        estimator API (e.g. ``predict_proba``) -- that is what the phase-2
        service does to report confidences.

    Raises:
        ModelLoadError: The run, the artifact, or the tracking store is missing.
        TrackingError: The tracking store path is unusable.
    """
    resolved_uri = resolve_tracking_uri(tracking_uri)
    prepare_tracking_backend(resolved_uri, must_exist=True)
    mlflow.set_tracking_uri(resolved_uri)

    uri = model_uri(run_id, artifact_name)
    try:
        return mlflow.pyfunc.load_model(uri)
    except MlflowException as exc:
        raise ModelLoadError(
            f"Could not load '{uri}' from tracking store '{resolved_uri}': {exc}. "
            "Check the run id (`mlflow ui`) and that the run finished successfully."
        ) from exc
    except OSError as exc:
        raise ModelLoadError(
            f"Cannot read the model artifact for '{uri}': {exc.strerror or exc}. "
            "The artifact directory may have been moved, deleted, or is unreadable."
        ) from exc


def predict(model: mlflow.pyfunc.PyFuncModel, features: pd.DataFrame) -> list[int]:
    """Run the model on ``features`` and return plain integer class labels.

    ``features`` is normalized through the shared feature contract first, so
    column order, naming and dtype match training exactly.

    Raises:
        InputDataError: The features violate the shared feature contract.
        PredictionError: The model rejected the (valid) input frame.
    """
    try:
        frame = build_feature_frame(features)
    except FeatureValidationError as exc:
        raise InputDataError(str(exc)) from exc

    try:
        raw = model.predict(frame)
    except MlflowException as exc:
        raise PredictionError(
            f"The model rejected the input features: {exc}. "
            "Column names, order and dtypes must match the training data."
        ) from exc
    return [int(value) for value in pd.Series(raw).tolist()]


def _load_iris_frame() -> pd.DataFrame:
    """Full iris frame (features + ``target`` column)."""
    try:
        return load_iris(as_frame=True).frame
    except Exception as exc:  # pragma: no cover - depends on a broken install
        raise InputDataError(
            f"Failed to load the built-in iris dataset ({type(exc).__name__}: {exc})."
        ) from exc


def load_demo_features(n_samples: int) -> tuple[pd.DataFrame, list[str]]:
    """Deterministic sample of the iris dataset, with its true label names."""
    if n_samples <= 0:
        raise InputDataError(f"--n-samples must be positive, got {n_samples}.")
    full_frame = _load_iris_frame()
    # Sample across the whole frame so all three classes can appear.
    frame = full_frame.sample(n=min(n_samples, len(full_frame)), random_state=0).sort_index()
    true_names = decode_labels(frame["target"].tolist())
    return build_feature_frame(frame.drop(columns=["target"])), true_names


def load_csv_features(path: Path) -> pd.DataFrame:
    """Read features from CSV and validate them against the shared contract.

    Raises:
        InputDataError: The file is missing, unreadable, empty, or its columns
            do not satisfy the feature contract.
    """
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise InputDataError(f"Input file '{path}' does not exist.") from exc
    except PermissionError as exc:
        raise InputDataError(f"Input file '{path}' is not readable: {exc.strerror}.") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise InputDataError(f"Input file '{path}' is not a valid CSV: {exc}.") from exc

    if frame.empty:
        raise InputDataError(f"Input file '{path}' contains no data rows.")

    try:
        return build_feature_frame(frame)
    except FeatureValidationError as exc:
        raise InputDataError(f"Input file '{path}': {exc}") from exc


def run(
    run_id: str,
    tracking_uri: Optional[str] = None,
    artifact_name: str = MODEL_ARTIFACT_NAME,
    input_csv: Optional[Path] = None,
    n_samples: int = 5,
) -> PredictionOutput:
    """Load the model for ``run_id`` and predict on CSV input or demo samples."""
    model = load_model(run_id, tracking_uri=tracking_uri, artifact_name=artifact_name)

    if input_csv is not None:
        features = load_csv_features(input_csv)
        true_names = None
    else:
        features, true_names = load_demo_features(n_samples)

    labels = predict(model, features)
    return PredictionOutput(
        features=features,
        labels=labels,
        label_names=decode_labels(labels),
        true_names=true_names,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="load_and_predict.py",
        description="Load a model from an MLflow run_id and print predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run id to load from.")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Falls back to $MLFLOW_TRACKING_URI, then the local store.",
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default=MODEL_ARTIFACT_NAME,
        help="Artifact name the model was logged under.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            f"CSV holding the {n_features()} feature columns "
            f"({', '.join(API_FIELD_TO_COLUMN)}, or their scikit-learn names). "
            "Omit to predict on a sample of the iris dataset."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of demo rows to predict when --input-csv is not given.",
    )
    return parser


def _report(run_id: str, output: PredictionOutput) -> None:
    print(f"Loaded model from run {run_id}; {len(output.labels)} prediction(s):\n")
    table = output.features.copy()
    table["predicted"] = output.label_names
    if output.true_names is not None:
        table["actual"] = output.true_names
    print(table.to_string())
    if output.true_names is not None:
        correct = sum(p == t for p, t in zip(output.label_names, output.true_names))
        print(f"\nMatched {correct}/{len(output.label_names)} known labels.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code instead of leaking tracebacks."""
    try:
        args = build_parser().parse_args(argv)
        output = run(
            run_id=args.run_id,
            tracking_uri=args.tracking_uri,
            artifact_name=args.artifact_name,
            input_csv=args.input_csv,
            n_samples=args.n_samples,
        )
        _report(args.run_id, output)
    except TrainingError as exc:
        print(f"load_and_predict.py: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("load_and_predict.py: interrupted by user.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
