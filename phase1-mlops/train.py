"""Reproducible RandomForest baseline on the iris dataset, tracked with MLflow.

Phase 1 of the MLOps learning project. Two properties matter here:

* **Traceable** -- every run records all hyperparameters, the ``accuracy`` and
  ``f1_macro`` metrics, the serialized model artifact, and the git commit the
  code was at, so an experiment can always be tied back to a code version.
* **Reproducible** -- every source of randomness is driven by ``--random-state``,
  so two runs with identical arguments produce bit-identical metrics.

Example::

    python train.py --n-estimators 200 --max-depth 5 --random-state 42

The tracking backend is local (no MLflow server required). It defaults to a
SQLite file next to this script and can be overridden with ``--tracking-uri``
or the ``MLFLOW_TRACKING_URI`` environment variable.

The feature contract (column names, order, dtype, label decoding) is *not*
defined here: it lives in :mod:`mlops_shared.features`, which the phase-2
inference service imports as well. That shared import is what prevents
training-serving skew.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlops_shared.errors import FeatureValidationError, MlopsError, TrackingError
from mlops_shared.features import TARGET_NAMES, build_feature_frame
from mlops_shared.tracking import (
    MODEL_ARTIFACT_NAME,
    default_artifact_location,
    default_local_tracking_uri,
    prepare_tracking_backend,
)
from mlops_shared.tracking import resolve_tracking_uri as _resolve_tracking_uri

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DEFAULT_EXPERIMENT_NAME: str = "iris-baseline"

GIT_COMMIT_TAG: str = "git_commit"
GIT_BRANCH_TAG: str = "git_branch"
GIT_DIRTY_TAG: str = "git_dirty"

#: Value used for git tags when the code is not running inside a git checkout.
GIT_UNKNOWN: str = "unknown"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
#: Alias of the shared base error, so ``except TrainingError`` in this CLI keeps
#: catching every expected failure (including the shared TrackingError).
TrainingError = MlopsError


class DataLoadError(TrainingError):
    """The dataset could not be loaded or is unusable."""


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingConfig:
    """Fully resolved run configuration. No hyperparameter is hardcoded."""

    n_estimators: int
    max_depth: Optional[int]
    random_state: int
    test_size: float
    experiment_name: str
    tracking_uri: str
    run_name: Optional[str] = None
    #: 可选：把产出的模型注册到 MLflow Model Registry（只创建新版本，
    #: **不设置任何别名/阶段**——提升为生产模型是第五阶段的人工动作）。
    registered_model_name: Optional[str] = None
    #: 可选：附加到 run 上的审计标签（如 triggered_by / reason / ci_run_url）。
    extra_tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    """Train/test split plus the label names needed for reporting."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_names: list[str]


@dataclass(frozen=True)
class TrainingResult:
    """What a finished run produced."""

    run_id: str
    metrics: dict[str, float]
    model_uri: str
    tracking_uri: str
    experiment_name: str
    #: 注册表里的模型名与版本号（仅在使用 --register-as 时有值）。
    registered_model_name: Optional[str] = None
    registered_model_version: Optional[str] = None


# --------------------------------------------------------------------------- #
# Tracking backend resolution
# --------------------------------------------------------------------------- #
def default_tracking_uri() -> str:
    """Local SQLite store next to this file (never a hardcoded absolute path)."""
    return default_local_tracking_uri(PROJECT_ROOT)


def resolve_tracking_uri(cli_value: Optional[str] = None) -> str:
    """Resolve the tracking URI: CLI flag > ``MLFLOW_TRACKING_URI`` > local default.

    Thin wrapper over :func:`mlops_shared.tracking.resolve_tracking_uri` that
    supplies this project's local default, so the precedence rules exist once.
    """
    resolved = _resolve_tracking_uri(cli_value, default=default_tracking_uri())
    assert resolved is not None  # a default is always supplied
    return resolved


# --------------------------------------------------------------------------- #
# Git metadata (must degrade gracefully outside a repository)
# --------------------------------------------------------------------------- #
def _git(args: Sequence[str], repo_dir: Path) -> Optional[str]:
    """Run a read-only git command, returning ``None`` on any failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # git missing, not executable, or hung
    if completed.returncode != 0:
        return None  # not a repository, or no commits yet
    return completed.stdout.strip() or None


def collect_git_metadata(repo_dir: Path = PROJECT_ROOT) -> dict[str, str]:
    """Best-effort git tags binding this run to a code version.

    Never raises: outside a git checkout the commit/branch tags fall back to
    ``"unknown"`` so the tag schema stays stable for downstream tooling.
    """
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    status = _git(["status", "--porcelain"], repo_dir)
    return {
        GIT_COMMIT_TAG: commit or GIT_UNKNOWN,
        GIT_BRANCH_TAG: branch or GIT_UNKNOWN,
        GIT_DIRTY_TAG: GIT_UNKNOWN if commit is None else str(bool(status)).lower(),
    }


# --------------------------------------------------------------------------- #
# Data, model, metrics
# --------------------------------------------------------------------------- #
def load_dataset(test_size: float, random_state: int) -> Dataset:
    """Load iris and split it deterministically (stratified on the label).

    The raw frame is passed through :func:`mlops_shared.features.build_feature_frame`
    -- the exact function the inference service applies to incoming requests --
    so the model is fit on the same column order and dtype it will be served with.

    Raises:
        DataLoadError: The bundled dataset is unavailable, malformed, violates
            the shared feature contract, or the requested split leaves a class
            without samples.
    """
    try:
        bunch = load_iris(as_frame=True)
        raw_features: pd.DataFrame = bunch.data
        target: pd.Series = bunch.target
        sklearn_target_names = [str(name) for name in bunch.target_names]
    except Exception as exc:  # pragma: no cover - depends on a broken install
        raise DataLoadError(
            "Failed to load the built-in iris dataset from scikit-learn "
            f"({type(exc).__name__}: {exc}). Reinstall dependencies with "
            "`pip install -r requirements.txt` and retry."
        ) from exc

    if raw_features.empty or len(raw_features) != len(target):
        raise DataLoadError(
            f"The iris dataset looks corrupted: {len(raw_features)} feature rows vs "
            f"{len(target)} labels. Reinstall scikit-learn and retry."
        )

    if sklearn_target_names != list(TARGET_NAMES):
        raise DataLoadError(
            f"Label order changed in scikit-learn ({sklearn_target_names}) and no longer "
            f"matches the shared contract ({list(TARGET_NAMES)}). Update "
            "mlops_shared.features.TARGET_NAMES before training, or serving would "
            "decode predictions with the wrong class names."
        )

    try:
        features = build_feature_frame(raw_features)
    except FeatureValidationError as exc:
        raise DataLoadError(
            f"The iris frame does not satisfy the shared feature contract: {exc}"
        ) from exc
    target_names = list(TARGET_NAMES)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=target,
        )
    except ValueError as exc:
        raise DataLoadError(
            f"Cannot split the dataset with --test-size {test_size}: {exc}. "
            "Choose a value that leaves at least one sample per class in both splits."
        ) from exc

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
    )


def build_model(config: TrainingConfig) -> RandomForestClassifier:
    """Instantiate the estimator. ``n_jobs=1`` keeps results machine-independent."""
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=1,
    )


def evaluate(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute the two tracked metrics on a held-out split."""
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
    }


# --------------------------------------------------------------------------- #
# Training run
# --------------------------------------------------------------------------- #
def _activate_experiment(config: TrainingConfig) -> str:
    """Point MLflow at the store and experiment, creating the latter if needed."""
    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        client = MlflowClient(tracking_uri=config.tracking_uri)
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            client.create_experiment(
                config.experiment_name,
                artifact_location=default_artifact_location(config.tracking_uri),
            )
        mlflow.set_experiment(config.experiment_name)
    except MlflowException as exc:
        raise TrackingError(
            f"MLflow rejected the tracking store '{config.tracking_uri}': {exc}. "
            "Verify the URI (--tracking-uri / MLFLOW_TRACKING_URI) and that the "
            "backend is reachable and writable."
        ) from exc
    except OSError as exc:
        raise TrackingError(
            f"Cannot access the MLflow tracking store '{config.tracking_uri}': "
            f"{exc.strerror or exc}. Check the path permissions or choose another URI."
        ) from exc
    return config.experiment_name


def train(config: TrainingConfig) -> TrainingResult:
    """Train, evaluate and log one fully tracked run.

    Raises:
        DataLoadError: The dataset could not be prepared.
        TrackingError: The MLflow backend could not be used.
    """
    prepare_tracking_backend(config.tracking_uri)
    _activate_experiment(config)

    dataset = load_dataset(test_size=config.test_size, random_state=config.random_state)
    model = build_model(config)

    try:
        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.log_params(
                {
                    "n_estimators": config.n_estimators,
                    "max_depth": config.max_depth,
                    "random_state": config.random_state,
                    "test_size": config.test_size,
                    "model_type": type(model).__name__,
                    "dataset": "sklearn.datasets.load_iris",
                    "n_train_samples": len(dataset.X_train),
                    "n_test_samples": len(dataset.X_test),
                }
            )
            tags = collect_git_metadata()
            tags.update({str(k): str(v) for k, v in config.extra_tags.items()})
            mlflow.set_tags(tags)

            model.fit(dataset.X_train, dataset.y_train)
            metrics = evaluate(model, dataset.X_test, dataset.y_test)
            mlflow.log_metrics(metrics)

            signature = infer_signature(dataset.X_train, model.predict(dataset.X_train))
            # registered_model_name 只创建一个新版本；不设置任何 alias/stage，
            # 「提升为生产模型」始终是第五阶段里的人工动作。
            model_info = mlflow.sklearn.log_model(
                model,
                name=MODEL_ARTIFACT_NAME,
                signature=signature,
                input_example=dataset.X_train.head(5),
                registered_model_name=config.registered_model_name,
            )
            registered_version = getattr(model_info, "registered_model_version", None)
            run_id = run.info.run_id
    except MlflowException as exc:
        raise TrackingError(
            f"Failed to log the run to '{config.tracking_uri}': {exc}. "
            "The store may be read-only, locked by another process, or out of disk space."
        ) from exc
    except OSError as exc:
        raise TrackingError(
            f"Filesystem error while writing MLflow data for '{config.tracking_uri}': "
            f"{exc.strerror or exc}. Check free disk space and directory permissions."
        ) from exc

    return TrainingResult(
        run_id=run_id,
        metrics=metrics,
        model_uri=f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}",
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        registered_model_name=config.registered_model_name,
        registered_model_version=(str(registered_version) if registered_version else None),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got '{value}'") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _optional_positive_int(value: str) -> Optional[int]:
    """Accept an integer or ``none``/``None`` for "grow trees fully"."""
    if value.strip().lower() in {"none", "null", ""}:
        return None
    return _positive_int(value)


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a float, got '{value}'") from None
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(f"expected a fraction in (0, 1), got {parsed}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a reproducible RandomForest baseline on iris and track it in MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-estimators", type=_positive_int, default=100, help="Number of trees in the forest."
    )
    parser.add_argument(
        "--max-depth",
        type=_optional_positive_int,
        default=None,
        help="Maximum tree depth; 'none' grows trees until leaves are pure.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for the split and the forest; identical seeds reproduce identical metrics.",
    )
    parser.add_argument(
        "--test-size", type=_unit_interval, default=0.2, help="Held-out fraction of the dataset."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment to log into.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional human-readable MLflow run name."
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help=(
            "MLflow tracking URI. Falls back to $MLFLOW_TRACKING_URI, then to "
            f"'{default_tracking_uri()}'."
        ),
    )
    parser.add_argument(
        "--register-as",
        dest="registered_model_name",
        type=str,
        default=None,
        help=(
            "Register the produced model under this name in the MLflow Model Registry. "
            "Creates a new version only -- no alias or stage is set, so promoting to "
            "production stays a human decision (phase 5)."
        ),
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra MLflow run tag for audit trails (repeatable), e.g. "
            "--tag triggered_by=alice --tag reason='drift alert 2026-09-08'."
        ),
    )
    return parser


def _parse_tags(raw_tags: Sequence[str]) -> dict[str, str]:
    """Turn ``KEY=VALUE`` strings into a tag mapping."""
    tags: dict[str, str] = {}
    for item in raw_tags:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"train.py: error: --tag expects KEY=VALUE, got '{item}'")
        tags[key.strip()] = value.strip()
    return tags


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    """Turn command-line arguments into a fully resolved :class:`TrainingConfig`."""
    args = build_parser().parse_args(argv)
    return TrainingConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        test_size=args.test_size,
        experiment_name=args.experiment_name,
        tracking_uri=resolve_tracking_uri(args.tracking_uri),
        run_name=args.run_name,
        registered_model_name=args.registered_model_name,
        extra_tags=_parse_tags(args.tags),
    )


def _report(result: TrainingResult) -> None:
    print("Run logged successfully.")
    print(f"  tracking_uri : {result.tracking_uri}")
    print(f"  experiment   : {result.experiment_name}")
    print(f"  run_id       : {result.run_id}")
    for name, value in result.metrics.items():
        print(f"  {name:<13}: {value:.6f}")
    print(f"  model_uri    : {result.model_uri}")
    if result.registered_model_name:
        version = result.registered_model_version or "?"
        print(f"  registered   : {result.registered_model_name} v{version} (no alias/stage set)")
    print(f"\nReload it with:\n  python load_and_predict.py --run-id {result.run_id}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code instead of leaking tracebacks."""
    try:
        config = parse_args(argv)
        _report(train(config))
    except TrainingError as exc:
        print(f"train.py: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("train.py: interrupted by user.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
