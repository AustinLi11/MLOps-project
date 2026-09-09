"""Tests for the phase-1 training baseline.

Each test runs against a throwaway SQLite tracking store under pytest's
``tmp_path``, so the project's own ``mlflow.db`` is never touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytest

# train.py / load_and_predict.py live one level above this tests/ directory.
PHASE1_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE1_ROOT))

import train  # noqa: E402
from load_and_predict import ModelLoadError, load_model, predict  # noqa: E402
from train import GIT_COMMIT_TAG, TrackingError, TrainingConfig, TrainingResult  # noqa: E402


def make_config(
    tracking_uri: str,
    n_estimators: int = 10,
    max_depth: Optional[int] = 3,
    random_state: int = 42,
) -> TrainingConfig:
    return TrainingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        test_size=0.2,
        experiment_name="test-iris-baseline",
        tracking_uri=tracking_uri,
    )


@pytest.fixture()
def tracking_uri(tmp_path: Path) -> str:
    """Isolated local tracking store for a single test."""
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture()
def trained_run(tracking_uri: str) -> TrainingResult:
    return train.train(make_config(tracking_uri))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_are_within_a_reasonable_range(trained_run: TrainingResult) -> None:
    metrics = trained_run.metrics
    assert set(metrics) == {"accuracy", "f1_macro"}
    # A RandomForest on iris should be far better than chance (~0.33) but is
    # still bounded by 1.0; anything outside this band signals a broken pipeline.
    assert 0.80 <= metrics["accuracy"] <= 1.0
    assert 0.80 <= metrics["f1_macro"] <= 1.0


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_identical_arguments_give_bit_identical_metrics(tracking_uri: str) -> None:
    first = train.train(make_config(tracking_uri))
    second = train.train(make_config(tracking_uri))

    assert first.run_id != second.run_id  # two independent runs were recorded
    assert first.metrics == second.metrics  # exact equality, zero tolerance


def test_different_seeds_are_allowed_to_differ(tracking_uri: str) -> None:
    metrics_a = train.train(make_config(tracking_uri, random_state=0)).metrics
    metrics_b = train.train(make_config(tracking_uri, random_state=1)).metrics
    assert set(metrics_a) == set(metrics_b)  # same metric schema regardless of seed


# --------------------------------------------------------------------------- #
# Model reload
# --------------------------------------------------------------------------- #
def test_model_can_be_reloaded_from_run_id_and_predicts(
    trained_run: TrainingResult, tracking_uri: str
) -> None:
    from load_and_predict import load_demo_features

    model = load_model(trained_run.run_id, tracking_uri=tracking_uri)
    features, _ = load_demo_features(n_samples=6)
    labels = predict(model, features)

    assert len(labels) == len(features)
    assert all(label in {0, 1, 2} for label in labels)


def test_unknown_run_id_raises_a_clear_error(trained_run: TrainingResult, tracking_uri: str) -> None:
    with pytest.raises(ModelLoadError) as excinfo:
        load_model("0" * 32, tracking_uri=tracking_uri)
    assert "Could not load" in str(excinfo.value)


def test_load_model_reports_a_missing_tracking_store(tmp_path: Path) -> None:
    missing = f"sqlite:///{(tmp_path / 'does-not-exist' / 'mlflow.db').as_posix()}"
    with pytest.raises(TrackingError) as excinfo:
        load_model("a" * 32, tracking_uri=missing)
    assert "No MLflow tracking store found" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Traceability: params, metrics and git tags land in MLflow
# --------------------------------------------------------------------------- #
def test_run_records_params_metrics_and_git_commit_tag(
    trained_run: TrainingResult, tracking_uri: str
) -> None:
    from mlflow.tracking import MlflowClient

    run = MlflowClient(tracking_uri=tracking_uri).get_run(trained_run.run_id)

    assert run.data.params["n_estimators"] == "10"
    assert run.data.params["max_depth"] == "3"
    assert run.data.params["random_state"] == "42"
    assert run.data.metrics["accuracy"] == pytest.approx(trained_run.metrics["accuracy"])
    assert run.data.metrics["f1_macro"] == pytest.approx(trained_run.metrics["f1_macro"])
    assert GIT_COMMIT_TAG in run.data.tags


def test_git_metadata_degrades_gracefully_outside_a_repository(tmp_path: Path) -> None:
    metadata = train.collect_git_metadata(repo_dir=tmp_path)
    assert metadata[GIT_COMMIT_TAG] == train.GIT_UNKNOWN
    assert set(metadata) == {train.GIT_COMMIT_TAG, train.GIT_BRANCH_TAG, train.GIT_DIRTY_TAG}


# --------------------------------------------------------------------------- #
# CLI wiring: no hyperparameter is hardcoded
# --------------------------------------------------------------------------- #
def test_hyperparameters_come_from_the_command_line() -> None:
    config = train.parse_args(
        [
            "--n-estimators",
            "77",
            "--max-depth",
            "4",
            "--random-state",
            "7",
            "--test-size",
            "0.3",
            "--tracking-uri",
            "sqlite:///tmp.db",
        ]
    )
    assert (config.n_estimators, config.max_depth, config.random_state) == (77, 4, 7)
    assert config.test_size == 0.3
    assert config.tracking_uri == "sqlite:///tmp.db"


def test_max_depth_accepts_none() -> None:
    assert train.parse_args(["--max-depth", "none"]).max_depth is None


@pytest.mark.parametrize("argv", [["--n-estimators", "0"], ["--test-size", "1.5"]])
def test_invalid_hyperparameters_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        train.parse_args(argv)


def test_tracking_uri_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = f"sqlite:///{(tmp_path / 'from-env.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", override)
    assert train.parse_args([]).tracking_uri == override
    # An explicit CLI flag still wins over the environment.
    assert train.parse_args(["--tracking-uri", "sqlite:///cli.db"]).tracking_uri == "sqlite:///cli.db"


def test_main_returns_zero_and_prints_the_run_id(
    capsys: pytest.CaptureFixture[str], tracking_uri: str
) -> None:
    exit_code = train.main(
        [
            "--n-estimators",
            "5",
            "--max-depth",
            "2",
            "--random-state",
            "3",
            "--experiment-name",
            "test-cli",
            "--tracking-uri",
            tracking_uri,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_id" in captured.out


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_unwritable_tracking_directory_gives_a_clear_error(tmp_path: Path) -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses filesystem permissions")

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # readable and traversable, but not writable
    try:
        with pytest.raises(TrackingError) as excinfo:
            train.train(make_config(f"sqlite:///{(locked / 'mlflow.db').as_posix()}"))
        message = str(excinfo.value)
        assert "not writable" in message or "Cannot" in message
        assert "Traceback" not in message
    finally:
        locked.chmod(0o700)


# --------------------------------------------------------------------------- #
# Shared feature contract: the guard against training-serving skew (phase 2)
# --------------------------------------------------------------------------- #
def test_training_features_follow_the_shared_contract() -> None:
    from mlops_shared.features import FEATURE_COLUMNS, TARGET_NAMES

    dataset = train.load_dataset(test_size=0.2, random_state=42)

    assert list(dataset.X_train.columns) == list(FEATURE_COLUMNS)
    assert list(dataset.X_test.columns) == list(FEATURE_COLUMNS)
    assert set(dataset.X_train.dtypes.astype(str)) == {"float64"}
    assert dataset.target_names == list(TARGET_NAMES)


def test_api_shaped_records_normalize_to_the_training_layout() -> None:
    """A serving-style snake_case record must yield the exact training frame."""
    from mlops_shared.features import FEATURE_COLUMNS, build_feature_frame, feature_example

    frame = build_feature_frame([feature_example()])

    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert set(frame.dtypes.astype(str)) == {"float64"}


def test_invalid_test_size_is_reported_as_a_data_error(tracking_uri: str) -> None:
    # 0.99 leaves too few samples per class for a stratified split.
    config = TrainingConfig(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        test_size=0.99,
        experiment_name="test-bad-split",
        tracking_uri=tracking_uri,
    )
    with pytest.raises(train.DataLoadError) as excinfo:
        train.train(config)
    assert "--test-size" in str(excinfo.value)
