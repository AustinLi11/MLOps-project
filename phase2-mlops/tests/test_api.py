"""API tests for the inference service.

The suite trains throwaway models by importing the phase-1 trainer, so it also
verifies the end-to-end contract between the two phases: a model logged by
``train.py`` is loadable by the service purely through ``MODEL_URI``, and the
HTTP prediction equals the offline prediction for the same rows.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

PHASE2_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = PHASE2_ROOT.parent / "phase1-mlops"
for _path in (PHASE2_ROOT, PHASE1_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train  # noqa: E402  (phase-1 trainer: creates the models under test)
from app.main import app  # noqa: E402
from app.model_loader import configured_model_uri  # noqa: E402
from app.schemas import MAX_BATCH_SIZE  # noqa: E402
from mlops_shared.errors import ModelLoadError  # noqa: E402
from mlops_shared.features import FEATURE_COLUMNS, feature_example  # noqa: E402
from mlops_shared.tracking import normalize_model_uri  # noqa: E402

SAMPLES: list[dict[str, float]] = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.0, "sepal_width": 2.2, "petal_length": 4.0, "petal_width": 1.0},
    {"sepal_length": 7.3, "sepal_width": 2.9, "petal_length": 6.3, "petal_width": 1.8},
]


# --------------------------------------------------------------------------- #
# Fixtures: a throwaway tracking store holding two model versions
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def store(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("tracking")
    tracking_uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"

    def _train(n_estimators: int, max_depth: Optional[int], seed: int) -> train.TrainingResult:
        return train.train(
            train.TrainingConfig(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                test_size=0.2,
                experiment_name="api-tests",
                tracking_uri=tracking_uri,
            )
        )

    return {
        "tracking_uri": tracking_uri,
        "v1": _train(10, 3, 42),
        "v2": _train(25, 2, 7),
    }


@contextmanager
def serving_client(
    monkeypatch: pytest.MonkeyPatch,
    model_uri: str,
    tracking_uri: Optional[str],
) -> Iterator[TestClient]:
    """Start the app exactly as the container does: configuration via env vars."""
    monkeypatch.setenv("MODEL_URI", model_uri)
    if tracking_uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    with TestClient(app) as client:  # entering runs the startup hook -> loads once
        yield client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, store: dict[str, Any]) -> Iterator[TestClient]:
    with serving_client(monkeypatch, store["v1"].run_id, store["tracking_uri"]) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_returns_200_with_the_model_identity(
    client: TestClient, store: dict[str, Any]
) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["run_id"] == store["v1"].run_id
    assert body["model"]["resolved_uri"] == f"runs:/{store['v1'].run_id}/model"
    assert body["model"]["requested_uri"] == store["v1"].run_id
    assert body["model"]["supports_confidence"] is True
    # Traceability inherited from phase 1: the training commit and hyperparameters.
    assert body["model"]["git_commit"]
    assert body["model"]["training_params"]["n_estimators"] == "10"


def test_health_is_degraded_when_the_model_cannot_be_loaded(
    monkeypatch: pytest.MonkeyPatch, store: dict[str, Any]
) -> None:
    unknown_run = "0" * 32
    with serving_client(monkeypatch, unknown_run, store["tracking_uri"]) as client:
        health = client.get("/health")
        assert health.status_code == 503
        body = health.json()
        assert body["status"] == "degraded"
        assert body["model"] is None
        assert body["detail"]
        assert "Traceback" not in body["detail"]

        # Requests are refused with the same envelope, not a 500.
        predicted = client.post("/predict", json={"instances": [feature_example()]})
        assert predicted.status_code == 503
        assert predicted.json()["error"] == "model_unavailable"


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_returns_labels_and_confidences(client: TestClient) -> None:
    response = client.post("/predict", json={"instances": SAMPLES})
    assert response.status_code == 200

    body = response.json()
    assert len(body["predictions"]) == len(SAMPLES)
    for prediction in body["predictions"]:
        assert prediction["label"] in {0, 1, 2}
        assert prediction["label_name"] in {"setosa", "versicolor", "virginica"}
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert prediction["confidence"] == pytest.approx(
            prediction["probabilities"][prediction["label_name"]]
        )
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_predict_response_identifies_the_serving_model(
    client: TestClient, store: dict[str, Any]
) -> None:
    body = client.post("/predict", json={"instances": SAMPLES}).json()
    assert body["model"]["run_id"] == store["v1"].run_id


def test_predict_matches_offline_prediction_for_the_same_model(
    client: TestClient, store: dict[str, Any]
) -> None:
    """Acceptance 3: HTTP and offline inference must agree exactly."""
    import mlflow
    import pandas as pd

    from load_and_predict import load_model, predict as offline_predict

    api_body = client.post("/predict", json={"instances": SAMPLES}).json()
    api_labels = [prediction["label"] for prediction in api_body["predictions"]]
    api_confidences = [prediction["confidence"] for prediction in api_body["predictions"]]

    frame = pd.DataFrame(
        [[record[field] for field in ("sepal_length", "sepal_width", "petal_length", "petal_width")]
         for record in SAMPLES],
        columns=list(FEATURE_COLUMNS),
    )

    offline_labels = offline_predict(
        load_model(store["v1"].run_id, tracking_uri=store["tracking_uri"]), frame
    )
    assert api_labels == offline_labels

    mlflow.set_tracking_uri(store["tracking_uri"])
    estimator = mlflow.sklearn.load_model(f"runs:/{store['v1'].run_id}/model")
    offline_confidences = [float(max(row)) for row in estimator.predict_proba(frame)]
    assert api_confidences == pytest.approx(offline_confidences)


def test_model_is_loaded_once_not_per_request(client: TestClient) -> None:
    first = client.post("/predict", json={"instances": SAMPLES}).json()
    second = client.post("/predict", json={"instances": SAMPLES}).json()
    # loaded_at is stamped when the model is read from the store.
    assert first["model"]["loaded_at"] == second["model"]["loaded_at"]
    assert first["model"]["model_uuid"] == second["model"]["model_uuid"]


def test_model_uri_switches_versions_without_code_changes(
    monkeypatch: pytest.MonkeyPatch, store: dict[str, Any]
) -> None:
    """Acceptance 4: only the env var changes between the two runs below."""
    seen = []
    for version in ("v1", "v2"):
        with serving_client(monkeypatch, store[version].run_id, store["tracking_uri"]) as client:
            body = client.get("/health").json()
            assert body["status"] == "ok"
            seen.append(body["model"]["run_id"])
            assert body["model"]["training_params"]["n_estimators"] == (
                "10" if version == "v1" else "25"
            )
    assert seen == [store["v1"].run_id, store["v2"].run_id]
    assert seen[0] != seen[1]


def test_local_directory_model_uri_needs_no_tracking_store(
    monkeypatch: pytest.MonkeyPatch, store: dict[str, Any], tmp_path: Path
) -> None:
    """The container's recommended setup: a mounted model directory."""
    import mlflow

    mlflow.set_tracking_uri(store["tracking_uri"])
    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{store['v1'].run_id}/model",
        dst_path=str(tmp_path / "model"),
    )

    with serving_client(monkeypatch, local_dir, None) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model"]["run_id"] is None  # no run context needed

        body = client.post("/predict", json={"instances": SAMPLES}).json()
        assert [prediction["label"] for prediction in body["predictions"]]


# --------------------------------------------------------------------------- #
# Input validation: 422, never 500
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"instances": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}]},
         "missing feature"),
        ({"instances": [{**feature_example(), "petal_width": "wide"}]}, "non-numeric feature"),
        ({"instances": [{**feature_example(), "extra_feature": 1.0}]}, "unexpected feature"),
        ({"instances": [{**feature_example(), "petal_width": -1.0}]}, "out-of-range feature"),
        ({"instances": []}, "empty batch"),
        ({"instances": [feature_example()] * (MAX_BATCH_SIZE + 1)}, "oversized batch"),
        ({"rows": [feature_example()]}, "unknown top-level key"),
        ({"instances": feature_example()}, "wrong container type"),
    ],
)
def test_invalid_input_returns_422(client: TestClient, payload: dict, reason: str) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"{reason} should be rejected with 422"

    body = response.json()
    assert body["error"] == "validation_error"
    assert body["detail"], "the client needs to know which field was wrong"
    assert "Traceback" not in repr(body["detail"])


def test_validation_errors_do_not_reach_the_model(client: TestClient) -> None:
    """A 422 must be produced by the schema, before any inference happens."""
    response = client.post("/predict", json={"instances": [{"sepal_length": 5.1}]})
    assert response.status_code == 422
    # The service is still healthy afterwards.
    assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_unset_model_uri_is_a_clear_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_URI", raising=False)
    with pytest.raises(ModelLoadError) as excinfo:
        configured_model_uri()
    assert "MODEL_URI is not set" in str(excinfo.value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a" * 32, f"runs:/{'a' * 32}/model"),
        ("runs:/abc/model", "runs:/abc/model"),
        ("models:/iris/3", "models:/iris/3"),
    ],
)
def test_model_uri_forms_are_normalized(raw: str, expected: str) -> None:
    assert normalize_model_uri(raw) == expected


def test_unusable_model_uri_is_rejected_with_a_clear_message() -> None:
    with pytest.raises(ModelLoadError) as excinfo:
        normalize_model_uri("/nonexistent/model/dir")
    assert "Cannot interpret model reference" in str(excinfo.value)
