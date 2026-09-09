"""MLflow location helpers shared by training and serving.

Pure path/URI logic on purpose: this module must not import ``mlflow``, so the
inference image can reason about model locations without pulling the training
stack in. The MLflow calls themselves live in the phase that needs them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .errors import ModelLoadError, TrackingError

#: Artifact name every phase logs/loads the model under.
#: Stable contract: a model is always addressable as ``runs:/<run_id>/model``.
MODEL_ARTIFACT_NAME: str = "model"

DEFAULT_DB_FILENAME: str = "mlflow.db"
DEFAULT_ARTIFACT_DIRNAME: str = "mlartifacts"

#: Backends that live somewhere else and need no local filesystem preparation.
REMOTE_SCHEMES: frozenset[str] = frozenset(
    {"http", "https", "databricks", "databricks-uc", "postgresql", "mysql", "mssql"}
)

#: URI schemes ``mlflow.*.load_model`` understands directly.
MODEL_URI_SCHEMES: tuple[str, ...] = (
    "runs:/",
    "models:/",
    "file://",
    "s3://",
    "gs://",
    "wasbs://",
    "abfss://",
    "http://",
    "https://",
)

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


# --------------------------------------------------------------------------- #
# Tracking URI resolution
# --------------------------------------------------------------------------- #
def default_local_tracking_uri(base_dir: Path) -> str:
    """SQLite tracking URI for a store kept inside ``base_dir``.

    Derived from the caller's own location, so nothing hardcodes an absolute path.
    """
    return f"sqlite:///{(base_dir / DEFAULT_DB_FILENAME).as_posix()}"


def resolve_tracking_uri(
    cli_value: Optional[str] = None, default: Optional[str] = None
) -> Optional[str]:
    """Resolve the tracking URI: explicit value > ``MLFLOW_TRACKING_URI`` > ``default``."""
    if cli_value:
        return cli_value
    env_value = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_value:
        return env_value
    return default


def sqlite_db_path(uri: str) -> Optional[Path]:
    """Filesystem path of a SQLite tracking store, or ``None`` for other backends."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return None
    raw = uri[len(prefix) :]
    return Path(raw) if raw else None


def file_store_dir(uri: str) -> Optional[Path]:
    """Directory of a legacy file-based tracking store, or ``None``."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if not parsed.scheme or len(parsed.scheme) == 1:  # bare path, incl. Windows drives
        return Path(uri)
    return None


def local_store_target(uri: str) -> Optional[Path]:
    """Path that must be writable for a local backend, or ``None`` if remote."""
    if urlparse(uri).scheme in REMOTE_SCHEMES:
        return None
    return sqlite_db_path(uri) or file_store_dir(uri)


def default_artifact_location(uri: str) -> Optional[str]:
    """Artifact directory to pin to a new experiment, kept next to the store.

    An explicit location makes artifact paths independent of the current working
    directory. ``None`` means "let MLflow decide" (remote backends and legacy
    file stores, which already co-locate artifacts).
    """
    db_path = sqlite_db_path(uri)
    if db_path is None:
        return None
    artifact_dir = (db_path.parent / DEFAULT_ARTIFACT_DIRNAME).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.as_uri()


def prepare_tracking_backend(uri: str, *, must_exist: bool = False) -> None:
    """Validate a local tracking backend up front, with actionable errors.

    Args:
        uri: Resolved tracking URI.
        must_exist: When true (read paths), the store must already exist
            instead of being created on demand.

    Raises:
        TrackingError: The store is missing, not creatable or not writable.
    """
    if file_store_dir(uri) is not None and sqlite_db_path(uri) is None:
        # MLflow >= 3.16 refuses file-based stores unless this opt-out is set.
        # Honour the user's explicit choice instead of crashing on it.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    target = local_store_target(uri)
    if target is None:
        return  # remote backend: nothing to prepare locally

    is_db_file = sqlite_db_path(uri) is not None
    directory = target.parent if is_db_file else target

    if must_exist:
        if is_db_file and not target.exists():
            raise TrackingError(
                f"No MLflow tracking store found at '{target}'. "
                "Run train.py first, or point MLFLOW_TRACKING_URI at the store "
                "that holds your runs."
            )
        if not is_db_file and not directory.exists():
            raise TrackingError(
                f"No MLflow tracking directory found at '{directory}'. "
                "Run train.py first, or set MLFLOW_TRACKING_URI."
            )
        return

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrackingError(
            f"Cannot create the MLflow tracking directory '{directory}': {exc.strerror or exc}. "
            "Pick a writable location via MLFLOW_TRACKING_URI "
            "(for example: MLFLOW_TRACKING_URI=sqlite:///$HOME/mlflow.db)."
        ) from exc

    if not os.access(directory, os.W_OK | os.X_OK):
        raise TrackingError(
            f"The MLflow tracking directory '{directory}' is not writable by the current user. "
            "Fix its permissions or set MLFLOW_TRACKING_URI to a writable path."
        )
    if is_db_file and target.exists() and not os.access(target, os.W_OK):
        raise TrackingError(
            f"The MLflow tracking database '{target}' is read-only for the current user. "
            "Fix its permissions or choose another tracking URI."
        )


# --------------------------------------------------------------------------- #
# Model URI resolution (used by the serving layer's MODEL_URI env var)
# --------------------------------------------------------------------------- #
def is_run_id(value: str) -> bool:
    """True if ``value`` is a bare MLflow run id (32 hex characters)."""
    return bool(_RUN_ID_PATTERN.match(value.strip().lower()))


def normalize_model_uri(raw: str, artifact_name: str = MODEL_ARTIFACT_NAME) -> str:
    """Turn a user-supplied model reference into a URI ``mlflow`` can load.

    Accepts a bare run id, any supported MLflow URI scheme, or an existing local
    path (a directory containing ``MLmodel``). Nothing about a specific model is
    hardcoded -- the value comes from configuration.

    Raises:
        ModelLoadError: The reference is empty or is neither a run id, a known
            URI scheme, nor an existing local path.
    """
    value = (raw or "").strip()
    if not value:
        raise ModelLoadError(
            "No model reference given. Set MODEL_URI to a run id, a "
            f"'runs:/<run_id>/{artifact_name}' URI, a 'models:/<name>/<version>' URI, "
            "or a local model directory."
        )

    if is_run_id(value):
        return f"runs:/{value.lower()}/{artifact_name}"
    if value.startswith(MODEL_URI_SCHEMES):
        return value

    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())

    raise ModelLoadError(
        f"Cannot interpret model reference '{raw}'. Expected a 32-character run id, "
        f"one of the URI schemes {list(MODEL_URI_SCHEMES)}, or an existing local "
        "model directory (the path above does not exist)."
    )


def run_id_from_uri(uri: str) -> Optional[str]:
    """Extract the run id from a ``runs:/<run_id>/...`` URI, else ``None``."""
    if not uri.startswith("runs:/"):
        return None
    remainder = uri[len("runs:/") :].strip("/")
    run_id = remainder.split("/", 1)[0]
    return run_id or None


def requires_tracking_store(uri: str) -> bool:
    """True if resolving ``uri`` needs a tracking/registry server to look it up."""
    return uri.startswith(("runs:/", "models:/"))
