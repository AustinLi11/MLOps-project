"""Error hierarchy shared by the training and serving code.

Every error here is *expected* and carries an operator-facing message. Callers
(CLI entrypoints, FastAPI handlers) turn them into exit codes or HTTP responses
instead of leaking tracebacks.
"""

from __future__ import annotations


class MlopsError(RuntimeError):
    """Base class for all user-facing failures in this project."""


class TrackingError(MlopsError):
    """The MLflow tracking backend is unreachable, unwritable or misconfigured."""


class ModelLoadError(MlopsError):
    """A model could not be resolved or loaded from the given URI."""


class FeatureValidationError(MlopsError):
    """Input features violate the shared feature contract (names/dtypes/shape)."""
