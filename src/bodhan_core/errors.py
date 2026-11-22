"""Typed errors returned by the Bodhan integration layer."""


class BodhanError(RuntimeError):
    """Base exception for expected Bodhan integration failures."""


class ConfigurationError(BodhanError):
    """Raised when a required setting or API key is missing."""


class InputFileError(BodhanError):
    """Raised for missing, unreadable, or unsupported input files."""


class BodhanAPIError(BodhanError):
    """Raised when a Bodhan API call fails or returns an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        cost_inr: float = 0.0,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        # Some operations contain several paid requests. Keep the cost of requests
        # that succeeded before a later chunk failed so the run report stays honest.
        self.cost_inr = max(0.0, float(cost_inr))
