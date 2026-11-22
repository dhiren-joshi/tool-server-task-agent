import pytest

from bodhan_core.config import Settings
from bodhan_core.errors import ConfigurationError


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"request_timeout_seconds": 0}, "positive number"),
        ({"request_timeout_seconds": float("inf")}, "positive number"),
        ({"max_retries": -1}, "cannot be negative"),
        ({"max_concurrent_requests": 0}, "at least 1"),
        ({"translation_chunk_chars": 0}, "chunk sizes must be positive"),
        ({"tts_chunk_chars": 0}, "chunk sizes must be positive"),
        ({"audio_chunk_seconds": 30}, "between 1 and 29"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        Settings(**kwargs)
