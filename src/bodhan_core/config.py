"""Environment-backed configuration. Secrets are never written to artifacts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from bodhan_core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    api_base_url: str = "https://api.bodhan.ai"
    transcribe_key: str | None = None
    translate_key: str | None = None
    speak_key: str | None = None
    ocr_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    max_concurrent_requests: int = 3
    translation_chunk_chars: int = 8000
    tts_chunk_chars: int = 240
    audio_chunk_seconds: int = 29
    mcp_allowed_root: str = "."

    def __post_init__(self) -> None:
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ConfigurationError("BODHAN_REQUEST_TIMEOUT_SECONDS must be a positive number.")
        if self.max_retries < 0:
            raise ConfigurationError("BODHAN_MAX_RETRIES cannot be negative.")
        if self.max_concurrent_requests < 1:
            raise ConfigurationError("BODHAN_MAX_CONCURRENT_REQUESTS must be at least 1.")
        if self.translation_chunk_chars < 1 or self.tts_chunk_chars < 1:
            raise ConfigurationError("Text chunk sizes must be positive.")
        if not 1 <= self.audio_chunk_seconds <= 29:
            raise ConfigurationError("AUDIO_CHUNK_SECONDS must be between 1 and 29.")

    @classmethod
    def from_env(cls, *, env_file: str | None = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file, override=False)
        return cls(
            api_base_url=os.getenv("BODHAN_API_BASE_URL", "https://api.bodhan.ai").rstrip("/"),
            transcribe_key=os.getenv("BODHAN_TRANSCRIBE_KEY"),
            translate_key=os.getenv("BODHAN_TRANSLATE_KEY"),
            speak_key=os.getenv("BODHAN_SPEAK_KEY"),
            ocr_key=os.getenv("BODHAN_OCR_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_model=os.getenv("LLM_MODEL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            request_timeout_seconds=float(os.getenv("BODHAN_REQUEST_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("BODHAN_MAX_RETRIES", "2")),
            max_concurrent_requests=int(os.getenv("BODHAN_MAX_CONCURRENT_REQUESTS", "3")),
            translation_chunk_chars=int(os.getenv("TRANSLATION_CHUNK_CHARS", "8000")),
            tts_chunk_chars=int(os.getenv("TTS_CHUNK_CHARS", "240")),
            audio_chunk_seconds=int(os.getenv("AUDIO_CHUNK_SECONDS", "29")),
            mcp_allowed_root=os.getenv("BODHAN_MCP_ALLOWED_ROOT", "."),
        )

    def require_key(self, model: str) -> str:
        keys = {
            "transcribe": self.transcribe_key,
            "translate": self.translate_key,
            "speak": self.speak_key,
            "ocr": self.ocr_key,
        }
        key = keys.get(model)
        if not key or key == "replace_me":
            raise ConfigurationError(
                f"Missing BODHAN_{model.upper()}_KEY. Add the model-specific key to your local .env file."
            )
        return key

    def require_llm(self) -> tuple[str, str, str]:
        if (
            not self.llm_base_url
            or not self.llm_model
            or not self.llm_api_key
            or self.llm_model == "replace_with_a_tool_calling_model"
            or self.llm_api_key == "replace_me"
        ):
            raise ConfigurationError(
                "LLM is not configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY in .env."
            )
        return self.llm_base_url.rstrip("/"), self.llm_model, self.llm_api_key
