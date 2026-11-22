"""HTTP integration for Bodhan's four supported models."""

from __future__ import annotations

import base64
import json
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import fitz
import httpx
from pydub import AudioSegment

from bodhan_core.chunking import split_text
from bodhan_core.config import Settings
from bodhan_core.costs import (
    ocr_cost,
    speech_cost,
    transcription_cost,
    translation_cost,
)
from bodhan_core.errors import BodhanAPIError, InputFileError

SUPPORTED_TRANSLATION_LANGUAGES = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "ks-Deva",
    "mai", "ml", "mni", "mni-Beng", "mr", "ne", "or", "pa", "sa", "sat",
    "sd", "sd-Arab", "ta", "te", "ur", "en",
}
SUPPORTED_SPEECH_LANGUAGES = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur", "en",
}

DEFAULT_VOICES = {
    "as": "Prastuti", "bn": "Ishita", "brx": "Gwrbw", "doi": "Preeti",
    "gu": "Dhara", "hi": "Kavya", "kn": "Deepika", "kok": "Anjali",
    "ks": "Zoon", "mai": "Vaidehi", "ml": "Lakshmi", "mni": "Thoibi",
    "mr": "Anagha", "ne": "Srijana", "or": "Itishree", "pa": "Kaur",
    "sa": "Bharati", "sat": "Phulmani", "sd": "Moomal", "ta": "Anitha",
    "te": "Sravani", "ur": "Saba", "en": "Arun",
}


class BodhanClient:
    """A synchronous Bodhan client with process-wide concurrency limits."""

    _semaphore_lock = threading.Lock()
    _semaphores: dict[int, threading.BoundedSemaphore] = {}

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._http = httpx.Client(timeout=httpx.Timeout(self.settings.request_timeout_seconds))
        # MCP creates a client per tool invocation, so the semaphore must be shared
        # across instances to enforce a real process-wide concurrency ceiling.
        limit = self.settings.max_concurrent_requests
        with self._semaphore_lock:
            self._semaphore = self._semaphores.setdefault(
                limit, threading.BoundedSemaphore(limit)
            )

    def close(self) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        model_key: str,
        **kwargs: Any,
    ) -> httpx.Response:
        key = self.settings.require_key(model_key)
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {key}"

        last_error: Exception | None = None
        with self._semaphore:
            for attempt in range(self.settings.max_retries + 1):
                try:
                    if attempt:
                        uploads = kwargs.get("files")
                        if isinstance(uploads, dict):
                            for upload in uploads.values():
                                if isinstance(upload, tuple) and len(upload) >= 2:
                                    file_object = upload[1]
                                    if hasattr(file_object, "seek"):
                                        file_object.seek(0)
                    response = self._http.request(method, url, headers=headers, **kwargs)
                    if self._should_retry_response(response) and attempt < self.settings.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = float(retry_after) if retry_after else 0.75 * (2**attempt)
                        except ValueError:
                            delay = 0.75 * (2**attempt)
                        time.sleep(min(max(delay, 0.0), 8.0))
                        continue
                    if response.is_error:
                        self._raise_api_error(response)
                    return response
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt >= self.settings.max_retries:
                        break
                    time.sleep(min(0.75 * (2**attempt), 8.0))
        raise BodhanAPIError(f"Bodhan request failed after retries: {last_error or 'unknown error'}")

    @staticmethod
    def _should_retry_response(response: httpx.Response) -> bool:
        if response.status_code in {500, 502, 503, 504}:
            return True
        if response.status_code != 429:
            return False
        # Bodhan uses 429 for both rate limits and exhausted account credit.
        # Retrying a balance failure cannot help and consumes more requests.
        try:
            body = response.json()
            error = body.get("error", {}) if isinstance(body, dict) else {}
            code = str(error.get("code", "")).lower() if isinstance(error, dict) else ""
            message = str(error.get("message", "")).lower() if isinstance(error, dict) else ""
        except ValueError:
            code, message = "", response.text.lower()
        if any(token in f"{code} {message}" for token in ("credit", "balance", "insufficient_funds", "budget")):
            return False
        return True

    @staticmethod
    def _raise_api_error(response: httpx.Response) -> None:
        code = None
        message = response.text[:1000]
        try:
            body = response.json()
            error = body.get("error", {}) if isinstance(body, dict) else {}
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message", message) if isinstance(error, dict) else message
        except ValueError:
            pass
        raise BodhanAPIError(
            f"Bodhan API returned HTTP {response.status_code}: {message}",
            status_code=response.status_code,
            code=str(code) if code is not None else None,
        )

    def transcribe_audio(self, input_path: str | Path, language: str | None = None) -> dict[str, Any]:
        """Transcribe an audio file, splitting it into <=29-second WAV windows."""
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise InputFileError(f"Audio file not found: {source}")

        try:
            audio = AudioSegment.from_file(source)
        except Exception as exc:  # pydub wraps ffmpeg/decoder errors in several types
            raise InputFileError(f"Could not decode audio file {source.name}: {exc}") from exc

        if len(audio) <= 0:
            raise InputFileError("Audio file is empty or has zero duration.")

        chunk_ms = self.settings.audio_chunk_seconds * 1000
        transcripts: list[str] = []
        segments: list[dict[str, Any]] = []
        total_seconds = len(audio) / 1000
        billed_seconds = 0.0
        started = time.perf_counter()

        try:
            with tempfile.TemporaryDirectory(prefix="bodhan_asr_") as temp_dir:
                for index, start_ms in enumerate(range(0, len(audio), chunk_ms), start=1):
                    chunk = audio[start_ms : start_ms + chunk_ms]
                    chunk_path = Path(temp_dir) / f"chunk_{index:04d}.wav"
                    chunk.export(chunk_path, format="wav")
                    data = {"model": "indic-transcribe"}
                    if language:
                        data["language"] = language
                    with chunk_path.open("rb") as handle:
                        response = self._request(
                            "POST",
                            f"{self.settings.api_base_url}/v1/audio/transcriptions",
                            model_key="transcribe",
                            data=data,
                            files={"file": (chunk_path.name, handle, "audio/wav")},
                        )
                    billed_seconds += len(chunk) / 1000
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise BodhanAPIError(
                            f"Transcription chunk {index} returned invalid JSON."
                        ) from exc
                    if not isinstance(body, dict):
                        raise BodhanAPIError(
                            f"Transcription chunk {index} returned an invalid response."
                        )
                    transcript = body.get("text")
                    if not isinstance(transcript, str) or not transcript.strip():
                        raise BodhanAPIError(f"Transcription chunk {index} returned no text.")
                    transcripts.append(transcript.strip())
                    segments.append({
                        "start_seconds": round(start_ms / 1000, 3),
                        "end_seconds": round(min(start_ms + len(chunk), len(audio)) / 1000, 3),
                        "text": transcript.strip(),
                    })
        except BodhanAPIError as exc:
            exc.cost_inr += transcription_cost(billed_seconds)
            raise

        elapsed = time.perf_counter() - started
        return {
            "text": "\n".join(transcripts),
            "segments": segments,
            "chunks": len(transcripts),
            "duration_seconds": round(total_seconds, 3),
            "latency_seconds": round(elapsed, 3),
            "cost_inr": round(transcription_cost(total_seconds), 6),
        }

    def translate_text(
        self,
        text: str,
        target_language: str,
        script: str = "native",
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Translate text in conservative chunks; fail explicitly if a chunk is truncated."""
        if not text.strip():
            raise ValueError("Translation input cannot be empty.")
        if target_language not in SUPPORTED_TRANSLATION_LANGUAGES:
            raise ValueError(f"Unsupported translation language: {target_language}")
        if script not in {"native", "roman", "codemix"}:
            raise ValueError("script must be one of: native, roman, codemix")
        if target_language == "en" and script != "native":
            raise ValueError("script modes are not supported when target_language is 'en'.")
        if script == "codemix" and target_language in {"ks-Deva", "sd-Arab"}:
            raise ValueError(f"codemix output is not supported for target_language '{target_language}'.")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive when supplied.")

        chunks = split_text(text, self.settings.translation_chunk_chars)
        translations: list[str] = []
        output_tokens = 0
        started = time.perf_counter()
        try:
            for index, chunk in enumerate(chunks, start=1):
                payload: dict[str, Any] = {
                    "text": chunk,
                    "target_language": target_language,
                }
                if target_language != "en":
                    payload["script"] = script
                token_limit = max_output_tokens
                if token_limit is None:
                    # Keep a request-level ceiling so long/misbehaving outputs cannot
                    # consume the entire task budget in a single translation call.
                    token_limit = min(8192, max(128, math.ceil(len(chunk) * 2.5)))
                payload["max_output_tokens"] = token_limit
                response = self._request(
                    "POST",
                    f"{self.settings.api_base_url}/translate",
                    model_key="translate",
                    json=payload,
                )
                try:
                    body = response.json()
                except ValueError as exc:
                    raise BodhanAPIError(
                        f"Translation chunk {index} returned invalid JSON."
                    ) from exc
                if not isinstance(body, dict):
                    raise BodhanAPIError(f"Translation chunk {index} returned an invalid response.")
                if "error" in body:
                    error = body["error"]
                    if isinstance(error, dict):
                        error_message = error.get("message", "unknown error")
                        error_code = error.get("code")
                    else:
                        error_message = str(error)[:500]
                        error_code = None
                    raise BodhanAPIError(
                        f"Translation chunk {index} failed: {error_message}",
                        status_code=response.status_code,
                        code=str(error_code) if error_code is not None else None,
                    )
                translated = body.get("translation")
                if not isinstance(translated, str) or not translated.strip():
                    raise BodhanAPIError(f"Translation chunk {index} returned no translation.")

                usage = body.get("usage") or {}
                token_count = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if not isinstance(token_count, int) or token_count < 0:
                    # The documented API always returns usage. Keep a conservative
                    # fallback for older gateways instead of silently reporting ₹0.
                    token_count = max(1, math.ceil(len(translated) / 3))
                output_tokens += token_count
                if body.get("truncated") is True:
                    raise BodhanAPIError(
                        f"Translation chunk {index}/{len(chunks)} was truncated; refusing to mark it complete."
                    )
                translations.append(translated)
        except BodhanAPIError as exc:
            exc.cost_inr += translation_cost(output_tokens)
            raise

        elapsed = time.perf_counter() - started
        return {
            "translation": "\n\n".join(translations),
            "chunks": len(chunks),
            "output_tokens": output_tokens,
            "latency_seconds": round(elapsed, 3),
            "cost_inr": round(translation_cost(output_tokens), 6),
        }

    def synthesize_speech(
        self,
        text: str,
        language: str,
        output_path: str | Path,
        voice: str | None = None,
    ) -> dict[str, Any]:
        """Synthesize short text chunks and concatenate the returned WAV audio."""
        if not text.strip():
            raise ValueError("Speech input cannot be empty.")
        if language not in SUPPORTED_SPEECH_LANGUAGES:
            raise ValueError(f"Unsupported speech language: {language}")
        chunks = split_text(text, self.settings.tts_chunk_chars)
        chosen_voice = voice or DEFAULT_VOICES.get(language)
        if not chosen_voice:
            raise ValueError(f"No default voice configured for language code '{language}'.")

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        combined = AudioSegment.empty()
        started = time.perf_counter()
        sent_characters = 0
        try:
            for index, chunk in enumerate(chunks, start=1):
                response = self._request(
                    "POST",
                    f"{self.settings.api_base_url}/v1/audio/speech",
                    model_key="speak",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "indic-speak",
                        "input": chunk,
                        "voice": chosen_voice,
                        "instructions": json.dumps({"lang": language}, ensure_ascii=False),
                    },
                )
                sent_characters += len(chunk)
                temp_path: str | None = None
                try:
                    # Close the file before FFmpeg opens it. This is required on Windows.
                    with tempfile.NamedTemporaryFile(
                        prefix=f"bodhan_tts_{index}_", suffix=".wav", delete=False
                    ) as temp:
                        temp.write(response.content)
                        temp_path = temp.name
                    combined += AudioSegment.from_file(temp_path, format="wav")
                except Exception as exc:
                    raise BodhanAPIError(
                        f"TTS chunk {index} did not contain a readable WAV: {exc}"
                    ) from exc
                finally:
                    if temp_path:
                        Path(temp_path).unlink(missing_ok=True)
        except BodhanAPIError as exc:
            exc.cost_inr += speech_cost(sent_characters)
            raise

        try:
            combined.export(output, format="wav")
        except Exception as exc:
            raise BodhanAPIError(
                f"Could not write synthesized audio to {output.name}: {exc}",
                cost_inr=speech_cost(sent_characters),
            ) from exc
        elapsed = time.perf_counter() - started
        return {
            "artifact_path": str(output),
            "chunks": len(chunks),
            "input_characters": len(text),
            "duration_seconds": round(len(combined) / 1000, 3),
            "latency_seconds": round(elapsed, 3),
            "cost_inr": round(speech_cost(len(text)), 6),
            "voice": chosen_voice,
        }

    def ocr_document(
        self,
        input_path: str | Path,
        output_path: str | Path,
        table_format: str = "html",
    ) -> dict[str, Any]:
        """OCR an image or every page of a PDF using Indic-OCR; no local OCR fallback."""
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise InputFileError(f"Document not found: {source}")
        if table_format not in {"markdown", "html"}:
            raise ValueError("table_format must be 'markdown' or 'html'.")

        pages: list[tuple[str, bytes]] = []
        if source.suffix.lower() == ".pdf":
            try:
                with fitz.open(source) as document:
                    for page_number, page in enumerate(document, start=1):
                        pixmap = page.get_pixmap(dpi=160, alpha=False)
                        pages.append((f"page_{page_number:04d}.png", pixmap.tobytes("png")))
            except Exception as exc:
                raise InputFileError(f"Could not render PDF {source.name}: {exc}") from exc
        elif source.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            try:
                from PIL import Image

                with Image.open(source) as image:
                    image.verify()
                pages.append((source.name, source.read_bytes()))
            except Exception as exc:
                raise InputFileError(f"Unsupported or unreadable image {source.name}: {exc}") from exc
        else:
            raise InputFileError("OCR accepts PDF, PNG or JPEG files.")

        if not pages:
            raise InputFileError("The PDF contains no pages.")

        page_texts: list[str] = []
        page_summaries: list[dict[str, Any]] = []
        total_output_tokens = 0
        started = time.perf_counter()
        billed_pages = 0
        try:
            for page_number, (page_name, image_bytes) in enumerate(pages, start=1):
                mime = "image/jpeg" if page_name.lower().endswith((".jpg", ".jpeg")) else "image/png"
                data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
                response = self._request(
                    "POST",
                    f"{self.settings.api_base_url}/v1/chat/completions",
                    model_key="ocr",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "indic-ocr",
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]}],
                        "table_format": table_format,
                        "max_tokens": 4096,
                    },
                )
                billed_pages += 1
                try:
                    body = response.json()
                except ValueError as exc:
                    raise BodhanAPIError(f"OCR page {page_number} returned invalid JSON.") from exc
                if not isinstance(body, dict):
                    raise BodhanAPIError(
                        f"OCR page {page_number} returned an invalid response shape."
                    )
                try:
                    choice = body["choices"][0]
                    if not isinstance(choice, dict):
                        raise TypeError("choice must be an object")
                    content = choice["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise BodhanAPIError(
                        f"OCR page {page_number} returned an invalid response shape."
                    ) from exc
                if choice.get("finish_reason") == "length":
                    raise BodhanAPIError(
                        f"OCR page {page_number} hit its output limit; increase max_tokens or review the page."
                    )
                if not isinstance(content, str) or not content.strip():
                    raise BodhanAPIError(f"OCR page {page_number} returned no text.")
                page_texts.append(f"<!-- {page_name} -->\n\n{content.strip()}")
                usage = body.get("usage")
                if isinstance(usage, dict):
                    try:
                        total_output_tokens += max(0, int(usage.get("completion_tokens", 0)))
                    except (TypeError, ValueError):
                        pass
                blocks = body.get("blocks")
                page_summaries.append({
                    "page": page_number,
                    "name": page_name,
                    "block_count": len(blocks) if isinstance(blocks, list) else 0,
                })
        except BodhanAPIError as exc:
            exc.cost_inr += ocr_cost(billed_pages)
            raise

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.write_text("\n\n---\n\n".join(page_texts) + "\n", encoding="utf-8")
        except OSError as exc:
            raise BodhanAPIError(
                f"OCR finished but could not write {output.name}: {exc}",
                cost_inr=ocr_cost(billed_pages),
            ) from exc
        elapsed = time.perf_counter() - started
        return {
            "artifact_path": str(output),
            "pages": len(pages),
            "output_tokens": total_output_tokens,
            "latency_seconds": round(elapsed, 3),
            "cost_inr": round(ocr_cost(len(pages)), 6),
            "page_details": page_summaries,
        }
