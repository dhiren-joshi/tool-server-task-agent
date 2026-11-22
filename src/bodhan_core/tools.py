"""File-reference based tool functions shared by MCP and the task agent."""

from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from bodhan_core.api_client import BodhanClient
from bodhan_core.config import Settings
from bodhan_core.errors import InputFileError


def inspect_input_files(directory: str) -> dict[str, Any]:
    """List available input files as metadata; file contents are not sent to the LLM."""
    root = Path(directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise InputFileError(f"Input directory not found: {root}")
    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in supported:
            files.append({
                "path": str(path),
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            })
    return {"directory": str(root), "files": files, "file_count": len(files)}


def transcribe_audio_tool(
    input_path: str,
    output_path: str,
    language: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    client = BodhanClient(settings)
    result: dict[str, Any] | None = None
    try:
        result = client.transcribe_audio(input_path, language=language)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result["text"] + "\n", encoding="utf-8")
        segments_path = Path(str(target) + ".segments.json")
        segments_path.write_text(
            json.dumps(result["segments"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {k: v for k, v in result.items() if k not in {"text", "segments"}}
        return {
            "status": "completed", "model": "indic-transcribe", "artifact_path": str(target),
            "segments_path": str(segments_path), "segment_count": len(result["segments"]), **summary,
        }
    except Exception as exc:
        if result is not None:
            _add_partial_cost(exc, float(result.get("cost_inr", 0.0)))
        raise
    finally:
        client.close()


def _detect_source_language(text: str) -> str | None:
    """Infer a likely source-language code from Unicode script; caller may override it."""
    # Devanagari is shared by Hindi, Marathi, Nepali, Sanskrit, Maithili and others;
    # guessing Hindi from script alone would make back-translation misleading.
    script_ranges = [
        (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"),
        (0x0A80, 0x0AFF, "gu"), (0x0B00, 0x0B7F, "or"),
        (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"),
        (0x0C80, 0x0CFF, "kn"), (0x0D00, 0x0D7F, "ml"),
        (0x0600, 0x06FF, "ur"),
    ]
    counts = {code: 0 for _, _, code in script_ranges}
    letters = 0
    latin = 0
    for char in text:
        if char.isalpha():
            letters += 1
            point = ord(char)
            if point <= 0x024F:
                latin += 1
            for lower, upper, code in script_ranges:
                if lower <= point <= upper:
                    counts[code] += 1
    # Script alone cannot reliably distinguish English from romanized Indian text,
    # or Bengali/Assamese and the languages written in Arabic-derived scripts.
    candidates = {code: count for code, count in counts.items() if code not in {"bn", "ur"}}
    best = max(candidates, key=candidates.get) if candidates else None
    if not best or not candidates[best] or candidates[best] / max(letters, 1) < 0.75:
        return None
    return best


def _add_partial_cost(exc: Exception, amount: float) -> None:
    """Keep charges from successful API calls when a later tool step fails."""
    try:
        current = float(getattr(exc, "cost_inr", 0.0) or 0.0)
        exc.cost_inr = current + max(0.0, amount)
    except (AttributeError, TypeError):
        pass


def translate_file_tool(
    input_path: str,
    output_path: str,
    target_language: str,
    script: str = "native",
    source_language: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise InputFileError(f"Text file not found: {source}")
    source_text = source.read_text(encoding="utf-8")
    client = BodhanClient(settings)
    started_at = time.perf_counter()
    primary_result: dict[str, Any] | None = None
    total_cost = 0.0
    try:
        result = client.translate_text(source_text, target_language, script)
        primary_result = result
        total_cost = float(result["cost_inr"])
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result["translation"] + "\n", encoding="utf-8")

        # Back-translate with Bodhan to catch empty/truncated or severely divergent output.
        detected_source = source_language or _detect_source_language(source_text)
        verification: dict[str, Any] = {
            "method": "Bodhan back-translation",
            "source_language_used": detected_source,
            "primary_chunks_complete": result["chunks"] > 0,
            "primary_output_nonempty": bool(result["translation"].strip()),
        }
        if detected_source and detected_source != target_language:
            back = client.translate_text(result["translation"], detected_source)
            total_cost += back["cost_inr"]
            back_path = Path(str(target) + ".backtranslation.txt")
            back_path.write_text(back["translation"] + "\n", encoding="utf-8")
            expected = re.sub(r"[^\w]", "", source_text.casefold(), flags=re.UNICODE)
            observed = re.sub(r"[^\w]", "", back["translation"].casefold(), flags=re.UNICODE)
            similarity = SequenceMatcher(None, expected, observed).ratio() if expected else 0.0
            verification.update({
                "status": "passed" if similarity >= 0.35 else "needs_review",
                "backtranslation_path": str(back_path),
                "backtranslation_nonempty": bool(back["translation"].strip()),
                "character_similarity_ratio": round(similarity, 4),
                "similarity_is_diagnostic_not_a_formal_quality_score": True,
                "backtranslation_chunks": back["chunks"],
            })
        elif detected_source == target_language:
            verification.update({"status": "passed", "note": "Back-translation skipped because source and target language match."})
        else:
            verification.update({
                "status": "primary_only",
                "note": "Back-translation skipped because the source language cannot be identified reliably from its script. Pass source_language when known.",
            })
        return {
            "status": "completed", "model": "indic-translate", "artifact_path": str(target),
            "chunks": result["chunks"], "output_tokens": result["output_tokens"],
            "latency_seconds": round(time.perf_counter() - started_at, 3),
            "cost_inr": round(total_cost, 6), "verification": verification,
        }
    except Exception as exc:
        if primary_result is not None:
            _add_partial_cost(exc, total_cost)
        raise
    finally:
        client.close()


def translate_segments_to_srt_tool(
    segments_path: str,
    output_path: str,
    target_language: str = "en",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Translate transcript segments with Bodhan and create a segment-timed SRT file."""
    source = Path(segments_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".json":
        raise InputFileError("segments_path must point to an existing transcript .segments.json file.")
    try:
        segments = json.loads(source.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise InputFileError(f"Could not read transcript segment file: {exc}") from exc
    if not isinstance(segments, list) or not segments:
        raise InputFileError("Transcript segment file contains no segments.")

    client = BodhanClient(settings)
    cues: list[str] = []
    cue_count = 0
    total_cost = 0.0
    translation_cache: dict[str, dict[str, Any]] = {}
    started_at = time.perf_counter()
    try:
        for segment in segments:
            if not isinstance(segment, dict):
                raise InputFileError("Transcript segment entries must be JSON objects.")
            source_text = str(segment.get("text", "")).strip()
            if not source_text:
                continue
            raw_start = float(segment["start_seconds"])
            raw_end = float(segment["end_seconds"])
            if raw_start < 0 or raw_end <= raw_start:
                raise InputFileError("Transcript segment has invalid start/end timestamps.")

            # Bodhan returns chunk-level timestamps, not word-level alignment. Split a
            # chunk into sentences where punctuation permits and interpolate approximate
            # cue durations by sentence length, keeping the source chunk's total duration.
            sentences = [part.strip() for part in re.split(r"(?<=[.!?।؟])\s+", source_text) if part.strip()]
            if not sentences:
                sentences = [source_text]
            weights = [max(1, len(sentence)) for sentence in sentences]
            weight_total = sum(weights)
            cursor = raw_start
            for index, sentence in enumerate(sentences):
                cache_key = f"{target_language}\0{sentence}"
                translated = translation_cache.get(cache_key)
                if translated is None:
                    translated = client.translate_text(sentence, target_language)
                    translation_cache[cache_key] = translated
                    total_cost += translated["cost_inr"]
                cue_count += 1
                cue_end = raw_end if index == len(sentences) - 1 else cursor + (raw_end - raw_start) * weights[index] / weight_total
                start = _format_srt_time(cursor)
                end = _format_srt_time(cue_end)
                cues.extend([str(cue_count), f"{start} --> {end}", translated["translation"], ""])
                cursor = cue_end

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(cues).rstrip() + "\n", encoding="utf-8")
        elapsed = time.perf_counter() - started_at
        return {
            "status": "completed", "model": "indic-translate", "artifact_path": str(output),
            "cue_count": cue_count, "cost_inr": round(total_cost, 6),
            "latency_seconds": round(elapsed, 3),
            "verification": {
                "status": "passed" if cue_count > 0 else "failed",
                "method": "Bodhan Indic-Translate applied to each sentence cue",
                "translated_cues": cue_count,
                "timing_note": "Approximate sentence-level timestamps are interpolated by character count inside source ASR chunk boundaries; this is not word-level forced alignment.",
            },
        }
    except Exception as exc:
        _add_partial_cost(exc, total_cost)
        raise
    finally:
        client.close()


def _format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{millis:03}"


def synthesize_file_tool(
    input_path: str,
    output_path: str,
    language: str,
    voice: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise InputFileError(f"Text file not found: {source}")
    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() not in {".wav", ".mp3"}:
        raise InputFileError("Speech output must use a .wav or .mp3 extension.")

    client = BodhanClient(settings)
    total_cost = 0.0
    temporary_wav: Path | None = None
    try:
        source_text = source.read_text(encoding="utf-8")
        if target.suffix.lower() == ".mp3":
            temporary_wav = target.with_name(f".{target.stem}.bodhan-intermediate.wav")
            synthesis_path = temporary_wav
        else:
            synthesis_path = target

        result = client.synthesize_speech(source_text, language, synthesis_path, voice=voice)
        total_cost += float(result.get("cost_inr", 0.0))
        if temporary_wav is not None:
            try:
                AudioSegment.from_file(temporary_wav, format="wav").export(
                    target, format="mp3", bitrate="192k"
                )
            except Exception as exc:
                raise InputFileError(
                    f"Could not convert the Bodhan WAV response to MP3: {exc}"
                ) from exc
            finally:
                temporary_wav.unlink(missing_ok=True)
            result["artifact_path"] = str(target)
        # The assignment asks that every generated audio file be checked. Re-transcribe the
        # generated WAV with Bodhan Indic-Transcribe; do not use a third-party ASR fallback.
        verification = client.transcribe_audio(result["artifact_path"], language=language)
        total_cost += float(verification.get("cost_inr", 0.0))
        transcript_path = Path(str(result["artifact_path"]) + ".verification.txt")
        transcript_path.write_text(verification["text"] + "\n", encoding="utf-8")
        label_path = Path(str(result["artifact_path"]) + ".ai-generated.txt")
        label_path.write_text(
            "AI-generated speech created with Bodhan Indic-Speak.\n", encoding="utf-8"
        )

        def normalize(value: str) -> str:
            return re.sub(r"[^\w]", "", value.casefold(), flags=re.UNICODE)

        expected = normalize(source_text)
        observed = normalize(verification["text"])
        similarity = SequenceMatcher(None, expected, observed).ratio() if expected else 0.0
        verification_summary = {
            "method": "Bodhan Indic-Transcribe back-transcription",
            "status": "passed" if similarity >= 0.35 else "needs_review",
            "similarity_ratio": round(similarity, 4),
            "transcript_path": str(transcript_path),
            "label_path": str(label_path),
            "transcription_cost_inr": verification["cost_inr"],
        }
        return {
            "status": "completed",
            "model": "indic-speak",
            **result,
            "cost_inr": round(total_cost, 6),
            "verification": verification_summary,
        }
    except Exception as exc:
        _add_partial_cost(exc, total_cost)
        raise
    finally:
        if temporary_wav is not None:
            temporary_wav.unlink(missing_ok=True)
        client.close()


def ocr_document_tool(
    input_path: str,
    output_path: str,
    table_format: str = "html",
    settings: Settings | None = None,
) -> dict[str, Any]:
    client = BodhanClient(settings)
    try:
        result = client.ocr_document(input_path, output_path, table_format=table_format)
        return {"status": "completed", "model": "indic-ocr", **result}
    finally:
        client.close()


def read_text_file_tool(input_path: str, max_chars: int = 12000, start_char: int = 0) -> dict[str, Any]:
    """Read text artifacts generated by Bodhan tools, not raw audio or image files."""
    source = Path(input_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {".txt", ".md"}:
        raise InputFileError("read_text_file accepts existing .txt or .md artifacts only.")
    if max_chars < 1 or max_chars > 20000:
        raise ValueError("max_chars must be between 1 and 20000.")
    if start_char < 0:
        raise ValueError("start_char cannot be negative.")
    text = source.read_text(encoding="utf-8")
    excerpt = text[start_char : start_char + max_chars]
    return {
        "path": str(source),
        "text": excerpt,
        "start_char": start_char,
        "characters_returned": len(excerpt),
        "total_characters": len(text),
        "truncated": start_char + len(excerpt) < len(text),
    }


def write_english_text_tool(output_path: str, text: str) -> dict[str, Any]:
    """Write English-only agent-authored prose; non-Latin scripts are rejected."""
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    # This intentionally conservative guard prevents the LLM from fabricating Indic script text.
    if any(ord(char) > 0x024F and char.isalpha() for char in text):
        raise ValueError("This tool accepts English/Latin-script prose only. Use Bodhan Translate for Indic text.")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.strip() + "\n", encoding="utf-8")
    return {"status": "completed", "artifact_path": str(target), "characters": len(text)}


def serialize_tool_error(exc: Exception) -> str:
    """Compact, non-secret error summary safe to put in an execution trace."""
    return json.dumps({"error": type(exc).__name__, "message": str(exc)[:1000]}, ensure_ascii=False)
