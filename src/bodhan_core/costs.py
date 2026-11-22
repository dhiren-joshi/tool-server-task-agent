"""Transparent cost estimates using the prices stated in the take-home brief."""

from __future__ import annotations

TRANSCRIBE_INR_PER_MINUTE = 0.10
TRANSLATE_INR_PER_10K_OUTPUT_TOKENS = 0.20
SPEAK_INR_PER_10K_INPUT_CHARS = 6.00
OCR_INR_PER_PAGE = 0.20


def transcription_cost(duration_seconds: float) -> float:
    return round(max(0.0, duration_seconds) / 60 * TRANSCRIBE_INR_PER_MINUTE, 6)


def translation_cost(output_tokens: int) -> float:
    return round(max(0, output_tokens) / 10_000 * TRANSLATE_INR_PER_10K_OUTPUT_TOKENS, 6)


def speech_cost(input_characters: int) -> float:
    return round(max(0, input_characters) / 10_000 * SPEAK_INR_PER_10K_INPUT_CHARS, 6)


def ocr_cost(page_count: int) -> float:
    return round(max(0, page_count) * OCR_INR_PER_PAGE, 6)
