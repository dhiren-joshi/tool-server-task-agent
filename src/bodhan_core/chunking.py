"""Conservative chunkers for Bodhan model request limits."""

from __future__ import annotations

import re


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text on natural boundaries without losing or changing any source character."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        end = limit
        if limit < len(text):
            window = text[start:limit]
            # Prefer paragraph boundaries, then sentence boundaries, then whitespace.
            paragraph = window.rfind("\n\n")
            sentence_matches = list(re.finditer(r"(?<=[.!?।])(?:[ \t]+|\n+)", window))
            whitespace = max((m.start() for m in re.finditer(r"\s", window)), default=-1)
            if paragraph >= max_chars // 3:
                end = start + paragraph + 2
            elif sentence_matches and sentence_matches[-1].end() >= max_chars // 3:
                end = start + sentence_matches[-1].end()
            elif whitespace >= max_chars // 3:
                end = start + whitespace + 1
        if end <= start:
            end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks
