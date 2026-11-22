"""MCP stdio server. Tools accept filesystem references, never base64 media payloads."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

from bodhan_core.config import Settings
from bodhan_core.errors import InputFileError
from bodhan_core.paths import resolve_workspace_path
from bodhan_core.schemas import (
    FileInfo,
    FileInventory,
    TextReadResult,
    ToolErrorInfo,
    ToolExecutionResult,
)
from bodhan_core.tools import (
    inspect_input_files,
    ocr_document_tool,
    read_text_file_tool,
    synthesize_file_tool,
    transcribe_audio_tool,
    translate_file_tool,
    translate_segments_to_srt_tool,
    write_english_text_tool,
)

server = MCPServer("bodhan-tool-server")


def _workspace() -> str:
    return Settings.from_env().mcp_allowed_root


def _input_path(path: str) -> str:
    return str(resolve_workspace_path(path, _workspace(), must_exist=True))


def _output_path(path: str) -> str:
    root = Path(_workspace()).expanduser().resolve()
    candidate = resolve_workspace_path(path, root)
    relative_parts = {part.lower() for part in candidate.relative_to(root).parts}
    if relative_parts.intersection({".git", ".venv", "__pycache__"}):
        raise InputFileError("MCP tools cannot write inside Git, virtual-environment or cache directories.")
    if candidate.name.lower().startswith(".env"):
        raise InputFileError("MCP tools cannot write environment or secret files.")

    # Existing project/workspace files are protected from accidental overwrite. Replacing
    # an existing artifact is allowed only under the dedicated outputs/ directory.
    outputs_root = (root / "outputs").resolve()
    try:
        candidate.relative_to(outputs_root)
        inside_outputs = True
    except ValueError:
        inside_outputs = False
    if candidate.exists() and not inside_outputs:
        raise InputFileError(
            "Refusing to overwrite an existing workspace file. Write new artifacts under outputs/."
        )
    return str(candidate)


def _execute(tool_name: str, model: str | None, operation: Callable[[], dict[str, Any]]) -> ToolExecutionResult:
    """Return a structured result for success and failure; do not crash the MCP client."""
    started_at = time.perf_counter()
    try:
        data = operation()
        return ToolExecutionResult(
            status="completed",
            tool=tool_name,
            model=model,
            artifact_path=data.get("artifact_path"),
            cost_inr=float(data.get("cost_inr", 0.0)),
            latency_seconds=float(data.get("latency_seconds", time.perf_counter() - started_at)),
            details={k: v for k, v in data.items() if k not in {
                "status", "model", "artifact_path", "cost_inr", "latency_seconds"
            }},
        )
    except Exception as exc:  # structured errors are part of the public MCP tool contract
        status_code = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        error_code = str(getattr(exc, "code", "") or "").lower()
        error_message = str(exc).lower()
        exhausted_credit = any(
            marker in f"{error_code} {error_message}"
            for marker in ("credit", "balance", "insufficient_funds", "budget")
        )
        retryable = not exhausted_credit and (
            status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
        )
        return ToolExecutionResult(
            status="failed",
            tool=tool_name,
            model=model,
            cost_inr=max(0.0, float(getattr(exc, "cost_inr", 0.0) or 0.0)),
            latency_seconds=round(time.perf_counter() - started_at, 3),
            error=ToolErrorInfo(
                error_type=type(exc).__name__,
                message=str(exc)[:1000],
                retryable=retryable,
                status_code=status_code,
                code=code,
            ),
        )


@server.tool()
def inspect_files(directory: str) -> FileInventory:
    """List supported audio, video, PDF, image and text files under a directory.

    Returns paths and metadata only. It does not decode media or reveal file contents.
    """
    try:
        safe_directory = resolve_workspace_path(
            directory, _workspace(), must_exist=True, kind="directory"
        )
        data = inspect_input_files(str(safe_directory))
        return FileInventory(
            directory=data["directory"],
            files=[FileInfo(path=item["path"], name=item["name"], suffix=item["extension"], size_bytes=item["size_bytes"]) for item in data["files"]],
            file_count=data["file_count"],
        )
    except Exception as exc:
        return FileInventory(
            status="failed", directory=directory,
            error=ToolErrorInfo(error_type=type(exc).__name__, message=str(exc)[:1000]),
        )


@server.tool()
def transcribe_audio(input_path: str, output_path: str, language: str | None = None) -> ToolExecutionResult:
    """Transcribe an audio file with Bodhan Indic-Transcribe.

    Audio is decoded and split into windows shorter than 30 seconds automatically.
    Pass a local file path; never pass audio bytes or base64 in the tool arguments.
    Optional language is an ISO code such as hi, ta, bn or te. Omit to auto-detect.
    """
    return _execute(
        "transcribe_audio", "indic-transcribe",
        lambda: transcribe_audio_tool(_input_path(input_path), _output_path(output_path), language),
    )


@server.tool()
def translate_text_file(
    input_path: str,
    output_path: str,
    target_language: str,
    script: str = "native",
    source_language: str | None = None,
) -> ToolExecutionResult:
    """Translate a UTF-8 text/Markdown file with Bodhan Indic-Translate.

    Supports native, roman or codemix script modes. Long files are chunked automatically.
    The tool rejects truncated chunks and back-translates to the supplied or script-inferred
    source language as a consistency check. source_language is used for verification only; the
    Bodhan translation endpoint infers the source language itself.
    """
    return _execute(
        "translate_text_file", "indic-translate",
        lambda: translate_file_tool(
            _input_path(input_path), _output_path(output_path), target_language, script,
            source_language=source_language,
        ),
    )


@server.tool()
def translate_transcript_segments_to_srt(
    segments_path: str,
    output_path: str,
    target_language: str = "en",
) -> ToolExecutionResult:
    """Translate Bodhan transcription segments and assemble a timed SRT subtitle file.

    Timing is inherited from the audio chunks; this is segment-level timing, not word-level
    forced alignment. Every subtitle segment is translated by Bodhan Indic-Translate.
    """
    return _execute(
        "translate_transcript_segments_to_srt", "indic-translate",
        lambda: translate_segments_to_srt_tool(
            _input_path(segments_path), _output_path(output_path), target_language
        ),
    )


@server.tool()
def synthesize_speech_from_file(
    input_path: str,
    output_path: str,
    language: str,
    voice: str | None = None,
) -> ToolExecutionResult:
    """Generate WAV or MP3 speech from a UTF-8 text file using Bodhan Indic-Speak.

    Use an .mp3 output_path when MP3 is requested; the Bodhan WAV response is converted locally.
    The text is chunked into short requests and concatenated. Omit voice to use the
    configured female default for that language (Bengali uses Ishita). Generated audio is
    back-transcribed with Bodhan Indic-Transcribe and labelled AI-generated. Use the text-file
    reference rather than passing a long Indic string to the LLM.
    """
    return _execute(
        "synthesize_speech_from_file", "indic-speak",
        lambda: synthesize_file_tool(
            _input_path(input_path), _output_path(output_path), language, voice
        ),
    )


@server.tool()
def ocr_document(
    input_path: str,
    output_path: str,
    table_format: str = "html",
) -> ToolExecutionResult:
    """Extract reading-ordered text/layout from an image or multi-page PDF with Bodhan Indic-OCR.

    PDF pages are rendered to temporary PNG images and sent one page at a time. OCR supports
    markdown or html table output. No local OCR engine or non-Bodhan fallback is used.
    """
    return _execute(
        "ocr_document", "indic-ocr",
        lambda: ocr_document_tool(
            _input_path(input_path), _output_path(output_path), table_format
        ),
    )


@server.tool()
def read_text_file(input_path: str, max_chars: int = 12000, start_char: int = 0) -> TextReadResult:
    """Read a bounded portion of an existing UTF-8 .txt/.md artifact into the agent context.

    Use this for text already extracted or translated by Bodhan tools, not raw media files.
    Set start_char to continue reading a large artifact in a later call.
    """
    try:
        return TextReadResult(**read_text_file_tool(_input_path(input_path), max_chars, start_char))
    except Exception as exc:
        return TextReadResult(
            status="failed", path=input_path,
            error=ToolErrorInfo(error_type=type(exc).__name__, message=str(exc)[:1000]),
        )


@server.tool()
def write_english_text(output_path: str, text: str) -> ToolExecutionResult:
    """Write English/Latin-script explanation or summary to a text artifact.

    This tool rejects non-Latin alphabetic scripts. Use translate_text_file for Indic output.
    """
    return _execute(
        "write_english_text", None,
        lambda: write_english_text_tool(_output_path(output_path), text),
    )


def main() -> None:
    """Run the independent MCP server on stdio."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
