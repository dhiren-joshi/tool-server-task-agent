"""Protocol-level MCP smoke test using the official Python SDK client."""

from __future__ import annotations

import asyncio
import os
import sys


def test_mcp_server_lists_and_executes_local_helper_tool(tmp_path):
    """Verify stdio connection and a no-credential helper tool call end to end."""
    import pytest

    pytest.importorskip("mcp")
    from mcp import Client, StdioServerParameters

    async def exercise() -> None:

        env = os.environ.copy()
        env["BODHAN_MCP_ALLOWED_ROOT"] = str(tmp_path)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "bodhan_mcp"],
            env=env,
        )

        async with Client(params) as client:
            result = await client.list_tools()
            discovered = {tool.name for tool in result.tools}
            assert {
                "transcribe_audio",
                "translate_text_file",
                "synthesize_speech_from_file",
                "ocr_document",
                "inspect_files",
                "read_text_file",
                "write_english_text",
                "translate_transcript_segments_to_srt",
            } <= discovered

            output_path = tmp_path / "artifacts" / "mcp-smoke.md"
            write_result = await client.call_tool(
                "write_english_text",
                {
                    "output_path": str(output_path),
                    "text": "The MCP server executed a helper tool successfully.",
                },
            )
            assert not write_result.is_error
            assert output_path.read_text(encoding="utf-8").strip() == (
                "The MCP server executed a helper tool successfully."
            )

            secret_path = tmp_path / ".env"
            secret_path.write_text("BODHAN_TRANSCRIBE_KEY=do-not-overwrite", encoding="utf-8")
            blocked_result = await client.call_tool(
                "write_english_text",
                {"output_path": str(secret_path), "text": "This must not overwrite secrets."},
            )
            assert not blocked_result.is_error
            assert secret_path.read_text(encoding="utf-8") == (
                "BODHAN_TRANSCRIBE_KEY=do-not-overwrite"
            )

    asyncio.run(exercise())
