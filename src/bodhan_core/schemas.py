"""Stable, typed shapes shared by the agent and MCP server."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArtifactResult(BaseModel):
    status: str = "completed"
    model: str
    artifact_path: str
    cost_inr: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    details: dict[str, object] = Field(default_factory=dict)


class FileInfo(BaseModel):
    path: str
    name: str
    suffix: str
    size_bytes: int = Field(ge=0)


class ToolErrorInfo(BaseModel):
    error_type: str
    message: str
    retryable: bool = False
    status_code: int | None = None
    code: str | None = None


class ToolExecutionResult(BaseModel):
    status: Literal["completed", "failed"]
    tool: str
    model: str | None = None
    artifact_path: str | None = None
    cost_inr: float = Field(default=0.0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0)
    details: dict[str, object] = Field(default_factory=dict)
    error: ToolErrorInfo | None = None


class FileInventory(BaseModel):
    status: Literal["completed", "failed"] = "completed"
    directory: str
    files: list[FileInfo] = Field(default_factory=list)
    file_count: int = Field(default=0, ge=0)
    error: ToolErrorInfo | None = None


class TextReadResult(BaseModel):
    status: Literal["completed", "failed"] = "completed"
    path: str | None = None
    text: str = ""
    start_char: int = Field(default=0, ge=0)
    characters_returned: int = Field(default=0, ge=0)
    total_characters: int = Field(default=0, ge=0)
    truncated: bool = False
    error: ToolErrorInfo | None = None
