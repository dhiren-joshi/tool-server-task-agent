"""Small path helpers used to keep MCP file access inside its workspace."""

from __future__ import annotations

from pathlib import Path

from bodhan_core.errors import InputFileError


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace_path(
    raw_path: str,
    workspace_root: str | Path,
    *,
    must_exist: bool = False,
    kind: str = "file",
) -> Path:
    """Resolve a path and reject paths that escape the configured MCP workspace."""
    if not raw_path or not str(raw_path).strip():
        raise InputFileError(f"{kind.capitalize()} path cannot be empty.")
    root = Path(workspace_root).expanduser().resolve()
    candidate = Path(raw_path).expanduser().resolve(strict=must_exist)
    if not _is_within(candidate, root):
        raise InputFileError(f"{kind.capitalize()} path must be inside the MCP workspace: {root}")
    if must_exist:
        if kind == "directory" and not candidate.is_dir():
            raise InputFileError(f"Directory does not exist: {candidate}")
        if kind == "file" and not candidate.is_file():
            raise InputFileError(f"File does not exist: {candidate}")
    return candidate
