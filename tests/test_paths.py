import pytest

from bodhan_core.errors import InputFileError
from bodhan_core.paths import resolve_workspace_path


def test_workspace_path_accepts_child_file(tmp_path):
    path = tmp_path / "inputs" / "note.txt"
    path.parent.mkdir()
    path.write_text("hello", encoding="utf-8")

    resolved = resolve_workspace_path(path, tmp_path, must_exist=True)

    assert resolved == path.resolve()


def test_workspace_path_rejects_parent_traversal(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("not in workspace", encoding="utf-8")
    try:
        with pytest.raises(InputFileError, match="inside the MCP workspace"):
            resolve_workspace_path(outside, tmp_path, must_exist=True)
    finally:
        outside.unlink(missing_ok=True)


def test_workspace_path_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-link-target.txt"
    outside.write_text("private", encoding="utf-8")
    link = tmp_path / "shortcut.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        outside.unlink(missing_ok=True)
        pytest.skip("Symlinks are not available on this platform.")

    try:
        with pytest.raises(InputFileError, match="inside the MCP workspace"):
            resolve_workspace_path(link, tmp_path, must_exist=True)
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_empty_workspace_path_is_rejected(tmp_path):
    with pytest.raises(InputFileError, match="cannot be empty"):
        resolve_workspace_path("", tmp_path)
