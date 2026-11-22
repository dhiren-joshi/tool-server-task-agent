from bodhan_core.chunking import split_text


def test_split_text_keeps_all_characters():
    source = ("A sentence. Another sentence.\n\n" * 20).strip()
    chunks = split_text(source, 30)
    assert len(chunks) > 1
    assert "".join(chunks) == source
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_split_text_handles_single_very_long_token():
    source = "x" * 101
    chunks = split_text(source, 20)
    assert "".join(chunks) == source
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_split_text_empty_input():
    assert split_text("", 10) == []


def test_split_text_preserves_multiple_spaces_and_tabs():
    source = ("word   word\tword.  next paragraph\n\n" * 8).strip()
    chunks = split_text(source, 17)
    assert "".join(chunks) == source
    assert all(len(chunk) <= 17 for chunk in chunks)
