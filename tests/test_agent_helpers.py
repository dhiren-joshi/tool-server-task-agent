import json
from pathlib import Path

from bodhan_agent.runner import AgentRunner
from bodhan_core.config import Settings
from bodhan_core.tools import _detect_source_language


def test_single_file_input_does_not_allow_sibling_files(tmp_path):
    selected = tmp_path / "selected.txt"
    sibling = tmp_path / "sibling.txt"
    selected.write_text("selected", encoding="utf-8")
    sibling.write_text("sibling", encoding="utf-8")
    runner = AgentRunner(
        task="summarize", files_path=str(selected), output_path=str(tmp_path / "out"),
        settings=Settings(),
    )

    assert runner._resolve_input(str(selected)) == selected.resolve()
    try:
        runner._resolve_input(str(sibling))
    except Exception as exc:
        assert "inside --files" in str(exc)
    else:
        raise AssertionError("Sibling file escaped the single-file input scope")


def test_source_language_detection_does_not_guess_shared_devanagari():
    assert _detect_source_language("नमस्कार, कसे आहात?") is None
    assert _detect_source_language("বাংলা ভাষা") is None
    assert _detect_source_language("This is an English sentence") is None
    assert _detect_source_language("தமிழ் மொழி") == "ta"


def test_missing_llm_settings_still_write_report_and_result(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("A test input", encoding="utf-8")
    output = tmp_path / "run"
    runner = AgentRunner(
        task="summarize this", files_path=str(source), output_path=str(output),
        settings=Settings(),
    )
    result = runner.run()

    assert result["status"] == "failed"
    assert (output / "result.md").is_file()
    assert (output / "report.json").is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "LLM is not configured" in report["stop_reason"]


def test_output_extension_is_checked(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("content", encoding="utf-8")
    runner = AgentRunner(
        task="summarize", files_path=str(source), output_path=str(tmp_path / "out"),
        settings=Settings(),
    )

    try:
        runner._resolve_output("summary.wav", ".md")
    except Exception as exc:
        assert ".md" in str(exc)
    else:
        raise AssertionError("A text tool accepted an audio extension")


def test_cache_hit_copies_result_to_requested_filename(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("content", encoding="utf-8")
    runner = AgentRunner(
        task="summarize", files_path=str(source), output_path=str(tmp_path / "out"),
        settings=Settings(),
    )
    first = runner._execute_tool("write_english_text", {
        "output_name": "first.md", "text": "A short summary."})
    second = runner._execute_tool("write_english_text", {
        "output_name": "second.md", "text": "A short summary."})

    assert Path(second["artifact_path"]).name == "second.md"
    assert Path(second["artifact_path"]).read_text(encoding="utf-8") == "A short summary.\n"
    assert second["cache_hit"] is True
    assert first["cache_hit"] is False


def test_speech_cache_does_not_mix_wav_and_mp3_formats(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("A short sentence.", encoding="utf-8")
    runner = AgentRunner(
        task="Read this aloud", files_path=str(source), output_path=str(tmp_path / "out"),
        settings=Settings(),
    )

    wav_key = runner._cache_key("synthesize_speech_from_file", {
        "input_path": str(source), "output_name": "speech.wav", "language": "bn",
    })
    mp3_key = runner._cache_key("synthesize_speech_from_file", {
        "input_path": str(source), "output_name": "speech.mp3", "language": "bn",
    })

    assert wav_key != mp3_key
