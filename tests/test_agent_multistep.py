"""Offline multi-step orchestration tests using fake provider responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz
from pydub import AudioSegment

from bodhan_agent import runner as agent_module
from bodhan_agent.runner import AgentRunner
from bodhan_core import tools as core_tools
from bodhan_core.config import Settings


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedLLM:
    def __init__(self, settings: Settings, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        del messages
        response = self.responses[self.calls]
        self.calls += 1
        return response

    def close(self) -> None:
        pass


def _settings() -> Settings:
    return Settings(
        llm_base_url="https://llm.example/v1",
        llm_model="offline-test-model",
        llm_api_key="test-only-not-a-real-key",
    )


def _make_pdf(path: Path, pages: int = 1) -> None:
    document = fitz.open()
    for page_number in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"Test page {page_number + 1}")
    document.save(path)
    document.close()


def test_mocked_multistep_ocr_translate_speak_pipeline(tmp_path, monkeypatch):
    """Exercise several real planner/tool turns without spending API credits."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source_pdf = input_dir / "chapter.pdf"
    _make_pdf(source_pdf, pages=2)

    output_dir = tmp_path / "out"
    artifacts = output_dir / "artifacts"
    ocr_path = artifacts / "chapter_ocr.md"
    translation_path = artifacts / "chapter_tamil.md"
    speech_path = artifacts / "chapter_summary.wav"

    def fake_ocr(input_path, output_path, table_format, settings):
        assert Path(input_path) == source_pdf.resolve()
        assert table_format == "markdown"
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("A short chapter extract.", encoding="utf-8")
        return {
            "status": "completed", "model": "indic-ocr",
            "artifact_path": str(target), "cost_inr": 0.40,
            "latency_seconds": 0.1, "pages": 2,
        }

    def fake_translate(
        input_path, output_path, target_language, script="native",
        source_language=None, settings=None,
    ):
        assert Path(input_path) == ocr_path.resolve()
        assert target_language == "ta"
        assert source_language == "en"
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # This is a provider fixture only; no Indic translation is authored by the test.
        target.write_text("[mock provider translation]", encoding="utf-8")
        return {
            "status": "completed", "model": "indic-translate",
            "artifact_path": str(target), "cost_inr": 0.01,
            "latency_seconds": 0.1,
            "verification": {"status": "passed", "method": "mocked back-translation"},
        }

    def fake_speak(input_path, output_path, language, voice=None, settings=None):
        assert Path(input_path) == translation_path.resolve()
        assert language == "ta"
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"offline fake wav content")
        transcript = Path(str(target) + ".verification.txt")
        label = Path(str(target) + ".ai-generated.txt")
        transcript.write_text("mock verification transcript", encoding="utf-8")
        label.write_text("AI-generated speech created with Bodhan Indic-Speak.\n", encoding="utf-8")
        return {
            "status": "completed", "model": "indic-speak",
            "artifact_path": str(target), "transcript_path": str(transcript),
            "label_path": str(label), "cost_inr": 0.03, "latency_seconds": 0.1,
            "verification": {"status": "passed", "method": "mocked Bodhan back-transcription"},
        }

    responses = [
        {
            "content": "I will extract the chapter first.",
            "tool_calls": [_tool_call("call-1", "ocr_document", {
                "input_path": str(source_pdf.resolve()),
                "output_name": "chapter_ocr.md",
                "table_format": "markdown",
            })],
        },
        {
            "content": "I will translate the extracted text.",
            "tool_calls": [_tool_call("call-2", "translate_text_file", {
                "input_path": str(ocr_path.resolve()),
                "output_name": "chapter_tamil.md",
                "target_language": "ta",
                "source_language": "en",
            })],
        },
        {
            "content": "I will synthesize the translated text.",
            "tool_calls": [_tool_call("call-3", "synthesize_speech_from_file", {
                "input_path": str(translation_path.resolve()),
                "output_name": "chapter_summary.wav",
                "language": "ta",
            })],
        },
        {"content": "The mocked multi-step pipeline completed.", "tool_calls": []},
    ]

    fake_llm = ScriptedLLM(_settings(), responses)
    monkeypatch.setattr(agent_module, "LLMClient", lambda settings: fake_llm)
    monkeypatch.setattr(agent_module, "ocr_document_tool", fake_ocr)
    monkeypatch.setattr(agent_module, "translate_file_tool", fake_translate)
    monkeypatch.setattr(agent_module, "synthesize_file_tool", fake_speak)

    result = AgentRunner(
        task="Extract a chapter, translate it to Tamil and create speech.",
        files_path=str(input_dir),
        output_path=str(output_dir),
        max_steps=3,
        max_cost_inr=2.0,
        settings=_settings(),
    ).run()

    assert result["status"] == "completed"
    assert result["steps"] == 3
    assert fake_llm.calls == 4  # Three tool turns plus the final summary turn.
    assert ocr_path.is_file()
    assert translation_path.is_file()
    assert speech_path.is_file()
    assert Path(str(speech_path) + ".verification.txt").is_file()
    assert Path(str(speech_path) + ".ai-generated.txt").is_file()

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["steps"] == 3
    assert report["total_bodhan_cost_inr"] == 0.44
    assert len(report["verification_results"]) == 2
    assert {Path(path).name for path in report["artifacts"]} >= {
        "chapter_ocr.md", "chapter_tamil.md", "chapter_summary.wav",
        "chapter_summary.wav.verification.txt", "chapter_summary.wav.ai-generated.txt",
    }

    trace_lines = (output_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in trace_lines]
    assert events.count("tool_start") == 3
    assert events.count("tool_end") == 3
    assert events[-1] == "run_finished"


def test_max_steps_limit_allows_partial_status_instead_of_false_completion(tmp_path, monkeypatch):
    input_file = tmp_path / "source.txt"
    input_file.write_text("Test input.", encoding="utf-8")
    responses = [
        {"content": "Writing a first result.", "tool_calls": [_tool_call(
            "call-1", "write_english_text", {"output_name": "first.md", "text": "First result."}
        )]},
        {"content": "I need another tool call.", "tool_calls": [_tool_call(
            "call-2", "write_english_text", {"output_name": "second.md", "text": "Second result."}
        )]},
    ]
    fake_llm = ScriptedLLM(_settings(), responses)
    monkeypatch.setattr(agent_module, "LLMClient", lambda settings: fake_llm)

    result = AgentRunner(
        task="Write two short notes.",
        files_path=str(input_file),
        output_path=str(tmp_path / "out"),
        max_steps=1,
        max_cost_inr=1.0,
        settings=_settings(),
    ).run()

    assert result["status"] == "partial"
    assert result["stop_reason"] == "max_steps_exceeded"
    assert result["steps"] == 1
    assert (tmp_path / "out" / "artifacts" / "first.md").is_file()
    assert not (tmp_path / "out" / "artifacts" / "second.md").exists()


def test_cost_budget_stops_before_ocr_api_operation(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source_pdf = input_dir / "large.pdf"
    _make_pdf(source_pdf, pages=2)
    fake_llm = ScriptedLLM(_settings(), [{
        "content": "I will OCR the PDF.",
        "tool_calls": [_tool_call("call-1", "ocr_document", {
            "input_path": str(source_pdf.resolve()),
            "output_name": "ocr.md",
        })],
    }])
    monkeypatch.setattr(agent_module, "LLMClient", lambda settings: fake_llm)

    def should_not_run(*args, **kwargs):
        raise AssertionError("OCR must not run after the preflight budget guard")

    monkeypatch.setattr(agent_module, "ocr_document_tool", should_not_run)

    result = AgentRunner(
        task="Extract all text from the PDF.",
        files_path=str(input_dir),
        output_path=str(tmp_path / "out"),
        max_steps=5,
        max_cost_inr=0.10,
        settings=_settings(),
    ).run()

    assert result["status"] == "budget_exceeded"
    assert result["stop_reason"] == "max_cost_inr_would_be_exceeded"
    assert result["total_cost_inr"] == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "budget_exceeded"
    assert not list((tmp_path / "out" / "artifacts").iterdir())


def test_speech_tool_exports_mp3_and_back_transcribes_final_file(tmp_path, monkeypatch):
    source = tmp_path / "bengali.txt"
    source.write_text("A short Bengali test phrase.", encoding="utf-8")
    output = tmp_path / "speech.mp3"
    calls: list[Path] = []

    class FakeBodhanClient:
        def __init__(self, settings=None):
            pass

        def synthesize_speech(self, text, language, output_path, voice=None):
            assert language == "bn"
            assert voice is None
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            AudioSegment.silent(duration=350).export(target, format="wav")
            return {"artifact_path": str(target), "cost_inr": 0.01, "duration_seconds": 0.35}

        def transcribe_audio(self, input_path, language=None):
            final_path = Path(input_path)
            calls.append(final_path)
            assert final_path.suffix == ".mp3"
            # Decoding here proves the final MP3 is valid before it is sent to Bodhan ASR.
            assert len(AudioSegment.from_file(final_path)) > 0
            return {"text": source.read_text(encoding="utf-8"), "cost_inr": 0.001}

        def close(self):
            pass

    monkeypatch.setattr(core_tools, "BodhanClient", FakeBodhanClient)
    result = core_tools.synthesize_file_tool(
        str(source), str(output), "bn", settings=_settings()
    )

    assert output.is_file()
    assert result["artifact_path"] == str(output.resolve())
    assert result["verification"]["status"] == "passed"
    assert Path(str(output) + ".verification.txt").is_file()
    assert Path(str(output) + ".ai-generated.txt").is_file()
    assert calls == [output.resolve()]
    assert not (tmp_path / ".speech.bodhan-intermediate.wav").exists()


def test_invalid_tool_arguments_are_recorded_in_trace(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("Input", encoding="utf-8")
    fake_llm = ScriptedLLM(_settings(), [
        {"content": "Trying a malformed call.", "tool_calls": [{
            "id": "call-invalid", "type": "function",
            "function": {"name": "write_english_text", "arguments": "not-json"},
        }]},
        {"content": "The malformed call could not be executed.", "tool_calls": []},
    ])
    monkeypatch.setattr(agent_module, "LLMClient", lambda settings: fake_llm)

    result = AgentRunner(
        task="Write a summary.", files_path=str(source), output_path=str(tmp_path / "out"),
        max_steps=1, max_cost_inr=1.0, settings=_settings(),
    ).run()

    assert result["status"] == "failed"
    trace = [json.loads(line) for line in (tmp_path / "out" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "tool_start" and row["tool"] == "write_english_text" for row in trace)
    assert any(row["event"] == "tool_end" and "Invalid tool arguments" in row["result"]["error"] for row in trace)
