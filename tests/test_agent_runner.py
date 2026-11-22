import json

from bodhan_agent import runner as agent_module
from bodhan_agent.runner import AgentRunner
from bodhan_core.config import Settings


class FakeLLM:
    def __init__(self, settings):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "I will write the English summary.",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "write_english_text",
                        "arguments": json.dumps({
                            "output_name": "summary.md",
                            "text": "A short English summary.",
                        }),
                    },
                }],
            }
        return {"content": "The summary file was written successfully.", "tool_calls": []}

    def close(self):
        pass


def test_agent_runner_writes_artifact_trace_and_report(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "source.txt").write_text("Source material", encoding="utf-8")
    monkeypatch.setattr(agent_module, "LLMClient", FakeLLM)
    settings = Settings(llm_base_url="https://llm.example/v1", llm_model="test", llm_api_key="test")

    result = AgentRunner(
        task="Create a short English summary", files_path=str(input_dir),
        output_path=str(tmp_path / "out"), settings=settings,
    ).run()

    artifact = tmp_path / "out" / "artifacts" / "summary.md"
    assert result["status"] == "completed"
    assert artifact.read_text(encoding="utf-8").strip() == "A short English summary."
    assert (tmp_path / "out" / "trace.jsonl").is_file()
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert str(artifact.resolve()) in report["artifacts"]
    assert report["steps"] == 1
