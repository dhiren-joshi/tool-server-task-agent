import json

from bodhan_core import tools


class FakeClient:
    def __init__(self, settings=None):
        pass

    def translate_text(self, text, target_language):
        return {"translation": f"EN: {text}", "cost_inr": 0.0001}

    def close(self):
        pass


def test_srt_sentence_cues_are_numbered_and_interpolated(tmp_path, monkeypatch):
    segments = tmp_path / "transcript.segments.json"
    segments.write_text(json.dumps([
        {"start_seconds": 0.0, "end_seconds": 10.0, "text": "First sentence. Second sentence!"},
        {"start_seconds": 10.0, "end_seconds": 20.0, "text": "Third sentence."},
    ]), encoding="utf-8")
    output = tmp_path / "captions.srt"
    monkeypatch.setattr(tools, "BodhanClient", FakeClient)

    result = tools.translate_segments_to_srt_tool(str(segments), str(output), "en")
    content = output.read_text(encoding="utf-8")

    assert result["cue_count"] == 3
    assert result["verification"]["status"] == "passed"
    assert "1\n00:00:00,000 -->" in content
    assert "2\n" in content
    assert "3\n00:00:10,000 --> 00:00:20,000" in content
    assert result["verification"]["timing_note"].startswith("Approximate sentence-level")
