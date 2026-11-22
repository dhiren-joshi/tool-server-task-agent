import io

import fitz
import httpx
import pytest
from pydub import AudioSegment

from bodhan_core.api_client import BodhanClient
from bodhan_core.config import Settings
from bodhan_core.errors import BodhanAPIError


def response_json(url: str, body: dict):
    return httpx.Response(200, json=body, request=httpx.Request("POST", url))


def make_settings(**overrides):
    values = {
        "transcribe_key": "test-transcribe",
        "translate_key": "test-translate",
        "speak_key": "test-speak",
        "ocr_key": "test-ocr",
        "translation_chunk_chars": 5,
        "audio_chunk_seconds": 29,
        "tts_chunk_chars": 100,
    }
    values.update(overrides)
    return Settings(**values)


def test_translation_splits_long_text_and_checks_every_chunk():
    client = BodhanClient(make_settings())
    calls = []

    def fake_request(method, url, *, model_key, **kwargs):
        calls.append(kwargs["json"]["text"])
        return response_json(url, {
            "translation": "translated:" + kwargs["json"]["text"],
            "target_language": "hi",
            "script": "native",
            "truncated": False,
            "usage": {"completion_tokens": 12},
        })

    client._request = fake_request
    try:
        result = client.translate_text("abcdefghijkl", "hi")
    finally:
        client.close()

    assert len(calls) == 3
    assert "".join(calls) == "abcdefghijkl"
    assert result["chunks"] == 3
    assert result["cost_inr"] == 0.00072


def test_translation_rejects_truncated_response():
    client = BodhanClient(make_settings())

    def fake_request(method, url, *, model_key, **kwargs):
        return response_json(url, {
            "translation": "cut off", "truncated": True,
            "usage": {"completion_tokens": 10},
        })

    client._request = fake_request
    try:
        with pytest.raises(BodhanAPIError, match="truncated"):
            client.translate_text("hello", "hi")
    finally:
        client.close()


def test_translation_sets_an_output_token_ceiling_and_rejects_english_script_modes():
    client = BodhanClient(make_settings())
    calls = []

    def fake_request(method, url, *, model_key, **kwargs):
        calls.append(kwargs["json"])
        return response_json(url, {
            "translation": "translated", "truncated": False,
            "usage": {"completion_tokens": 4},
        })

    client._request = fake_request
    try:
        client.translate_text("hello", "en")
        assert calls[0]["max_output_tokens"] == 128
        assert "script" not in calls[0]
        with pytest.raises(ValueError, match="not supported"):
            client.translate_text("hello", "en", script="roman")
    finally:
        client.close()


def test_translation_preserves_cost_if_a_later_chunk_fails():
    client = BodhanClient(make_settings())
    calls = 0

    def fake_request(method, url, *, model_key, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BodhanAPIError("temporary failure", status_code=503)
        return response_json(url, {
            "translation": "translated", "truncated": False,
            "usage": {"completion_tokens": 12},
        })

    client._request = fake_request
    try:
        with pytest.raises(BodhanAPIError) as error:
            client.translate_text("abcdefghij", "hi")
        assert error.value.cost_inr == 0.00024
    finally:
        client.close()


def test_audio_transcription_splits_at_29_seconds(tmp_path):
    audio_path = tmp_path / "long.wav"
    AudioSegment.silent(duration=61_000).export(audio_path, format="wav")
    client = BodhanClient(make_settings())
    calls = []

    def fake_request(method, url, *, model_key, **kwargs):
        calls.append(kwargs["data"])
        return response_json(url, {"text": f"chunk {len(calls)}"})

    client._request = fake_request
    try:
        result = client.transcribe_audio(audio_path, language="hi")
    finally:
        client.close()

    assert len(calls) == 3
    assert result["chunks"] == 3
    assert len(result["segments"]) == 3
    assert result["segments"][-1]["end_seconds"] == 61.0
    assert "chunk 1" in result["text"]


def test_pdf_ocr_calls_bodhan_once_per_page(tmp_path):
    pdf_path = tmp_path / "two_pages.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    output_path = tmp_path / "ocr.md"
    client = BodhanClient(make_settings())
    calls = []

    def fake_request(method, url, *, model_key, **kwargs):
        calls.append(kwargs["json"])
        return response_json(url, {
            "choices": [{"message": {"content": f"Page {len(calls)} extracted text"}}],
            "blocks": [{"label": "Paragraph", "text": "fixture"}],
            "usage": {"completion_tokens": 8},
        })

    client._request = fake_request
    try:
        result = client.ocr_document(pdf_path, output_path)
    finally:
        client.close()

    assert result["pages"] == 2
    assert len(calls) == 2
    assert calls[0]["table_format"] == "html"
    assert "Page 1 extracted text" in output_path.read_text(encoding="utf-8")
    assert "Page 2 extracted text" in output_path.read_text(encoding="utf-8")
    # The OCR image data URI is only in the outbound Bodhan HTTP body, not in MCP arguments.
    image_url = calls[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_transcription_preserves_cost_when_a_later_chunk_fails(tmp_path):
    import pytest

    from bodhan_core.errors import BodhanAPIError

    audio_path = tmp_path / "long.wav"
    AudioSegment.silent(duration=61_000).export(audio_path, format="wav")
    client = BodhanClient(make_settings())
    calls = 0

    def fake_request(method, url, *, model_key, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BodhanAPIError("API unavailable", status_code=503)
        return response_json(url, {"text": "first chunk"})

    client._request = fake_request
    try:
        with pytest.raises(BodhanAPIError) as error:
            client.transcribe_audio(audio_path, language="hi")
        assert error.value.cost_inr == pytest.approx(29 / 60 * 0.10, abs=1e-6)
    finally:
        client.close()


def test_rate_limit_and_exhausted_credit_are_treated_differently():
    rate_limit = httpx.Response(
        429, json={"error": {"code": "rate_limit", "message": "Too many requests"}},
        request=httpx.Request("POST", "https://api.bodhan.ai/test"),
    )
    no_credit = httpx.Response(
        429, json={"error": {"code": "insufficient_funds", "message": "Insufficient balance"}},
        request=httpx.Request("POST", "https://api.bodhan.ai/test"),
    )

    assert BodhanClient._should_retry_response(rate_limit) is True
    assert BodhanClient._should_retry_response(no_credit) is False


def test_transcription_upload_is_rewound_before_retry(monkeypatch):
    client = BodhanClient(make_settings(max_retries=1))
    uploaded = []

    class FakeHTTP:
        def request(self, method, url, *, headers, **kwargs):
            uploaded.append(kwargs["files"]["file"][1].read())
            status = 503 if len(uploaded) == 1 else 200
            return httpx.Response(
                status, json={"text": "ok"}, request=httpx.Request(method, url)
            )

        def close(self):
            pass

    client._http.close()
    client._http = FakeHTTP()
    monkeypatch.setattr("bodhan_core.api_client.time.sleep", lambda _: None)
    upload = io.BytesIO(b"wave-data")
    try:
        response = client._request(
            "POST", "https://api.bodhan.ai/v1/audio/transcriptions",
            model_key="transcribe",
            files={"file": ("audio.wav", upload, "audio/wav")},
        )
        assert response.status_code == 200
        assert uploaded == [b"wave-data", b"wave-data"]
    finally:
        client.close()


def _unexpected_api_call(*args, **kwargs):
    pytest.fail("Bodhan should not be called")


def test_unsupported_translation_language_fails_before_api_call():
    client = BodhanClient(make_settings())
    client._request = _unexpected_api_call
    try:
        with pytest.raises(ValueError, match="Unsupported translation language: fr"):
            client.translate_text("hello", "fr")
    finally:
        client.close()


def test_unsupported_speech_language_fails_before_api_call(tmp_path):
    client = BodhanClient(make_settings())
    client._request = _unexpected_api_call
    try:
        with pytest.raises(ValueError, match="Unsupported speech language: fr"):
            client.synthesize_speech("bonjour", "fr", tmp_path / "speech.wav")
    finally:
        client.close()


@pytest.mark.parametrize("language", ["ks-Deva", "sd-Arab"])
def test_codemix_is_rejected_for_languages_without_codemix_support(language):
    client = BodhanClient(make_settings())
    client._request = _unexpected_api_call
    try:
        with pytest.raises(ValueError, match="codemix output is not supported"):
            client.translate_text("hello", language, script="codemix")
    finally:
        client.close()
