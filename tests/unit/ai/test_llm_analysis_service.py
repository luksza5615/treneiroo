from __future__ import annotations

from types import SimpleNamespace

from garmin_buddy.ai.llm_analysis_service import LLMService, TokenUsageTotals


class _FakeModels:
    def __init__(self, response) -> None:
        self._response = response

    def generate_content(self, *, model, contents, config):
        return self._response


class _FakeClient:
    def __init__(self, response) -> None:
        self.models = _FakeModels(response)


def test_llm_service_captures_usage_metadata(monkeypatch) -> None:
    response = SimpleNamespace(
        text='{"ok":true}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=21,
            tool_use_prompt_token_count=4,
            candidates_token_count=9,
        ),
    )
    monkeypatch.setattr(
        "garmin_buddy.ai.llm_analysis_service.genai.Client",
        lambda api_key: _FakeClient(response),
    )
    service = LLMService("api-key")
    usage = TokenUsageTotals()

    text = service.generate("prompt", usage_tracker=usage)

    assert text == '{"ok":true}'
    assert usage.total_input_tokens == 25
    assert usage.total_output_tokens == 9


def test_llm_service_ignores_missing_usage_metadata(monkeypatch) -> None:
    response = SimpleNamespace(text="plain text", usage_metadata=None)
    monkeypatch.setattr(
        "garmin_buddy.ai.llm_analysis_service.genai.Client",
        lambda api_key: _FakeClient(response),
    )
    service = LLMService("api-key")
    usage = TokenUsageTotals()

    text = service.generate("prompt", usage_tracker=usage)

    assert text == "plain text"
    assert usage.total_input_tokens == 0
    assert usage.total_output_tokens == 0
