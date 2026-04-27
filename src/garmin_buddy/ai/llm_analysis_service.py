from dataclasses import dataclass
from typing import Any, Mapping

from google import genai
from google.genai import types

_LLM_MODEL = "gemini-2.5-flash-lite"


@dataclass
class TokenUsageTotals:
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_usage(self, usage_metadata: Any) -> None:
        if usage_metadata is None:
            return

        self.total_input_tokens += _usage_value(usage_metadata, "prompt_token_count")
        self.total_input_tokens += _usage_value(
            usage_metadata, "tool_use_prompt_token_count"
        )
        self.total_output_tokens += _usage_value(
            usage_metadata, "candidates_token_count"
        )


class LLMService:
    def __init__(self, api_key: str, *, model_name: str | None = None) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or _LLM_MODEL

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        response_json_schema: Mapping[str, Any] | None = None,
        usage_tracker: TokenUsageTotals | None = None,
    ) -> str:
        """Keep one provider boundary so new workflows do not fork provider code."""

        return self._generate_response(
            prompt,
            model=model,
            system_instruction=system_instruction,
            response_json_schema=response_json_schema,
            usage_tracker=usage_tracker,
        )

    def _generate_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        response_json_schema: Mapping[str, Any] | None = None,
        usage_tracker: TokenUsageTotals | None = None,
    ) -> str:
        config = None
        if system_instruction is not None or response_json_schema is not None:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type=(
                    "application/json" if response_json_schema is not None else None
                ),
                response_json_schema=response_json_schema,
            )

        response = self.client.models.generate_content(
            model=model or self.model_name,
            contents=prompt,
            config=config,
        )
        if usage_tracker is not None:
            usage_tracker.add_usage(response.usage_metadata)

        return response.text or ""


def _usage_value(usage_metadata: Any, field_name: str) -> int:
    value = getattr(usage_metadata, field_name, None)
    if not isinstance(value, int) or value < 0:
        return 0
    return value
