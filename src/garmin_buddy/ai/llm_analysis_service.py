from typing import Any, Mapping

from google import genai
from google.genai import types

_LLM_MODEL = "gemini-2.5-flash-lite"


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
    ) -> str:
        """Keep one provider boundary so new workflows do not fork provider code."""

        return self._generate_response(
            prompt,
            model=model,
            system_instruction=system_instruction,
            response_json_schema=response_json_schema,
        )

    def _generate_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        response_json_schema: Mapping[str, Any] | None = None,
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

        return response.text or ""
