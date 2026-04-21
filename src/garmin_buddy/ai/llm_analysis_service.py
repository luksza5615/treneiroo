from google import genai

_LLM_MODEL = "gemini-3-flash-preview"


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
    ) -> str:
        """Keep one provider boundary so new workflows do not fork provider code."""

        contents = prompt
        if system_instruction:
            contents = f"{system_instruction}\n\n{prompt}"

        return self._generate_response(contents, model=model)

    def _generate_response(self, prompt: str, *, model: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=model or self.model_name,
            contents=prompt,
        )

        return response.text or ""
