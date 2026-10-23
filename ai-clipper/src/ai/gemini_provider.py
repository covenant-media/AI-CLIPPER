from .base import AIProvider


class GeminiProvider(AIProvider):
    name = "gemini"
    default_model = "gemini-1.5-pro"
    model_env = "GEMINI_MODEL"
    model_prefixes = ("gemini",)

    def __init__(self, api_key: str):
        import google.generativeai as genai
        self._genai = genai
        genai.configure(api_key=api_key)

    @classmethod
    def is_available(cls, cfg) -> bool:
        return bool(cfg.gemini_key)

    def complete(self, system: str, user: str, model: str = None) -> str:
        name = (model or self.default_model).removeprefix("models/")
        gm = self._genai.GenerativeModel(name)
        resp = gm.generate_content(
            [system, user],
            generation_config={"response_mime_type": "application/json"},
        )
        return resp.text
