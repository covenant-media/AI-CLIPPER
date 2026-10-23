from .base import AIProvider, env_max_retries


def _json_mode_unsupported(err: Exception) -> bool:
    """Some OpenAI-compatible gateways reject response_format=json_object."""
    text = str(err).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text


class OpenAIProvider(AIProvider):
    name = "openai"
    default_model = "gpt-4o"
    model_env = "OPENAI_MODEL"
    model_prefixes = ("gpt", "o1", "o3", "o4", "chatgpt")

    def __init__(self, api_key: str):
        from openai import OpenAI
        # base_url is picked up from the OPENAI_BASE_URL env var by the SDK.
        self.client = OpenAI(api_key=api_key, max_retries=env_max_retries())

    @classmethod
    def is_available(cls, cfg) -> bool:
        return bool(cfg.openai_key)

    def complete(self, system: str, user: str, model: str = None) -> str:
        model = model or self.default_model
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as e:
            if not _json_mode_unsupported(e):
                raise
            resp = self.client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""
