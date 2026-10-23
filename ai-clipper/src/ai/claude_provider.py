from .base import AIProvider, env_max_retries


class ClaudeProvider(AIProvider):
    name = "claude"
    default_model = "claude-sonnet-4-5"
    model_env = "ANTHROPIC_MODEL"
    model_prefixes = ("claude",)

    def __init__(self, api_key: str):
        from anthropic import Anthropic
        # base_url is picked up from the ANTHROPIC_BASE_URL env var by the SDK.
        self.client = Anthropic(api_key=api_key, max_retries=env_max_retries())

    @classmethod
    def is_available(cls, cfg) -> bool:
        return bool(cfg.anthropic_key)

    def complete(self, system: str, user: str, model: str = None) -> str:
        resp = self.client.messages.create(
            model=model or self.default_model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
