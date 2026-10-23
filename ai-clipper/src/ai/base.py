import os
from abc import ABC, abstractmethod
from typing import List, Optional


def env_max_retries(default: int = 0) -> int:
    """How many times the HTTP SDK may retry the *same* request.

    Defaults to 0 so a dead/unauthorised provider/model is skipped immediately
    instead of burning seconds (and quota) retrying a 503/404.
    """
    raw = os.getenv("AI_MAX_RETRIES")
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class AIProvider(ABC):
    name: str = "base"
    # Model used when nothing is configured via env.
    default_model: str = ""
    # Env var that overrides the model for this provider only.
    model_env: str = ""
    # Prefixes used to decide whether a model from AI_FALLBACK_MODELS could
    # plausibly belong to this provider (avoids pointless 404 round trips).
    model_prefixes: tuple = ()

    @abstractmethod
    def complete(self, system: str, user: str, model: Optional[str] = None) -> str:
        raise NotImplementedError

    @classmethod
    def is_available(cls, cfg) -> bool:
        return False

    @classmethod
    def model_candidates(cls, cfg) -> List[str]:
        """Ordered list of models to try for this provider.

        Order: <PROVIDER>_MODEL env -> AI_MODEL -> built-in default ->
        AI_FALLBACK_MODELS entries that match this provider's prefixes.
        """
        ordered = [
            os.getenv(cls.model_env) if cls.model_env else None,
            getattr(cfg, "ai_model", None),
            cls.default_model,
        ]
        names: List[str] = []
        for name in ordered:
            name = (name or "").strip()
            if name and name not in names:
                names.append(name)

        for name in getattr(cfg, "ai_fallback_models", None) or []:
            name = (name or "").strip()
            if not name or name in names:
                continue
            if cls.model_prefixes and not name.lower().startswith(cls.model_prefixes):
                continue
            names.append(name)
        return names
