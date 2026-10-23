import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml
from dotenv import dotenv_values

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# `.env` is the project's source of truth: a non-blank value there *overrides*
# the ambient shell environment, otherwise a global ANTHROPIC_BASE_URL /
# ANTHROPIC_MODEL exported elsewhere would silently win over this project.
# Blank entries (KEY=) are ignored so shell-provided secrets keep working and
# no key ever has to be committed to the repo.
# Plain load_dotenv() cannot do this - it never overrides existing variables.
if _ENV_PATH.exists():
    for _key, _value in dotenv_values(_ENV_PATH).items():
        if _value is not None and _value.strip():
            os.environ[_key] = _value.strip()


def _env_list(name: str) -> List[str]:
    """Parse a comma separated env var into a clean list."""
    raw = os.getenv(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None
    ai_provider: str = "auto"

    # --- AI highlight selection ------------------------------------------
    # Optional model override; must be a model the endpoint actually serves.
    ai_model: Optional[str] = None
    # Extra model names to try (in order) before giving up on the AI step.
    ai_fallback_models: List[str] = field(default_factory=list)
    # 0 = never retry the same provider/model, jump straight to the next one.
    ai_max_retries: int = 0
    # When every AI provider fails, use a local transcript heuristic so the
    # pipeline still produces clips instead of crashing.
    ai_heuristic_fallback: bool = True

    # --- Transcription ---------------------------------------------------
    whisper_model: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    # Force a language (e.g. "yo") to skip Whisper's detection pass entirely.
    whisper_language: Optional[str] = None
    whisper_batch_size: int = 16
    # VAD backend for WhisperX: "pyannote" (default, offline, slow on CPU) or
    # "silero" (much faster on CPU, but torch.hub downloads it on first use).
    whisper_vad_method: str = "pyannote"
    # CPU threads for Whisper decoding. 0 = use every logical core (WhisperX
    # itself defaults to just 4, which wastes most of a modern CPU).
    whisper_threads: int = 0
    # Print a "Progress: N%" line as Whisper decodes each VAD segment. Stage 2
    # is the long one, and without this it prints nothing for many minutes on
    # CPU. Set false to keep logs quiet.
    whisper_progress: bool = True
    # HuggingFace wav2vec2 alignment model override.
    align_model: Optional[str] = None
    hf_token: Optional[str] = None

    clip_min: int = 20
    clip_max: int = 90
    max_clips: int = 10
    scene_threshold: float = 27.0
    scene_min_frames: int = 15
    # Process every Nth+1 frame; speeds up scene detection on high framerate
    # video at the cost of cut precision.
    scene_frame_skip: int = 0
    # When PySceneDetect finds no cuts at all, generate evenly spaced fallback
    # boundaries so clip edges can still snap to "scenes". 0 = disable.
    scene_fallback_interval: float = 0.0
    # Skip scene detection altogether. Single-continuous-shot footage (sermons,
    # talking heads, screen recordings) has no cuts, so the full decode is pure
    # runtime with no effect on the output. true = detect (default).
    scene_detect: bool = True
    cut_mode: str = "lossless"
    fade_seconds: float = 0.0
    snap_seconds: float = 1.5
    out_w: Optional[int] = None
    out_h: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1

    # --- Run provenance (filled in by load(); informational only) ----------
    # Which config.yaml / .env were actually applied. Printed at startup so
    # "did my .env get picked up?" never has to be guessed again.
    config_path: Optional[str] = None
    env_path: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "Config":
        cfg = cls(
            anthropic_key=os.getenv("ANTHROPIC_API_KEY") or None,
            openai_key=os.getenv("OPENAI_API_KEY") or None,
            gemini_key=os.getenv("GEMINI_API_KEY") or None,
            ai_provider=os.getenv("AI_PROVIDER") or "auto",
            ai_model=os.getenv("AI_MODEL") or None,
            ai_fallback_models=_env_list("AI_FALLBACK_MODELS"),
            ai_max_retries=_env_int("AI_MAX_RETRIES", 0),
            ai_heuristic_fallback=_env_bool("AI_HEURISTIC_FALLBACK", True),
            whisper_model=os.getenv("WHISPER_MODEL") or "large-v3",
            whisper_device=os.getenv("WHISPER_DEVICE") or "cpu",
            whisper_compute=os.getenv("WHISPER_COMPUTE") or "int8",
            whisper_language=os.getenv("WHISPER_LANGUAGE") or None,
            whisper_batch_size=_env_int("WHISPER_BATCH_SIZE", 16),
            whisper_progress=_env_bool("WHISPER_PROGRESS", True),
            whisper_threads=_env_int("WHISPER_THREADS", 0),
            whisper_vad_method=(
                os.getenv("WHISPER_VAD_METHOD") or "pyannote"
            ).strip().lower(),
            align_model=os.getenv("ALIGN_MODEL") or None,
            hf_token=os.getenv("HF_TOKEN") or None,
        )
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            v = data.get("video", {})
            cfg.clip_min = v.get("target_clip_min_seconds", cfg.clip_min)
            cfg.clip_max = v.get("target_clip_max_seconds", cfg.clip_max)
            cfg.max_clips = v.get("max_clips", cfg.max_clips)
            s = data.get("scenes", {})
            cfg.scene_threshold = s.get("threshold", cfg.scene_threshold)
            cfg.scene_min_frames = s.get("min_scene_len_frames", cfg.scene_min_frames)
            cfg.scene_frame_skip = s.get("frame_skip", cfg.scene_frame_skip)
            cfg.scene_fallback_interval = s.get(
                "fallback_interval_seconds", cfg.scene_fallback_interval
            )
            e = data.get("edit", {})
            cfg.cut_mode = e.get("cut_mode", cfg.cut_mode)
            cfg.fade_seconds = e.get("fade_seconds", cfg.fade_seconds)
            cfg.snap_seconds = e.get("snap_to_scene_seconds", cfg.snap_seconds)
            cfg.out_w = e.get("output_width")
            cfg.out_h = e.get("output_height")
            a = data.get("audio", {})
            cfg.sample_rate = a.get("sample_rate", cfg.sample_rate)
            cfg.channels = a.get("channels", cfg.channels)

        # Env wins over config.yaml so a run can be tuned without editing files.
        cfg.scene_frame_skip = _env_int("SCENE_FRAME_SKIP", cfg.scene_frame_skip)
        cfg.scene_fallback_interval = _env_float(
            "SCENE_FALLBACK_INTERVAL", cfg.scene_fallback_interval
        )
        cfg.scene_detect = _env_bool("SCENE_DETECT", cfg.scene_detect)
        cfg.config_path = str(path) if path.exists() else None
        cfg.env_path = str(_ENV_PATH) if _ENV_PATH.exists() else None
        return cfg
