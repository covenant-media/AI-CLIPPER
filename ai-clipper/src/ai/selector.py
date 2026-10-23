import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from ..utils import log, write_json, read_json


PROVIDERS = [ClaudeProvider, OpenAIProvider, GeminiProvider]


def _load_prompt() -> str:
    p = Path(__file__).parent.parent.parent / "prompts" / "highlight_system.txt"
    return p.read_text(encoding="utf-8")


def _build_transcript_text(transcript: Dict[str, Any], scenes: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("SCENE CUTS (visual):")
    for s in scenes[:80]:
        lines.append(f"  {s['start']:.1f}s -> {s['end']:.1f}s")
    lines.append("")
    lines.append("TRANSCRIPT (timestamped segments):")
    for seg in transcript["segments"]:
        speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        lines.append(
            f"[{seg['start']:.2f} - {seg['end']:.2f}] {speaker}{seg['text']}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    return json.loads(text)


def _validate_edl(edl: Dict[str, Any], cfg, total_duration: float) -> Dict[str, Any]:
    clips = edl.get("clips", [])
    # A clip must be at least half the target minimum... unless the whole video
    # is shorter than that, in which case the video's own length is the floor
    # (otherwise an 8s source could never yield a clip).
    floor = min(cfg.clip_min * 0.5, total_duration)
    valid = []
    for c in clips:
        try:
            s = float(c["start"])
            e = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        s = max(0.0, s)
        e = min(total_duration, e)
        if (e - s) < floor:
            continue
        valid.append({
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "category": c.get("category", "highlight"),
            "reason": c.get("reason", ""),
            "score": float(c.get("score", 0) or 0),
        })
    valid.sort(key=lambda x: x["score"], reverse=True)
    valid = valid[: cfg.max_clips]
    return {"clips": valid}


def _snap_to_scenes(edl: Dict[str, Any], scenes: List[Dict[str, Any]], tol: float) -> Dict[str, Any]:
    boundaries = [s["start"] for s in scenes] + [s["end"] for s in scenes]
    boundaries = sorted(set(round(b, 3) for b in boundaries))

    def snap(t):
        if not boundaries:
            return t
        nearest = min(boundaries, key=lambda b: abs(b - t))
        return nearest if abs(nearest - t) <= tol else t

    for c in edl["clips"]:
        c["start"] = round(snap(c["start"]), 3)
        c["end"] = round(snap(c["end"]), 3)
    return edl


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------
# Status codes that mean "this provider/model cannot serve us" - never retry.
_SKIP_STATUS = {400, 401, 402, 403, 404, 405, 408, 409, 413, 422, 429,
                500, 502, 503, 504, 522, 524, 529}
# Substrings (lowercase) seen from OpenAI-compatible relays / gateways.
_SKIP_TEXT = (
    "no available channel", "no channel", "无可用渠道", "渠道",
    "no available upstream", "upstream", "model not found", "model_not_found",
    "does not exist", "not a valid model", "unsupported model",
    "invalid model", "unknown model",
    "quota", "insufficient", "billing", "exceeded your current",
    "rate limit", "rate_limit", "too many requests",
    "overloaded", "unavailable", "temporarily", "try again later",
    "invalid api key", "authentication", "unauthorized", "permission",
)


def _brief(err: Exception, limit: int = 200) -> str:
    """One tidy line for the console (SDK errors can be multi-line JSON)."""
    text = str(err).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit] + ("..." if len(text) > limit else "")


def _is_skip_error(err: Exception) -> bool:
    """True when we should move straight to the next provider/model.

    Covers provider-side 5xx ("no available channel"), auth failures, unknown
    models and quota errors - i.e. retrying the *same* target is pointless.
    """
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(status, int) and status in _SKIP_STATUS:
        return True
    text = str(err).lower()
    return any(p in text for p in _SKIP_TEXT)


# --------------------------------------------------------------------------
# Local heuristic fallback (used only when every AI provider fails)
# --------------------------------------------------------------------------
_HOOK_WORDS = (
    "how", "why", "what", "when", "because", "secret", "never", "always",
    "best", "worst", "mistake", "important", "remember", "listen", "look",
    "imagine", "truth", "story", "money", "free", "stop", "start", "first",
    "understand", "realize", "wisdom", "key", "lesson", "prove", "proven",
)


def _score_window(text: str, start: float, dur: float,
                  total_duration: float, cfg) -> float:
    low = text.lower()
    score = sum(5.0 for w in _HOOK_WORDS if w in low)
    score += 12.0 if "?" in text else 0.0
    score += 6.0 if "!" in text else 0.0
    # Prefer clip lengths near the middle of the requested range.
    mid = (cfg.clip_min + cfg.clip_max) / 2.0
    score += max(0.0, 15.0 - abs(dur - mid) / max(mid, 1.0) * 15.0)
    # Prefer the middle of the video over the very start/end.
    center = (start + dur / 2.0) / max(total_duration, 1.0)
    score += 10.0 * (1.0 - abs(center - 0.5) * 2.0)
    # Favour windows with a decent amount of speech.
    score += min(len(text) / 60.0, 8.0)
    return score


def _heuristic_edl(transcript, cfg, total_duration) -> Dict[str, Any]:
    """Duration/punctuation based highlight picker.

    Groups consecutive transcript segments into windows that fit the requested
    clip length, scores them, and keeps the best ones. This guarantees the
    pipeline still produces a reel when no AI provider is reachable.
    """
    segs = transcript.get("segments") or []
    clip_min = float(cfg.clip_min)
    clip_max = float(cfg.clip_max)

    windows = []
    i = 0
    while i < len(segs):
        start = float(segs[i].get("start", 0) or 0)
        end = float(segs[i].get("end", 0) or 0)
        text = segs[i].get("text") or ""
        j = i
        while j + 1 < len(segs) and (end - start) < clip_min:
            nxt_end = float(segs[j + 1].get("end", end) or end)
            if (nxt_end - start) > clip_max:
                break
            j += 1
            end = nxt_end
            text = f"{text} {segs[j].get('text') or ''}".strip()
        dur = end - start
        if dur > 0:
            windows.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(dur, 3),
                "category": "heuristic",
                "reason": "Longest high-signal transcript window (local fallback).",
                "score": round(min(100.0, _score_window(
                    text, start, dur, total_duration, cfg
                )), 1),
            })
        i = j + 1

    log.info(f"Heuristic candidate windows: {len(windows)}")
    return {"clips": windows}


def select_highlights(transcript, scenes, workdir, cfg, total_duration):
    edl_path = workdir / "edl.json"
    if edl_path.exists():
        log.info("EDL cached")
        return read_json(edl_path)

    system = _load_prompt()
    user = (
        f"Target clip length: {cfg.clip_min}-{cfg.clip_max} seconds.\n"
        f"Return at most {cfg.max_clips} clips, ordered best-first.\n"
        f"Total video duration: {total_duration:.2f}s.\n\n"
        + _build_transcript_text(transcript, scenes)
    )

    if cfg.ai_provider == "auto":
        providers = [P for P in PROVIDERS if P.is_available(cfg)]
    else:
        providers = [P for P in PROVIDERS
                     if P.name == cfg.ai_provider and P.is_available(cfg)]

    # Build every (provider, model) pair we are willing to try, in order.
    attempts = []
    for P in providers:
        for model in P.model_candidates(cfg):
            attempts.append((P, model))

    tried = []
    last_err = None
    for P, model in attempts:
        key = (
            cfg.anthropic_key if P.name == "claude"
            else cfg.openai_key if P.name == "openai"
            else cfg.gemini_key
        )
        try:
            log.info(f"Asking {P.name} ({model}) to select highlights...")
            prov = P(key)
            raw = prov.complete(system, user, model=model)
            edl = _extract_json(raw)
            edl = _validate_edl(edl, cfg, total_duration)
            if not edl["clips"]:
                raise ValueError("model returned no usable clips")
            edl = _snap_to_scenes(edl, scenes, cfg.snap_seconds)
            edl["source"] = f"{P.name}:{model}"
            write_json(edl_path, edl)
            log.ok(f"{P.name} ({model}) returned {len(edl['clips'])} clips")
            return edl
        except Exception as e:
            last_err = e
            label = f"{P.name}:{model}"
            tried.append(label)
            if _is_skip_error(e):
                # Provider/model is unreachable (503 "no available channel",
                # bad key, missing model, quota...) - do not retry it.
                log.warn(f"{label} unavailable, trying next model/provider - {_brief(e)}")
            else:
                log.warn(f"{label} failed: {_brief(e)}")

    # ---------------------------------------------------------------- failure
    if not providers:
        log.error(
            "No AI provider is configured. Set one of ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY or GEMINI_API_KEY (see .env.example)."
        )
    else:
        log.error("All AI providers failed.")
        if last_err is not None:
            log.error(f"Last error: {_brief(last_err, limit=400)}")
        log.error(
            "Checked: " + (", ".join(tried) if tried else "none") + "\n"
            "  AI_MODEL=" + str(cfg.ai_model or "(unset)") + "\n"
            "  ANTHROPIC_BASE_URL=" + str(os.getenv("ANTHROPIC_BASE_URL") or "(default)") + "\n"
            "  OPENAI_BASE_URL=" + str(os.getenv("OPENAI_BASE_URL") or "(default)") + "\n"
            "Fix by pointing at a model YOUR endpoint actually serves:\n"
            "  AI_MODEL=gpt-4o  (or claude-3-5-sonnet-20241022, gemini-1.5-pro, ...)\n"
            "  AI_FALLBACK_MODELS=modelA,modelB\n"
            "  If the base URL above is a relay, check that its 'default' group has\n"
            "  a channel for that model name, and that it accepts requests from the\n"
            "  official SDKs (some relay to Claude-Code clients only -> 401\n"
            "  'unauthorized client detected')."
        )

    if getattr(cfg, "ai_heuristic_fallback", True):
        log.warn("Falling back to the local transcript heuristic.")
        edl = _validate_edl(_heuristic_edl(transcript, cfg, total_duration),
                            cfg, total_duration)
        if not edl["clips"]:
            log.error(
                "Heuristic found nothing usable - the transcript is empty or "
                "every candidate window was too short to use."
            )
            return {"clips": [], "source": "heuristic"}
        edl = _snap_to_scenes(edl, scenes, cfg.snap_seconds)
        edl["source"] = "heuristic"
        write_json(edl_path, edl)
        log.ok(f"Heuristic fallback returned {len(edl['clips'])} clips")
        return edl

    raise RuntimeError(
        "All AI providers failed and the heuristic fallback is disabled "
        "(AI_HEURISTIC_FALLBACK=false). Last error: "
        f"{_brief(last_err, limit=400)}"
    )
