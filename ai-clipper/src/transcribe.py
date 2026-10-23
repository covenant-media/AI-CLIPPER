import os
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Any
from .utils import write_json, log, read_json

# pyannote.audio probes `import torchcodec` while being imported and shouts when
# the Windows FFmpeg build has no shared libraries for it. WhisperX never uses
# torchcodec (it hands pyannote an in-memory waveform dict), so this warning is
# pure noise - filter it before whisperx pulls pyannote in.
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"pyannote\.audio\.core\.io")
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly.*")


def _brief(err: Exception, limit: int = 200) -> str:
    """One tidy line for the console (SDK errors can be multi-line)."""
    text = " ".join(str(err).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def transcribe(audio: Path, workdir: Path, cfg) -> Dict[str, Any]:
    json_path = workdir / "transcript.json"
    srt_path = workdir / "transcript.srt"

    if json_path.exists():
        log.info("Transcript already exists (cached)")
        return read_json(json_path)

    try:
        result = _whisperx(audio, workdir, cfg)
    except Exception as e:
        log.warn(f"WhisperX failed ({e}). Falling back to faster-whisper.")
        result = _faster_whisper(audio, cfg)

    write_json(json_path, result)
    _write_srt(result, srt_path)
    return result


def _whisperx(audio: Path, workdir: Path, cfg) -> Dict[str, Any]:
    import whisperx
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

    log.info(f"WhisperX loading model={cfg.whisper_model} device={cfg.whisper_device}")
    device = cfg.whisper_device
    compute_type = cfg.whisper_compute

    model = _load_whisperx_model(whisperx, cfg, device, compute_type)
    audio_data = whisperx.load_audio(str(audio))
    log.info(
        f"Running VAD + transcription on {len(audio_data) / 16000 / 60:.1f} min "
        f"of audio (vad={getattr(cfg, 'whisper_vad_method', 'pyannote')}). "
        f"On CPU this is the slow stage - progress lines start once VAD has "
        f"swept the whole file."
    )
    result = model.transcribe(
        audio_data,
        batch_size=cfg.whisper_batch_size,
        print_progress=bool(getattr(cfg, "whisper_progress", True)),
    )

    result = _align(
        whisperx, result, audio_data, device, cfg,
        DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF,
    )

    if cfg.hf_token:
        try:
            from whisperx.diarize import DiarizationPipeline
            diar = DiarizationPipeline(token=cfg.hf_token, device=device)
            diar_segments = diar(audio_data)
            result = whisperx.assign_word_speakers(diar_segments, result)
        except Exception as e:
            log.warn(f"Diarization skipped: {_brief(e)}")

    return _normalize(result)


def _load_whisperx_model(whisperx, cfg, device, compute_type):
    """Load the WhisperX pipeline, falling back to pyannote VAD on VAD failure.

    pyannote VAD is the only backend WhisperX can always build offline (the
    checkpoint ships inside the wheel), but its sliding-window model is the slow
    part of Stage 2 on CPU. Silero is far cheaper on CPU - however torch.hub
    has to download it from GitHub on first use, so a locked-down or offline
    machine must not lose the whole run over it.
    """
    method = getattr(cfg, "whisper_vad_method", "pyannote") or "pyannote"
    # WhisperX defaults to threads=4 no matter how many cores exist, which
    # leaves most CPUs idle during the int8 decode.
    threads = int(getattr(cfg, "whisper_threads", 0) or 0) or (os.cpu_count() or 4)
    log.info(
        f"WhisperX threads={threads} "
        f"(cpu_count={os.cpu_count()}), vad={method}"
    )
    kwargs = dict(
        compute_type=compute_type,
        language=cfg.whisper_language,
        vad_method=method,
        threads=threads,
    )
    try:
        return whisperx.load_model(cfg.whisper_model, device, **kwargs)
    except Exception as e:
        if method == "pyannote":
            raise
        log.warn(
            f"VAD method '{method}' could not be initialised ({_brief(e)}). "
            f"Falling back to pyannote (slower on CPU)."
        )
        kwargs["vad_method"] = "pyannote"
        return whisperx.load_model(cfg.whisper_model, device, **kwargs)


def _align(whisperx, result, audio_data, device, cfg, torch_models, hf_models):
    """Word-level alignment, skipped cleanly when the language is unsupported.

    WhisperX only ships default wav2vec2 align models for a fixed set of
    languages. For anything else (e.g. Yoruba "yo") it logs an ERROR and raises,
    which used to pollute the run - we now skip silently-but-visibly and keep
    Whisper's own segment timestamps.
    """
    language = (result.get("language") or "en").lower()
    override = cfg.align_model
    # Explicit opt-out. Word-level timestamps are unused by the clip heuristic
    # (and only marginally useful to the AI prompt), while the wav2vec2 align
    # model is a large download plus a full CPU pass over the audio.
    if override and override.strip().lower() in ("none", "off", "skip", "disable"):
        log.info("Alignment disabled (ALIGN_MODEL=none) - segment timestamps kept.")
        return result
    if override:
        model_name = override
    elif language in torch_models:
        model_name = torch_models[language]
    elif language in hf_models:
        model_name = hf_models[language]
    else:
        log.warn(
            f"Skipping alignment: WhisperX has no default wav2vec2 align model "
            f"for language '{language}'. Segment timestamps from Whisper are "
            f"kept as-is. To enable word-level alignment, set "
            f"ALIGN_MODEL=<hf-model-id> (for Yoruba try "
            f"chrisjay/wav2vec2-large-xlsr-53-yoruba)."
        )
        return result

    try:
        log.info(f"Aligning with {model_name}")
        align_model, meta = whisperx.load_align_model(
            language_code=language, device=device, model_name=model_name
        )
        aligned = whisperx.align(
            result["segments"], align_model, meta, audio_data, device,
            return_char_alignments=False,
        )
        # whisperx.align() drops the "language" key - keep it for downstream use.
        aligned["language"] = result.get("language", language)
        return aligned
    except Exception as e:
        log.warn(f"Alignment skipped: {_brief(e)}")
        return result


def _decode_pcm(audio: Path, sample_rate: int = 16000):
    """Decode any audio file to a float32 mono array using ffmpeg.

    faster-whisper's own file loader goes through PyAV, and the `av` build in
    this venv no longer accepts the `metadata_errors` argument that
    faster-whisper 1.2.1 still passes, so opening a *path* raises
    "TypeError: open() got an unexpected keyword argument 'metadata_errors'".
    WhisperX already decodes with ffmpeg; the fallback path does the same and
    hands the model an in-memory array, which needs no PyAV at all.
    """
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-f", "f32le",
         "-ac", "1", "-ar", str(sample_rate), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {audio}: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _faster_whisper(audio: Path, cfg) -> Dict[str, Any]:
    from faster_whisper import WhisperModel

    log.info(f"faster-whisper loading model={cfg.whisper_model}")
    model = WhisperModel(
        cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute,
        cpu_threads=int(getattr(cfg, "whisper_threads", 0) or 0),
    )
    pcm = _decode_pcm(audio, sample_rate=cfg.sample_rate)
    segments, info = model.transcribe(
        pcm,
        word_timestamps=True,
        vad_filter=True,
        language=cfg.whisper_language,
    )

    out = {"language": info.language, "segments": []}
    for s in segments:
        seg = {
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [],
        }
        if s.words:
            for w in s.words:
                seg["words"].append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                })
        out["segments"].append(seg)
    return out


def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    segs = []
    for s in result.get("segments", []):
        segs.append({
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "text": (s.get("text") or "").strip(),
            "speaker": s.get("speaker"),
            "words": [
                {
                    "word": w.get("word", ""),
                    "start": float(w.get("start", 0) or 0),
                    "end": float(w.get("end", 0) or 0),
                }
                for w in (s.get("words") or [])
            ],
        })
    return {"language": result.get("language", "en"), "segments": segs}


def _fmt_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(transcript: Dict[str, Any], path: Path):
    lines = []
    for i, seg in enumerate(transcript["segments"], 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        lines.append(speaker + seg["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
