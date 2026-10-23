import os
from datetime import datetime
from pathlib import Path
from .config import Config
from .utils import log, ffprobe_duration, Timer, read_json
from .audio import extract_audio
from .transcribe import transcribe
from .scenes import detect_scenes
from .ai import select_highlights
from .editor import cut_clips, merge_clips


def _log_run_config(cfg: Config) -> None:
    """Print the settings this run will really use - and the code it runs.

    Two questions have cost real debugging time on this project: "was my .env
    picked up?" (a run that silently used large-v3/pyannote) and "is this
    process running my edits?" (a long-lived process started before the fix).
    Both are answered in one block, before any heavy work starts.
    """
    src_dir = Path(__file__).resolve().parent
    try:
        newest = max(src_dir.rglob("*.py"), key=lambda p: p.stat().st_mtime)
        code = (f"{newest.name} @ "
                f"{datetime.fromtimestamp(newest.stat().st_mtime):%m-%d %H:%M}")
    except ValueError:
        code = "unknown"

    cpus = os.cpu_count()
    threads = cfg.whisper_threads or (cpus or 4)
    log.info(
        f"Config: {cfg.config_path or 'config.yaml MISSING'}  |  "
        f".env: {cfg.env_path or 'NOT FOUND - shell + code defaults only'}\n"
        f"  Code: {code}  (newest source file)\n"
        f"  Whisper: {cfg.whisper_model} on {cfg.whisper_device}/"
        f"{cfg.whisper_compute}, language={cfg.whisper_language or 'auto'}, "
        f"vad={cfg.whisper_vad_method}, threads={threads}/{cpus}, "
        f"align={cfg.align_model or 'default'}\n"
        f"  Scenes: detect={cfg.scene_detect}, threshold={cfg.scene_threshold}, "
        f"frame_skip={cfg.scene_frame_skip}, "
        f"fallback_interval={cfg.scene_fallback_interval:g}s  |  "
        f"Edit: {cfg.cut_mode}, snap={cfg.snap_seconds}s\n"
        f"  AI: provider={cfg.ai_provider}, model={cfg.ai_model or '(unset)'}, "
        f"fallbacks={len(cfg.ai_fallback_models)}, "
        f"heuristic_fallback={cfg.ai_heuristic_fallback}, "
        f"ANTHROPIC_BASE_URL={os.getenv('ANTHROPIC_BASE_URL') or '(default)'}"
    )


class Pipeline:
    def __init__(self, cfg: Config, video: Path):
        self.cfg = cfg
        self.video = video.resolve()
        self.workdir = Path("workspace") / self.video.stem
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.workdir / "clips"
        self.output_dir = Path("output")

    def run(self, skip_transcribe=False, skip_scenes=False,
            skip_ai=False, skip_edit=False, do_merge=True):

        log.step(f"AI Clipper - {self.video.name}")
        duration = ffprobe_duration(self.video)
        log.info(f"Source duration: {duration:.1f}s ({duration/60:.1f} min)")
        _log_run_config(self.cfg)

        with Timer("Stage 1: Audio extraction"):
            wav = extract_audio(
                self.video,
                self.workdir / "audio.wav",
                sample_rate=self.cfg.sample_rate,
                channels=self.cfg.channels,
            )

        if skip_transcribe and (self.workdir / "transcript.json").exists():
            transcript = read_json(self.workdir / "transcript.json")
            log.info("Skipping transcription (using cache)")
        else:
            with Timer("Stage 2: Transcription"):
                transcript = transcribe(wav, self.workdir, self.cfg)
        log.info(f"Transcript: {len(transcript['segments'])} segments")

        if skip_scenes and (self.workdir / "scenes.json").exists():
            scenes = read_json(self.workdir / "scenes.json")
            log.info("Skipping scene detection (using cache)")
        else:
            with Timer("Stage 3: Scene detection"):
                scenes = detect_scenes(self.video, self.workdir, self.cfg)

        if skip_ai and (self.workdir / "edl.json").exists():
            edl = read_json(self.workdir / "edl.json")
            log.info("Skipping AI selection (using cache)")
        else:
            with Timer("Stage 4: AI highlight selection"):
                edl = select_highlights(
                    transcript, scenes, self.workdir, self.cfg, duration
                )
        log.info(f"EDL contains {len(edl['clips'])} clips")

        if not edl["clips"]:
            log.warn("No clips selected - stopping.")
            return

        if skip_edit and self.clips_dir.exists():
            clips = sorted(self.clips_dir.glob("clip_*.mp4"))
            log.info(f"Skipping cut (using {len(clips)} cached clips)")
        else:
            with Timer("Stage 5: Cutting clips"):
                clips = cut_clips(self.video, edl, self.clips_dir, self.cfg)

        if do_merge:
            with Timer("Stage 6: Merging final reel"):
                final = self.output_dir / f"{self.video.stem}_final.mp4"
                merge_clips(clips, final, self.workdir, self.cfg)
                log.ok(f"FINAL: {final}")

        log.step("Pipeline complete")
