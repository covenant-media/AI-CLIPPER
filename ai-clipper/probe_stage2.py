"""A/B the WHISPER_THREADS setting on one already-extracted audio file.

Usage: python probe_stage2.py <threads> [audio.wav]
WhisperX hard-codes threads=4, so this shows what the extra cores are worth.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.transcribe import transcribe

threads = int(sys.argv[1])
audio = Path(sys.argv[2] if len(sys.argv) > 2 else "workspace/test_short/audio.wav")

cfg = Config.load(Path("config.yaml"))
cfg.whisper_threads = threads
workdir = Path(f"workspace/_probe/stage2_t{threads}")
shutil.rmtree(workdir, ignore_errors=True)
workdir.mkdir(parents=True, exist_ok=True)

print(f"--- threads={threads} model={cfg.whisper_model} "
      f"vad={cfg.whisper_vad_method} compute={cfg.whisper_compute} ---",
      flush=True)
t0 = time.perf_counter()
result = transcribe(audio, workdir, cfg)
dt = time.perf_counter() - t0
print(f"RESULT threads={threads}: {dt:.1f}s wall, "
      f"{len(result['segments'])} segments, language={result['language']}",
      flush=True)
