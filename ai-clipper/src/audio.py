from pathlib import Path
from .utils import run, log


def extract_audio(video: Path, out_wav: Path, sample_rate=16000, channels=1) -> Path:
    if out_wav.exists() and out_wav.stat().st_size > 0:
        log.info(f"Audio already extracted: {out_wav.name}")
        return out_wav
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting audio → {out_wav.name}")
    run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    return out_wav
