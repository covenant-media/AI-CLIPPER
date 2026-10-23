"""Which language is really spoken? Detects it away from a musical intro.

Usage: python probe_lang.py "video clippings/x.mp4" [start_sec] [duration_sec]

Whisper/WhisperX detect the language from the *first 30s* of audio, so a video
that opens with music or singing gets mislabelled (an English sermon was once
tagged "yo"). This takes a slice from the middle of the video and prints the
auto-detected language plus the transcript it produces with that language and
with a forced WHISPER_LANGUAGE, so you can pick the right value in seconds
instead of after a 13-minute Stage 2.
"""
import itertools
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    video = Path(sys.argv[1])
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "slice.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur),
             "-i", str(video), "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )

        from faster_whisper import WhisperModel

        # Decode with ffmpeg rather than letting faster-whisper open the file:
        # its PyAV loader breaks with the `av` build installed here.
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(wav), "-f", "f32le",
             "-ac", "1", "-ar", "16000", "-"],
            stdout=subprocess.PIPE, check=True,
        ).stdout
        import numpy as np

        pcm = np.frombuffer(raw, dtype=np.float32)

        model = WhisperModel("small", device="cpu", compute_type="int8",
                             cpu_threads=8)
        for lang in (None, sys.argv[4] if len(sys.argv) > 4 else "en"):
            t = time.time()
            segs, info = model.transcribe(
                pcm, language=lang, vad_filter=True, beam_size=1,
                condition_on_previous_text=False,
            )
            print(f"\nlanguage={lang or 'auto'} -> detected={info.language} "
                  f"({info.language_probability:.2f}), probe took "
                  f"{time.time() - t:.0f}s")
            # Decoding is lazy: only the first few segments are pulled, so the
            # probe stays fast even on a long slice.
            t = time.time()
            text = " ".join(s.text.strip() for s in itertools.islice(segs, 6))
            print(f"  first segments ({time.time() - t:.0f}s): {text[:300]}")


if __name__ == "__main__":
    main()
