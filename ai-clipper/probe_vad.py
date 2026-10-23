"""Times the two WhisperX VAD backends on a real audio slice.

pyannote VAD is threaded through the whole file before transcription starts, so
its cost is a flat additive delay on every run. This measures that delay.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import whisperx  # noqa: E402

# WhisperX hands the VAD a (channel, time) tensor dict; mirror that exactly.
_orig_load = whisperx.load_audio


def load_audio(path):
    return torch.from_numpy(_orig_load(path)).unsqueeze(0)


source = Path(sys.argv[1] if len(sys.argv) > 1 else "workspace/test/audio.wav")
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0

print(f"loading {source} (first {seconds:.0f}s)...")
audio = load_audio(str(source))
audio = audio[:, : int(16000 * seconds)]
print(f"slice: {audio.shape[1] / 16000:.1f}s @16kHz")

from whisperx.vads.pyannote import Pyannote  # noqa: E402


def span(seg):
    """pyannote yields (Segment, speaker); silero yields SegmentX objects."""
    if hasattr(seg, "end"):
        return seg.end - seg.start
    if isinstance(seg, (tuple, list)) and seg and hasattr(seg[0], "end"):
        return seg[0].end - seg[0].start
    return 0.0


def timed(label, fn):
    t0 = time.perf_counter()
    try:
        segs = fn()
    except Exception as e:
        print(f"{label}: FAILED after {time.perf_counter() - t0:.1f}s - {e}",
              flush=True)
        return None
    dt = time.perf_counter() - t0
    segs = list(segs or [])
    print(f"{label}: {dt:.1f}s wall for {seconds:.0f}s of audio "
          f"({dt / seconds * 100:.0f}% of realtime)", flush=True)
    try:
        print(f"{label}: {len(segs)} segments, {sum(map(span, segs)):.0f}s speech",
              flush=True)
    except Exception as e:  # noqa: BLE001 - diagnostics must never hide the time
        print(f"{label}: (segment stats unavailable: {e})", flush=True)
    return dt


py = Pyannote(device="cpu", token=None, vad_onset=0.5, vad_offset=0.363,
              chunk_size=30)
d_py = timed("pyannote", lambda: py({"waveform": audio, "sample_rate": 16000}))

print("building silero (downloads from torch.hub on first use)...")
from whisperx.vads.silero import Silero  # noqa: E402

try:
    si = Silero(vad_onset=0.5, vad_offset=0.363, chunk_size=30)
except Exception as e:
    print(f"silero unavailable: {e}")
    raise SystemExit(0)
d_si = timed("silero  ", lambda: si({"waveform": audio, "sample_rate": 16000}))

if d_py and d_si:
    print(f"speedup: {d_py / d_si:.1f}x faster with silero")
    print(f"extrapolated to {seconds:.0f}s of audio: "
          f"pyannote {d_py:.1f}s vs silero {d_si:.1f}s")
print("VAD OK")
