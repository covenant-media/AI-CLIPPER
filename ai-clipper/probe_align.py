"""Validates transcribe._align: unsupported languages skip cleanly, supported
ones run, ALIGN_MODEL overrides, and failures never propagate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.transcribe import _align

TORCH = {"en": "WAV2VEC2_ASR_BASE_960H", "fr": "VOXPOPULI_ASR_BASE_10K_FR"}
HF = {"ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"}

calls = []
loads = []


class FakeWhisperX:
    def load_align_model(self, language_code, device, model_name=None):
        calls.append(("load", language_code, model_name))
        loads.append(model_name)
        if model_name == "boom":
            raise RuntimeError("bad model")
        return object(), {"chars": ["a"]}

    def align(self, segments, model, meta, audio, device, return_char_alignments=False):
        calls.append(("align", language_code_seen[-1]))
        return {"segments": [{"text": "hi", "words": []}], "word_segments": []}


language_code_seen = []
fake = FakeWhisperX()
cfg = Config.load(Path("config.yaml"))
cfg.align_model = None


def run(lang, override=None):
    cfg.align_model = override
    result = {"language": lang, "segments": [{"start": 0, "end": 1, "text": "x"}]}
    language_code_seen.append(lang)
    return _align(fake, result, object(), "cpu", cfg, TORCH, HF)


# 1. unsupported language -> unchanged result, no model load attempted
out = run("yo")
assert out["language"] == "yo" and "word_segments" not in out, out
assert not loads, loads
print("[1] unsupported 'yo' skipped cleanly, original result returned")

# 2. torch-map language -> aligned, language key preserved
out = run("en")
assert out["language"] == "en", out
assert out["word_segments"] == []
print("[2] 'en' aligned via", loads[-1])

# 3. hf-map language
out = run("ja")
assert loads[-1] == HF["ja"] and out["language"] == "ja"
print("[3] 'ja' aligned via", loads[-1])

# 4. explicit override for an unsupported language
out = run("yo", override="chrisjay/wav2vec2-large-xlsr-53-yoruba")
assert loads[-1] == "chrisjay/wav2vec2-large-xlsr-53-yoruba"
assert out["language"] == "yo"
print("[4] ALIGN_MODEL override honoured for 'yo'")

# 5. align failure -> falls back to the unaligned result instead of raising
out = run("en", override="boom")
assert out["segments"] and "word_segments" not in out
print("[5] alignment failure degraded gracefully")

print("ALIGN OK")
