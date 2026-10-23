"""Stage-4 smoke test: proves select_highlights degrades gracefully.

Runs the real selector against a cached transcript using the real environment
(including any ambient ANTHROPIC_BASE_URL). Expects either real AI clips or the
local heuristic fallback - never a crash.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.utils import read_json, ffprobe_duration, log
from src.ai import select_highlights

ws = Path("workspace/20260920_082508-2")
video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.mp4")

transcript = read_json(ws / "transcript.json")
scenes = read_json(ws / "scenes.json")
duration = ffprobe_duration(video)

cfg = Config.load(Path("config.yaml"))
cfg.ai_max_retries = 0

edl_path = ws / "edl.probe.json"
if edl_path.exists():
    edl_path.unlink()

log.info(f"transcript segments: {len(transcript['segments'])}")
log.info(f"scenes: {len(scenes)}")
log.info(f"duration: {duration:.2f}s")

# Work in a scratch dir so we never poison the real cache.
scratch = Path("workspace/_probe")
scratch.mkdir(parents=True, exist_ok=True)
edl = select_highlights(transcript, scenes, scratch, cfg, duration)

log.ok(f"source={edl.get('source')} clips={len(edl['clips'])}")
for c in edl["clips"]:
    log.info(f"  {c['start']:8.3f} -> {c['end']:8.3f} ({c['duration']:6.2f}s) "
             f"score={c['score']} [{c['category']}]")

assert edl["clips"], "selector produced no clips at all"
print("STAGE 4 OK")
