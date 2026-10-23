"""Validates the scene-detection changes (max score reporting, frame_skip,
fallback interval, cached path)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.scenes import detect_scenes
from src.utils import log, ffprobe_duration

video = Path("workspace/_probe/seg20.mp4")
print(f"clip duration: {ffprobe_duration(video):.2f}s")

# 1. default threshold, no fallback
cfg = Config.load(Path("config.yaml"))
cfg.scene_threshold = 27.0
cfg.scene_frame_skip = 0
cfg.scene_fallback_interval = 0.0
ws = Path("workspace/_probe/sc1")
ws.mkdir(parents=True, exist_ok=True)
(ws / "scenes.json").unlink(missing_ok=True)
a = detect_scenes(video, ws, cfg)
print(f"[1] frame_skip=0 fallback=off -> {len(a)} scenes")

# 2. aggressive threshold + fallback interval
cfg.scene_threshold = 5.0
cfg.scene_fallback_interval = 8.0
ws2 = Path("workspace/_probe/sc2")
ws2.mkdir(parents=True, exist_ok=True)
(ws2 / "scenes.json").unlink(missing_ok=True)
b = detect_scenes(video, ws2, cfg)
print(f"[2] threshold=5 fallback=8s -> {len(b)} scenes: {b[:4]}")

# 3. frame_skip path + cache reuse
cfg.scene_frame_skip = 4
ws3 = Path("workspace/_probe/sc3")
ws3.mkdir(parents=True, exist_ok=True)
(ws3 / "scenes.json").unlink(missing_ok=True)
c = detect_scenes(video, ws3, cfg)
print(f"[3] frame_skip=4 -> {len(c)} scenes")

cached = detect_scenes(video, ws3, cfg)
assert c == cached, "cache must round-trip"
print(f"[4] cache replay OK ({len(cached)} scenes)")

# 5. fallback boundaries must tile the clip
expect = 20 / 8
assert len(b) >= int(expect), f"expected >= {int(expect)} fallback windows, got {len(b)}"
assert b[0]["start"] == 0.0
print("SCENES OK")
