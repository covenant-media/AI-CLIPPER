import sys
import numpy as np
from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector

path = sys.argv[1]
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 27.0

stats = StatsManager()
v = open_video(path)
sm = SceneManager(stats_manager=stats)
sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))
n = sm.detect_scenes(v, show_progress=False)
scenes = sm.get_scene_list()

vals = [stats.get_metrics(i, ["content_val"])[0] for i in range(1, 1 + n)]
vals = [x for x in vals if x is not None]
print("frames processed:", n)
print("video fps:", v.frame_rate)
print("max content_val:", max(vals) if vals else None)
print("mean content_val:", float(np.mean(vals)) if vals else None)
print("p99 content_val:", float(np.percentile(vals, 99)) if vals else None)
print("num frames over threshold", threshold, ":", sum(1 for x in vals if x >= threshold))
print("scenes:", len(scenes))
