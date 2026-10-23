from pathlib import Path
from typing import List, Dict, Any
from .utils import write_json, read_json, log, ffprobe_duration


def detect_scenes(video: Path, workdir: Path, cfg) -> List[Dict[str, Any]]:
    path = workdir / "scenes.json"
    if path.exists():
        log.info("Scenes cached")
        return read_json(path)

    if not getattr(cfg, "scene_detect", True):
        # Nothing is written: leaving no scenes.json means re-enabling detection
        # later actually re-detects instead of reusing an empty cache.
        log.info(
            "Scene detection disabled (SCENE_DETECT=false) - clip edges keep "
            "their transcript timestamps and are not snapped to cuts."
        )
        return []

    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    class TrackingContentDetector(ContentDetector):
        """ContentDetector that also remembers the biggest frame-to-frame change.

        Lets us report *why* nothing was detected (score vs. threshold) without
        paying for a full StatsManager.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_score = 0.0

        def _calculate_frame_score(self, timecode, frame_img):
            score = super()._calculate_frame_score(timecode, frame_img)
            if score is not None and score > self.max_score:
                self.max_score = score
            return score

    frame_skip = int(getattr(cfg, "scene_frame_skip", 0) or 0)
    log.info(
        f"Detecting scenes with PySceneDetect "
        f"(threshold={cfg.scene_threshold}, "
        f"min_scene_len={cfg.scene_min_frames} frames, frame_skip={frame_skip})..."
    )
    v = open_video(str(video))
    detector = TrackingContentDetector(
        threshold=cfg.scene_threshold,
        min_scene_len=cfg.scene_min_frames,
    )
    sm = SceneManager()
    sm.add_detector(detector)
    sm.detect_scenes(v, frame_skip=frame_skip, show_progress=True)
    scene_list = sm.get_scene_list()

    scenes = [
        {"start": s.get_seconds(), "end": e.get_seconds()}
        for s, e in scene_list
    ]

    if scenes:
        log.ok(
            f"Found {len(scenes)} scenes "
            f"(largest frame-change score was {detector.max_score:.1f})"
        )
    else:
        log.warn(
            f"No scene cuts detected. Largest frame-change score was "
            f"{detector.max_score:.1f} vs a threshold of {cfg.scene_threshold} - "
            f"this is normal for a single continuous shot (no hard cuts)."
        )
        interval = float(getattr(cfg, "scene_fallback_interval", 0.0) or 0.0)
        if interval > 0:
            duration = ffprobe_duration(video)
            t = 0.0
            while t < duration:
                scenes.append({"start": round(t, 3),
                               "end": round(min(t + interval, duration), 3)})
                t += interval
            log.warn(
                f"Generated {len(scenes)} evenly spaced fallback boundaries "
                f"every {interval:.0f}s so clip edges can still snap. "
                f"If you expected real cuts, lower 'scenes.threshold' in "
                f"config.yaml (it is currently {cfg.scene_threshold}; try 15)."
            )

    write_json(path, scenes)
    return scenes
