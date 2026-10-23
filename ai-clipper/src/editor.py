from pathlib import Path
from typing import Dict, Any, List
from .utils import run, log


def cut_clips(video: Path, edl: Dict[str, Any], clips_dir: Path, cfg) -> List[Path]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for i, c in enumerate(edl["clips"], 1):
        out = clips_dir / f"clip_{i:02d}.mp4"
        if out.exists() and out.stat().st_size > 0:
            log.info(f"Clip exists: {out.name}")
            outputs.append(out)
            continue

        start = c["start"]
        dur = c["end"] - c["start"]
        log.info(f"Cutting clip {i}: {start:.2f}s -> {c['end']:.2f}s ({dur:.1f}s)")

        if cfg.cut_mode == "lossless" and cfg.fade_seconds == 0.0:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video),
                "-t", f"{dur:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(out),
            ]
        else:
            vf = []
            if cfg.fade_seconds > 0:
                vf.append(f"fade=t=in:st=0:d={cfg.fade_seconds}")
                vf.append(
                    f"fade=t=out:st={max(0, dur - cfg.fade_seconds):.3f}"
                    f":d={cfg.fade_seconds}"
                )
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video),
                "-t", f"{dur:.3f}",
            ]
            if vf:
                cmd += ["-vf", ",".join(vf), "-af",
                        f"afade=t=in:st=0:d={cfg.fade_seconds},"
                        f"afade=t=out:st={max(0, dur - cfg.fade_seconds):.3f}"
                        f":d={cfg.fade_seconds}"]
            cmd += [
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                str(out),
            ]

        run(cmd)
        outputs.append(out)

    return outputs


def merge_clips(clips: List[Path], out_path: Path, workdir: Path, cfg):
    if not clips:
        raise RuntimeError("No clips to merge")

    list_file = workdir / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c.resolve().as_posix()}'\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.cut_mode == "lossless" and cfg.fade_seconds == 0.0:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ]
        try:
            run(cmd)
            log.ok(f"Merged (lossless) -> {out_path}")
            return
        except RuntimeError as e:
            log.warn(f"Lossless concat failed, re-encoding. ({e})")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    run(cmd)
    log.ok(f"Merged -> {out_path}")
