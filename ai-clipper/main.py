#!/usr/bin/env python3
"""CLI entrypoint for the AI clipper pipeline."""
import argparse
import sys
from pathlib import Path

from src.pipeline import Pipeline
from src.config import Config
from src.utils import log


def main():
    ap = argparse.ArgumentParser(description="Local AI Clipping Automation")
    ap.add_argument("video", type=Path, help="Path to source video")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--skip-transcribe", action="store_true")
    ap.add_argument("--skip-scenes", action="store_true")
    ap.add_argument("--skip-ai", action="store_true")
    ap.add_argument("--skip-edit", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--provider", choices=["auto", "claude", "openai", "gemini"],
                    default=None)
    args = ap.parse_args()

    if not args.video.exists():
        log.error(f"Video not found: {args.video}")
        sys.exit(1)

    cfg = Config.load(args.config)
    if args.provider:
        cfg.ai_provider = args.provider

    pipe = Pipeline(cfg, args.video)
    pipe.run(
        skip_transcribe=args.skip_transcribe,
        skip_scenes=args.skip_scenes,
        skip_ai=args.skip_ai,
        skip_edit=args.skip_edit,
        do_merge=not args.no_merge,
    )


if __name__ == "__main__":
    main()
