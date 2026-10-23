# AI Clipper — Local Clipping Automation

Transcribes long videos, asks an AI to select highlights, then uses FFmpeg
to cut and merge them into a final reel.

## Quickstart

    bash setup.sh
    cp .env.example .env      # add your API keys
    bash run.sh path/to/video.mp4

## Pipeline

1. Extract audio (FFmpeg)
2. Transcribe (WhisperX / faster-whisper)
3. Scene detect (PySceneDetect)
4. AI selection (Claude / GPT / Gemini) → EDL JSON
5. Cut clips (FFmpeg lossless)
6. Merge final reel (FFmpeg concat)

Outputs land in workspace/<video>/ and output/<video>_final.mp4.
All stages cache — delete a file to re-run just that stage.

## Configuration

Settings are read from **`.env`** first (a non-blank value there overrides your
shell environment), then `config.yaml`, then built-in defaults.
Run `python main.py video.mp4 --provider claude` to force one provider.

See `.env.example` for every switch. The ones that matter most:

| Variable | Default | Why |
| --- | --- | --- |
| `AI_MODEL` | provider default | A model **your endpoint actually serves**. |
| `AI_FALLBACK_MODELS` | – | Comma separated models tried in order. |
| `AI_MAX_RETRIES` | `0` | `0` = on 401/404/429/5xx jump straight to the next model. |
| `AI_HEURISTIC_FALLBACK` | `true` | Pick clips locally if every AI target fails. |
| `WHISPER_MODEL` | `small` | `small` is roughly 10x faster than `large-v3` on CPU. |
| `WHISPER_LANGUAGE` | auto | Leave blank: pinning the wrong code makes Whisper *translate*. |
| `WHISPER_BATCH_SIZE` | `16` | Segments per decode batch. Lower = less RAM. |
| `WHISPER_VAD_METHOD` | `pyannote` | `silero` is much faster on CPU (auto-falls back). |
| `WHISPER_PROGRESS` | `true` | `Progress: N%` line per decoded segment during Stage 2. |
| `WHISPER_THREADS` | `0` (all cores) | WhisperX alone uses only 4 threads. |
| `ALIGN_MODEL` | – | wav2vec2 model for languages WhisperX has no default for. |
| `ALIGN_MODEL=none` | – | Skip word-level alignment entirely (saves a download + CPU pass). |
| `SCENE_FRAME_SKIP` | `0` | Skip frame *processing*; measured no gain on PyAV (see below). |
| `SCENE_DETECT` | `true` | `false` skips Stage 3 entirely (continuous-shot footage). |
| `SCENE_FALLBACK_INTERVAL` | `0` | Split a cut-free timeline every N seconds. |

### Speeding up stages 2 and 3

* `WHISPER_MODEL=small` (or `base`) instead of `large-v3`.
* `WHISPER_LANGUAGE=en` skips the detection pass — only worth it when you know
  the language; a wrong code makes Whisper translate the audio instead.
* `WHISPER_COMPUTE=int8` on CPU; `float16` + `WHISPER_DEVICE=cuda` on a GPU.
* `WHISPER_THREADS=0` (the default here) — WhisperX otherwise hard-codes 4
  decode threads regardless of how many cores you have. Set a lower number if
  you want to keep the machine usable while it runs.
* `WHISPER_VAD_METHOD=silero` — WhisperX runs VAD across the *whole* file
  before transcribing a single segment, and pyannote (the default) is the
  slowest part of Stage 2 on CPU. Silero is dramatically cheaper there. It is
  fetched once from GitHub via `torch.hub`; if that is blocked the run logs a
  warning and continues with pyannote instead of failing.
  * First use is slow and needs network access — delete
    `%USERPROFILE%\.cache\torch\hub` to force a re-download if it goes bad.
* `pip install hf_xet` — Whisper and wav2vec2 weights come from HuggingFace;
  the `xet` transport is several times faster than plain HTTPS. Without it
  Stage 2 prints a "Xet Storage is enabled ... not installed" notice.
* `SCENE_DETECT=false` — the real Stage 3 lever. On single-continuous-shot
  footage (sermons, talking heads, screen recordings) there are no cuts to
  find, so the whole decode is runtime with nothing to show for it: measured
  850s on `20260920_082508-2.mp4` (58 fps, 41055 frames) that produced 0
  scenes. Clip edges keep their transcript timestamps instead of snapping.
* `SCENE_FRAME_SKIP=2` is **not** a speed-up here: PySceneDetect skips frame
  *processing* but the PyAV stream still walks every packet, so the stage took
  the same ~950s and the detector saw 1/3 of the frames. Left at `0`.

## Troubleshooting

**"Which settings did this run actually use?"** The first lines print a
`Config:` block: the `config.yaml` path, the `.env` path (or `NOT FOUND`), the
newest source file it is running, and the resolved Whisper/scene/edit/AI values.
Reach for it first — it exposes both a `.env` that was not picked up and a
long-lived process that was started before your edits.

**Stage 4 dies with "no available channel" / 503 / 401.**
Your endpoint has no channel for the requested model, or rejects the official
SDKs. The run now logs every candidate it tried plus the active base URLs. Fix
whichever applies:

* Point `AI_MODEL` at a model the endpoint serves, and list alternatives in
  `AI_FALLBACK_MODELS`.
* `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` pointing at a relay? Some relays
  only accept requests whose user agent claims to be the Claude Code CLI and
  answer everything else with `401 unauthorized client detected`. Set the base
  URL to a working endpoint in `.env` (or leave it blank to use the official
  API), and remember a *non-blank* `.env` value overrides your shell.
* Nothing works? `AI_HEURISTIC_FALLBACK=true` still produces a reel, and
  `edl.json` records `"source": "heuristic"` so you can tell it apart.

**Stage 4 always falls back to "heuristic".** Expected when no provider is
reachable — check the error block printed just above the fallback line.

**Warnings about `torchcodec`.** Harmless. WhisperX hands pyannote an in-memory
waveform and never calls torchcodec; the check happens at import time. The
warning is filtered. If you want it gone at the source, repair your FFmpeg
shared libraries (`torchcodec` needs FFmpeg 4-7 with DLLs on `PATH`).

**"Skipping alignment: no default wav2vec2 align model for language 'yo'".**
WhisperX only ships default aligners for ~25 languages. Whisper's own segment
timestamps are used instead, which is fine for clipping. To get word-level
timings set `ALIGN_MODEL=chrisjay/wav2vec2-large-xlsr-53-yoruba`.

**"No scene cuts detected ... vs a threshold of 27.0".** Correct for a single
continuous shot. Lower `scenes.threshold` in `config.yaml` (try `15`) to catch
softer transitions, or set `SCENE_FALLBACK_INTERVAL=60` to split the timeline
every 60s so clip edges can still snap.

**`UnicodeEncodeError` / garbled text in the console.** Fixed — console output
is now sanitised for non-UTF-8 code pages. Provider errors often contain
Chinese text, which used to crash rich's legacy Windows renderer.

**Clips are a few seconds longer than `edl.json` says.** That is `cut_mode:
lossless` doing its job: with `-c copy` ffmpeg cannot start mid-GOP, so each cut
begins at the keyframe *before* the requested start (typically 2-10s earlier on
screen recordings). Set `video.cut_mode` in `config.yaml` to anything other than
`lossless` (e.g. `accurate`) to re-encode and get exact boundaries — slower, but
frame-accurate.

**"The log shows a surprise language" / the transcript reads like bad English.**
Whisper detects the language from the *first 30s*, so music or singing at the
start can mislead it. Check the text in `workspace/<video>/transcript.srt`
before "fixing" anything: on this repo's videos the detector is usually right —
`20260920_082508-2.mp4` is Yoruba, and it reports `yo` at 0.87 from the musical
intro *and* at 0.97 from a slice taken mid-video. Pinning the wrong language is
worse than a wrong guess: Whisper then **translates** instead of transcribing,
so forcing `en` on that video replaced the Yoruba transcript with
"better 10,000. So, please, you can support the...". Leave `WHISPER_LANGUAGE`
blank unless you are certain, and use `python probe_lang.py <video> <start> <sec>`
to test a slice from the middle of the media.

**Stage 2 is taking a very long time.** WhisperX's own default is 4 decode
threads no matter how many cores you have; `WHISPER_THREADS=0` (the default
here) gives it every logical core. Measured on a 4-minute clip with
`WHISPER_MODEL=small`/`int8` on an 8-core CPU: **608s → 335s (1.8x)**. Also
`WHISPER_VAD_METHOD=silero` and `WHISPER_LANGUAGE=<code>` cut real time.

