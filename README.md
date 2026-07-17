# AV LangGraph Pipeline

An agentic data-curation pipeline that removes a target sounding object from a
short audio-visual clip — in **both the video and the audio** — and keeps only
the samples where the removal is clean. Built on [LangGraph](https://github.com/langchain-ai/langgraph):
each stage is a graph node, quality gates route to `discard_sample`, and every
stage uses a real pretrained model run out-of-process in its own conda env.

## What it does

Given an 8-second AV clip, the pipeline:

1. captions what is seen and heard,
2. picks the sounding object to remove,
3. segments it,
4. erases it from the video (inpainting),
5. verifies it's actually gone (re-segment + compare),
6. removes its sound (two SAM-Audio passes),
7. emits a "paired AV output" if every gate passed, else discards the clip.

The output is `data/logs/state.jsonl` — one line per clip with all artifact
paths and a `status` (`passed` clips are the curated set).

## Pipeline (graph)

```
 AV Pair
   │
   ▼
 av_caption_generation        Qwen3-Omni  (caption with audio)
   │
   ▼
 sounding_object_extraction   GPT-4o-mini ──► [objects found?] ──no──► discard
   │ yes
   ▼
 target_object_segmentation   SAM3 ──► [mask > 15%?] ──no──► discard
   │ yes
   ▼
 effect_erase_inpainting      EffectErase (Wan2.1 + LoRA)
   │
   ▼
 inpainted_video_check        SAM3 re-seg ──► [removal > 80%?] ──no──► discard
   │ yes
   ▼
 samaudio_remove_target       SAM-Audio (mask-conditioned)  → residual_1, target_1
   │
   ▼
 samaudio_text_remove         SAM-Audio (text-conditioned)  → residual_2, target_2
   │
   ▼
 audio_removal_check          ──► [audio > 80%?] ──no──► discard
   │ yes
   ▼
 paired_av_output             status = passed
```

Thresholds: `mask_area_ratio > 0.15`, `removal_ratio > 0.80`, `audio_removal_score > 0.80`.
The removal check re-runs SAM3 on the inpainted video and compares mask-area
fractions against the original mask: `removal_ratio = 1 - inpaint_area / original_area`.

## Architecture

The **orchestration** runs in a lightweight `avgraph` conda env (LangGraph +
OpenAI client, **no torch**). Heavy models live in their own envs and are driven
**out-of-process** via subprocess workers, so their conflicting dependencies
never touch the orchestrator.

| Stage | Model | Env | How | Needs |
|-------|-------|-----|-----|-------|
| caption | Qwen3-Omni-30B | `qwen3omni` | subprocess (`models/qwen3_omni_worker.py`) | GPU |
| object extraction | GPT-4o-mini | `avgraph` | in-process (OpenAI API) | `OPENAI_API_KEY` |
| segmentation + removal verify | SAM3 | `sam3` | subprocess (`models/sam3_worker.py`) | GPU |
| inpainting | EffectErase (Wan2.1-Fun-1.3B-InP + LoRA) | `effecterase` | subprocess (`models/effecterase_worker.py`) | GPU |
| audio removal | SAM-Audio-large | `samaudio311` | subprocess (`models/sam_audio_worker.py`) | GPU |
| audio check | — (mock) | `avgraph` | placeholder | — |

Each `models/<x>_model.py` is a thin, stdlib-only wrapper that builds a command,
launches its worker with that env's python, and parses a JSON result. Workers
set `LD_LIBRARY_PATH` to their env's `lib/` so torchcodec finds FFmpeg.

## Layout

```
av_langgraph_pipeline/
  generate_inputs.py      folder of mp4s -> input jsonl
  preprocess.py           extract audio (wav) from each video
  run.py                  batch runner: jsonl -> graph -> state.jsonl
  main.py                 build_graph() + a single-sample mock demo
  state.py                AVState (the dict threaded through the graph)
  nodes.py                the graph nodes (+ mock/real toggle, per-step tracking)
  routes.py               conditional-edge functions (the gates)
  utils.py                state.jsonl + per-video state writers
  models/
    caption_model.py            / qwen3_omni_worker.py
    object_extraction_model.py
    segmentation_model.py       / sam3_worker.py      (segment + verify modes)
    inpainting_model.py         / effecterase_worker.py
    audio_removal_model.py      / sam_audio_worker.py (mask + text modes)
  pretrained_weight/      qwen3_omni/  sam3/  sam_audio/  inpainting/
  data/
    raw/                  input videos (bundled samples)
    work/                 source_audio/ masks/ inpainted/ audio/ outputs/ state/ captions/
    logs/                 state.jsonl
  tests/                  test_1..5 (mock, CPU)
```

## Usage

```bash
AVPY=/home/weihan.xu/miniconda3/envs/avgraph/bin/python

# 1. list every video in a folder -> input jsonl  {"video_id","video"}
$AVPY generate_inputs.py --video_dir /group2/ct/weihanx/8sec_raw_video --output inputs.jsonl

# 2. extract the audio track from each video  -> inputs_preprocessed.jsonl  {..., "audio"}
$AVPY preprocess.py --input_jsonl inputs.jsonl --audio_dir data/work/source_audio

# 3a. mock run (CPU, login node) — sanity-check the wiring
$AVPY run.py --input_jsonl inputs_preprocessed.jsonl

# 3b. real run (GPU node) — all models live
srun -p sharedp --gres=gpu:1 --pty bash -l
export OPENAI_API_KEY=sk-...
AVGRAPH_USE_REAL_MODELS=1 $AVPY run.py --input_jsonl inputs_preprocessed.jsonl
```

`run.py` threads the video into caption/SAM3/EffectErase and the extracted audio
into SAM-Audio.

### Mock vs. real

`AVGRAPH_USE_REAL_MODELS=1` switches every node from a fast mock (stub paths /
default scores) to its real model. Mock is the default, so the tests and a dry
run work on a login node with no GPU. Knobs:

- `OPENAI_API_KEY` — required for real object extraction.
- `SAM3_FIRST_FRAME_THRESHOLD` (default `0.1`) — SAM3 first-frame gate.
- `INPAINT_NUM_FRAMES` (default `192`) — EffectErase max frames (clamped to the largest valid `4n+1`).

## Output: `state.jsonl`

Per-video state is written to `data/work/state/<video_id>.json` after **every
node** (live progress), and each finished clip is appended as one line to
`data/logs/state.jsonl`:

```json
{
  "video_id": "sample_0001",
  "object_name": "toilet",
  "mask_path": "data/work/masks/sample_0001/sample_0001_mask.mp4",
  "inpainted_video_path": "data/work/inpainted/sample_0001_without_toilet.mp4",
  "mask_residual_audio_path": "...", "mask_target_audio_path": "...",
  "text_residual_audio_path": "...", "text_target_audio_path": "...",
  "paired_av_output_path": "...",
  "status": "passed",
  "discard_stage": null, "discard_reason": null,
  "metrics": { "mask_area_ratio": 0.24, "visual_removal_score": 0.91, "audio_removal_score": 0.9 }
}
```

The curated set is `status == "passed"`; discards keep `discard_stage` /
`discard_reason` so you can see which gate dropped them.

## Tests

```bash
$AVPY tests/test_1_graph_structure.py     # graph compiles, all nodes present
$AVPY tests/test_2_route_behavior.py      # each gate routes correctly
$AVPY tests/test_3_artifact_writing.py    # artifacts written and non-empty
$AVPY tests/test_4_discard_logging.py     # state.jsonl records for discards
$AVPY tests/test_5_one_real_video_e2e.py  # one clip end-to-end (mock)
```

All run on CPU in mock mode.

## Environment & cluster notes

- **Cluster**: Slurm. Login nodes have no GPU — grab one with
  `srun -p sharedp --gres=gpu:1 --pty bash -l`. Compute nodes have internet
  (HF sub-models download/cache on the first real run).
- **`/tmp` is node-local** — driver scripts and logs must live on shared disk.
- **Weights** are local: `pretrained_weight/{qwen3_omni,sam3,sam_audio,inpainting}`;
  EffectErase's base Wan model is at `/group2/ct/weihanx/Wan-AI/Wan2.1-Fun-1.3B-InP`.
- **torch must be cu128**, not cu130 (the cluster driver is CUDA 12.x).
- **torchcodec** needs the env's FFmpeg libs on `LD_LIBRARY_PATH` (the workers
  set this automatically).

## Scaling to large batches

For 100k+ clips: `preprocess.py` is CPU-bound (one ffmpeg per clip) and the real
`run.py` is a large GPU job — shard the jsonl (`split -l`) across array jobs, one
shard per GPU. `state.jsonl` is append-only, so shards can write to separate
files and be concatenated.

## Status

Real & validated: caption, object extraction, segmentation, removal-verify,
inpainting, audio removal (both passes). Mock: `audio_removal_check` (no real
audio verifier wired yet).
