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
6. removes its sound (best-of-10 SAM-Audio separation, ImageBind-ranked),
7. jointly refines the video+audio pair with the LTX-2 denoising enhancer,
8. emits a "paired AV output" if every gate passed, else discards the clip.

The output is `data/logs/state.jsonl` — one line per clip with all artifact
paths and a `status` (`passed` clips are the curated set).

**Showcase**: stage-by-stage examples (successes and gate rejections) at
<https://huggingface.co/spaces/WitneyWW/av-agent-pipeline-stages>.

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
 samaudio_best_of_remove      SAM-Audio best-of-10 (visual+text × 5 seeds),
   │                          ImageBind-ranked → best residual + target
   ▼
 audio_removal_check          ──► [audio > 80%?] ──no──► discard
   │ yes
   ▼
 av_quality_enhancement       LTX-2.3 22B: mux inpainted video + residual audio,
   │                          SDEdit-refine BOTH modalities jointly
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
| audio removal | SAM-Audio-large (best-of-10) | `samaudio311` | subprocess (`models/sam_audio_worker.py --mode best_of`) | GPU |
| audio candidate ranking | ImageBind (JavisDiT) | `javisdit` | subprocess (`models/ib_select_worker.py`) | GPU |
| AV enhancement | LTX-2.3 22B distilled + Gemma-3-12B | LTX-2 uv venv | subprocess (`models/ltx_enhance_worker.py`) | GPU (80 GB, or `AVENHANCE_OFFLOAD_MODE=cpu`) |
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
    audio_removal_model.py      / sam_audio_worker.py (mask/text/best_of modes)
                                / ib_select_worker.py (ImageBind ranking)
    av_enhance_model.py         / ltx_enhance_worker.py (LTX-2 joint AV enhance)
  pretrained_weight/      qwen3_omni/  sam3/  sam_audio/  inpainting/
  data/
    raw/                  input videos (bundled samples)
    work/                 source_audio/ masks/ inpainted/ audio/ outputs/ state/ captions/
    logs/                 state.jsonl
  tests/                  test_1..5 (mock, CPU)
```

## Using the agent on GPUs

Everything heavy runs on a GPU node; the orchestrator itself is CPU-only.
One H100/A100 (80 GB) covers every stage sequentially. A full real pass takes
roughly 20–40 min per clip (caption ~10 min, inpaint ~10 min, best-of-10 audio
~10 min, LTX-2 enhance ~7 min incl. the 46 GB checkpoint load).

### 1. Prepare inputs (CPU, login node)

```bash
AVPY=/home/weihan.xu/miniconda3/envs/avgraph/bin/python

# list every video in a folder -> input jsonl  {"video_id","video"}
$AVPY generate_inputs.py --video_dir /group2/ct/weihanx/8sec_raw_video --output inputs.jsonl

# extract the audio track from each video  -> inputs_preprocessed.jsonl  {..., "audio"}
$AVPY preprocess.py --input_jsonl inputs.jsonl --audio_dir data/work/source_audio
```

### 2. Sanity-check the wiring (CPU, no GPU needed)

```bash
$AVPY run.py --input_jsonl inputs_preprocessed.jsonl   # mock mode is the default
```

### 3. Real run on a GPU node

Interactive:

```bash
srun -p sharedp --gres=gpu:h100:1 --pty bash -l
export OPENAI_API_KEY=sk-...            # or pin targets, see _e2e_showcase_batch.py
export AVGRAPH_USE_REAL_MODELS=1
$AVPY run.py --input_jsonl inputs_preprocessed.jsonl
```

Batch via SLURM (recommended — see the `run_*.sh` scripts for ready-made
`sbatch` templates):

- `run_e2e_batch.sh <id> [<id> ...]` — full pipeline for a list of samples
  (uses cached captions + pinned targets from `_e2e_showcase_batch.py`).
- `run_enhance_test.sh` — the LTX-2 enhancement stage alone, on existing artifacts.
- `run_best_of_driver_test.sh` — the best-of-10 audio stage alone.

All of them log to `slurm-logs/` and print machine-parsable `SAMPLE_RESULT`
lines.

### Mock vs. real

`AVGRAPH_USE_REAL_MODELS=1` switches every node from a fast mock (stub paths /
default scores) to its real model. Mock is the default, so the tests and a dry
run work on a login node with no GPU.

### Gate thresholds (env vars)

| Var | Production | Dev runs here | Meaning |
|---|---|---|---|
| `SAM3_FIRST_FRAME_THRESHOLD` | 0.80 (intended) | 0.05 | first-frame fail-fast before full segmentation |
| `MASK_AREA_THRESHOLD` | 0.15 | 0.05 | full mask must cover this frame fraction |
| `VISUAL_SCORE_THRESHOLD` | 0.80 | 0.40 | SAM3 re-seg removal score after inpainting |

Other knobs: `OPENAI_API_KEY` (real object extraction),
`INPAINT_NUM_FRAMES` (default `192`), and the LTX-2 enhancement set
`AVENHANCE_DENOISE_STRENGTH` (default `0.4219`, the lightest step on the
distilled sigma grid), `AVENHANCE_AUDIO_DENOISE_STRENGTH`,
`AVENHANCE_FINE_STEPS`, `AVENHANCE_OFFLOAD_MODE=cpu` (GPUs < 60 GB).

### Running on a different host

Every interpreter/weight/repo path has an env-var override
(`SAM3_PYTHON`, `LTX_CKPT`, `AVGRAPH_FFMPEG`, …) — see `SELF_HOSTING.md` for
the full table and per-model environment requirements.

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
inpainting, best-of-10 audio removal + ImageBind ranking, LTX-2 joint AV
enhancement. Mock: `audio_removal_check` (no real audio verifier wired yet —
quality control on audio currently comes from the ImageBind tournament).
