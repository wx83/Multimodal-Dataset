# Self-hosting the AV Data-Curation Agent demo

Run the interactive demo (`demo_app.py`) on **your own GPU server** — your
hardware, your model weights, your OpenAI key. Nothing talks to the authors'
infrastructure.

## What you need

| Component | Role | Weights / repo to obtain |
|---|---|---|
| Qwen3-Omni | AV captioning | Qwen3-Omni checkpoint (HF) |
| GPT (optional) | sounding-object extraction | your `OPENAI_API_KEY`; or type the target in the UI and skip it |
| SAM3 | text-prompted video segmentation + removal verification | SAM3 repo + checkpoint |
| EffectErase | video object-removal inpainting | Wan2.1-Fun-1.3B-InP base + EffectErase LoRA + EffectErase repo |
| SAM-Audio | target-sound separation (best-of-10) | SAM-Audio checkpoint |
| ImageBind (JavisDiT) | ranking the 10 audio candidates | JavisDiT repo + `imagebind_huge.pth` |
| LTX-2.3 | joint AV denoising enhancement | `ltx-2.3-22b-distilled-1.1.safetensors` (~46 GB) + Gemma-3-12B, in the LTX-2 uv workspace |

Hardware: one 80 GB GPU (H100/A100) covers everything, ~20–40 min per clip.
ffmpeg/ffprobe are also required.

Each heavy model runs in its **own Python environment** (conda env or venv) and
is driven as a subprocess by the orchestrator — you don't need one env that
satisfies everything. Create an env per row above following each upstream
repo's install instructions, then point the pipeline at your interpreters and
weights via the env vars below.

## Configuration (env vars)

The orchestrator env (`avgraph`) needs: `langgraph`, `gradio`, `openai`.
All paths default to the authors' cluster; override every one for your host:

```bash
# interpreters (one per model env)
export QWEN3_PYTHON=/path/to/envs/qwen3omni/bin/python
export SAM3_PYTHON=/path/to/envs/sam3/bin/python
export EFFECTERASE_PYTHON=/path/to/envs/effecterase/bin/python
export SAMAUDIO_PYTHON=/path/to/envs/samaudio/bin/python
export JAVISDIT_PYTHON=/path/to/envs/javisdit/bin/python
export LTX_PYTHON=/path/to/LTX-2/.venv/bin/python

# weights & repos
export QWEN3_OMNI_PATH=/path/to/qwen3_omni_weights
export SAM3_MODEL_DIR=/path/to/sam3_weights
export SAM3_REPO=/path/to/sam3_repo
export WAN_BASE_MODEL_DIR=/path/to/Wan2.1-Fun-1.3B-InP
export EFFECTERASE_LORA=/path/to/EffectErase.ckpt
export EFFECTERASE_REPO=/path/to/EffectErase_repo
export SAMAUDIO_MODEL_DIR=/path/to/sam_audio_weights
export JAVISDIT_ROOT=/path/to/JavisDiT          # with ./checkpoints/imagebind_huge.pth
export LTX_CKPT=/path/to/ltx-2.3-22b-distilled-1.1.safetensors
export LTX_GEMMA_ROOT=/path/to/gemma-3-12b

# tools
export AVGRAPH_FFMPEG=$(which ffmpeg)
export AVGRAPH_FFPROBE=$(which ffprobe)

# pipeline behavior
export AVGRAPH_USE_REAL_MODELS=1
export SAM3_FIRST_FRAME_THRESHOLD=0.05   # demo-friendly gates; production uses stricter
export MASK_AREA_THRESHOLD=0.05
export VISUAL_SCORE_THRESHOLD=0.40
```

Optional LTX-2 enhancement knobs: `AVENHANCE_DENOISE_STRENGTH` (default
0.4219), `AVENHANCE_AUDIO_DENOISE_STRENGTH`, `AVENHANCE_FINE_STEPS`,
`AVENHANCE_OFFLOAD_MODE=cpu` (for GPUs < 60 GB).

## Launch

```bash
python demo_app.py
```

Gradio prints a local URL and (if outbound internet is available) a temporary
public `*.gradio.live` share link. To keep the demo private to your network,
edit the last line of `demo_app.py` to `launch(share=False)`.

On a SLURM cluster, `run_demo_server.sh` is a ready-made launcher
(`sbatch run_demo_server.sh`; the URL appears in the job log).

## Notes

- Your OpenAI key: the UI's key field is used for a single `gpt-4o-mini` call
  per request and is not stored or logged. Typing the target object in the UI
  avoids needing a key at all. When self-hosting you can also
  `export OPENAI_API_KEY=...` server-side and leave the field empty.
- Requests are processed one at a time (`concurrency_limit=1`); each full pass
  occupies the GPU for the duration.
- Mock mode (no GPU, no weights): unset `AVGRAPH_USE_REAL_MODELS` and every
  stage returns stubs — useful to check the wiring before downloading weights.
