#!/usr/bin/env bash
#SBATCH --job-name=bestof_samaudio
#SBATCH --partition=sharedp_l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/best_of_test/%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/best_of_test/%N.%j.log
# Single-sample test of SAM-Audio best-of-10 seed selection (text + visual prompts):
# stage A generates 2 methods x 5 seeds in the samaudio311 env, stage B scores all
# candidates with ImageBind (ib_ta max, ib_ta_res tiebreak) in the javisdit env.
set -euo pipefail

SAMPLE=9NcjRFu6C4I
PROMPT="train engine"
SEEDS=1132891577,1778986134,240868205,1453635084,1335522078
VIDEO=/group2/ct/weihanx/8sec_raw_video/$SAMPLE.mp4
MASK=/group2/ct/weihanx/ranking_test/$SAMPLE/mask.mp4
OUT_DIR=/group2/ct/weihanx/av_langgraph_pipeline/data/work/best_of_test/$SAMPLE
PIPE=/group2/ct/weihanx/av_langgraph_pipeline

SAM_PY=/home/weihan.xu/miniconda3/envs/samaudio311/bin/python
JAV_PY=/home/weihan.xu/miniconda3/envs/javisdit/bin/python
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true
mkdir -p "$OUT_DIR"

echo "=== stage A: best_of generation (samaudio311) on $(hostname) at $(date) ==="
LD_LIBRARY_PATH=/home/weihan.xu/miniconda3/envs/samaudio311/lib:${LD_LIBRARY_PATH:-} \
$SAM_PY "$PIPE/models/sam_audio_worker.py" \
  --mode best_of \
  --video "$VIDEO" \
  --mask "$MASK" \
  --prompt "$PROMPT" \
  --seeds "$SEEDS" \
  --sample_id "$SAMPLE" \
  --output_dir "$OUT_DIR" \
  --out "$OUT_DIR/stageA_result.json"

echo "=== stage B: ImageBind selection (javisdit) at $(date) ==="
cd /group2/ct/weihanx/JavisDiT
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/group2/ct/weihanx/JavisDiT \
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/weihan.xu/miniconda3/envs/javisdit/lib \
$JAV_PY "$PIPE/models/ib_select_worker.py" \
  --candidates "$OUT_DIR/candidates.json" \
  --out "$OUT_DIR/selection.json"

echo "=== finished at $(date) ==="
