#!/usr/bin/env bash
#SBATCH --job-name=e2e_showcase
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/e2e_batch/e2e.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/e2e_batch/e2e.%N.%j.log
# Batch E2E for the HF showcase: sbatch run_e2e_batch.sh <sample_id> ...
# Same dev thresholds as run_e2e_9NcjRFu6C4I.sh.
set -euo pipefail

cd /group2/ct/weihanx/av_langgraph_pipeline
export AVGRAPH_USE_REAL_MODELS=1
export SAM3_FIRST_FRAME_THRESHOLD=0.05
export MASK_AREA_THRESHOLD=0.05
export VISUAL_SCORE_THRESHOLD=0.40
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/weihan.xu/miniconda3/envs/avgraph/bin/python _e2e_showcase_batch.py "$@"
