#!/usr/bin/env bash
#SBATCH --job-name=avagent_demo
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/demo/demo.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/demo/demo.%N.%j.log
# Live Gradio demo of the full agent. The public share URL appears in the log
# (line "Running on public URL: https://….gradio.live"); it dies with the job.
set -euo pipefail

cd /group2/ct/weihanx/av_langgraph_pipeline
export AVGRAPH_USE_REAL_MODELS=1
export SAM3_FIRST_FRAME_THRESHOLD=0.05
export MASK_AREA_THRESHOLD=0.05
export VISUAL_SCORE_THRESHOLD=0.40
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIO_ANALYTICS_ENABLED=False

/home/weihan.xu/miniconda3/envs/avgraph/bin/python demo_app.py
