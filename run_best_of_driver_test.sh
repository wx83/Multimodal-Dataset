#!/usr/bin/env bash
#SBATCH --job-name=bestof_driver
#SBATCH --partition=sharedp_l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/best_of_test/driver.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/best_of_test/driver.%N.%j.log
# Integration test: AudioRemovalModel.remove_best_of() — the exact code path the
# samaudio_best_of_remove graph node uses (both subprocess stages + parsing).
set -euo pipefail

cd /group2/ct/weihanx/av_langgraph_pipeline
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true

/home/weihan.xu/miniconda3/envs/avgraph/bin/python - <<'EOF'
from models.audio_removal_model import AudioRemovalModel

m = AudioRemovalModel(mock=False)
res = m.remove_best_of(
    av_pair_path="/group2/ct/weihanx/8sec_raw_video/9NcjRFu6C4I.mp4",
    target_object="train engine",
    sample_id="9NcjRFu6C4I_driver",
    mask_path="/group2/ct/weihanx/ranking_test/9NcjRFu6C4I/mask.mp4",
)
print("BEST_OF_DRIVER_OK", res)
EOF
