#!/usr/bin/env bash
#SBATCH --job-name=verify_new_inpaint
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/verify_new/verify.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/verify_new/verify.%N.%j.log
# Score the /group2/ct/weihanx/8_sec_inpainted versions of the two visual-gate
# failures with the pipeline's own SAM3 verify_removal metric, for comparison
# against the recorded EffectErase scores (-0.105 and 0.477).
set -euo pipefail

cd /group2/ct/weihanx/av_langgraph_pipeline
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true

/home/weihan.xu/miniconda3/envs/avgraph/bin/python - <<'EOF'
from models.segmentation_model import SegmentationModel

m = SegmentationModel(mock=False)
CASES = [
    ("--8Bq81udbw", "flowing water", -0.1046),
    ("--BFPeFaj2o", "train", 0.4768),
]
for sid, target, old_score in CASES:
    score = m.verify_removal(
        inpainted_video=f"/group2/ct/weihanx/8_sec_inpainted/{sid}.mp4",
        original_mask=f"data/work/masks/{sid}/{sid}_mask.mp4",
        target_object=target,
        sample_id=f"{sid}_newinpaint",
    )
    print(f"VERIFY_RESULT {sid}: new={score:.4f} vs effecterase={old_score:.4f}", flush=True)
print("VERIFY_DONE")
EOF
