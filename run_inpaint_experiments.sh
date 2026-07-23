#!/usr/bin/env bash
#SBATCH --job-name=inpaint_exp
#SBATCH --partition=sharedp_l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/e2e/inpaint_exp.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/e2e/inpaint_exp.%N.%j.log
# Why does EffectErase fail on 9NcjRFu6C4I, and does the seed matter?
#  A. same SAM3 mask, seed 7 (baseline seed 2025 scored 0.4352) -> verify
#  B. denser mask: SAM3 "blue and yellow locomotive" -> EffectErase -> verify
set -euo pipefail

PIPE=/group2/ct/weihanx/av_langgraph_pipeline
cd "$PIPE"
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true

EE_PY=/group2/ct/weihanx/miniconda3/envs/effecterase/bin/python
SAM3_PY=/group2/ct/weihanx/miniconda3/envs/sam3/bin/python
VIDEO=/group2/ct/weihanx/8sec_raw_video/9NcjRFu6C4I.mp4
MASK_A="$PIPE/data/work/masks/9NcjRFu6C4I/9NcjRFu6C4I_mask.mp4"
EXP="$PIPE/data/work/inpaint_exp"
mkdir -p "$EXP"

echo "=== A: EffectErase seed 7, original mask, $(hostname) $(date) ==="
$EE_PY models/effecterase_worker.py \
  --video "$VIDEO" --mask "$MASK_A" \
  --output_path "$EXP/inpainted_seed7.mp4" --sample_id 9NcjRFu6C4I_seed7 \
  --seed 7 --out "$EXP/seed7_result.json"

echo "=== A: verify seed 7 at $(date) ==="
$SAM3_PY models/sam3_worker.py --mode verify \
  --inpainted_video "$EXP/inpainted_seed7.mp4" --original_mask "$MASK_A" \
  --prompt "train engine" --sample_id 9NcjRFu6C4I_seed7 \
  --model_dir pretrained_weight/sam3 --sam3_repo /group2/ct/weihanx/sam3 \
  --output_dir "$EXP" --out "$EXP/seed7_verify.json"

echo "=== B: SAM3 dense-prompt segmentation at $(date) ==="
$SAM3_PY models/sam3_worker.py \
  --video "$VIDEO" --prompt "blue and yellow locomotive" \
  --sample_id 9NcjRFu6C4I_dense --model_dir pretrained_weight/sam3 \
  --sam3_repo /group2/ct/weihanx/sam3 --output_dir "$EXP/masks_dense" \
  --first_frame_threshold 0.05 --max_seconds 8 --fps 24 --batch_size 4 \
  --out "$EXP/dense_seg.json"

MASK_B="$EXP/masks_dense/9NcjRFu6C4I_dense_mask.mp4"
echo "=== B: EffectErase on dense mask (seed 2025) at $(date) ==="
$EE_PY models/effecterase_worker.py \
  --video "$VIDEO" --mask "$MASK_B" \
  --output_path "$EXP/inpainted_dense.mp4" --sample_id 9NcjRFu6C4I_dense \
  --seed 2025 --out "$EXP/dense_result.json"

echo "=== B: verify dense at $(date) ==="
$SAM3_PY models/sam3_worker.py --mode verify \
  --inpainted_video "$EXP/inpainted_dense.mp4" --original_mask "$MASK_B" \
  --prompt "blue and yellow locomotive" --sample_id 9NcjRFu6C4I_dense \
  --model_dir pretrained_weight/sam3 --sam3_repo /group2/ct/weihanx/sam3 \
  --output_dir "$EXP" --out "$EXP/dense_verify.json"

echo "=== results ==="
for f in seed7_verify dense_seg dense_verify; do
  echo "--- $f"; cat "$EXP/$f.json" 2>/dev/null || true; echo
done
echo "=== finished at $(date) ==="
