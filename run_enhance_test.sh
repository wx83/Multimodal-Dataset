#!/usr/bin/env bash
#SBATCH --job-name=avenhance_driver
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/enhance_test/driver.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/enhance_test/driver.%N.%j.log
# Integration test: AVEnhanceModel.enhance() — the exact code path the
# av_quality_enhancement graph node uses (ffmpeg conform+mux, LTX-2 worker,
# result parsing), driven on the already-passed 9NcjRFu6C4I artifacts.
# The 22B DiT loads ~46 GB bf16 on one H100; if it OOMs, resubmit with
# AVENHANCE_OFFLOAD_MODE=cpu exported below.
set -euo pipefail

cd /group2/ct/weihanx/av_langgraph_pipeline
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export AVENHANCE_OFFLOAD_MODE=cpu   # uncomment if the default (none) run OOMs

/home/weihan.xu/miniconda3/envs/avgraph/bin/python - <<'EOF'
from models.av_enhance_model import AVEnhanceModel

m = AVEnhanceModel(mock=False)
res = m.enhance(
    inpainted_video_path="data/work/inpainted/9NcjRFu6C4I_without_train engine.mp4",
    residual_audio_path="data/work/audio/9NcjRFu6C4I/best_of/9NcjRFu6C4I_best_residual.wav",
    sample_id="9NcjRFu6C4I_enh_test",
    target_object="train engine",
)
print("ENHANCE_DRIVER_OK", res)
assert res.frames == 185, f"expected 185 conformed frames, got {res.frames}"
assert res.audio_enhanced, "audio was not enhanced"
EOF

# Post-checks: both mp4s must carry video (185f) + audio streams, and the
# enhanced audio must differ from the mux input (proves both modalities changed).
FFPROBE=/group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/ffprobe
FFMPEG=/group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/ffmpeg
PAIRED="data/work/enhanced/9NcjRFu6C4I_enh_test/paired_input.mp4"
ENHANCED="data/work/enhanced/9NcjRFu6C4I_enh_test/9NcjRFu6C4I_enh_test_enhanced.mp4"

for f in "$PAIRED" "$ENHANCED"; do
  echo "--- $f"
  "$FFPROBE" -v error -select_streams v:0 -count_frames \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames -of default=nw=1 "$f"
  "$FFPROBE" -v error -select_streams a:0 \
    -show_entries stream=codec_name,sample_rate,channels -of default=nw=1 "$f"
done

echo "audio md5 paired_input : $("$FFMPEG" -v error -i "$PAIRED" -map 0:a:0 -f md5 -)"
echo "audio md5 enhanced     : $("$FFMPEG" -v error -i "$ENHANCED" -map 0:a:0 -f md5 -)"
echo "ENHANCE_TEST_DONE"
