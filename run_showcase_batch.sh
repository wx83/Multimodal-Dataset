#!/usr/bin/env bash
#SBATCH --job-name=showcase_batch
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/showcase_batch/batch.%N.%j.log
#SBATCH --error=/group2/ct/weihanx/av_langgraph_pipeline/slurm-logs/showcase_batch/batch.%N.%j.log
# Showcase batch: complete 4 samples for the HF page.
#  - best-of-10 audio removal for the two samples that were discarded before
#    reaching the audio stage (--8Bq81udbw, --BFPeFaj2o)
#  - LTX-2 joint AV enhancement for all four samples
set -euo pipefail

cd /group2/ct/weihanx/av_langgraph_pipeline
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/weihan.xu/miniconda3/envs/avgraph/bin/python - <<'EOF'
import json

from models.audio_removal_model import AudioRemovalModel
from models.av_enhance_model import AVEnhanceModel

RAW = "/group2/ct/weihanx/8sec_raw_video"

# sample_id -> (target_object, inpainted_video, residual_wav or None to compute)
SAMPLES = {
    "--8Bq81udbw": ("flowing water",
                    "data/work/inpainted/--8Bq81udbw_without_flowing water.mp4", None),
    "--BFPeFaj2o": ("train",
                    "data/work/inpainted/--BFPeFaj2o_without_train.mp4", None),
    "--IOqc-XgYo_000000": ("black electronic keyboard",
                           "data/work/inpainted/--IOqc-XgYo_000000_without_black electronic keyboard.mp4",
                           "data/work/audio/--IOqc-XgYo_000000/--IOqc-XgYo_000000_text_residual_black_electronic_keyboard.wav"),
    "--SQyOb8eS0_000030": ("woman",
                           "data/work/inpainted/--SQyOb8eS0_000030_without_woman.mp4",
                           "data/work/audio/--SQyOb8eS0_000030/--SQyOb8eS0_000030_text_residual_woman.wav"),
}

audio_model = AudioRemovalModel(mock=False)
enhance_model = AVEnhanceModel(mock=False)
summary = {}

for sid, (target, inpainted, residual) in SAMPLES.items():
    print(f"\n===== {sid} (target: {target}) =====", flush=True)
    info = {"target": target}
    if residual is None:
        print(f"[{sid}] best-of-10 audio removal...", flush=True)
        res = audio_model.remove_best_of(
            av_pair_path=f"{RAW}/{sid}.mp4",
            target_object=target,
            sample_id=sid,
            mask_path=f"data/work/masks/{sid}/{sid}_mask.mp4",
            audio_path=f"data/work/source_audio/{sid}.wav",
        )
        residual = res.residual_path
        info["bestof"] = {"method": res.method, "seed": res.seed,
                          "ib_ta": res.ib_ta, "ib_ta_res": res.ib_ta_res,
                          "residual": res.residual_path, "target_wav": res.target_path}
        print(f"[{sid}] best-of winner: {res.method}/seed={res.seed} "
              f"ib_ta={res.ib_ta:.4f} -> {residual}", flush=True)
    print(f"[{sid}] LTX-2 enhancement...", flush=True)
    enh = enhance_model.enhance(
        inpainted_video_path=inpainted,
        residual_audio_path=residual,
        sample_id=sid,
        target_object=target,
    )
    info["enhance"] = {"paired_input": enh.paired_input_path,
                       "enhanced": enh.enhanced_path,
                       "frames": enh.frames, "audio_rms": enh.audio_rms}
    print(f"[{sid}] enhanced -> {enh.enhanced_path} "
          f"(frames={enh.frames}, audio_rms={enh.audio_rms})", flush=True)
    summary[sid] = info

with open("data/work/enhanced/showcase_batch_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSHOWCASE_BATCH_OK")
print(json.dumps(summary, indent=2))
EOF
echo "SHOWCASE_BATCH_DONE"
