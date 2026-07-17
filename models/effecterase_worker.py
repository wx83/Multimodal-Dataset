"""EffectErase video object-removal worker (single video).

Runs INSIDE the dedicated `effecterase` conda env (diffsynth + diffusers
0.30-0.31 + transformers<5). Invoked as a subprocess by
models/inpainting_model.py — never imported by the avgraph pipeline.

Given the original video + the SAM3 mask video, it removes the masked object
(and its effects) and writes the inpainted video. Reuses EffectErase's own
helper functions (read_video_frames / crop_square_from_pil / save) and the
WanRemovePipeline, mirroring examples/remove_wan/infer_remove_wan.py.

Alignment: EffectErase reads the first `num_frames` (81) frames of both the
mask and fg_bg at the given fps. The SAM3 mask is 24fps, so we resample the
original video to 24fps (ffmpeg) to keep fg_bg[i] aligned with mask[i].
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

EE_REPO = "/group2/ct/weihanx/EffectErase"
sys.path.insert(0, EE_REPO)
sys.path.insert(0, os.path.join(EE_REPO, "examples/remove_wan"))

import torch

from infer_remove_wan import (  # noqa: E402  (path injected above)
    read_video_frames,
    crop_square_from_pil,
    save_frames_as_video_with_ref_fps,
)
from diffsynth import ModelManager, WanRemovePipeline  # noqa: E402

RESULT_MARKER = "EFFECTERASE_RESULT "
REMOVE_PROMPT = "Remove the specified object and all related effects, then restore a clean background."
NEGATIVE_PROMPT = (
    "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"
)


def resample_to_fps(src: str, dst: str, fps: int):
    """Resample `src` to `fps` (drop audio) so it aligns with the SAM3 mask."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-r", str(fps), "-an", dst],
        check=True, capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Original input video.")
    ap.add_argument("--mask", required=True, help="SAM3 binary mask video (24fps).")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--sample_id", required=True)
    ap.add_argument("--base_model_dir", default="/group2/ct/weihanx/Wan-AI/Wan2.1-Fun-1.3B-InP")
    ap.add_argument("--lora_path",
                    default="/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/inpainting/EffectErase.ckpt")
    ap.add_argument("--num_frames", type=int, default=192,
                    help="Max frames; clamped to the largest 4n+1 that fits the mask/fg_bg.")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    base = args.base_model_dir

    # 1. fg_bg = original resampled to the mask's fps, so frames align.
    tmp_fgbg = tempfile.NamedTemporaryFile(suffix="_fgbg.mp4", delete=False).name
    resample_to_fps(args.video, tmp_fgbg, args.fps)

    try:
        # Pick the largest valid frame count: Wan's VAE needs num_frames % 4 == 1,
        # and we can't read more frames than the mask/fg_bg actually have.
        import cv2

        def _frame_count(p):
            cap = cv2.VideoCapture(p)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n

        avail = min(_frame_count(args.mask), _frame_count(tmp_fgbg), args.num_frames)
        num_frames = max(1, ((avail - 1) // 4) * 4 + 1)  # largest 4n+1 <= avail
        print(f"[INFO] mask/fgbg available={avail}, requested={args.num_frames} -> num_frames={num_frames}")

        # 2. read mask + fg_bg, and the first-frame object crop (image cond).
        mask_t, mask_first = read_video_frames(args.mask, num_frames, 1, args.height, args.width)
        fgbg_t, fgbg_first = read_video_frames(tmp_fgbg, num_frames, 1, args.height, args.width)
        fg_first = crop_square_from_pil(mask_first, fgbg_first, target_size=224, video_mask_path=args.mask)

        # 3. build the pipeline (base Wan model + EffectErase LoRA).
        mm = ModelManager(device="cuda")
        mm.load_models(
            [
                os.path.join(base, "diffusion_pytorch_model.safetensors"),
                os.path.join(base, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(base, "Wan2.1_VAE.pth"),
                os.path.join(base, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ],
            torch_dtype=torch.bfloat16,
        )
        mm.load_lora_v2(args.lora_path, lora_alpha=args.lora_alpha)
        pipe = WanRemovePipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device="cuda")
        pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

        mask_t = mask_t.to(device)
        fgbg_t = fgbg_t.to(device)
        fg_first = fg_first.to(device)

        # 4. run removal.
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            remove_video, _ = pipe(
                video_mask=mask_t,
                video_fg_bg=fgbg_t,
                video_bg=None,
                task="remove",
                fg_first_img=fg_first,
                prompt_remove=REMOVE_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=num_frames,  # must match the input video frame count
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg,
                seed=args.seed,
                tiled=False,
                height=args.height,
                width=args.width,
                tea_cache_l1_thresh=None,
                tea_cache_model_id=None,
            )

        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        save_frames_as_video_with_ref_fps(
            remove_video, output_path=args.output_path, ref_video_path=tmp_fgbg, default_fps=args.fps,
        )
    finally:
        if os.path.exists(tmp_fgbg):
            os.remove(tmp_fgbg)

    res = {"sample_id": args.sample_id, "output_path": args.output_path}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    sys.stdout.write(RESULT_MARKER + json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
