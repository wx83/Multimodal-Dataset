"""SAM3 segmentation worker (single video).

Runs INSIDE the dedicated `sam3` conda env (transformers Sam3 + torchcodec +
cv2 + torch). Invoked as a subprocess by models/segmentation_model.py — never
imported by the avgraph pipeline.

Logic (per the user's spec + filter_first_frame.py / batch_process_unbalanced_speedup.py):
  1. Load the (downsampled) video once with torchcodec via load_video_safe.
  2. Run SAM3 on the FIRST frame with the object-name prompt; compute the
     first-frame mask ratio = largest connected component area / frame area.
  3. Only if that ratio > --first_frame_threshold (default 0.80, i.e. the object
     covers >80% of the first frame) do we run SAM3 over the WHOLE video and
     write the binary mask video (plus overlay + masked for inspection).
  4. Otherwise the sample fails the gate and no whole-video work is done.

Writes a JSON result to --out and a marker line to stdout for the parent.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from torchcodec.decoders import VideoDecoder

RESULT_MARKER = "SAM3_RESULT "
CONF_THRESHOLD = 0.5  # SAM3 mask confidence (matches the reference scripts)


def load_model(model_dir: str, device: str):
    model = Sam3Model.from_pretrained(model_dir).to(device)
    processor = Sam3Processor.from_pretrained(model_dir)
    return model, processor


def load_frames(video_path: str, max_seconds: int, fps: int, sam3_repo: str):
    """Load downsampled frames with torchcodec. Returns (frames_bgr, width, height).

    frames_bgr: list of HxWx3 uint8 BGR numpy arrays (cv2 convention).
    """
    sys.path.insert(0, sam3_repo)
    from load_video import load_video_safe

    vd = VideoDecoder(video_path)
    meta = vd.metadata
    width, height = meta.width, meta.height
    raw_frames, _ = load_video_safe(
        vd, start_sec=0.0, end_sec=float(max_seconds), sampling_fps=float(fps)
    )
    # raw_frames: [N, C, H, W] uint8 RGB
    frames_bgr = [
        cv2.cvtColor(f.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR) for f in raw_frames
    ]
    return frames_bgr, width, height


def segment_top_mask(model, processor, frames_bgr, prompt, height, width, device, batch_size):
    """Run SAM3 over frames_bgr in batches; yield the top-instance binary mask per frame.

    Returns (masks, inst_counts): masks is a list of (H, W) uint8 (None where nothing
    detected); inst_counts is how many instances SAM3 returned per frame.

    2026-08-30: inst_counts 是新增的。此前只取 masks[0] 丢弃其余实例，而 SAM-Audio
    按文本分离**全部**实例——两侧口径不一致。实测已交付样本中 8.8% 命中多实例
    （95%CI 4.3-17.0，n=80），推算全量 1945 条里约 170 条两侧删的不是同一个物体。
    不上报实例数就无法察觉这件事。
    """
    masks = []
    inst_counts = []
    for start in range(0, len(frames_bgr), batch_size):
        batch_bgr = frames_bgr[start:start + batch_size]
        batch_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_bgr]
        inputs = processor(images=batch_imgs, text=[prompt] * len(batch_imgs),
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=CONF_THRESHOLD, target_sizes=[(height, width)] * len(batch_imgs)
        )
        for r in results:
            inst_counts.append(len(r["masks"]))
            if len(r["masks"]) > 0:
                m = r["masks"][0].cpu().numpy().astype(np.uint8)
                m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
                masks.append(m)
            else:
                masks.append(None)
    return masks, inst_counts


def largest_cc_ratio(mask_np, frame_area) -> float:
    """Largest-connected-component area / frame area, as a fraction in [0, 1].

    Matches filter_first_frame.py (which reports the same value *100 as a percent).
    """
    if mask_np is None:
        return 0.0
    _, binary = cv2.threshold(mask_np * 255, 128, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:  # label 0 is background
        return 0.0
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_area / frame_area


def write_mask_videos(frames_bgr, masks, out_dir, sample_id, fps, width, height):
    os.makedirs(out_dir, exist_ok=True)
    out_mask = os.path.join(out_dir, f"{sample_id}_mask.mp4")
    out_overlay = os.path.join(out_dir, f"{sample_id}_overlay.mp4")
    out_masked = os.path.join(out_dir, f"{sample_id}_masked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_mask = cv2.VideoWriter(out_mask, fourcc, fps, (width, height))
    w_ov = cv2.VideoWriter(out_overlay, fourcc, fps, (width, height))
    w_masked = cv2.VideoWriter(out_masked, fourcc, fps, (width, height))

    for orig, m in zip(frames_bgr, masks):
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
        masked_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if m is not None:
            color = np.zeros_like(orig)
            color[m > 0] = [0, 255, 0]
            overlay = cv2.addWeighted(orig, 1.0, color, 0.5, 0)
            mask_frame[m > 0] = [255, 255, 255]
            masked_frame[m > 0] = orig[m > 0]
        else:
            overlay = orig.copy()
        w_ov.write(overlay)
        w_mask.write(mask_frame)
        w_masked.write(masked_frame)

    w_mask.release()
    w_ov.release()
    w_masked.release()
    return {"mask_path": out_mask, "overlay_path": out_overlay, "masked_path": out_masked}


def read_mask_areas(mask_video_path, max_seconds, fps, sam3_repo):
    """Read an original mask video; return (per-frame white-pixel areas, frame_px)."""
    sys.path.insert(0, sam3_repo)
    from load_video import load_video_safe

    vd = VideoDecoder(mask_video_path)
    raw_frames, _ = load_video_safe(
        vd, start_sec=0.0, end_sec=float(max_seconds), sampling_fps=float(fps)
    )
    areas, frame_px = [], None
    for f in raw_frames:
        arr = f.permute(1, 2, 0).numpy()  # (H, W, C) RGB
        gray = arr.mean(axis=2)
        areas.append(int(np.sum(gray > 127)))
        if frame_px is None:
            frame_px = arr.shape[0] * arr.shape[1]
    return areas, (frame_px or 1)


def run_segment(model, processor, args, device):
    """Segment mode: first-frame gate, then whole-video mask if it passes."""
    frames_bgr, width, height = load_frames(args.video, args.max_seconds, args.fps, args.sam3_repo)
    if not frames_bgr:
        raise RuntimeError(f"No frames decoded from {args.video}")
    frame_area = width * height

    masks0, inst0 = segment_top_mask(
        model, processor, frames_bgr[:1], args.prompt, height, width, device, args.batch_size
    )
    ratio = largest_cc_ratio(masks0[0], frame_area)
    n_inst = inst0[0] if inst0 else 0

    result = {
        "mode": "segment",
        "video": args.video,
        "prompt": args.prompt,
        "sample_id": args.sample_id,
        "first_frame_ratio": round(ratio, 4),
        "threshold": args.first_frame_threshold,
        "passed": ratio > args.first_frame_threshold,
        "num_frames": len(frames_bgr),
        "mask_path": None,
        # 首帧检出的实例数。>1 表示该词在画面里指向多个物体，而音频侧会把它们全部
        # 分离——两侧口径不一致。只上报，是否据此拦截由 routes.py 决定。
        "n_instances": n_inst,
    }
    if result["passed"]:
        t0 = time.time()
        masks, _ = segment_top_mask(
            model, processor, frames_bgr, args.prompt, height, width, device, args.batch_size
        )
        result.update(write_mask_videos(
            frames_bgr, masks, args.output_dir, args.sample_id, args.fps, width, height
        ))
        result["seconds"] = round(time.time() - t0, 1)
    return result


def run_verify(model, processor, args, device):
    """Verify mode: re-segment the inpainted video, compare with the original mask.

    removal_ratio = 1 - mean(inpaint_area_frac) / mean(original_area_frac), using
    area *fractions* so the (480x832) inpaint and original-res mask are comparable.
    """
    frames_bgr, width, height = load_frames(
        args.inpainted_video, args.max_seconds, args.fps, args.sam3_repo
    )
    if not frames_bgr:
        raise RuntimeError(f"No frames decoded from {args.inpainted_video}")
    inp_px = width * height
    masks, _ = segment_top_mask(
        model, processor, frames_bgr, args.prompt, height, width, device, args.batch_size
    )
    inpaint_frac = [(int(np.sum(m > 0)) / inp_px) if m is not None else 0.0 for m in masks]

    orig_areas, orig_px = read_mask_areas(
        args.original_mask, args.max_seconds, args.fps, args.sam3_repo
    )
    orig_frac = [a / orig_px for a in orig_areas]

    n = min(len(inpaint_frac), len(orig_frac))
    inp = np.array(inpaint_frac[:n], dtype=np.float64)
    org = np.array(orig_frac[:n], dtype=np.float64)
    orig_mean = float(org.mean()) if n > 0 else 0.0
    inpaint_mean = float(inp.mean()) if n > 0 else 0.0
    removal_ratio = 1.0 - (inpaint_mean / orig_mean) if orig_mean > 0 else 1.0

    return {
        "mode": "verify",
        "sample_id": args.sample_id,
        "prompt": args.prompt,
        "removal_ratio": round(float(removal_ratio), 4),
        "original_mask_area_frac_mean": orig_mean,
        "inpaint_mask_area_frac_mean": inpaint_mean,
        "object_detected_in_inpaint": bool(inpaint_mean > 0),
        "n_frames": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["segment", "verify"], default="segment")
    ap.add_argument("--video", help="[segment] source video for mask extraction.")
    ap.add_argument("--inpainted_video", help="[verify] inpainted video to re-segment.")
    ap.add_argument("--original_mask", help="[verify] original SAM3 mask video.")
    ap.add_argument("--prompt", required=True, help="Target object name (SAM3 text prompt).")
    ap.add_argument("--sample_id", required=True)
    ap.add_argument("--model_dir", default="/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/sam3")
    ap.add_argument("--sam3_repo", default="/group2/ct/weihanx/sam3")
    ap.add_argument("--output_dir", help="[segment] where to write the mask videos.")
    ap.add_argument("--first_frame_threshold", type=float, default=0.80)
    ap.add_argument("--max_seconds", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=None, help="Path to write the JSON result.")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.model_dir, device)

    if args.mode == "verify":
        result = run_verify(model, processor, args, device)
    else:
        result = run_segment(model, processor, args, device)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    sys.stdout.write(RESULT_MARKER + json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
