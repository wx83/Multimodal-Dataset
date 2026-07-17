"""SAM-Audio mask-conditioned separation worker (single video).

Runs INSIDE the `samaudio311` conda env (sam_audio + torchaudio + torchcodec).
Invoked as a subprocess by models/audio_removal_model.py — never imported by
the avgraph pipeline.

Given the original video (audio source) and the SAM3 mask video, it:
  1. Loads the video frames at 24fps/8s (torchcodec + load_video_safe) so the
     frame count matches the SAM3 mask video.
  2. Loads the SAM3 mask video (torchcodec) and aligns frame counts.
  3. Builds the masked video (processor.mask_videos zeroes the object region),
     runs SAM-Audio separation conditioned on the object description.
  4. Saves target (object sound) and residual (object-removed) wavs.

Writes a JSON result to --out and a marker line to stdout.
Reference: /group2/ct/weihanx/sam-audio/process_sam_audio_batch.py
"""

import argparse
import json
import os
import sys

# SAM-Audio pulls several sub-models (judge, pe-a-frame-large, roberta-base) from
# the HF hub on first load. Compute nodes here have internet, so we let it
# download/cache them rather than forcing offline (which needs every repo +
# revision pre-cached). Cached repos are reused on subsequent runs.

import torch
import torchaudio
from torchcodec.decoders import VideoDecoder

from sam_audio import SAMAudio, SAMAudioProcessor

RESULT_MARKER = "SAM_AUDIO_RESULT "


def _patch_hf_hub_compat():
    """Make sam_audio's ModelHubMixin loading work under huggingface_hub 1.x.

    1.x no longer passes proxies/resume_download to _from_pretrained, and its
    snapshot_download dropped those kwargs. Patch both (idempotent). This covers
    the top SAMAudio model AND its internal judge/ranker sub-models, which all
    inherit BaseModel._from_pretrained.
    """
    import sys

    # Find BaseModel (defines _from_pretrained) via SAMAudio's MRO, and its
    # module via sys.modules — avoids a fragile `import sam_audio.model.base`.
    base_cls = next(k for k in SAMAudio.__mro__ if k.__name__ == "BaseModel")
    base_mod = sys.modules[base_cls.__module__]
    if getattr(base_mod, "_avgraph_patched", False):
        return

    orig_snapshot = base_mod.snapshot_download

    def snapshot(*a, **kw):
        kw.pop("resume_download", None)
        kw.pop("proxies", None)
        return orig_snapshot(*a, **kw)

    base_mod.snapshot_download = snapshot

    orig_fp = base_cls._from_pretrained.__func__

    def patched_fp(cls, *, proxies=None, resume_download=False, **kw):
        return orig_fp(cls, proxies=proxies, resume_download=resume_download, **kw)

    base_cls._from_pretrained = classmethod(patched_fp)
    base_mod._avgraph_patched = True


def load_sam_audio(model_dir):
    """Load SAM-Audio (and its internal judge/ranker) under huggingface_hub 1.x."""
    _patch_hf_hub_compat()
    return SAMAudio.from_pretrained(model_dir)


def load_frames_24fps(video_path, max_seconds, fps, sam3_repo):
    """Load video frames at the same 24fps/8s sampling the SAM3 mask used."""
    sys.path.insert(0, sam3_repo)
    from load_video import load_video_safe

    vd = VideoDecoder(video_path)
    frames, _ = load_video_safe(vd, start_sec=0.0, end_sec=float(max_seconds), sampling_fps=float(fps))
    return frames  # (N, C, H, W) uint8 RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mask", "text"], default="mask",
                    help="mask: condition on the SAM3 mask video; text: condition on the object name.")
    ap.add_argument("--video", help="Original mp4 (frames source). [mask mode]")
    ap.add_argument("--audio", help="Audio source wav (preprocess output). [mask mode; falls back to --video]")
    ap.add_argument("--mask", help="SAM3 binary mask mp4. [mask mode]")
    ap.add_argument("--input_audio", help="Input wav to separate (e.g. prior residual). [text mode]")
    ap.add_argument("--prompt", required=True, help="Target object description.")
    ap.add_argument("--sample_id", required=True)
    ap.add_argument("--model_dir", default="/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/sam_audio")
    ap.add_argument("--sam3_repo", default="/group2/ct/weihanx/sam3", help="For load_video_safe.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_seconds", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sam_audio(args.model_dir).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.model_dir)

    n = None
    if args.mode == "mask":
        # Frames + mask, aligned to the same frame count.
        frames = load_frames_24fps(args.video, args.max_seconds, args.fps, args.sam3_repo)
        mask = VideoDecoder(args.mask)[:]  # (M, C, H, W)
        n = min(frames.shape[0], mask.shape[0])
        frames, mask = frames[:n], mask[:n]
        batch = processor(
            audios=[args.audio or args.video],   # extracted wav (preprocess) or the video's own track
            descriptions=[args.prompt],
            masked_videos=processor.mask_videos([frames], [mask]),
        )
        prefix = ""
    else:  # text mode: condition on the object name only, separate the input wav.
        batch = processor(audios=[args.input_audio], descriptions=[args.prompt])
        prefix = "text_"
    batch = batch.to(device)

    with torch.no_grad():
        result = model.separate(batch)

    target_audio = result.target[0] if isinstance(result.target, list) else result.target
    residual_audio = result.residual[0] if isinstance(result.residual, list) else result.residual

    os.makedirs(args.output_dir, exist_ok=True)
    sr = processor.audio_sampling_rate
    safe_prompt = args.prompt.replace(" ", "_")
    target_path = os.path.join(args.output_dir, f"{args.sample_id}_{prefix}target_{safe_prompt}.wav")
    residual_path = os.path.join(args.output_dir, f"{args.sample_id}_{prefix}residual_{safe_prompt}.wav")
    torchaudio.save(target_path, target_audio.cpu(), sr)
    torchaudio.save(residual_path, residual_audio.cpu(), sr)

    res = {
        "sample_id": args.sample_id,
        "mode": args.mode,
        "prompt": args.prompt,
        "target_path": target_path,
        "residual_path": residual_path,
        "sample_rate": sr,
        "num_frames": n,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    sys.stdout.write(RESULT_MARKER + json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
