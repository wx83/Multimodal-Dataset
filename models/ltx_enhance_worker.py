"""LTX-2 joint AV denoising-enhancement worker (single clip).

Runs INSIDE the LTX-2 uv venv (/group2/ct/weihanx/LTX-2/.venv, Python 3.13,
editable ltx_core/ltx_pipelines installs), never imported by the avgraph env.
Invoked as a subprocess by models/av_enhance_model.py.

SDEdit-refines BOTH modalities of an already-muxed mp4: VAE-encode video+audio,
re-noise to --denoise-strength, denoise back down through the 22B LTX-2.3 DiT
(joint cross-modal attention keeps audio and video coupled). The wrapper is
responsible for conforming the input first (frames = 8k+1, dims /32, AAC audio).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

RESULT_MARKER = "LTX_ENHANCE_RESULT "
DEFAULT_CKPT = "/group2/ct/weihanx/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_GEMMA = "/group2/ct/weihanx/models/gemma-3-12b"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="conformed paired-input mp4 (video+audio)")
    ap.add_argument("--out-video", required=True, help="enhanced mp4 output path")
    ap.add_argument("--out", default=None, help="result JSON path")
    ap.add_argument("--sample_id", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--gemma-root", default=DEFAULT_GEMMA)
    ap.add_argument("--prompt", default="", help="empty (default) refines without steering")
    ap.add_argument("--denoise-strength", type=float, default=0.4219)
    ap.add_argument("--audio-denoise-strength", type=float, default=None)
    ap.add_argument("--fine-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload-mode", default="none", choices=["none", "cpu", "disk"])
    args = ap.parse_args()

    # Surface the quantized-sigma warning + per-step logs in the wrapper's stderr.
    logging.basicConfig(level=logging.INFO)

    import torch
    torch.set_grad_enabled(False)
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.av_enhance import AVEnhancePipeline
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
    from ltx_pipelines.utils.types import OffloadMode

    src = get_videostream_metadata(args.video)
    print(f"[INFO] source: {src.width}x{src.height} {src.frames}f @ {src.fps}fps "
          f"({src.frames / src.fps:.2f}s)", flush=True)
    # Fail fast on the VAE shape rules before the 46 GB model load.
    if (src.frames - 1) % 8 != 0:
        snapped = ((src.frames - 1) // 8) * 8 + 1
        raise SystemExit(f"frame count must be 8k+1; got {src.frames} (nearest valid: {snapped})")
    if src.width % 32 or src.height % 32:
        raise SystemExit(f"width/height must be multiples of 32; got {src.width}x{src.height}")

    pipe = AVEnhancePipeline(
        checkpoint_path=args.ckpt,
        gemma_root=args.gemma_root,
        loras=[],
        distilled=True,
        offload_mode=OffloadMode(args.offload_mode),
    )
    video_iter, audio = pipe(
        prompt=args.prompt,
        seed=args.seed,
        video_path=args.video,
        denoise_strength=args.denoise_strength,
        audio_denoise_strength=args.audio_denoise_strength,
        fine_steps=args.fine_steps,
        enhance_audio=True,
    )

    audio_rms = audio_sr = None
    if audio is not None:
        audio_rms = float(audio.waveform.float().pow(2).mean().sqrt())
        audio_sr = int(audio.sampling_rate)
        print(f"[INFO] enhanced audio {tuple(audio.waveform.shape)} @ {audio_sr} Hz "
              f"rms={audio_rms:.4f}", flush=True)
    else:
        print("[WARN] no audio stream in input -- only video was enhanced", flush=True)

    encode_video(
        video=video_iter,
        fps=int(src.fps),
        audio=audio,
        output_path=args.out_video,
        video_chunks_number=get_video_chunks_number(src.frames, None),
    )
    print(f"[INFO] saved -> {args.out_video}", flush=True)

    res = {
        "sample_id": args.sample_id,
        "output_path": args.out_video,
        "frames": src.frames,
        "width": src.width,
        "height": src.height,
        "fps": float(src.fps),
        "denoise_strength": args.denoise_strength,
        "audio_denoise_strength": args.audio_denoise_strength,
        "audio_enhanced": audio is not None,
        "audio_rms": audio_rms,
        "audio_sampling_rate": audio_sr,
        "seed": args.seed,
        "offload_mode": args.offload_mode,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    sys.stdout.write(RESULT_MARKER + json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
