"""ImageBind best-candidate selection for SAM-Audio best_of outputs.

Runs INSIDE the `javisdit` conda env with cwd=/group2/ct/weihanx/JavisDiT (so
./checkpoints/imagebind_huge.pth resolves) and that repo on PYTHONPATH. Invoked
as a subprocess — never imported by the avgraph pipeline.

Scores every candidate from sam_audio_worker.py --mode best_of with the
ranking_test_eval metrics (compute_ranking_metrics.py):
  ib_ta     : ImageBind text <-> target audio similarity   (higher = better)
  ib_ta_res : ImageBind text <-> residual audio similarity (lower = cleaner removal)
and selects the winner by highest ib_ta, tiebreak lowest ib_ta_res. The winner's
wavs are copied to {output_dir}/{sample_id}_best_{target,residual}.wav.

Writes a JSON result to --out and a marker line to stdout.
"""

import argparse
import json
import os
import shutil
import sys

import torch

from eval.javisbench.src.metrics import calc_imagebind_score

RESULT_MARKER = "IB_SELECT_RESULT "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="candidates.json from sam_audio_worker.py --mode best_of")
    ap.add_argument("--video", default=None,
                    help="Source video for the ImageBind vision tower; defaults to "
                         "the candidates.json video_path.")
    ap.add_argument("--text", default=None,
                    help="Prompt text; defaults to the candidates.json prompt.")
    ap.add_argument("--out", default=None, help="Where to write selection.json.")
    args = ap.parse_args()

    with open(args.candidates, encoding="utf-8") as f:
        cand = json.load(f)
    rows = cand["rows"]
    video = args.video or cand["video_path"]
    text = args.text or cand["prompt"]
    out_dir = os.path.dirname(os.path.abspath(args.candidates))
    sample_id = cand.get("sample_id", "sample")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    videos = [video] * len(rows)
    prompts = [text] * len(rows)
    _, ib_ta, _ = calc_imagebind_score(
        videos, [r["target_wav"] for r in rows], prompts, prompts, device)
    _, ib_ta_res, _ = calc_imagebind_score(
        videos, [r["residual_wav"] for r in rows], prompts, prompts, device)
    for i, r in enumerate(rows):
        r["ib_ta"] = ib_ta[i].item()
        r["ib_ta_res"] = ib_ta_res[i].item()

    ranked = sorted(rows, key=lambda r: (-r["ib_ta"], r["ib_ta_res"]))
    best = ranked[0]
    best_target = os.path.join(out_dir, f"{sample_id}_best_target.wav")
    best_residual = os.path.join(out_dir, f"{sample_id}_best_residual.wav")
    shutil.copyfile(best["target_wav"], best_target)
    shutil.copyfile(best["residual_wav"], best_residual)

    res = {
        "sample_id": sample_id,
        "text": text,
        "video_path": video,
        "best": {**best, "best_target": best_target, "best_residual": best_residual},
        "candidates": ranked,
    }
    out_path = args.out or os.path.join(out_dir, "selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    sys.stdout.write(RESULT_MARKER + json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
