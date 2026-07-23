"""E2E run of 9NcjRFu6C4I with two substitutions (see run_e2e_9NcjRFu6C4I.sh):

1. The Qwen3-Omni caption from job 386777 is passed in as mock_caption so the
   caption node reuses it instead of re-running ~10 min of inference (the
   caption IS real model output, just cached).
2. The GPT sounding-object step runs the offline keyword matcher with a vocab
   extended for this clip (the OpenAI key has no quota - 429 insufficient_quota).
   Everything downstream (SAM3, EffectErase, SAM-Audio best-of-10 + ImageBind
   selection) runs real.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import nodes
from models.object_extraction_model import ObjectExtractionModel
from main import build_graph

CAPTION = Path("data/work/captions/9NcjRFu6C4I_caption.txt").read_text().strip()


class _FixedExtractor(ObjectExtractionModel):
    """Offline stand-in for the GPT step (key has no quota). Returns
    "train engine" — the prompt used by every prior experiment on this clip,
    and the natural answer given the caption's "train horn is heard". The
    one-word "train" from the plain keyword matcher gives SAM3 a first-frame
    ratio of only 0.0016 vs 0.0951 for "train engine"."""

    def extract(self, caption: str) -> list[str]:
        return ["train engine"] if "train" in caption.lower() else []


nodes._get_object_model = lambda: _FixedExtractor(mock=True)

graph = build_graph()
initial = {
    "sample_id": "9NcjRFu6C4I",
    "av_pair_path": "/group2/ct/weihanx/8sec_raw_video/9NcjRFu6C4I.mp4",
    "audio_path": "data/work/source_audio/9NcjRFu6C4I.wav",
    "mock_caption": CAPTION,
    "retry_count": 0,
    "max_retries": 0,
    "status": "running",
}
t0 = time.time()
final = graph.invoke(initial)
print(f"--- finished in {time.time() - t0:.1f}s ---", flush=True)
for k in ["status", "caption", "sounding_objects", "target_object", "mask_path",
          "mask_area_ratio", "inpainted_video_path", "visual_removal_score",
          "text_residual_audio_path", "text_target_audio_path",
          "best_audio_method", "best_audio_seed", "best_audio_ib_ta",
          "best_audio_ib_ta_res", "audio_selection_path", "audio_removal_score",
          "paired_av_output_path", "discard_stage", "discard_reason"]:
    print(f"  {k}: {final.get(k)}")
