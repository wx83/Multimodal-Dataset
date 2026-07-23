"""Batch E2E runs for the HF showcase page (see run_e2e_batch.sh).

Same substitutions as _e2e_9NcjRFu6C4I_fallback.py, generalized to many samples:
1. Cached Qwen3-Omni captions are passed as mock_caption (real model output,
   reused instead of ~10 min of inference each).
2. The GPT sounding-object step is replaced by a per-sample pinned target
   (the OpenAI key has no quota - 429 insufficient_quota); targets were chosen
   from each caption's explicitly named sound source.
Everything downstream (SAM3, EffectErase, verify, SAM-Audio best-of-10 +
ImageBind selection, LTX-2 enhancement) runs real.

Usage: python _e2e_showcase_batch.py <sample_id> [<sample_id> ...]
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import nodes
from models.object_extraction_model import ObjectExtractionModel
from main import build_graph

# sample_id -> target object (the caption's named sound source)
TARGETS = {
    "---lTs1dxhU": "race car",
    "--34LejG4cE": "brass band",
    "--56QUhyDQM_000185": "tennis player",
    "--5A5ZCa1dE": "airplane wing",
    "--65x-naOz0": "dragster",
    "--AErD4Wx6c": "white chicken",
    "--CudrykwoE_000078": "cricket",
    "--I3Bjp_ptc_000254": "bassoon",
    "--Lj4Y_96f0_000120": "bees",
    "--NVTlZnl00": "wheel loader",
}

_current_target = [""]


class _PinnedExtractor(ObjectExtractionModel):
    def extract(self, caption: str) -> list[str]:
        return [_current_target[0]] if _current_target[0] else []


nodes._get_object_model = lambda: _PinnedExtractor(mock=True)

graph = build_graph()

for sample_id in sys.argv[1:]:
    if sample_id not in TARGETS:
        print(f"SAMPLE_RESULT {sample_id} status=skipped reason=no_target", flush=True)
        continue
    _current_target[0] = TARGETS[sample_id]
    caption = Path(f"data/work/captions/{sample_id}_caption.txt").read_text().strip()
    initial = {
        "sample_id": sample_id,
        "av_pair_path": f"/group2/ct/weihanx/8sec_raw_video/{sample_id}.mp4",
        "audio_path": f"data/work/source_audio/{sample_id}.wav",
        "mock_caption": caption,
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
    }
    print(f"SAMPLE_START {sample_id} target={TARGETS[sample_id]!r}", flush=True)
    t0 = time.time()
    try:
        final = graph.invoke(initial)
        print(
            f"SAMPLE_RESULT {sample_id} status={final.get('status')} "
            f"discard_stage={final.get('discard_stage')} "
            f"discard_reason={final.get('discard_reason')} "
            f"mask_area={final.get('mask_area_ratio')} "
            f"visual={final.get('visual_removal_score')} "
            f"audio={final.get('audio_removal_score')} "
            f"enhanced={final.get('enhanced_video_path')} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"SAMPLE_RESULT {sample_id} status=crashed error={e!r} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

print("BATCH_DONE", flush=True)
