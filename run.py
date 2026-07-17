"""Batch runner: run each video in a (preprocessed) jsonl through the AV-curation
graph. Per-video state lands in data/work/state/<video_id>.json (updated each
step); finished videos are appended to data/logs/state.jsonl with a `status`
(passed videos = status=="passed").

Input jsonl — resolved entries from preprocess.py:
    {"video_id", "video", "audio"}

Usage:
    python preprocess.py --input_jsonl inputs.jsonl           # -> inputs_preprocessed.jsonl
    AVGRAPH_USE_REAL_MODELS=1 python run.py --input_jsonl inputs_preprocessed.jsonl
(real models also need OPENAI_API_KEY and a GPU node.)
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Resolved jsonl from preprocess.py.")
    args = ap.parse_args()

    graph = build_graph()
    with open(args.input_jsonl) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Running {len(entries)} videos through the pipeline...")

    passed = discarded = errored = 0
    for e in entries:
        vid = e["video_id"]
        initial = {
            "sample_id": vid,
            "av_pair_path": e["video"],
            "audio_path": e.get("audio", e["video"]),
            "retry_count": 0,
            "max_retries": 0,
            "status": "running",
        }
        t0 = time.time()
        try:
            final = graph.invoke(initial)
            status = final.get("status")
            passed += status == "passed"
            discarded += status == "discarded"
            reason = f" ({final.get('discard_reason')})" if status == "discarded" else ""
            print(f"  {vid}: {status}{reason}  [{time.time()-t0:.1f}s]")
        except Exception as ex:
            errored += 1
            print(f"  {vid}: ERROR {type(ex).__name__}: {ex}")

    print(f"\nDone: {passed} passed, {discarded} discarded, {errored} errored "
          f"-> data/logs/state.jsonl")


if __name__ == "__main__":
    main()
