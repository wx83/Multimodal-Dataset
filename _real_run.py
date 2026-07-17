import sys, time
from pathlib import Path

ROOT = "/music-shared-disk/group/ct/weihanx/av_langgraph_pipeline"
sys.path.insert(0, ROOT)
from main import build_graph

SAMPLES = [
    ("sample_0001", f"{ROOT}/data/raw/sample_0001.mp4"),
    ("sample_0002", f"{ROOT}/data/raw/sample_0002.mp4"),
]

graph = build_graph()

for sid, path in SAMPLES:
    print(f"\n{'='*72}\n REAL RUN: {sid}\n{'='*72}", flush=True)
    initial = {
        "sample_id": sid,
        "av_pair_path": path,
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
        # downstream stages are still mock -> give passing scores
        "mock_visual_removal_score": 0.90,
        "mock_audio_removal_score": 0.90,
    }
    t0 = time.time()
    try:
        final = graph.invoke(initial)
        print(f"--- finished in {time.time()-t0:.1f}s ---", flush=True)
        for k in ["status", "caption", "target_object", "mask_path",
                  "mask_area_ratio", "discard_stage", "discard_reason",
                  "paired_av_output_path"]:
            print(f"  {k}: {final.get(k)}")
        tj = Path(ROOT) / "data/work/tracking" / f"{sid}.json"
        if tj.exists():
            print("  tracking JSON ->", tj)
            print(tj.read_text())
    except Exception as e:
        print(f"!! ERROR on {sid}: {type(e).__name__}: {e}", flush=True)
