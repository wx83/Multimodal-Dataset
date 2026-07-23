import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_graph

EXPECTED_NODES = [
    "av_caption_generation",
    "sounding_object_extraction",
    "target_object_segmentation",
    "effect_erase_inpainting",
    "inpainted_video_check",
    "samaudio_best_of_remove",
    "audio_removal_check",
    "av_quality_enhancement",
    "paired_av_output",
    "discard_sample",
]


def main():
    graph = build_graph()
    print("Graph compiled successfully.")

    mermaid = graph.get_graph().draw_mermaid()
    print("\n========== MERMAID ==========")
    print(mermaid)

    for node in EXPECTED_NODES:
        assert node in mermaid, f"Missing node in Mermaid graph: {node}"

    out_dir = Path("data/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = out_dir / "graph.mmd"
    mmd_path.write_text(mermaid)
    print(f"Saved Mermaid graph to: {mmd_path}")

    try:
        png = graph.get_graph().draw_mermaid_png()
        png_path = out_dir / "graph.png"
        png_path.write_bytes(png)
        print(f"Saved PNG graph to: {png_path}")
    except Exception as e:
        print("Could not generate PNG. Mermaid text was still generated.")
        print("Error:", repr(e))

    print("All expected nodes are present.")


if __name__ == "__main__":
    main()
