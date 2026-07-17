import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_graph


def main():
    input_video = Path("data/raw/sample_0001.mp4")
    assert input_video.exists(), f"Missing input video: {input_video}"
    assert input_video.stat().st_size > 0, f"Input video is empty: {input_video}"

    graph = build_graph()
    initial_state = {
        "sample_id": "sample_0001",
        "av_pair_path": str(input_video),
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
        "mock_mask_area_ratio": 0.22,
        "mock_visual_removal_score": 0.90,
        "mock_audio_removal_score": 0.90,
    }

    print("\n========== Running one real video E2E ==========")
    final_state = None

    for state in graph.stream(initial_state, stream_mode="values"):
        final_state = state
        print("\n--- STATE SNAPSHOT ---")
        for key in [
            "sample_id",
            "av_pair_path",
            "caption",
            "target_object",
            "mask_path",
            "mask_area_ratio",
            "inpainted_video_path",
            "visual_removal_score",
            "mask_residual_audio_path",
            "mask_target_audio_path",
            "text_residual_audio_path",
            "text_target_audio_path",
            "audio_removal_score",
            "paired_av_output_path",
            "status",
            "discard_stage",
            "discard_reason",
        ]:
            if key in state:
                print(f"{key}: {state[key]}")

    assert final_state is not None, "Graph produced no final state."
    assert final_state["status"] in ["passed", "discarded"], (
        f"Unexpected final status: {final_state.get('status')}"
    )

    if final_state["status"] == "passed":
        output_path = Path(final_state["paired_av_output_path"])
        assert output_path.exists(), f"Missing paired output: {output_path}"
        assert output_path.stat().st_size > 0, f"Empty paired output: {output_path}"
        print(f"\nE2E passed. Output: {output_path}")
    else:
        assert "discard_stage" in final_state
        assert "discard_reason" in final_state
        print("\nE2E ended in discard.")
        print("discard_stage:", final_state["discard_stage"])
        print("discard_reason:", final_state["discard_reason"])

    print("\nOne real video E2E test passed.")


if __name__ == "__main__":
    main()
