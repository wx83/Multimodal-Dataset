import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_graph


def run_case(case_name, mock_mask, mock_visual, mock_audio, expected_status, expected_discard_stage=None):
    graph = build_graph()
    initial_state = {
        "sample_id": case_name,
        "av_pair_path": f"data/raw/{case_name}.mp4",
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
        "mock_mask_area_ratio": mock_mask,
        "mock_visual_removal_score": mock_visual,
        "mock_audio_removal_score": mock_audio,
    }

    print(f"\n========== Running {case_name} ==========")
    result = graph.invoke(initial_state)
    print("Final result:")
    print(result)

    assert result["status"] == expected_status, (
        f"{case_name}: expected status {expected_status}, got {result['status']}"
    )

    if expected_discard_stage is not None:
        assert result.get("discard_stage") == expected_discard_stage, (
            f"{case_name}: expected discard_stage {expected_discard_stage}, got {result.get('discard_stage')}"
        )

    print(f"{case_name} passed.")


def main():
    run_case("case_1_mask_low", 0.10, 0.90, 0.90, "discarded", "mask_check")
    run_case("case_2_visual_low", 0.22, 0.60, 0.90, "discarded", "visual_removal_check")
    run_case("case_3_audio_low", 0.22, 0.90, 0.55, "discarded", "audio_removal_check")
    run_case("case_4_all_pass", 0.22, 0.90, 0.90, "passed")
    print("\nAll route behavior tests passed.")


if __name__ == "__main__":
    main()
