import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_graph

ARTIFACT_KEYS = [
    "mask_path",
    "inpainted_video_path",
    "mask_residual_audio_path",
    "text_residual_audio_path",
    "paired_av_output_path",
]


def main():
    graph = build_graph()
    initial_state = {
        "sample_id": "artifact_test_all_pass",
        "av_pair_path": "data/raw/artifact_test_all_pass.mp4",
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
        "mock_mask_area_ratio": 0.22,
        "mock_visual_removal_score": 0.90,
        "mock_audio_removal_score": 0.90,
    }

    result = graph.invoke(initial_state)
    print("Final result:")
    print(result)

    assert result["status"] == "passed", "Artifact test should pass."

    for key in ARTIFACT_KEYS:
        assert key in result, f"Missing artifact key in state: {key}"
        path = Path(result[key])
        assert path.exists(), f"Artifact does not exist: {key}={path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {key}={path}"
        print(f"{key}: exists, size={path.stat().st_size}, path={path}")

    print("\nAll artifact writing tests passed.")


if __name__ == "__main__":
    main()
