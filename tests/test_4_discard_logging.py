import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_graph

STATE_JSONL = Path("data/logs/state.jsonl")


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_case(graph, sample_id, mock_mask, mock_visual, mock_audio):
    initial_state = {
        "sample_id": sample_id,
        "av_pair_path": f"data/raw/{sample_id}.mp4",
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
        "mock_mask_area_ratio": mock_mask,
        "mock_visual_removal_score": mock_visual,
        "mock_audio_removal_score": mock_audio,
    }
    return graph.invoke(initial_state)


def assert_discard_record(record, expected_stage, expected_reason):
    assert record["status"] == "discarded"
    assert record["discard_stage"] == expected_stage
    assert record["discard_reason"] == expected_reason
    assert "metrics" in record
    for k in ("mask_area_ratio", "visual_removal_score", "audio_removal_score"):
        assert k in record["metrics"], f"missing metric {k}"
    # state.jsonl schema: paths live at the top level
    for k in (
        "video_id",
        "object_name",
        "mask_path",
        "inpainted_video_path",
        "mask_residual_audio_path",
        "mask_target_audio_path",
        "text_residual_audio_path",
        "text_target_audio_path",
        "paired_av_output_path",
    ):
        assert k in record, f"missing field {k}"


def main():
    if STATE_JSONL.exists():
        STATE_JSONL.unlink()

    graph = build_graph()
    cases = [
        {
            "sample_id": "log_case_mask_low",
            "mock_mask": 0.10,
            "mock_visual": 0.90,
            "mock_audio": 0.90,
            "expected_stage": "mask_check",
            "expected_reason": "mask_area_below_threshold",
        },
        {
            "sample_id": "log_case_visual_low",
            "mock_mask": 0.22,
            "mock_visual": 0.60,
            "mock_audio": 0.90,
            "expected_stage": "visual_removal_check",
            "expected_reason": "visual_removal_score_below_threshold",
        },
        {
            "sample_id": "log_case_audio_low",
            "mock_mask": 0.22,
            "mock_visual": 0.90,
            "mock_audio": 0.55,
            "expected_stage": "audio_removal_check",
            "expected_reason": "audio_removal_score_below_threshold",
        },
    ]

    for case in cases:
        result = run_case(
            graph,
            sample_id=case["sample_id"],
            mock_mask=case["mock_mask"],
            mock_visual=case["mock_visual"],
            mock_audio=case["mock_audio"],
        )
        print("Result:", result)
        assert result["status"] == "discarded"

    records = read_jsonl(STATE_JSONL)
    print("\nstate.jsonl records:")
    for r in records:
        print(r)

    assert len(records) == 3, f"Expected 3 state records, got {len(records)}"
    records_by_id = {r["video_id"]: r for r in records}

    for case in cases:
        sample_id = case["sample_id"]
        assert sample_id in records_by_id, f"Missing state record for {sample_id}"
        assert_discard_record(
            records_by_id[sample_id],
            expected_stage=case["expected_stage"],
            expected_reason=case["expected_reason"],
        )

    print("\nAll discard logging tests passed.")


if __name__ == "__main__":
    main()
