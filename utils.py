import json
from pathlib import Path
from typing import Any, Dict


# Aggregate log: one line per finished video (passed or discarded), with status.
STATE_JSONL = Path("data/logs/state.jsonl")
# Per-video live state, written/overwritten after every node (status="running"
# until a terminal node sets passed/discarded).
STATE_DIR = Path("data/work/state")


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text_artifact(path: str | Path, content: str) -> str:
    path = ensure_parent(path)
    path.write_text(content)
    return str(path)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_state_record(state: Dict[str, Any]) -> Dict[str, Any]:
    """Per-video state snapshot. Keeps every artifact path the pipeline produces;
    fields not reached yet are null. video_id<-sample_id, object_name<-target_object."""
    return {
        "video_id": state.get("sample_id"),
        "object_name": state.get("target_object"),
        "mask_path": state.get("mask_path"),
        "inpainted_video_path": state.get("inpainted_video_path"),
        # SAM-Audio: mask-pass then text-pass (all four kept)
        "mask_residual_audio_path": state.get("mask_residual_audio_path"),
        "mask_target_audio_path": state.get("mask_target_audio_path"),
        "text_residual_audio_path": state.get("text_residual_audio_path"),
        "text_target_audio_path": state.get("text_target_audio_path"),
        "paired_av_output_path": state.get("paired_av_output_path"),
        "status": state.get("status"),
        "discard_stage": state.get("discard_stage"),
        "discard_reason": state.get("discard_reason"),
        "metrics": {
            "mask_area_ratio": state.get("mask_area_ratio"),
            "visual_removal_score": state.get("visual_removal_score"),
            "audio_removal_score": state.get("audio_removal_score"),
        },
    }


def write_video_state(state: Dict[str, Any]) -> str:
    """Write/overwrite data/work/state/<video_id>.json with the latest snapshot."""
    record = build_state_record(state)
    video_id = record.get("video_id") or "unknown"
    path = ensure_parent(STATE_DIR / f"{video_id}.json")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return str(path)


def append_state_jsonl(state: Dict[str, Any]) -> None:
    """Append one line to data/logs/state.jsonl for a finished video (terminal node)."""
    append_jsonl(STATE_JSONL, build_state_record(state))
