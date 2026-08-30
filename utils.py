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
        # 2026-08-30: caption 与 sounding_objects 原本都没落盘。后果是 8907 条历史运行
        # （约 1484 GPU-小时的 caption）无法复用，任何「换个 prompt 会怎样」的对照
        # 都必须重新生成 caption；而 sounding_objects 缺失使得「同一条片子里是否本来
        # 就有可分割的备选发声物体」这个问题在历史数据上无法回答。两者都近乎零成本。
        "caption": state.get("caption"),
        "sounding_objects": state.get("sounding_objects"),
        "mask_path": state.get("mask_path"),
        "inpainted_video_path": state.get("inpainted_video_path"),
        # SAM-Audio: mask-pass then text-pass (all four kept)
        "mask_residual_audio_path": state.get("mask_residual_audio_path"),
        "mask_target_audio_path": state.get("mask_target_audio_path"),
        "text_residual_audio_path": state.get("text_residual_audio_path"),
        "text_target_audio_path": state.get("text_target_audio_path"),
        # LTX-2 joint AV denoising enhancement
        "paired_input_video_path": state.get("paired_input_video_path"),
        "enhanced_video_path": state.get("enhanced_video_path"),
        "paired_av_output_path": state.get("paired_av_output_path"),
        "status": state.get("status"),
        "discard_stage": state.get("discard_stage"),
        "discard_reason": state.get("discard_reason"),
        "metrics": {
            "mask_area_ratio": state.get("mask_area_ratio"),
            # 2026-08-30: first_frame_ratio 之前没落盘，而 segmentation_model.py 在
            # 首帧比例低于门槛时会把 mask_area_ratio 硬写成 0.0（真实值只留在
            # first_frame_ratio）。后果是历史 8907 条里 4789 条记为 mask=0，
            # 看上去像「SAM3 完全没检出」，实际是「检出了但低于门槛被抹零」——
            # 两者对应完全不同的修法（换目标 vs 调门槛），而数据无法区分。
            "first_frame_ratio": state.get("first_frame_ratio"),
            # 实例数同样要落盘：不落就无法回测「加多实例闸会损失多少交付量」，
            # 而这正是 S23 只能靠重跑 80 条 SAM3 才量出 8.8% 的原因。
            "n_instances": state.get("n_instances"),
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
