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


def effective_config() -> Dict[str, Any]:
    """Gate thresholds and model identities in effect for this run.

    2026-08-30: this is the fix for tonight's most expensive lesson. The 8907
    historical runs did not record the gate setting, so h0 = 21.84% cannot be
    compared across configurations -- it came from a first-frame gate of 0.05,
    while the value the code intends is 0.80. Recomputing at the intended value
    drops delivered samples from 1945 to 45: the same metric differs 40x between
    the two settings. Nobody noticed, because the setting was not in the log.

    For the goal of "ship the architecture, users swap in their own model" this
    is fatal: the entire point of swapping a model is comparison, and comparing
    two runs is meaningless if you do not know what configuration each ran under.

    Thresholds are always **read from routes / nodes**; no second copy of the
    defaults is kept here -- two copies drift, which is exactly the problem
    commit 3bae3f0 fixed. Lazy import: nodes/models depend on utils, so a
    module-level import would be circular; by call time they are already loaded.
    """
    cfg: Dict[str, Any] = {}
    try:
        import routes
        cfg["mask_area_threshold"] = routes.MASK_AREA_THRESHOLD
        cfg["visual_score_threshold"] = routes.VISUAL_SCORE_THRESHOLD
        cfg["audio_score_threshold"] = routes.AUDIO_SCORE_THRESHOLD
        cfg["require_single_instance"] = routes.REQUIRE_SINGLE_INSTANCE
    except Exception as e:                      # a logging failure must never take down the pipeline
        cfg["routes_error"] = f"{type(e).__name__}: {e}"
    try:
        import nodes
        cfg["sam3_first_frame_threshold"] = nodes.SAM3_FIRST_FRAME_THRESHOLD
        cfg["sam3_first_frame_intended"] = nodes.SAM3_FIRST_FRAME_INTENDED
        # Which gates actually measure. A gate that reads a constant and a gate
        # that really checks look identical in the pass rate; without this flag
        # you cannot tell "audio side passed" from "audio side was never measured".
        # The cross_modal one is declared in models/__init__.py as the last gate
        # but was never wired into the graph, so the correctness constraint "both
        # sides must remove the same object" is currently enforced by nobody.
        cfg["gates_measuring"] = {
            "mask": True,
            "visual": True,
            "audio": nodes.AUDIO_SCORE_IS_MEASURED,
            "cross_modal": False,
        }
        # Mark explicitly when the gate is relaxed, so a dev-setting run does not
        # later look like a normal run
        cfg["gate_relaxed"] = (
            nodes.SAM3_FIRST_FRAME_THRESHOLD < nodes.SAM3_FIRST_FRAME_INTENDED)
    except Exception as e:
        cfg["nodes_error"] = f"{type(e).__name__}: {e}"
    try:
        from models.caption_model import CaptionModel
        from models.object_extraction_model import DEFAULT_MODEL as EXTRACT_MODEL
        import inspect
        sig = inspect.signature(CaptionModel.__init__).parameters
        cfg["caption_model"] = sig["model_name"].default
        cfg["extraction_model"] = EXTRACT_MODEL
    except Exception as e:
        cfg["models_error"] = f"{type(e).__name__}: {e}"
    return cfg


def build_state_record(state: Dict[str, Any]) -> Dict[str, Any]:
    """Per-video state snapshot. Keeps every artifact path the pipeline produces;
    fields not reached yet are null. video_id<-sample_id, object_name<-target_object."""
    return {
        "video_id": state.get("sample_id"),
        "object_name": state.get("target_object"),
        # 2026-08-30: neither caption nor sounding_objects used to be written to
        # disk. As a result the 8907 historical runs (about 1484 GPU-hours of
        # captioning) cannot be reused: any "what if we changed the prompt"
        # comparison has to regenerate captions from scratch. And the missing
        # sounding_objects makes the question "did this clip have an alternative
        # separable sounding object at all" unanswerable on historical data.
        # Both are near-zero cost to record.
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
            # 2026-08-30: first_frame_ratio was not written to disk before, while
            # segmentation_model.py hard-writes mask_area_ratio to 0.0 whenever the
            # first-frame ratio is below the threshold (the real value survives only
            # in first_frame_ratio). As a result 4789 of the 8907 historical runs are
            # recorded as mask=0, which looks like "SAM3 detected nothing at all" but
            # is really "it detected something, below threshold, and got zeroed" --
            # two situations calling for completely different fixes (change the target
            # vs. retune the threshold), and the data could not tell them apart.
            "first_frame_ratio": state.get("first_frame_ratio"),
            # The instance count must be written to disk too: without it you cannot
            # backtest "how much delivered volume would a multi-instance gate cost",
            # which is exactly why S23 had to rerun SAM3 on 80 samples to measure 8.8%.
            "n_instances": state.get("n_instances"),
            "visual_removal_score": state.get("visual_removal_score"),
            "audio_removal_score": state.get("audio_removal_score"),
            # Whether the score was actually measured. Without this bit, downstream
            # cannot tell "passed" from "never measured".
            "audio_score_measured": state.get("audio_score_measured"),
        },
        # The effective configuration is written to disk with every record. Without it
        # pass rates from two runs are not comparable -- see effective_config's docstring.
        "config": effective_config(),
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
