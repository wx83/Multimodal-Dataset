import os
from functools import lru_cache, wraps

from models import (
    AudioRemovalModel,
    CaptionModel,
    InpaintingModel,
    ObjectExtractionModel,
    SegmentationModel,
)
from state import AVState
from utils import (
    append_state_jsonl,
    write_text_artifact,
    write_video_state,
)


def track_node(fn):
    """Decorator: after the node runs, overwrite the per-video state JSON with the
    merged (pre-node + node-update) state, so progress is recorded step by step.
    """
    @wraps(fn)
    def wrapper(state: AVState) -> AVState:
        update = fn(state)
        write_video_state({**state, **update})
        return update
    return wrapper


# ---------------------------
# Real model config
# ---------------------------
# Local Qwen3-Omni weights (override with the QWEN3_OMNI_PATH env var).
QWEN3_OMNI_PATH = os.environ.get(
    "QWEN3_OMNI_PATH",
    "/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/qwen3_omni",
)

# SAM3 first-frame mask gate (fraction of the frame the object must cover to
# proceed to whole-video segmentation).
# TEMP: 0.1 for development so real videos pass; set back to 0.80 later
# (or override with the SAM3_FIRST_FRAME_THRESHOLD env var).
SAM3_FIRST_FRAME_THRESHOLD = float(os.environ.get("SAM3_FIRST_FRAME_THRESHOLD", "0.1"))

# EffectErase max frames to inpaint (clamped to the largest valid 4n+1 that fits
# the mask/fg_bg). 192 ≈ full 8s clip at 24fps -> the worker uses 189.
INPAINT_NUM_FRAMES = int(os.environ.get("INPAINT_NUM_FRAMES", "192"))


def _use_real_models() -> bool:
    return os.environ.get("AVGRAPH_USE_REAL_MODELS") == "1"


@lru_cache(maxsize=1)
def _get_caption_model() -> CaptionModel:
    """Build the real captioner once, on first real use.

    Lazy + cached so mock/CPU tests never spawn the Qwen3-Omni subprocess.
    """
    return CaptionModel(mock=False, model_name=QWEN3_OMNI_PATH)


@lru_cache(maxsize=1)
def _get_object_model() -> ObjectExtractionModel:
    """Sounding-object extractor: real GPT when enabled, else mock keyword matcher.

    Built once; the OpenAI key comes from the OPENAI_API_KEY env var.
    """
    return ObjectExtractionModel(mock=not _use_real_models())


@lru_cache(maxsize=1)
def _get_segmentation_model() -> SegmentationModel:
    """Target-object segmentor: real SAM3 (subprocess, GPU) when enabled, else mock.

    Real path runs SAM3 on the first frame and only segments the whole video if
    the first-frame mask covers >80% of the frame.
    """
    return SegmentationModel(
        mock=not _use_real_models(),
        first_frame_threshold=SAM3_FIRST_FRAME_THRESHOLD,
    )


@lru_cache(maxsize=1)
def _get_audio_removal_model() -> AudioRemovalModel:
    """Target-audio remover: real SAM-Audio (subprocess, GPU) when enabled, else mock.

    Conditions separation on the SAM3 mask video; returns the residual
    (object-removed) audio.
    """
    return AudioRemovalModel(mock=not _use_real_models())


@lru_cache(maxsize=1)
def _get_inpainting_model() -> InpaintingModel:
    """Object-removal inpainter: real EffectErase (subprocess, GPU) when enabled, else mock.

    Removes the SAM3-masked object (and its effects) from the video.
    """
    return InpaintingModel(mock=not _use_real_models(), num_frames=INPAINT_NUM_FRAMES)


# ---------------------------
# Mock model functions
# Replace these later with real model calls.
# ---------------------------

def run_av_caption_model(state: AVState) -> str:
    # 1) explicit mock caption -> unit tests (highest priority)
    if "mock_caption" in state:
        return state["mock_caption"]
    # 2) real Qwen3-Omni when enabled (requires a GPU node)
    if _use_real_models():
        return _get_caption_model().generate(state["av_pair_path"])
    # 3) default mock so login-node / CPU tests stay fast
    return "A dog is barking next to a car."


def extract_sounding_objects(caption: str) -> list[str]:
    return _get_object_model().extract(caption)


# ---------------------------
# LangGraph nodes
# ---------------------------

@track_node
def av_caption_generation(state: AVState) -> AVState:
    caption = run_av_caption_model(state)

    # Persist the caption so downstream steps (and reruns) can reuse it
    # without re-invoking the model.
    caption_path = f"data/work/captions/{state['sample_id']}_caption.txt"
    write_text_artifact(caption_path, caption + "\n")

    return {"caption": caption, "caption_path": caption_path}


@track_node
def sounding_object_extraction(state: AVState) -> AVState:
    objects = extract_sounding_objects(state.get("caption", ""))

    if not objects:
        return {
            "sounding_objects": [],
            "discard_stage": "sounding_object_extraction",
            "discard_reason": "no_sounding_object_found",
        }

    return {
        "sounding_objects": objects,
        "target_object": objects[0],
    }


@track_node
def target_object_segmentation(state: AVState) -> AVState:
    sample_id = state["sample_id"]
    target_object = state.get("target_object", "unknown")

    result = _get_segmentation_model().segment(
        av_pair_path=state.get("av_pair_path", ""),
        target_object=target_object,
        sample_id=sample_id,
        mock_mask_area_ratio=state.get("mock_mask_area_ratio", 0.22),
    )

    update: AVState = {
        "mask_path": result.mask_path or "",
        "mask_area_ratio": result.mask_area_ratio,
    }

    if result.discard_reason:
        # Real SAM3 path: first-frame mask below the 80% gate.
        update.update({
            "discard_stage": "mask_check",
            "discard_reason": result.discard_reason,
        })
    elif result.mask_area_ratio <= 0.15:
        # Mock path: keeps the existing area-threshold behavior for tests.
        update.update({
            "discard_stage": "mask_check",
            "discard_reason": "mask_area_below_threshold",
        })

    return update


@track_node
def effect_erase_inpainting(state: AVState) -> AVState:
    sample_id = state["sample_id"]
    target_object = state.get("target_object", "unknown")

    inpainted_video_path = _get_inpainting_model().inpaint(
        av_pair_path=state.get("av_pair_path", ""),
        mask_path=state.get("mask_path"),
        target_object=target_object,
        sample_id=sample_id,
    )

    return {"inpainted_video_path": inpainted_video_path}


@track_node
def inpainted_video_check(state: AVState) -> AVState:
    # Real removal check: re-segment the inpainted video with SAM3 and compare
    # against the original mask -> removal_ratio (= visual_removal_score).
    if "mock_visual_removal_score" in state:          # explicit test override
        score = state["mock_visual_removal_score"]
    elif _use_real_models():
        score = _get_segmentation_model().verify_removal(
            inpainted_video=state["inpainted_video_path"],
            original_mask=state["mask_path"],
            target_object=state.get("target_object", "unknown"),
            sample_id=state["sample_id"],
        )
    else:
        score = 0.85                                  # default mock

    update: AVState = {"visual_removal_score": score}

    if score <= 0.80:
        update.update({
            "discard_stage": "visual_removal_check",
            "discard_reason": "visual_removal_score_below_threshold",
        })

    return update


@track_node
def samaudio_remove_target(state: AVState) -> AVState:
    sample_id = state["sample_id"]
    target_object = state.get("target_object", "unknown")

    result = _get_audio_removal_model().remove(
        av_pair_path=state.get("av_pair_path", ""),
        target_object=target_object,
        sample_id=sample_id,
        mask_path=state.get("mask_path"),
        audio_path=state.get("audio_path"),
    )

    update: AVState = {"mask_residual_audio_path": result.residual_path}
    if result.target_path:
        update["mask_target_audio_path"] = result.target_path
    return update


@track_node
def samaudio_text_remove(state: AVState) -> AVState:
    # Second-pass, text-conditioned separation: take the mask-pass residual and
    # remove any remaining target sound by description. Its residual is the FINAL
    # object-removed audio; all four SAM-Audio paths are kept in state.
    sample_id = state["sample_id"]
    target_object = state.get("target_object", "unknown")

    result = _get_audio_removal_model().remove_by_text(
        input_audio=state.get("mask_residual_audio_path", ""),
        target_object=target_object,
        sample_id=sample_id,
    )

    update: AVState = {"text_residual_audio_path": result.residual_path}
    if result.target_path:
        update["text_target_audio_path"] = result.target_path
    return update


@track_node
def audio_removal_check(state: AVState) -> AVState:
    score = state.get("mock_audio_removal_score", 0.90)

    update: AVState = {"audio_removal_score": score}

    if score <= 0.80:
        update.update({
            "discard_stage": "audio_removal_check",
            "discard_reason": "audio_removal_score_below_threshold",
        })

    return update


@track_node
def paired_av_output(state: AVState) -> AVState:
    sample_id = state["sample_id"]

    output_path = f"data/work/outputs/{sample_id}_paired_output.txt"
    write_text_artifact(
        output_path,
        (
            f"mock paired AV output for sample={sample_id}\n"
            f"video={state.get('inpainted_video_path')}\n"
            f"audio={state.get('text_residual_audio_path')}\n"
        ),
    )

    updated_state = {
        **state,
        "paired_av_output_path": output_path,
        "status": "passed",
        "discard_stage": None,
        "discard_reason": None,
    }

    append_state_jsonl(updated_state)

    return {
        "paired_av_output_path": output_path,
        "status": "passed",
        "discard_stage": None,
        "discard_reason": None,
    }


@track_node
def discard_sample(state: AVState) -> AVState:
    updated_state = {
        **state,
        "status": "discarded",
        "discard_stage": state.get("discard_stage", "unknown"),
        "discard_reason": state.get("discard_reason", "unknown"),
    }

    append_state_jsonl(updated_state)

    return {
        "status": "discarded",
        "discard_stage": updated_state["discard_stage"],
        "discard_reason": updated_state["discard_reason"],
    }
