import os
from typing import Literal
from state import AVState

# Whole-video mask gate (fraction of frame the tracked object must cover).
# Overridable for dev/demo runs on clips SAM3 tracks weakly.
MASK_AREA_THRESHOLD = float(os.environ.get("MASK_AREA_THRESHOLD", "0.15"))
# Visual-removal gate (SAM3 re-segmentation removal ratio on the inpainted video).
VISUAL_SCORE_THRESHOLD = float(os.environ.get("VISUAL_SCORE_THRESHOLD", "0.80"))
# Audio-removal gate (SAM-Audio separation score). Was hardcoded as a bare 0.80
# in both this module and nodes.py; the two copies could drift and neither was
# overridable, which made threshold sweeps impossible.
AUDIO_SCORE_THRESHOLD = float(os.environ.get("AUDIO_SCORE_THRESHOLD", "0.80"))


def route_after_object_extraction(state: AVState) -> Literal["continue", "discard"]:
    if len(state.get("sounding_objects", [])) == 0:
        return "discard"
    return "continue"


def route_after_mask_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("mask_area_ratio", 0.0) > MASK_AREA_THRESHOLD:
        return "continue"
    return "discard"


def route_after_visual_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("visual_removal_score", 0.0) > VISUAL_SCORE_THRESHOLD:
        return "continue"
    return "discard"


def route_after_audio_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("audio_removal_score", 0.0) > AUDIO_SCORE_THRESHOLD:
        return "continue"
    return "discard"
