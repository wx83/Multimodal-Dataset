from typing import Literal
from state import AVState


def route_after_object_extraction(state: AVState) -> Literal["continue", "discard"]:
    if len(state.get("sounding_objects", [])) == 0:
        return "discard"
    return "continue"


def route_after_mask_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("mask_area_ratio", 0.0) > 0.15:
        return "continue"
    return "discard"


def route_after_visual_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("visual_removal_score", 0.0) > 0.80:
        return "continue"
    return "discard"


def route_after_audio_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("audio_removal_score", 0.0) > 0.80:
        return "continue"
    return "discard"
