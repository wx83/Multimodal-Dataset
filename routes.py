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


# Multi-instance consistency gate. **Off** by default -- turning it on reduces
# delivered volume, which is a human's call and must not take effect silently.
# Rationale: the visual side takes only masks[0] (sam3_worker.py), while on the
# audio side SAM-Audio separates every instance matching the text. When the word
# points at multiple objects the two sides remove different things, violating the
# correctness constraint "both sides remove the same object". Measured: about 8.8%
# of the 1945 delivered samples hit this (7/80 sampled, 95%CI 4.3-17.0).
REQUIRE_SINGLE_INSTANCE = os.environ.get("REQUIRE_SINGLE_INSTANCE", "0") == "1"


def route_after_mask_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("mask_area_ratio", 0.0) <= MASK_AREA_THRESHOLD:
        return "discard"
    # None = mock, or an older worker that does not report it; do not block in that
    # case (missing data is not proof of a single instance, but it is no reason to
    # throw samples away either)
    n = state.get("n_instances")
    if REQUIRE_SINGLE_INSTANCE and n is not None and n > 1:
        return "discard"
    return "continue"


def route_after_visual_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("visual_removal_score", 0.0) > VISUAL_SCORE_THRESHOLD:
        return "continue"
    return "discard"


def route_after_audio_check(state: AVState) -> Literal["continue", "discard"]:
    if state.get("audio_removal_score", 0.0) > AUDIO_SCORE_THRESHOLD:
        return "continue"
    return "discard"


# ---------------------------------------------------------------------------
# Gate identity, declared where the routing lives (manifest contract v2).
# Each entry names the node the gate sits after, the discard_stage that node
# writes when the gate rejects, the live condition, and whether the score it
# reads is actually measured. Tools render and audit from this; nobody
# maintains a copy elsewhere. `measures` for the audio gate is looked up lazily
# because nodes imports routes.
# ---------------------------------------------------------------------------
from manifest import GATES, gate  # noqa: E402


def _audio_measured() -> bool:
    import nodes
    return bool(nodes.AUDIO_SCORE_IS_MEASURED)


GATES[:] = [
    gate(after="sounding_object_extraction", discard_stage="sounding_object_extraction",
         condition=lambda: "objects ≠ ∅", measures=lambda: True, files=["routes.py"]),
    gate(after="target_object_segmentation", discard_stage="mask_check",
         condition=lambda: f"mask_area_ratio > {MASK_AREA_THRESHOLD}"
                           + (" ∧ n_instances ≤ 1" if REQUIRE_SINGLE_INSTANCE else ""),
         measures=lambda: True, files=["routes.py"]),
    gate(after="inpainted_video_check", discard_stage="visual_removal_check",
         condition=lambda: f"visual_removal_score > {VISUAL_SCORE_THRESHOLD}",
         measures=lambda: True, files=["routes.py", "models/visual_checker.py"]),
    gate(after="audio_removal_check", discard_stage="audio_removal_check",
         condition=lambda: f"audio_removal_score > {AUDIO_SCORE_THRESHOLD}",
         measures=_audio_measured, files=["routes.py", "models/audio_checker.py"]),
]
