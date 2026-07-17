from typing import TypedDict, List, Optional, Literal


class AVState(TypedDict, total=False):
    # input
    sample_id: str
    av_pair_path: str         # the AV video (with audio) — used by caption / SAM3 / EffectErase
    audio_path: str           # audio extracted from the video by preprocess.py — SAM-Audio input

    # caption / target
    caption: str
    caption_path: str
    sounding_objects: List[str]
    target_object: str

    # segmentation
    mask_path: str
    mask_area_ratio: float

    # video branch
    inpainted_video_path: str
    visual_removal_score: float

    # audio branch (SAM-Audio: mask-pass then text-pass, all four kept)
    mask_residual_audio_path: str     # mask-pass residual (object removed via visual mask)
    mask_target_audio_path: str       # mask-pass isolated target sound
    text_residual_audio_path: str     # text-pass residual = FINAL object-removed audio
    text_target_audio_path: str       # text-pass isolated target sound
    audio_removal_score: float

    # final output
    paired_av_output_path: str

    # mock controls for tests
    mock_mask_area_ratio: float
    mock_visual_removal_score: float
    mock_audio_removal_score: float
    mock_caption: str

    # control
    retry_count: int
    max_retries: int
    status: Literal["running", "passed", "discarded", "failed"]
    discard_stage: Optional[str]
    discard_reason: Optional[str]
    failure_reason: Optional[str]
