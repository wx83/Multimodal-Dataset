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
    first_frame_ratio: float   # raw first-frame ratio; mask_area_ratio is zeroed below threshold, only this keeps the true value
    n_instances: int           # instances SAM3 detected in the first frame; >1 means the two sides may remove different objects

    # video branch
    inpainted_video_path: str
    visual_removal_score: float

    # audio branch (SAM-Audio: mask-pass then text-pass, all four kept)
    mask_residual_audio_path: str     # mask-pass residual (object removed via visual mask)
    mask_target_audio_path: str       # mask-pass isolated target sound
    text_residual_audio_path: str     # text-pass residual = FINAL object-removed audio
    text_target_audio_path: str       # text-pass isolated target sound
    audio_removal_score: float

    # best-of-10 SAM-Audio selection (visual+text prompts x 5 seeds, ImageBind-ranked;
    # the winner's residual/target land in text_{residual,target}_audio_path)
    best_audio_method: str            # "visual" or "text"
    best_audio_seed: int
    best_audio_ib_ta: float           # ImageBind text<->target (higher = better)
    best_audio_ib_ta_res: float       # ImageBind text<->residual (lower = cleaner)
    audio_selection_path: str         # selection.json with all candidate scores

    # LTX-2 joint AV denoising enhancement (post-processing)
    paired_input_video_path: str      # conformed mux of inpainted video + residual audio
    enhanced_video_path: str          # LTX-2 enhanced AV mp4 (video+audio)

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
