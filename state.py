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
    # 目标的声学描述（同一物体的声音，如 "footsteps"），供音频侧使用；视觉侧仍用 target_object。
    # 来源：ACOUSTIC_DESC=clap 时由原始音频的 CLAP 零样本标签给出（strategy-lab S55/S56）。
    acoustic_desc: str
    acoustic_desc_score: float
    acoustic_desc_source: str
    speech_score: float        # CLAP 对 speech 标签的相似度；speech 不是目标，只作 has_speech 标注（A02）
    speech_excluded: bool      # 人物类目标：声学描述已排除 speech 标签

    # segmentation
    mask_path: str
    mask_area_ratio: float
    first_frame_ratio: float   # 首帧比例原值；低于门槛时 mask_area_ratio 会被抹成 0，只有这里保留真值
    n_instances: int           # SAM3 首帧检出的实例数；>1 意味着两侧删的可能不是同一个物体

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
    mock_acoustic_desc: str

    # control
    retry_count: int
    max_retries: int
    status: Literal["running", "passed", "discarded", "failed"]
    discard_stage: Optional[str]
    discard_reason: Optional[str]
    failure_reason: Optional[str]
