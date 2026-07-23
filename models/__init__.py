"""Model wrappers for the AV data-curation pipeline.

Each module exposes one model class that owns its weights and a single
predict-style method returning exactly what the matching graph node needs.
Every model accepts `mock=True` (the default) so the pipeline runs end-to-end
without real checkpoints; flip to `mock=False` once weights are wired in.

Stage -> module:
    av_caption_generation     -> caption_model.CaptionModel
    sounding_object_extraction-> object_extraction_model.ObjectExtractionModel
    target_object_segmentation-> segmentation_model.SegmentationModel
    effect_erase_inpainting   -> inpainting_model.InpaintingModel
    samaudio_remove_target    -> audio_removal_model.AudioRemovalModel
    av_quality_enhancement    -> av_enhance_model.AVEnhanceModel
    inpainted_video_check     -> visual_checker.VisualChecker
    audio_removal_check       -> audio_checker.AudioChecker
    (av consistency gate)     -> cross_modal_checker.CrossModalChecker
"""

from .audio_checker import AudioChecker
from .audio_removal_model import AudioRemovalModel
from .av_enhance_model import AVEnhanceModel
from .caption_model import CaptionModel
from .cross_modal_checker import CrossModalChecker
from .inpainting_model import InpaintingModel
from .object_extraction_model import ObjectExtractionModel
from .segmentation_model import SegmentationModel
from .visual_checker import VisualChecker

__all__ = [
    "CaptionModel",
    "ObjectExtractionModel",
    "SegmentationModel",
    "InpaintingModel",
    "AudioRemovalModel",
    "AVEnhanceModel",
    "VisualChecker",
    "AudioChecker",
    "CrossModalChecker",
]
