"""Cross-modal consistency checker.

After both the target's visuals and sound are removed, verify the remaining
video and audio still agree with each other (e.g. AV sync / semantic
alignment via ImageBind or AV-CLIP). Useful as a final gate before writing
the paired output. Not yet wired into the graph — add a node + route when ready.
"""

from __future__ import annotations


class CrossModalChecker:
    def __init__(self, model_name: str = "mock-cross-modal", device: str = "cuda",
                 threshold: float = 0.80, mock: bool = True):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.mock = mock
        self._model = None
        if not self.mock:
            self._load()

    def _load(self) -> None:
        # from imagebind import imagebind_model
        # self._model = imagebind_model.imagebind_huge(pretrained=True).to(self.device).eval()
        raise NotImplementedError("Wire up ImageBind/AV-CLIP here, then set mock=False.")

    def score(self, inpainted_video_path: str, residual_audio_path: str,
              caption: str | None = None, mock_score: float = 0.88) -> float:
        """Return AV-consistency score in [0, 1]."""
        if self.mock:
            return mock_score
        raise NotImplementedError

    def passes(self, score: float) -> bool:
        return score > self.threshold
