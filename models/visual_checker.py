"""Visual-removal quality checker.

Scores how completely the target object was removed from the inpainted video
(1.0 = no trace left, 0.0 = clearly still present). The graph gate keeps the
sample when the score is above threshold. Feeds `inpainted_video_check`.
"""

from __future__ import annotations


class VisualChecker:
    def __init__(self, model_name: str = "mock-visual-checker", device: str = "cuda",
                 threshold: float = 0.80, mock: bool = True):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.mock = mock
        self._model = None
        if not self.mock:
            self._load()

    def _load(self) -> None:
        # e.g. re-run an open-vocab detector on the inpainted frames and
        # measure residual confidence for `target_object`
        raise NotImplementedError("Wire up the visual checker here, then set mock=False.")

    def score(self, inpainted_video_path: str, target_object: str,
              mock_score: float = 0.85) -> float:
        """Return visual-removal score in [0, 1]."""
        if self.mock:
            return mock_score
        raise NotImplementedError

    def passes(self, score: float) -> bool:
        return score > self.threshold
