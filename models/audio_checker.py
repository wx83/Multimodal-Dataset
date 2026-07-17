"""Audio-removal quality checker.

Scores how completely the target object's sound was removed from the residual
audio (1.0 = fully gone, 0.0 = still clearly audible). The graph gate keeps the
sample when the score is above threshold. Feeds `audio_removal_check`.
"""

from __future__ import annotations


class AudioChecker:
    def __init__(self, model_name: str = "mock-audio-checker", device: str = "cuda",
                 threshold: float = 0.80, mock: bool = True):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.mock = mock
        self._model = None
        if not self.mock:
            self._load()

    def _load(self) -> None:
        # e.g. run an audio tagger / CLAP similarity against `target_object`
        # on the residual track and convert to a removal score
        raise NotImplementedError("Wire up the audio checker here, then set mock=False.")

    def score(self, residual_audio_path: str, target_object: str,
              mock_score: float = 0.90) -> float:
        """Return audio-removal score in [0, 1]."""
        if self.mock:
            return mock_score
        raise NotImplementedError

    def passes(self, score: float) -> bool:
        return score > self.threshold
