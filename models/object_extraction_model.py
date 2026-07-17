"""Sounding-object extraction.

Given a video caption, return the objects/entities that are ACTIVELY producing
sound. Feeds the `sounding_object_extraction` node.

Two backends:
  * mock=True  -> offline keyword matcher over a small vocab (deterministic;
                  used by CPU/login-node tests, no API key needed).
  * mock=False -> GPT via the OpenAI API (an audio-scene-analyst prompt).

Unlike the captioner, this is a remote HTTP API call: no GPU, no subprocess,
no conda env. It runs in-process in avgraph and only needs `openai` + the
OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import os

DEFAULT_VOCAB = ["dog", "car", "person", "bird", "instrument", "baby"]
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are an audio scene analyst. Given a video caption, identify objects or entities "
    "that are ACTIVELY producing sound in the scene. "
    "An object qualifies only if it has a clear physical mechanism for sound production "
    "(e.g. vibrating strings, flowing water, a running engine, a speaking person, "
    "flapping wings, crackling fire). "
    "Exclude static background objects that do not themselves emit sound "
    "(e.g. buildings, walls, trees, roads, shacks). "
    "Output ONLY a comma-separated list of the sounding objects. "
    "If nothing in the caption is actively producing sound, output: none."
)


class ObjectExtractionModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str | None = None,
                 vocab: list[str] | None = None, mock: bool = True):
        self.model_name = model_name
        # Falls back to the OPENAI_API_KEY env var (the "global" key).
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.vocab = vocab or DEFAULT_VOCAB
        self.mock = mock
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai  # lazy: only needed for the real backend
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it (e.g. `export OPENAI_API_KEY=sk-...`) "
                    "or pass api_key=... to ObjectExtractionModel."
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def extract(self, caption: str) -> list[str]:
        """Return the list of sounding objects mentioned in `caption`."""
        if self.mock:
            caption_lower = caption.lower()
            return [obj for obj in self.vocab if obj in caption_lower]
        return self._extract_gpt(caption)

    def _extract_gpt(self, caption: str) -> list[str]:
        # Strip null bytes / invalid chars that break the JSON request.
        caption = caption.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": caption},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if content.lower() == "none":
            return []
        return [s.strip() for s in content.split(",") if s.strip()]
