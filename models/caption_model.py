"""Audio-visual captioning with Qwen3-Omni, via subprocess.

Qwen3-Omni (30B + torch + transformers>=5.2) lives in its own conda env so it
never pollutes the lightweight avgraph orchestration env. This module imports
only stdlib and drives the model out-of-process by launching
`qwen3_omni_worker.py` with the qwen3omni interpreter.

Usage in nodes.py:

    from models import CaptionModel
    caption_model = CaptionModel(mock=False)      # real model
    caption = caption_model.generate(state["av_pair_path"])

Notes:
- The worker needs a GPU, so the pipeline must run on a GPU node
  (e.g. inside `srun -p sharedp --gres=gpu:1 --pty bash -l`). If you launch the
  pipeline from a login node, pass `launcher=["srun", "-p", "sharedp",
  "--gres=gpu:1"]` to wrap the worker call in its own allocation.
- attn_implementation defaults to "sdpa" (works everywhere). Switch to
  "flash_attention_2" on a capable GPU once flash-attn is installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "qwen3_omni_worker.py"
# Env override makes the pipeline portable to other hosts (see SELF_HOSTING.md).
DEFAULT_QWEN_PYTHON = os.environ.get(
    "QWEN3_PYTHON", "/group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/python")
RESULT_MARKER = "QWEN3_OMNI_RESULT "
MOCK_CAPTION = "A dog is barking next to a car."


class CaptionModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        python_bin: str = DEFAULT_QWEN_PYTHON,
        worker: str = str(WORKER),
        max_new_tokens: int = 256,
        attn_implementation: str = "sdpa",
        use_audio_in_video: bool = True,
        timeout: int = 1800,
        launcher: list[str] | None = None,
        mock: bool = True,
    ):
        self.model_name = model_name
        self.python_bin = python_bin
        self.worker = worker
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.use_audio_in_video = use_audio_in_video
        self.timeout = timeout
        self.launcher = launcher or []
        self.mock = mock

    def _build_cmd(self, av_pair_path: str, out_path: str) -> list[str]:
        cmd = list(self.launcher) + [
            self.python_bin,
            self.worker,
            "--model", self.model_name,
            "--video", str(av_pair_path),
            "--max_new_tokens", str(self.max_new_tokens),
            "--attn_implementation", self.attn_implementation,
            "--out", out_path,
        ]
        cmd.append("--use_audio_in_video" if self.use_audio_in_video else "--no_audio_in_video")
        return cmd

    def generate(self, av_pair_path: str) -> str:
        """Return an audio-visual caption for the clip at `av_pair_path`."""
        if self.mock:
            return MOCK_CAPTION

        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"qwen3omni interpreter not found: {self.python_bin}. "
                "Set up the env or pass python_bin=..."
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            out_path = tmp.name
            cmd = self._build_cmd(av_pair_path, out_path)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Qwen3-Omni worker failed (exit {proc.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDERR (tail):\n{proc.stderr[-4000:]}"
                )

            return self._parse_caption(out_path, proc.stdout)

    @staticmethod
    def _parse_caption(out_path: str, stdout: str) -> str:
        # Prefer the JSON file written by the worker.
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)["caption"]
        except (OSError, KeyError, json.JSONDecodeError):
            pass
        # Fall back to the marker line on stdout.
        for line in stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                return json.loads(line[len(RESULT_MARKER):])["caption"]
        raise RuntimeError(
            "Could not parse a caption from the worker output.\n"
            f"STDOUT (tail):\n{stdout[-2000:]}"
        )
