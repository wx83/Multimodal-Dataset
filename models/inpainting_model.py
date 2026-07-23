"""Video inpainting / object removal with EffectErase, via subprocess.

EffectErase (diffsynth + Wan2.1-Fun-1.3B-InP, diffusers 0.30-0.31, transformers<5)
lives in the dedicated `effecterase` conda env, so it never pollutes the avgraph
orchestration env. This module imports only stdlib and drives the model
out-of-process via `effecterase_worker.py`.

mock=False removes the SAM3-masked object from the video and returns the
inpainted video path. mock=True keeps the old offline behavior (write a .txt).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from utils import write_text_artifact

WORKER = Path(__file__).resolve().parent / "effecterase_worker.py"
# Env overrides make the pipeline portable to other hosts (see SELF_HOSTING.md).
DEFAULT_EE_PYTHON = os.environ.get(
    "EFFECTERASE_PYTHON", "/group2/ct/weihanx/miniconda3/envs/effecterase/bin/python")
DEFAULT_BASE_MODEL_DIR = os.environ.get(
    "WAN_BASE_MODEL_DIR", "/group2/ct/weihanx/Wan-AI/Wan2.1-Fun-1.3B-InP")
DEFAULT_LORA_PATH = os.environ.get(
    "EFFECTERASE_LORA",
    "/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/inpainting/EffectErase.ckpt")
RESULT_MARKER = "EFFECTERASE_RESULT "


class InpaintingModel:
    def __init__(
        self,
        python_bin: str = DEFAULT_EE_PYTHON,
        worker: str = str(WORKER),
        base_model_dir: str = DEFAULT_BASE_MODEL_DIR,
        lora_path: str = DEFAULT_LORA_PATH,
        output_base: str = "data/work/inpainted",
        num_frames: int = 192,  # worker clamps to the largest valid 4n+1 that fits
        height: int = 480,
        width: int = 832,
        fps: int = 24,
        num_inference_steps: int = 50,
        timeout: int = 5400,
        mock: bool = True,
    ):
        self.python_bin = python_bin
        self.worker = worker
        self.base_model_dir = base_model_dir
        self.lora_path = lora_path
        self.output_base = output_base
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.timeout = timeout
        self.mock = mock

    def inpaint(self, av_pair_path: str, mask_path: str, target_object: str, sample_id: str) -> str:
        """Remove the masked object; return the inpainted video path."""
        if self.mock:
            inpainted_path = f"data/work/inpainted/{sample_id}_without_{target_object}.txt"
            write_text_artifact(
                inpainted_path,
                f"mock inpainted video for sample={sample_id}, removed={target_object}\n",
            )
            return inpainted_path
        return self._inpaint_effecterase(av_pair_path, mask_path, target_object, sample_id)

    def _inpaint_effecterase(self, av_pair_path, mask_path, target_object, sample_id) -> str:
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"effecterase interpreter not found: {self.python_bin}. Set up the env or pass python_bin=..."
            )
        if not mask_path or not Path(mask_path).exists():
            raise FileNotFoundError(
                f"EffectErase needs the SAM3 mask video; got mask_path={mask_path!r} (missing)."
            )
        output_path = f"{self.output_base}/{sample_id}_without_{target_object}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [
                self.python_bin, self.worker,
                "--video", str(av_pair_path),
                "--mask", str(mask_path),
                "--output_path", output_path,
                f"--sample_id={sample_id}",
                "--base_model_dir", self.base_model_dir,
                "--lora_path", self.lora_path,
                "--num_frames", str(self.num_frames),
                "--height", str(self.height),
                "--width", str(self.width),
                "--fps", str(self.fps),
                "--num_inference_steps", str(self.num_inference_steps),
                "--out", tmp.name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"EffectErase worker failed (exit {proc.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDERR (tail):\n{proc.stderr[-4000:]}"
                )
            data = self._parse(tmp.name, proc.stdout)
        return data.get("output_path", output_path)

    @staticmethod
    def _parse(out_path: str, stdout: str) -> dict:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        for line in stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                return json.loads(line[len(RESULT_MARKER):])
        raise RuntimeError(f"Could not parse EffectErase result.\nSTDOUT (tail):\n{stdout[-2000:]}")
