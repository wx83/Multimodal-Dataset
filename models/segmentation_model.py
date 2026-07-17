"""Target-object segmentation with SAM3, via subprocess.

SAM3 (transformers Sam3 + torchcodec + GPU) lives in its own `sam3` conda env,
so it never pollutes the avgraph orchestration env. This module imports only
stdlib and drives the model out-of-process via `sam3_worker.py`.

Behavior (mock=False):
  1. SAM3 on the first frame -> first-frame mask ratio (largest CC / frame area).
  2. If ratio > first_frame_threshold (default 0.80), run SAM3 over the whole
     video and write the mask video; the sample continues.
  3. Otherwise -> no mask, sample is discarded at the mask gate.

mock=True keeps the old offline behavior (write a .txt mask, use the supplied
mock ratio) so CPU/login-node tests stay fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from utils import write_text_artifact

WORKER = Path(__file__).resolve().parent / "sam3_worker.py"
DEFAULT_SAM3_PYTHON = "/group2/ct/weihanx/miniconda3/envs/sam3/bin/python"
DEFAULT_SAM3_MODEL_DIR = "/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/sam3"
DEFAULT_SAM3_REPO = "/group2/ct/weihanx/sam3"
RESULT_MARKER = "SAM3_RESULT "


@dataclass
class SegmentationResult:
    mask_path: str | None
    mask_area_ratio: float
    first_frame_ratio: float | None = None
    discard_reason: str | None = None


class SegmentationModel:
    def __init__(
        self,
        model_dir: str = DEFAULT_SAM3_MODEL_DIR,
        python_bin: str = DEFAULT_SAM3_PYTHON,
        worker: str = str(WORKER),
        sam3_repo: str = DEFAULT_SAM3_REPO,
        output_base: str = "data/work/masks",
        first_frame_threshold: float = 0.80,
        max_seconds: int = 8,
        fps: int = 24,
        batch_size: int = 8,
        timeout: int = 3600,
        mock: bool = True,
    ):
        self.model_dir = model_dir
        self.python_bin = python_bin
        self.worker = worker
        self.sam3_repo = sam3_repo
        self.output_base = output_base
        self.first_frame_threshold = first_frame_threshold
        self.max_seconds = max_seconds
        self.fps = fps
        self.batch_size = batch_size
        self.timeout = timeout
        self.mock = mock

    def segment(self, av_pair_path: str, target_object: str, sample_id: str,
                mock_mask_area_ratio: float = 0.22) -> SegmentationResult:
        if self.mock:
            mask_path = f"data/work/masks/{sample_id}_mask.txt"
            write_text_artifact(
                mask_path,
                f"mock mask for sample={sample_id}, target={target_object}, "
                f"mask_area_ratio={mock_mask_area_ratio}\n",
            )
            return SegmentationResult(mask_path=mask_path, mask_area_ratio=mock_mask_area_ratio)
        return self._segment_sam3(av_pair_path, target_object, sample_id)

    def _segment_sam3(self, av_pair_path: str, target_object: str, sample_id: str) -> SegmentationResult:
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"sam3 interpreter not found: {self.python_bin}. Set up the env or pass python_bin=..."
            )
        out_dir = f"{self.output_base}/{sample_id}"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [
                self.python_bin, self.worker,
                "--video", str(av_pair_path),
                "--prompt", target_object,
                f"--sample_id={sample_id}",
                "--model_dir", self.model_dir,
                "--sam3_repo", self.sam3_repo,
                "--output_dir", out_dir,
                "--first_frame_threshold", str(self.first_frame_threshold),
                "--max_seconds", str(self.max_seconds),
                "--fps", str(self.fps),
                "--batch_size", str(self.batch_size),
                "--out", tmp.name,
            ]
            data = self._run(cmd)

        ratio = float(data.get("first_frame_ratio", 0.0))
        if data.get("passed"):
            return SegmentationResult(
                mask_path=data.get("mask_path"),
                mask_area_ratio=ratio,
                first_frame_ratio=ratio,
            )
        return SegmentationResult(
            mask_path=None,
            mask_area_ratio=0.0,
            first_frame_ratio=ratio,
            discard_reason="first_frame_mask_below_threshold",
        )

    def verify_removal(self, inpainted_video: str, original_mask: str,
                       target_object: str, sample_id: str) -> float:
        """Re-segment the inpainted video with SAM3 and compare against the original
        mask; return removal_ratio in [0, 1] (higher = object more fully removed)."""
        if self.mock:
            return 1.0
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"sam3 interpreter not found: {self.python_bin}. Set up the env or pass python_bin=..."
            )
        for label, p in [("inpainted_video", inpainted_video), ("original_mask", original_mask)]:
            if not p or not Path(p).exists():
                raise FileNotFoundError(f"verify_removal: {label} missing -> {p!r}")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [
                self.python_bin, self.worker,
                "--mode", "verify",
                "--inpainted_video", str(inpainted_video),
                "--original_mask", str(original_mask),
                "--prompt", target_object,
                f"--sample_id={sample_id}",
                "--model_dir", self.model_dir,
                "--sam3_repo", self.sam3_repo,
                "--max_seconds", str(self.max_seconds),
                "--fps", str(self.fps),
                "--batch_size", str(self.batch_size),
                "--out", tmp.name,
            ]
            data = self._run(cmd)
        return float(data.get("removal_ratio", 0.0))

    def _run(self, cmd: list[str]) -> dict:
        # torchcodec needs the env's FFmpeg shared libs on the loader path.
        env = dict(os.environ)
        env_lib = str(Path(self.python_bin).resolve().parent.parent / "lib")
        env["LD_LIBRARY_PATH"] = f"{env_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"SAM3 worker failed (exit {proc.returncode}).\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR (tail):\n{proc.stderr[-4000:]}"
            )
        out_path = cmd[cmd.index("--out") + 1] if "--out" in cmd else ""
        return self._parse(out_path, proc.stdout)

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
        raise RuntimeError(f"Could not parse SAM3 result.\nSTDOUT (tail):\n{stdout[-2000:]}")
