"""Target-audio removal with SAM-Audio, via subprocess.

SAM-Audio (sam_audio + torchaudio + torchcodec) lives in the `samaudio311`
conda env, so it never pollutes the avgraph orchestration env. This module
imports only stdlib and drives the model out-of-process via `sam_audio_worker.py`.

mock=False conditions separation on the SAM3 mask video and returns the
residual (object-removed) audio path. mock=True keeps the old offline behavior
(write a .txt) so CPU/login-node tests stay fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from utils import write_text_artifact

WORKER = Path(__file__).resolve().parent / "sam_audio_worker.py"
DEFAULT_SAMAUDIO_PYTHON = "/home/weihan.xu/miniconda3/envs/samaudio311/bin/python"
DEFAULT_SAMAUDIO_MODEL_DIR = "/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/sam_audio"
DEFAULT_SAM3_REPO = "/group2/ct/weihanx/sam3"
RESULT_MARKER = "SAM_AUDIO_RESULT "


@dataclass
class AudioRemovalResult:
    residual_path: str | None
    target_path: str | None = None


class AudioRemovalModel:
    def __init__(
        self,
        model_dir: str = DEFAULT_SAMAUDIO_MODEL_DIR,
        python_bin: str = DEFAULT_SAMAUDIO_PYTHON,
        worker: str = str(WORKER),
        sam3_repo: str = DEFAULT_SAM3_REPO,
        output_base: str = "data/work/audio",
        max_seconds: int = 8,
        fps: int = 24,
        timeout: int = 3600,
        mock: bool = True,
    ):
        self.model_dir = model_dir
        self.python_bin = python_bin
        self.worker = worker
        self.sam3_repo = sam3_repo
        self.output_base = output_base
        self.max_seconds = max_seconds
        self.fps = fps
        self.timeout = timeout
        self.mock = mock

    def remove(self, av_pair_path: str, target_object: str, sample_id: str,
               mask_path: str | None = None, audio_path: str | None = None) -> AudioRemovalResult:
        if self.mock:
            residual_path = f"data/work/audio/{sample_id}_without_{target_object}.txt"
            write_text_artifact(
                residual_path,
                f"mock residual audio for sample={sample_id}, removed={target_object}\n",
            )
            return AudioRemovalResult(residual_path=residual_path)
        return self._remove_sam_audio(av_pair_path, target_object, sample_id, mask_path, audio_path)

    def remove_by_text(self, input_audio: str, target_object: str, sample_id: str) -> AudioRemovalResult:
        """Text-conditioned 2nd-pass separation: remove `target_object` from `input_audio`
        (typically the mask-pass residual). Returns the new residual + target."""
        if self.mock:
            residual_path = f"data/work/audio/{sample_id}_text_without_{target_object}.txt"
            write_text_artifact(
                residual_path,
                f"mock text-conditioned residual for sample={sample_id}, removed={target_object}\n",
            )
            return AudioRemovalResult(residual_path=residual_path)
        return self._remove_text(input_audio, target_object, sample_id)

    def _subprocess_env(self) -> dict:
        # torchcodec (imported by the worker) needs the env's FFmpeg libs on the loader path.
        env = dict(os.environ)
        env_lib = str(Path(self.python_bin).resolve().parent.parent / "lib")
        env["LD_LIBRARY_PATH"] = f"{env_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        return env

    def _remove_text(self, input_audio, target_object, sample_id) -> AudioRemovalResult:
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"samaudio interpreter not found: {self.python_bin}. Set up the env or pass python_bin=..."
            )
        if not input_audio or not Path(input_audio).exists():
            raise FileNotFoundError(
                f"text-conditioned SAM-Audio needs an input wav; got input_audio={input_audio!r} (missing)."
            )
        out_dir = f"{self.output_base}/{sample_id}"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [
                self.python_bin, self.worker,
                "--mode", "text",
                "--input_audio", str(input_audio),
                "--prompt", target_object,
                f"--sample_id={sample_id}",
                "--model_dir", self.model_dir,
                "--output_dir", out_dir,
                "--out", tmp.name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  env=self._subprocess_env())
            if proc.returncode != 0:
                raise RuntimeError(
                    f"SAM-Audio (text) worker failed (exit {proc.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDERR (tail):\n{proc.stderr[-4000:]}"
                )
            data = self._parse(tmp.name, proc.stdout)
        return AudioRemovalResult(
            residual_path=data.get("residual_path"),
            target_path=data.get("target_path"),
        )

    def _remove_sam_audio(self, av_pair_path, target_object, sample_id, mask_path,
                          audio_path=None) -> AudioRemovalResult:
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"samaudio interpreter not found: {self.python_bin}. Set up the env or pass python_bin=..."
            )
        if not mask_path or not Path(mask_path).exists():
            raise FileNotFoundError(
                f"SAM-Audio needs the SAM3 mask video; got mask_path={mask_path!r} (missing)."
            )
        out_dir = f"{self.output_base}/{sample_id}"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [
                self.python_bin, self.worker,
                "--video", str(av_pair_path),
                "--mask", str(mask_path),
                "--prompt", target_object,
                f"--sample_id={sample_id}",
                "--model_dir", self.model_dir,
                "--sam3_repo", self.sam3_repo,
                "--output_dir", out_dir,
                "--max_seconds", str(self.max_seconds),
                "--fps", str(self.fps),
                "--out", tmp.name,
            ]
            if audio_path:
                cmd += ["--audio", str(audio_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  env=self._subprocess_env())
            if proc.returncode != 0:
                raise RuntimeError(
                    f"SAM-Audio worker failed (exit {proc.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDERR (tail):\n{proc.stderr[-4000:]}"
                )
            data = self._parse(tmp.name, proc.stdout)

        return AudioRemovalResult(
            residual_path=data.get("residual_path"),
            target_path=data.get("target_path"),
        )

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
        raise RuntimeError(f"Could not parse SAM-Audio result.\nSTDOUT (tail):\n{stdout[-2000:]}")
