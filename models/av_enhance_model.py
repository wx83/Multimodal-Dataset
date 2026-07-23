"""LTX-2 joint AV denoising enhancement, via subprocess.

The LTX-2 enhancer (AVEnhancePipeline, 22B DiT + Gemma text encoder) lives in
the LTX-2 uv venv, so it never pollutes the avgraph orchestration env. This
module imports only stdlib and drives the model out-of-process via
`ltx_enhance_worker.py`.

mock=False muxes the inpainted video with the residual audio (conforming to the
VAE's frames=8k+1 / div-32 rules), then SDEdit-refines BOTH modalities jointly
and returns the enhanced mp4 path. mock=True writes .txt stubs so
CPU/login-node tests stay fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from utils import write_text_artifact

WORKER = Path(__file__).resolve().parent / "ltx_enhance_worker.py"
# Env overrides make the pipeline portable to other hosts (see SELF_HOSTING.md).
DEFAULT_LTX_PYTHON = os.environ.get(
    "LTX_PYTHON", "/group2/ct/weihanx/LTX-2/.venv/bin/python")
DEFAULT_LTX_CKPT = os.environ.get(
    "LTX_CKPT", "/group2/ct/weihanx/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors")
DEFAULT_GEMMA_ROOT = os.environ.get("LTX_GEMMA_ROOT", "/group2/ct/weihanx/models/gemma-3-12b")
# ffmpeg/ffprobe are not on PATH in the orchestrator env; borrow the qwen3omni ones.
DEFAULT_FFMPEG = os.environ.get(
    "AVGRAPH_FFMPEG", "/group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/ffmpeg")
DEFAULT_FFPROBE = os.environ.get(
    "AVGRAPH_FFPROBE", "/group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/ffprobe")
# 0.4219 is the lightest strength reachable on the distilled sigma grid (one
# denoise step); lighter refinement needs AVENHANCE_FINE_STEPS.
DEFAULT_DENOISE_STRENGTH = 0.4219
RESULT_MARKER = "LTX_ENHANCE_RESULT "


@dataclass
class AVEnhanceResult:
    paired_input_path: str
    enhanced_path: str
    frames: int | None = None
    audio_enhanced: bool | None = None
    audio_rms: float | None = None
    denoise_strength: float | None = None


class AVEnhanceModel:
    def __init__(
        self,
        python_bin: str = DEFAULT_LTX_PYTHON,
        worker: str = str(WORKER),
        ckpt: str = DEFAULT_LTX_CKPT,
        gemma_root: str = DEFAULT_GEMMA_ROOT,
        ffmpeg: str = DEFAULT_FFMPEG,
        ffprobe: str = DEFAULT_FFPROBE,
        output_base: str = "data/work/enhanced",
        denoise_strength: float | None = None,
        audio_denoise_strength: float | None = None,
        fine_steps: int | None = None,
        offload_mode: str | None = None,
        seed: int = 0,
        prompt: str = "",
        timeout: int = 7200,
        mock: bool = True,
    ):
        self.python_bin = python_bin
        self.worker = worker
        self.ckpt = ckpt
        self.gemma_root = gemma_root
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.output_base = output_base
        self.seed = seed
        self.prompt = prompt
        self.timeout = timeout
        self.mock = mock
        # Env knobs win over defaults so sbatch exports tune runs without code edits.
        self.denoise_strength = (
            denoise_strength if denoise_strength is not None
            else float(os.environ.get("AVENHANCE_DENOISE_STRENGTH", str(DEFAULT_DENOISE_STRENGTH)))
        )
        _ads = os.environ.get("AVENHANCE_AUDIO_DENOISE_STRENGTH")
        self.audio_denoise_strength = (
            audio_denoise_strength if audio_denoise_strength is not None
            else (float(_ads) if _ads else None)
        )
        _fs = os.environ.get("AVENHANCE_FINE_STEPS")
        self.fine_steps = fine_steps if fine_steps is not None else (int(_fs) if _fs else None)
        self.offload_mode = offload_mode or os.environ.get("AVENHANCE_OFFLOAD_MODE", "none")

    def enhance(self, inpainted_video_path: str, residual_audio_path: str, sample_id: str,
                target_object: str = "unknown") -> AVEnhanceResult:
        if self.mock:
            paired_path = f"{self.output_base}/{sample_id}/paired_input.txt"
            enhanced_path = f"{self.output_base}/{sample_id}/{sample_id}_enhanced.txt"
            write_text_artifact(
                paired_path,
                f"mock paired AV input for sample={sample_id}, removed={target_object}\n",
            )
            write_text_artifact(
                enhanced_path,
                f"mock LTX-2 enhanced AV for sample={sample_id}, removed={target_object}\n",
            )
            return AVEnhanceResult(paired_input_path=paired_path, enhanced_path=enhanced_path,
                                   frames=185, audio_enhanced=True,
                                   denoise_strength=self.denoise_strength)
        return self._enhance_ltx(inpainted_video_path, residual_audio_path, sample_id)

    def _enhance_ltx(self, inpainted_video_path, residual_audio_path, sample_id) -> AVEnhanceResult:
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"LTX-2 interpreter not found: {self.python_bin}. Set up the venv or pass python_bin=..."
            )
        if not inpainted_video_path or not Path(inpainted_video_path).exists():
            raise FileNotFoundError(
                f"AV enhance needs the inpainted video; got {inpainted_video_path!r} (missing)."
            )
        if not residual_audio_path or not Path(residual_audio_path).exists():
            raise FileNotFoundError(
                f"AV enhance needs the residual audio; got {residual_audio_path!r} (missing)."
            )
        out_dir = f"{self.output_base}/{sample_id}"
        os.makedirs(out_dir, exist_ok=True)
        paired_path = os.path.join(out_dir, "paired_input.mp4")
        frames = self._conform_and_mux(inpainted_video_path, residual_audio_path, paired_path)

        enhanced_path = os.path.join(out_dir, f"{sample_id}_enhanced.mp4")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [
                self.python_bin, self.worker,
                "--video", paired_path,
                "--out-video", enhanced_path,
                f"--sample_id={sample_id}",
                "--ckpt", self.ckpt,
                "--gemma-root", self.gemma_root,
                "--prompt", self.prompt,
                "--denoise-strength", str(self.denoise_strength),
                "--seed", str(self.seed),
                "--offload-mode", self.offload_mode,
                "--out", tmp.name,
            ]
            if self.audio_denoise_strength is not None:
                cmd += ["--audio-denoise-strength", str(self.audio_denoise_strength)]
            if self.fine_steps is not None:
                cmd += ["--fine-steps", str(self.fine_steps)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  env=self._subprocess_env())
            if proc.returncode != 0:
                hint = ("\nHINT: CUDA OOM -- retry with AVENHANCE_OFFLOAD_MODE=cpu"
                        if "out of memory" in proc.stderr.lower() else "")
                raise RuntimeError(
                    f"LTX enhance worker failed (exit {proc.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDERR (tail):\n{proc.stderr[-4000:]}{hint}"
                )
            data = self._parse(tmp.name, proc.stdout)

        if not data.get("audio_enhanced"):
            raise RuntimeError(
                f"LTX enhance produced no audio for {sample_id}; the paired input mux "
                f"likely lost the audio stream: {paired_path}"
            )
        return AVEnhanceResult(
            paired_input_path=paired_path,
            enhanced_path=data.get("output_path", enhanced_path),
            frames=data.get("frames"),
            audio_enhanced=data.get("audio_enhanced"),
            audio_rms=data.get("audio_rms"),
            denoise_strength=data.get("denoise_strength"),
        )

    def _conform_and_mux(self, video_in: str, audio_in: str, out_path: str) -> int:
        """Mux video + wav into one mp4 satisfying the LTX VAE input rules:
        frames trimmed to the largest 8k+1, dims cropped to /32, audio as AAC
        (mp4 cannot carry pcm_s16le). Returns the conformed frame count."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        n, w, h = self._probe_video(video_in)
        frames = ((n - 1) // 8) * 8 + 1
        if frames < 9:
            raise RuntimeError(f"video too short for LTX enhance: {n} frames in {video_in}")
        cmd = [
            self.ffmpeg, "-y",
            "-i", video_in,
            "-i", audio_in,
            "-map", "0:v:0", "-map", "1:a:0",
            "-frames:v", str(frames),
        ]
        if w % 32 or h % 32:
            cmd += ["-vf", f"crop={w // 32 * 32}:{h // 32 * 32}"]
        cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conform/mux failed (exit {proc.returncode}).\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR (tail):\n{proc.stderr[-4000:]}"
            )
        got_frames, _, _ = self._probe_video(out_path)
        if got_frames != frames:
            raise RuntimeError(
                f"conform produced {got_frames} frames, expected {frames}: {out_path}"
            )
        if not self._has_audio_stream(out_path):
            raise RuntimeError(f"conform lost the audio stream: {out_path}")
        return frames

    def _probe_video(self, path: str) -> tuple[int, int, int]:
        cmd = [
            self.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed on {path}:\n{proc.stderr[-2000:]}")
        stream = json.loads(proc.stdout)["streams"][0]
        return int(stream["nb_read_frames"]), int(stream["width"]), int(stream["height"])

    def _has_audio_stream(self, path: str) -> bool:
        cmd = [
            self.ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name", "-of", "json", path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return proc.returncode == 0 and bool(json.loads(proc.stdout).get("streams"))

    def _subprocess_env(self) -> dict:
        # Mirror LTX-2/script/submit_infer_dmd.sh: the uv venv must not inherit
        # the orchestrator's PYTHONPATH/site-packages.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        return env

    @staticmethod
    def _parse(out_path: str, stdout: str, marker: str = RESULT_MARKER) -> dict:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        for line in stdout.splitlines():
            if line.startswith(marker):
                return json.loads(line[len(marker):])
        raise RuntimeError(f"Could not parse LTX enhance result.\nSTDOUT (tail):\n{stdout[-2000:]}")
