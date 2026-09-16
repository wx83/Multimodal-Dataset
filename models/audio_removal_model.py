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
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from utils import write_text_artifact

WORKER = Path(__file__).resolve().parent / "sam_audio_worker.py"
SELECT_WORKER = Path(__file__).resolve().parent / "ib_select_worker.py"
# Env overrides make the pipeline portable to other hosts (see SELF_HOSTING.md).
DEFAULT_SAMAUDIO_PYTHON = os.environ.get(
    "SAMAUDIO_PYTHON", "/home/weihan.xu/miniconda3/envs/samaudio311/bin/python")
DEFAULT_SAMAUDIO_MODEL_DIR = os.environ.get(
    "SAMAUDIO_MODEL_DIR", "/group2/ct/weihanx/av_langgraph_pipeline/pretrained_weight/sam_audio")
DEFAULT_SAM3_REPO = os.environ.get("SAM3_REPO", "/group2/ct/weihanx/sam3")
# ib_select_worker.py needs the JavisDiT repo (calc_imagebind_score + ./checkpoints).
DEFAULT_JAVISDIT_PYTHON = os.environ.get(
    "JAVISDIT_PYTHON", "/home/weihan.xu/miniconda3/envs/javisdit/bin/python")
DEFAULT_JAVISDIT_ROOT = os.environ.get("JAVISDIT_ROOT", "/group2/ct/weihanx/JavisDiT")
# Fixed seeds from /group2/ct/weihanx/ranking_test/seeds.json (reproducible best-of-10).
DEFAULT_BEST_OF_SEEDS = (1132891577, 1778986134, 240868205, 1453635084, 1335522078)
# Which stage-B selector picks the best-of winner. "ib" is the historical
# behaviour (max ib_ta, tiebreak min ib_ta_res). "clap" ranks by CLAP
# text<->audio removal (s_mix - s_res) — see clap_select_worker.py for why.
# Default stays "ib" so every existing run is reproduced bit-for-bit; flip it
# per run once an A/B on fresh generations says so.
DEFAULT_BESTOF_SELECTOR = os.environ.get("BESTOF_SELECTOR", "ib")
CLAP_SELECT_WORKER = Path(__file__).resolve().parent / "clap_select_worker.py"
DEFAULT_CLAP_SELECT_PYTHON = os.environ.get("CLAP_SELECT_PYTHON", sys.executable)
RESULT_MARKER = "SAM_AUDIO_RESULT "
IB_RESULT_MARKER = "IB_SELECT_RESULT "
CLAP_RESULT_MARKER = "CLAP_SELECT_RESULT "


@dataclass
class AudioRemovalResult:
    residual_path: str | None
    target_path: str | None = None
    # best-of-10 selection metadata (remove_best_of only)
    method: str | None = None       # "visual" or "text"
    seed: int | None = None
    ib_ta: float | None = None      # ImageBind text<->target (higher = better)
    ib_ta_res: float | None = None  # ImageBind text<->residual (lower = cleaner)
    selection_path: str | None = None
    selector: str | None = None     # "ib" | "clap" — which rule picked the winner
    clap_removal: float | None = None  # CLAP s_mix - s_res of the winner (clap selector only)


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
        javisdit_python: str = DEFAULT_JAVISDIT_PYTHON,
        javisdit_root: str = DEFAULT_JAVISDIT_ROOT,
        select_worker: str = str(SELECT_WORKER),
        best_of_seeds: tuple[int, ...] = DEFAULT_BEST_OF_SEEDS,
        selector: str = DEFAULT_BESTOF_SELECTOR,
        clap_select_python: str = DEFAULT_CLAP_SELECT_PYTHON,
        clap_select_worker: str = str(CLAP_SELECT_WORKER),
    ):
        if selector not in ("ib", "clap"):
            raise ValueError(f"BESTOF_SELECTOR must be 'ib' or 'clap', got {selector!r}")
        self.selector = selector
        self.clap_select_python = clap_select_python
        self.clap_select_worker = clap_select_worker
        self.model_dir = model_dir
        self.python_bin = python_bin
        self.worker = worker
        self.sam3_repo = sam3_repo
        self.output_base = output_base
        self.max_seconds = max_seconds
        self.fps = fps
        self.timeout = timeout
        self.mock = mock
        self.javisdit_python = javisdit_python
        self.javisdit_root = javisdit_root
        self.select_worker = select_worker
        self.best_of_seeds = tuple(best_of_seeds)

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

    def remove_best_of(self, av_pair_path: str, target_object: str, sample_id: str,
                       mask_path: str | None = None,
                       audio_path: str | None = None) -> AudioRemovalResult:
        """Best-of-10 separation: visual(mask) + text prompts x 5 seeds, then pick
        the winner by ImageBind score (max ib_ta, tiebreak min ib_ta_res).

        Stage A: sam_audio_worker.py --mode best_of (samaudio311 env) writes all
        candidates. Stage B: ib_select_worker.py (javisdit env, cwd=JavisDiT) scores
        them and copies the winner. The winner's residual is the FINAL
        object-removed audio.
        """
        if self.mock:
            residual_path = f"data/work/audio/{sample_id}_bestof_without_{target_object}.txt"
            write_text_artifact(
                residual_path,
                f"mock best-of residual for sample={sample_id}, removed={target_object}\n",
            )
            return AudioRemovalResult(residual_path=residual_path,
                                      method="visual", seed=self.best_of_seeds[0])
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"samaudio interpreter not found: {self.python_bin}. Set up the env or pass python_bin=..."
            )
        if not mask_path or not Path(mask_path).exists():
            raise FileNotFoundError(
                f"SAM-Audio best_of needs the SAM3 mask video; got mask_path={mask_path!r} (missing)."
            )
        # Absolute: stage B runs with cwd=JavisDiT, and candidates.json stores the
        # wav paths exactly as the worker built them from --output_dir.
        out_dir = os.path.abspath(f"{self.output_base}/{sample_id}/best_of")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            self.python_bin, self.worker,
            "--mode", "best_of",
            "--video", str(av_pair_path),
            "--mask", str(mask_path),
            "--prompt", target_object,
            "--seeds", ",".join(str(s) for s in self.best_of_seeds),
            f"--sample_id={sample_id}",
            "--model_dir", self.model_dir,
            "--sam3_repo", self.sam3_repo,
            "--output_dir", out_dir,
            "--max_seconds", str(self.max_seconds),
            "--fps", str(self.fps),
        ]
        if audio_path:
            cmd += ["--audio", str(audio_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                              env=self._subprocess_env())
        if proc.returncode != 0:
            raise RuntimeError(
                f"SAM-Audio (best_of) worker failed (exit {proc.returncode}).\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR (tail):\n{proc.stderr[-4000:]}"
            )

        selection_path = os.path.join(out_dir, "selection.json")
        if self.selector == "clap":
            return self._select_clap(out_dir, selection_path)
        cmd = [
            self.javisdit_python, self.select_worker,
            "--candidates", os.path.join(out_dir, "candidates.json"),
            "--out", selection_path,
        ]
        env = dict(os.environ)
        env_lib = str(Path(self.javisdit_python).resolve().parent.parent / "lib")
        env.update({
            "PYTHONPATH": self.javisdit_root,
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LD_LIBRARY_PATH": f"{env.get('LD_LIBRARY_PATH', '').rstrip(':')}:{env_lib}".lstrip(":"),
        })
        # cwd=JavisDiT root so ./checkpoints/imagebind_huge.pth resolves.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                              env=env, cwd=self.javisdit_root)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ImageBind selection worker failed (exit {proc.returncode}).\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR (tail):\n{proc.stderr[-4000:]}"
            )
        data = self._parse(selection_path, proc.stdout, marker=IB_RESULT_MARKER)
        best = data.get("best", {})
        return AudioRemovalResult(
            residual_path=best.get("best_residual"),
            target_path=best.get("best_target"),
            method=best.get("method"),
            seed=best.get("seed"),
            ib_ta=best.get("ib_ta"),
            ib_ta_res=best.get("ib_ta_res"),
            selection_path=selection_path,
            selector="ib",
        )

    def _select_clap(self, out_dir: str, selection_path: str) -> AudioRemovalResult:
        """Stage B with the CLAP selector (BESTOF_SELECTOR=clap). Same inputs and
        output files as the ImageBind path, so downstream nodes see no difference
        except the winner — and selection.json says which rule chose it."""
        cmd = [
            self.clap_select_python, self.clap_select_worker,
            "--candidates", os.path.join(out_dir, "candidates.json"),
            "--out", selection_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"CLAP selection worker failed (exit {proc.returncode}).\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR (tail):\n{proc.stderr[-4000:]}"
            )
        data = self._parse(selection_path, proc.stdout, marker=CLAP_RESULT_MARKER)
        best = data.get("best", {})
        return AudioRemovalResult(
            residual_path=best.get("best_residual"),
            target_path=best.get("best_target"),
            method=best.get("method"),
            seed=best.get("seed"),
            selection_path=selection_path,
            selector="clap",
            clap_removal=best.get("clap_removal"),
        )

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
    def _parse(out_path: str, stdout: str, marker: str = RESULT_MARKER) -> dict:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        for line in stdout.splitlines():
            if line.startswith(marker):
                return json.loads(line[len(marker):])
        raise RuntimeError(f"Could not parse SAM-Audio result.\nSTDOUT (tail):\n{stdout[-2000:]}")
