"""Acoustic description of the target's sound, via subprocess (acoustic_desc_worker.py).

The pipeline's extraction node emits a VISUAL noun (object_name). The audio side
needs a description of the SOUND: "man" is right for segmentation and wrong for
separation when the clip contains footsteps and no speech (strategy-lab S55/S56).
This wrapper adds `acoustic_desc` next to `target_object` without touching how
the visual side works. mock=True (the default in tests) never spawns a process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "acoustic_desc_worker.py"
DEFAULT_PYTHON = os.environ.get("ACOUSTIC_DESC_PYTHON", os.environ.get("CLAP_SELECT_PYTHON", sys.executable))
RESULT_MARKER = "ACOUSTIC_DESC_RESULT "
MOCK_LABEL = "footsteps"


@dataclass
class AcousticDescResult:
    acoustic_desc: str | None
    score: float | None = None
    source: str | None = None       # "clap-zeroshot" | "mock"
    top: list | None = None
    speech_score: float | None = None   # CLAP similarity to speech labels — reported, never the target
    speech_excluded: bool = False       # True for person-class targets (owner decision, A02)


class AcousticDescModel:
    def __init__(self, python_bin: str = DEFAULT_PYTHON, worker: str = str(WORKER),
                 timeout: int = 600, mock: bool = True):
        self.python_bin = python_bin
        self.worker = worker
        self.timeout = timeout
        self.mock = mock

    def describe(self, audio_path: str, mock_label: str | None = None,
                 target: str | None = None) -> AcousticDescResult:
        if self.mock:
            from clap_select_worker import is_person
            return AcousticDescResult(acoustic_desc=mock_label or MOCK_LABEL, score=1.0, source="mock",
                                      speech_excluded=is_person(target))
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"acoustic_desc needs the original audio; got {audio_path!r} (missing).")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            cmd = [self.python_bin, self.worker, "--audio", str(audio_path), "--out", tmp.name]
            if target:
                cmd += ["--target", target]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"acoustic_desc worker failed (exit {proc.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\nSTDERR (tail):\n{proc.stderr[-4000:]}")
            data = self._parse(tmp.name, proc.stdout)
        return AcousticDescResult(acoustic_desc=data.get("acoustic_desc"), score=data.get("acoustic_desc_score"),
                                  source=data.get("source"), top=data.get("top"),
                                  speech_score=data.get("speech_score"),
                                  speech_excluded=bool(data.get("speech_excluded", False)))

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
        raise RuntimeError(f"Could not parse acoustic_desc result.\nSTDOUT (tail):\n{stdout[-2000:]}")
