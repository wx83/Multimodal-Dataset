"""Interactive Gradio demo for the AV data-curation agent (see run_demo_server.sh).

Visitors upload a short clip (with audio) and watch the agent run for real:
caption -> sounding object -> SAM3 mask -> EffectErase inpaint -> gates ->
SAM-Audio best-of removal -> LTX-2 joint AV enhancement.

The OpenAI key field is optional and used ONLY for the single sounding-object
extraction call of that request (never stored or logged); visitors can instead
type the target object directly and skip the LLM step entirely.

Runs on a GPU node (launched via sbatch, gradio share link printed to the log).
Requests are processed one at a time; a full pass takes ~20-40 min.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr

import nodes
from main import build_graph
from models.object_extraction_model import ObjectExtractionModel

FFMPEG = os.environ.get(
    "AVGRAPH_FFMPEG", "/group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/ffmpeg")
DEMO_DIR = ROOT / "data" / "work" / "demo"
GRAPH = build_graph()

NODE_LABELS = {
    "av_caption_generation": "1/8 captioning (Qwen3-Omni, ~10 min)",
    "sounding_object_extraction": "2/8 extracting sounding object",
    "target_object_segmentation": "3/8 segmenting target (SAM3)",
    "effect_erase_inpainting": "4/8 inpainting video (EffectErase, ~10 min)",
    "inpainted_video_check": "5/8 visual removal check (SAM3)",
    "samaudio_best_of_remove": "6/8 audio separation (SAM-Audio best-of-10, ~10 min)",
    "audio_removal_check": "7/8 audio removal check",
    "av_quality_enhancement": "8/8 LTX-2 joint AV enhancement (~8 min)",
}


class _PinnedExtractor(ObjectExtractionModel):
    def __init__(self, target: str):
        super().__init__(mock=True)
        self._target = target

    def extract(self, caption: str) -> list[str]:
        return [self._target]


def _sh(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-1500:]}")


def _h264(src: str | Path, dst: Path) -> str | None:
    """Browser-safe copy of a pipeline artifact (SAM3 overlays are mpeg4)."""
    if not src or not Path(src).exists():
        return None
    _sh([FFMPEG, "-y", "-v", "error", "-i", str(src),
         "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dst)])
    return str(dst)


def run_agent(video, openai_key, target_override):
    outs = {"status": "", "caption": "", "target": "", "overlay": None,
            "inpainted": None, "residual": None, "enhanced": None}

    def emit():
        return (outs["status"], outs["caption"], outs["target"], outs["overlay"],
                outs["inpainted"], outs["residual"], outs["enhanced"])

    if not video:
        outs["status"] = "Please upload a video (with audio)."
        yield emit(); return

    sample_id = f"demo_{int(time.time())}"
    work = DEMO_DIR / sample_id
    work.mkdir(parents=True, exist_ok=True)

    # Normalize the upload: <=8 s, 24 fps, even dims, keep audio; extract the
    # 48 kHz mono wav SAM-Audio expects (mirrors preprocess.py).
    norm_mp4 = work / "input.mp4"
    wav = work / "input.wav"
    try:
        _sh([FFMPEG, "-y", "-v", "error", "-i", str(video), "-t", "8", "-r", "24",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", str(norm_mp4)])
        _sh([FFMPEG, "-y", "-v", "error", "-i", str(norm_mp4),
             "-vn", "-ac", "1", "-ar", "48000", str(wav)])
    except RuntimeError as e:
        outs["status"] = f"Could not read the upload ({e}). Does the clip have both video and audio?"
        yield emit(); return

    # Sounding-object extraction backend for THIS request only (requests are
    # serialized, so swapping the singleton getter is safe).
    target_override = (target_override or "").strip()
    key = (openai_key or "").strip()
    if target_override:
        extractor = _PinnedExtractor(target_override)
        mode = f"pinned target: {target_override!r}"
    elif key:
        extractor = ObjectExtractionModel(mock=False, api_key=key)
        mode = "GPT via your OpenAI key (used once, not stored)"
    else:
        extractor = ObjectExtractionModel(mock=True)
        mode = "offline keyword matcher (limited vocab — consider typing a target)"
    nodes._get_object_model = lambda: extractor

    initial = {
        "sample_id": sample_id,
        "av_pair_path": str(norm_mp4),
        "audio_path": str(wav),
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
    }

    outs["status"] = f"Starting agent (object extraction: {mode})…"
    yield emit()

    state = {}
    try:
        for step in GRAPH.stream(initial, stream_mode="updates"):
            for node_name, update in step.items():
                state.update(update or {})
                outs["status"] = f"Running: {NODE_LABELS.get(node_name, node_name)} — done"
                outs["caption"] = state.get("caption", outs["caption"])
                if state.get("target_object"):
                    outs["target"] = state["target_object"]
                if state.get("mask_path") and outs["overlay"] is None:
                    overlay_src = Path(state["mask_path"]).parent / f"{sample_id}_overlay.mp4"
                    outs["overlay"] = _h264(overlay_src, work / "overlay_h264.mp4")
                if state.get("inpainted_video_path") and outs["inpainted"] is None:
                    outs["inpainted"] = _h264(state["inpainted_video_path"], work / "inpainted_h264.mp4")
                if state.get("text_residual_audio_path") and outs["residual"] is None:
                    p = state["text_residual_audio_path"]
                    if p and Path(p).exists():
                        outs["residual"] = str(shutil.copy(p, work / "residual.wav"))
                if state.get("enhanced_video_path") and outs["enhanced"] is None:
                    src = state["enhanced_video_path"]
                    if src and Path(src).exists() and str(src).endswith(".mp4"):
                        outs["enhanced"] = str(shutil.copy(src, work / "enhanced.mp4"))
                yield emit()
    except Exception as e:
        outs["status"] = f"Pipeline error: {e}"
        yield emit(); return

    if state.get("status") == "passed":
        outs["status"] = ("✅ Passed every gate! "
                          f"mask={state.get('mask_area_ratio')}, "
                          f"visual={state.get('visual_removal_score')}, "
                          f"audio={state.get('audio_removal_score')}")
    else:
        outs["status"] = (f"⚠️ Discarded at {state.get('discard_stage')}: "
                          f"{state.get('discard_reason')} "
                          f"(mask={state.get('mask_area_ratio')}, "
                          f"visual={state.get('visual_removal_score')}) — "
                          "that's the curation gates doing their job.")
    yield emit()


with gr.Blocks(title="AV Data-Curation Agent — Live Demo") as demo:
    gr.Markdown(
        "# 🎬🔊 AV Data-Curation Agent — Live Demo\n"
        "Upload a short clip **with audio** (first 8 s are used). The agent captions it, "
        "picks the sounding object, erases it from the video (SAM3 + EffectErase), separates "
        "its sound (SAM-Audio best-of-10), and jointly refines the pair with LTX-2.\n\n"
        "⏱️ A full pass takes **20–40 min** on one H100; requests run one at a time. "
        "Gate failures are reported honestly — most clips are discarded, that's the point.\n\n"
        "🔑 The OpenAI key is optional (one `gpt-4o-mini` call for object extraction; "
        "never stored or logged). **Tip:** type the target object instead and skip the key."
    )
    with gr.Row():
        with gr.Column(scale=1):
            in_video = gr.Video(label="Your clip (with audio)", sources=["upload"])
            in_target = gr.Textbox(label="Target object (optional — skips the LLM step)",
                                   placeholder="e.g. dog, car engine, violin")
            in_key = gr.Textbox(label="OpenAI API key (optional, used once, not stored)",
                                type="password", placeholder="sk-…")
            btn = gr.Button("Run the agent", variant="primary")
            out_status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            out_caption = gr.Textbox(label="1. Caption (Qwen3-Omni)", interactive=False)
            out_target = gr.Textbox(label="2. Target object", interactive=False)
            with gr.Row():
                out_overlay = gr.Video(label="3. SAM3 mask overlay")
                out_inpainted = gr.Video(label="4. Inpainted (silent)")
            with gr.Row():
                out_residual = gr.Audio(label="5. Residual audio (target removed)")
                out_enhanced = gr.Video(label="6. LTX-2 enhanced AV pair")

    btn.click(run_agent, inputs=[in_video, in_key, in_target],
              outputs=[out_status, out_caption, out_target, out_overlay,
                       out_inpainted, out_residual, out_enhanced],
              concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=5).launch(share=True, server_name="0.0.0.0", show_error=True)
