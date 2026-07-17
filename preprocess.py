"""Preprocess: extract the audio track from each input AV video before the pipeline.

Input jsonl — one object per line with a video id and the AV video (with audio):
    {"video_id": "sample_0001", "video": "data/raw/sample_0001.mp4"}
Also accepts "video_path", or "video_dir" (+ "<video_id>.mp4").

For each entry it extracts a 48 kHz mono wav to <audio_dir>/<video_id>.wav (the
SAM-Audio input) and writes a resolved jsonl that run.py consumes:
    {"video_id", "video", "audio"}

Usage:
    python preprocess.py --input_jsonl inputs.jsonl --audio_dir data/work/source_audio
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def resolve_video(entry: dict) -> str:
    vid = entry["video_id"]
    for key in ("video", "video_path", "av_pair_path"):
        if entry.get(key):
            return entry[key]
    if entry.get("video_dir"):
        return str(Path(entry["video_dir"]) / f"{vid}.mp4")
    raise KeyError(f"no video path for {vid!r} (need 'video'/'video_path'/'video_dir')")


def extract_audio(ffmpeg: str, video: str, out_wav: str, sample_rate: int = 48000) -> None:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(sample_rate), out_wav],
        check=True, capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--audio_dir", default="data/work/source_audio")
    ap.add_argument("--output_jsonl", default=None,
                    help="Resolved jsonl for run.py (default: <input>_preprocessed.jsonl).")
    ap.add_argument("--sample_rate", type=int, default=48000)
    args = ap.parse_args()

    out_jsonl = args.output_jsonl or (os.path.splitext(args.input_jsonl)[0] + "_preprocessed.jsonl")
    ffmpeg = _ffmpeg()

    with open(args.input_jsonl) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Preprocessing {len(entries)} videos -> {out_jsonl}")

    results = []
    for e in entries:
        vid = e["video_id"]
        try:
            video = resolve_video(e)
            if not os.path.exists(video):
                print(f"[skip] {vid}: video not found: {video}")
                continue
            audio = os.path.join(args.audio_dir, f"{vid}.wav")
            extract_audio(ffmpeg, video, audio, args.sample_rate)
            results.append({"video_id": vid, "video": video, "audio": audio})
            print(f"[ok] {vid} -> {audio}")
        except Exception as ex:
            print(f"[skip] {vid}: {ex}")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)}/{len(entries)} resolved entries to {out_jsonl}")


if __name__ == "__main__":
    main()
