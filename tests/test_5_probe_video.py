import json
import subprocess
from pathlib import Path


def ffprobe_video(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    video_path = Path("data/raw/sample_0001.mp4")
    assert video_path.exists(), f"Missing video: {video_path}. Put a real video here first."
    assert video_path.stat().st_size > 0, f"Empty video file: {video_path}"

    info = ffprobe_video(video_path)
    print(json.dumps(info, indent=2))

    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    assert has_video, "No video stream found."
    print("Video stream found.")
    if has_audio:
        print("Audio stream found.")
    else:
        print("Warning: no audio stream found.")

    print("Video probe passed.")


if __name__ == "__main__":
    main()
