"""Generate the pipeline input jsonl: one line per video file in a folder.

    python generate_inputs.py --video_dir /group2/ct/weihanx/8sec_raw_video \
        --output inputs.jsonl

Each line: {"video_id": <filename stem>, "video": <absolute path>}
This feeds preprocess.py (which extracts audio) and then run.py.
"""

import argparse
import json
import os


def iter_videos(root: str, ext: str, recursive: bool):
    ext = ext.lower()
    if recursive:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(ext):
                    yield os.path.join(dirpath, fn)
    else:
        with os.scandir(root) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(ext):
                    yield e.path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--output", default="inputs.jsonl")
    ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    args = ap.parse_args()

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in iter_videos(args.video_dir, args.ext, args.recursive):
            video_id = os.path.splitext(os.path.basename(path))[0]
            out.write(json.dumps(
                {"video_id": video_id, "video": os.path.abspath(path)},
                ensure_ascii=False,
            ) + "\n")
            n += 1
    print(f"Wrote {n} entries to {args.output}")


if __name__ == "__main__":
    main()
