"""Qwen3-Omni captioning worker.

This script runs INSIDE the dedicated `qwen3omni` conda env (which has torch,
transformers>=5.2, qwen-omni-utils, ffmpeg). It is invoked as a subprocess by
models/caption_model.py — never imported by the avgraph pipeline.

It loads Qwen3-Omni once, captions a single video (using the embedded audio
track), and writes {"caption": ...} as JSON, both to --out (if given) and to
stdout on a line prefixed with the RESULT_MARKER so the parent can parse it
robustly even amid model-loading logs.

Run (on a GPU node, e.g. inside `srun --gres=gpu:1 --pty bash -l`):

    /group2/ct/weihanx/miniconda3/envs/qwen3omni/bin/python \
        models/qwen3_omni_worker.py --video data/raw/sample_0001.mp4
"""

import argparse
import json
import sys

import transformers

# 模型类不再写死。换 captioner 是这套架构的卖点之一（"他们可以自己换 model"），
# 而写死类名会让换模型必须改代码——传 Qwen2.5-Omni 的 checkpoint 给 Qwen3 的 MoE 类
# 只会加载失败。改成按名字解析，换模型只需多传两个参数。
#
# **但这只解开一处耦合，不等于 captioner 可以随便换。** caption_video() 里还有
# 两处 Qwen-Omni 家族特有的东西：process_mm_info（来自 qwen_omni_utils），
# 以及 generate 的 thinker_return_dict_in_generate / return_audio 参数（thinker-talker
# 双头架构特有）。Qwen2.5-Omni 同属该家族，大概率能跑但未在真机验证过；
# 换到非 Qwen 的 omni 模型必须写新 worker。换模型的完整说明见 WORKER_CONTRACT.md。
DEFAULT_MODEL_CLASS = "Qwen3OmniMoeForConditionalGeneration"
DEFAULT_PROCESSOR_CLASS = "Qwen3OmniMoeProcessor"
from qwen_omni_utils import process_mm_info

RESULT_MARKER = "QWEN3_OMNI_RESULT "

CAPTION_PROMPT = """You will be given a VIDEO WITH AUDIO.
Write a concise audiovisual caption that describes:
1) what is happening visually (key entities + actions),
2) what is heard (key sounds),

Rules:
- Prefer concrete nouns/verbs.
- Ignore narration/background music unless it dominates or is clearly relevant.
- If important sounds are offscreen, mention that.
"""


def _resolve(class_name: str):
    """按名字从 transformers 取类。取不到就当场报清楚，别等到 from_pretrained 才炸。"""
    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise SystemExit(
            f"transformers {transformers.__version__} 里没有 {class_name}。\n"
            f"  换模型时 --model_class / --processor_class 要与 checkpoint 匹配，"
            f"且当前 transformers 版本要支持该模型。"
        )
    return cls


def load(model_name: str, attn_implementation: str,
         model_class: str = DEFAULT_MODEL_CLASS,
         processor_class: str = DEFAULT_PROCESSOR_CLASS):
    model = _resolve(model_class).from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    processor = _resolve(processor_class).from_pretrained(model_name, use_fast=False)
    return model, processor


def caption_video(model, processor, video: str, max_new_tokens: int,
                  use_audio_in_video: bool) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text,
        audio=audios,        # extracted from the video's audio track
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # return_audio=False -> text-only: skip the speech "talker" head, which is
    # what we want for captioning (and it avoids a talker/transformers API clash).
    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        thinker_return_dict_in_generate=True,
        return_audio=False,
        use_audio_in_video=use_audio_in_video,
    )
    # With return_audio=False the model returns the thinker output, either
    # directly or as the first element of a tuple.
    thinker_out = generation[0] if isinstance(generation, (tuple, list)) else generation

    caption = processor.batch_decode(
        thinker_out.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--video", required=True,
                    help="Local path or URL to a video file with an audio track.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--attn_implementation", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="Use flash_attention_2 on a capable GPU node; sdpa is the safe default.")
    ap.add_argument("--use_audio_in_video", action="store_true", default=True,
                    help="Use the audio embedded in the video (critical for AV captions).")
    ap.add_argument("--no_audio_in_video", dest="use_audio_in_video", action="store_false")
    ap.add_argument("--out", default=None, help="Optional path to write the JSON result.")
    ap.add_argument("--model_class", default=DEFAULT_MODEL_CLASS,
                    help="transformers 里的模型类名。换 captioner 时必须与 checkpoint 匹配"
                         "（Qwen2.5-Omni 与 Qwen3-Omni 不是同一套类）。")
    ap.add_argument("--processor_class", default=DEFAULT_PROCESSOR_CLASS,
                    help="transformers 里的 processor 类名，同上。")
    args = ap.parse_args()

    model, processor = load(args.model, args.attn_implementation,
                            args.model_class, args.processor_class)
    caption = caption_video(
        model, processor, args.video, args.max_new_tokens, args.use_audio_in_video
    )

    # 记下用的是哪个模型和哪个类：换模型做对比时，没有这个就无法事后区分两次运行
    result = {"caption": caption, "video": args.video, "model": args.model,
              "model_class": args.model_class}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    # Parent process parses this marker line from stdout.
    sys.stdout.write(RESULT_MARKER + json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
