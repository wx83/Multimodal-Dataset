"""六个 worker 是否真的遵守同一份子进程契约。

为什么要有这条测试：`WORKER_CONTRACT.md` 声称换模型的入口是这份契约，
而不是类名注入。但文档会过时，而且我写下「6/6 一致」时只 grep 了 marker
和 --out 的存在，没核 wrapper 那一侧——这正是前一步刚犯过的错
（改了 load() 就说 captioner 可换，没读 caption_video）。

契约（见 WORKER_CONTRACT.md）：
  worker 侧   —— 接受 --out <path> 写结果 JSON；同时 stdout 打一行
                 `<NAME>_RESULT {json}` 并 flush
  wrapper 侧 —— 先读 --out 指向的文件，读不到再从 stdout 找 marker 行，
                 都失败才抛错

这里做的是静态检查：不加载模型，只读源码。目的是让新增 worker 无法
悄悄破坏契约，而不是验证模型跑得对。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

# worker 文件 -> 驱动它的 wrapper 文件。
# audio_removal_model.py 同时驱动两个 worker（SAM-Audio 分离 + ImageBind 选优），
# 所以它在 _parse 时显式传 marker=；一个 wrapper 对多个 worker 是允许的。
PAIRS = {
    "qwen3_omni_worker.py": "caption_model.py",
    "sam3_worker.py": "segmentation_model.py",
    "sam_audio_worker.py": "audio_removal_model.py",
    "effecterase_worker.py": "inpainting_model.py",
    "ltx_enhance_worker.py": "av_enhance_model.py",
    "ib_select_worker.py": "audio_removal_model.py",
}


def src(name):
    with open(os.path.join(MODELS, name), encoding="utf-8") as f:
        return f.read()


def test_every_worker_declares_a_result_marker():
    for w in PAIRS:
        s = src(w)
        assert re.search(r'RESULT_MARKER\s*=\s*"[A-Z0-9_]+_RESULT "', s), \
            f"{w} 没有声明形如 <NAME>_RESULT 的 marker"


def test_every_worker_accepts_out_and_writes_it():
    for w in PAIRS:
        s = src(w)
        assert '"--out"' in s, f"{w} 不接受 --out"


def test_every_worker_also_emits_the_marker_line():
    """兜底通道：文件写失败时 stdout 那行是唯一的结果来源，不能省。"""
    for w in PAIRS:
        s = src(w)
        assert "RESULT_MARKER +" in s or "RESULT_MARKER+" in s, \
            f"{w} 没有把结果打到 stdout"


def test_every_wrapper_parses_file_first_then_marker():
    """顺序很重要：stdout 可能被模型库的日志污染，文件才是主通道。"""
    for w, wrapper in PAIRS.items():
        s = src(wrapper)
        assert "RESULT_MARKER" in s, f"{wrapper} 没有用 marker 解析"
        assert "json.load" in s or "json.loads" in s, f"{wrapper} 没有解析 JSON"
        i_file = s.find("json.load(")
        i_marker = s.find("RESULT_MARKER):")
        if i_file != -1 and i_marker != -1:
            assert i_file < i_marker, \
                f"{wrapper} 似乎先找 stdout marker 再读文件，与契约相反"


def test_markers_are_unique_per_worker():
    """两个 worker 用同一个 marker 会导致解析串台。"""
    seen = {}
    for w in PAIRS:
        m = re.search(r'RESULT_MARKER\s*=\s*"([A-Z0-9_]+_RESULT) "', src(w))
        assert m, f"{w} 的 marker 取不到"
        name = m.group(1)
        assert name not in seen, f"{w} 与 {seen[name]} 用了同一个 marker {name}"
        seen[name] = w


def test_contract_doc_exists_and_lists_every_marker():
    """文档漏掉某个 marker，就说明有 worker 没被记进说明书。"""
    doc_path = os.path.join(ROOT, "WORKER_CONTRACT.md")
    assert os.path.exists(doc_path), "WORKER_CONTRACT.md 不存在"
    doc = open(doc_path, encoding="utf-8").read()
    for w in PAIRS:
        m = re.search(r'RESULT_MARKER\s*=\s*"([A-Z0-9_]+_RESULT) "', src(w))
        assert m.group(1) in doc, f"{m.group(1)}（来自 {w}）没写进 WORKER_CONTRACT.md"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
