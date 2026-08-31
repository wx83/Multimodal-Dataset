"""生效配置是否随每条记录落盘。

这条测试对应 2026-08-30 最贵的那个教训：8907 条历史运行没记录闸档，
导致 h₀ = 21.84% 无法跨配置比较——它产生自首帧闸 0.05，而代码预期值是 0.80，
按预期值回算交付量从 1945 条掉到 45 条。同一个指标在两档之间差 40 倍，
而没人发现，因为档位不在日志里。

对「提供架构、使用者自己换 model」这个目标，这一条是必需品而非锦上添花：
换模型的意义全在比较，两次运行不知道各自的配置就没法比。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import build_state_record, effective_config  # noqa: E402

# 每条都必须落盘：少任何一条，某类跨运行比较就失效
REQUIRED = [
    "mask_area_threshold",
    "visual_score_threshold",
    "audio_score_threshold",
    "require_single_instance",
    "sam3_first_frame_threshold",
    "sam3_first_frame_intended",
    "gate_relaxed",
    "caption_model",
    "extraction_model",
]


def test_all_required_keys_present():
    cfg = effective_config()
    missing = [k for k in REQUIRED if k not in cfg]
    assert not missing, f"缺字段：{missing}"


def test_no_group_failed_to_load():
    """惰性 import 任一组失败都会留下 *_error，静默降级比缺字段更危险。"""
    cfg = effective_config()
    errors = {k: v for k, v in cfg.items() if k.endswith("_error")}
    assert not errors, f"有分组加载失败：{errors}"


def test_thresholds_come_from_the_single_source():
    """阈值必须与 routes/nodes 逐位一致——另抄一份就会漂移（见 commit 3bae3f0）。"""
    import nodes
    import routes
    cfg = effective_config()
    assert cfg["mask_area_threshold"] == routes.MASK_AREA_THRESHOLD
    assert cfg["visual_score_threshold"] == routes.VISUAL_SCORE_THRESHOLD
    assert cfg["audio_score_threshold"] == routes.AUDIO_SCORE_THRESHOLD
    assert cfg["require_single_instance"] == routes.REQUIRE_SINGLE_INSTANCE
    assert cfg["sam3_first_frame_threshold"] == nodes.SAM3_FIRST_FRAME_THRESHOLD


def test_relaxed_gate_is_flagged():
    """闸被放宽时必须显式标记，否则开发档的运行事后看起来像正常运行。"""
    import nodes
    cfg = effective_config()
    expected = nodes.SAM3_FIRST_FRAME_THRESHOLD < nodes.SAM3_FIRST_FRAME_INTENDED
    assert cfg["gate_relaxed"] is expected


def test_record_carries_config():
    rec = build_state_record({"sample_id": "t", "target_object": "dog", "status": "passed"})
    assert "config" in rec, "记录里没有 config 块"
    assert all(k in rec["config"] for k in REQUIRED)


def test_logging_never_breaks_the_pipeline():
    """空 state 也要能建记录——落盘失败绝不能拖垮 pipeline。"""
    rec = build_state_record({})
    assert "config" in rec


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
