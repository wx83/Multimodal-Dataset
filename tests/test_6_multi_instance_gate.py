"""多实例一致性闸的行为测试。

这个闸拦的是「视觉侧只删了一个实例、音频侧删了一类」的样本。视觉侧
`sam3_worker.segment_top_mask` 只取 `masks[0]`，音频侧 SAM-Audio 按文本分离
全部实例；当一个词在画面里指向多个物体时，两侧删的就不是同一个东西，
违反「音频侧移除的对象与视觉侧移除的对象必须是同一个」这条 correctness 约束。

默认关闭。打开会减少交付量（实测约 8.8%），该由人决定要不要付这个代价，
所以这里最重要的一条测试是 test_off_by_default_changes_nothing——
闸关着时行为必须与加闸前逐位相同。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _routes(require_single):
    """按给定环境变量重新加载 routes（阈值是模块级常量，只能靠 reload 生效）。"""
    old = os.environ.get("REQUIRE_SINGLE_INSTANCE")
    os.environ["REQUIRE_SINGLE_INSTANCE"] = "1" if require_single else "0"
    import routes
    importlib.reload(routes)
    if old is None:
        os.environ.pop("REQUIRE_SINGLE_INSTANCE", None)
    else:
        os.environ["REQUIRE_SINGLE_INSTANCE"] = old
    return routes


PASSES_AREA = {"mask_area_ratio": 0.30}


def test_off_by_default_changes_nothing():
    r = _routes(False)
    for n in (None, 0, 1, 2, 7):
        assert r.route_after_mask_check({**PASSES_AREA, "n_instances": n}) == "continue"


def test_on_discards_multi_instance():
    r = _routes(True)
    assert r.route_after_mask_check({**PASSES_AREA, "n_instances": 2}) == "discard"
    assert r.route_after_mask_check({**PASSES_AREA, "n_instances": 1}) == "continue"


def test_missing_count_is_not_treated_as_multi():
    """mock 路径与旧 worker 不上报实例数。缺数据不等于多实例，不能凭空丢样本。"""
    r = _routes(True)
    assert r.route_after_mask_check({**PASSES_AREA, "n_instances": None}) == "continue"
    assert r.route_after_mask_check(PASSES_AREA) == "continue"


def test_area_gate_still_runs_first():
    """面积不达标的样本，无论实例数多少都该被拦——加闸不该反过来放行。"""
    r = _routes(True)
    assert r.route_after_mask_check({"mask_area_ratio": 0.01, "n_instances": 1}) == "discard"


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
