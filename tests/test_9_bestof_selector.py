"""best-of 选优开关与 CLAP 排序规则。

为什么要有这条测试：S52/S53 证明现行 ImageBind 键与移除质量不相关，CLAP 键有增益。
但增益是回测出来的上限，没在新生成上 A/B 过，所以新选优器必须**默认关闭**、
行为逐位不变；开了之后排序规则里的两条约束（不许把一切删光、tiebreak 看 target）
也得有据可查。这里不加载任何模型。
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from clap_select_worker import rank_candidates, pick_acoustic_label, ACOUSTIC_LABELS  # noqa: E402  纯 Python，不需要 torch


def test_default_text_source_is_object_and_invalid_rejected():
    os.environ.pop("BESTOF_TEXT", None)
    import audio_removal_model as arm
    importlib.reload(arm)
    assert arm.AudioRemovalModel(mock=True).bestof_text == "object"
    try:
        arm.AudioRemovalModel(mock=True, bestof_text="caption")
    except ValueError:
        pass
    else:
        raise AssertionError("未知的 BESTOF_TEXT 应该报错")


def test_env_switch_selects_acoustic_text():
    os.environ["BESTOF_TEXT"] = "acoustic"
    try:
        import audio_removal_model as arm
        importlib.reload(arm)
        assert arm.AudioRemovalModel(mock=True).bestof_text == "acoustic"
    finally:
        os.environ.pop("BESTOF_TEXT", None)
        importlib.reload(arm)


def test_pick_acoustic_label_is_argmax_over_table():
    sims = [0.0] * len(ACOUSTIC_LABELS)
    i = ACOUSTIC_LABELS.index("footsteps")
    sims[i] = 0.9
    label, score = pick_acoustic_label(sims)
    assert label == "footsteps" and score == 0.9
    assert len(set(ACOUSTIC_LABELS)) == len(ACOUSTIC_LABELS), "标签表有重复"


def _row(name, removal, energy, s_target=0.0):
    return {"name": name, "clap_removal": removal, "residual_energy": energy, "s_target": s_target}


def test_default_selector_is_ib_and_behaviour_unchanged():
    os.environ.pop("BESTOF_SELECTOR", None)
    import audio_removal_model as arm
    importlib.reload(arm)
    m = arm.AudioRemovalModel(mock=True)
    assert m.selector == "ib"
    # mock 路径不受开关影响：与加开关前逐位相同的返回
    r = m.remove_best_of("x.mp4", "dog", "s1")
    assert r.method == "visual" and r.seed == arm.DEFAULT_BEST_OF_SEEDS[0]


def test_env_switch_selects_clap():
    os.environ["BESTOF_SELECTOR"] = "clap"
    try:
        import audio_removal_model as arm
        importlib.reload(arm)
        assert arm.AudioRemovalModel(mock=True).selector == "clap"
    finally:
        os.environ.pop("BESTOF_SELECTOR", None)
        importlib.reload(arm)


def test_invalid_selector_is_rejected():
    import audio_removal_model as arm
    try:
        arm.AudioRemovalModel(mock=True, selector="imagebind")
    except ValueError:
        return
    raise AssertionError("未知的 selector 应该报错，而不是静默退回 ib")


def test_rank_picks_max_removal():
    rows = [_row("a", 0.01, 0.9), _row("b", 0.08, 0.9), _row("c", 0.05, 0.9)]
    assert [r["name"] for r in rank_candidates(rows)] == ["b", "c", "a"]


def test_rank_demotes_near_silent_residual():
    """removal 最大但残余几乎删光的候选必须排到后面——frozen_judge 警告的作弊形状。"""
    rows = [_row("silent", 0.50, 0.05), _row("honest", 0.08, 0.8)]
    assert rank_candidates(rows)[0]["name"] == "honest"


def test_rank_falls_back_when_all_violate_energy():
    rows = [_row("a", 0.10, 0.1), _row("b", 0.30, 0.2)]
    assert rank_candidates(rows)[0]["name"] == "b"


def test_rank_tiebreak_prefers_higher_target_similarity():
    rows = [_row("a", 0.05, 0.9, s_target=0.1), _row("b", 0.05, 0.9, s_target=0.4)]
    assert rank_candidates(rows)[0]["name"] == "b"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
