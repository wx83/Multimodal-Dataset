"""acoustic_desc：目标的声学描述贯通抽取节点与音频侧三处。

为什么要有这条测试：S55/S56 证明音频侧用视觉名词（man）拿不到信号、用声学标签可以；
但这是在分支上先落代码、等 GPU 才能做分离对照的改动，所以必须：
  · 两个开关默认关，历史行为逐位不变（抽取节点不多写字段，音频侧仍用 target_object）
  · 开了之后 acoustic_desc 真的流到分离 prompt 里，而视觉侧（分割/修复）仍用 target_object
  · 记录与 config 带上来源，两次运行可比
全部 mock，不加载模型。
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _reload_nodes(**env):
    for k in ("ACOUSTIC_DESC", "AUDIO_TEXT", "AVGRAPH_USE_REAL_MODELS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    import nodes
    importlib.reload(nodes)
    nodes._get_acoustic_desc_model.cache_clear()
    nodes._get_audio_removal_model.cache_clear()
    return nodes


def _teardown():
    _reload_nodes()


def test_defaults_off_and_extraction_output_unchanged():
    nodes = _reload_nodes()
    try:
        assert nodes.ACOUSTIC_DESC == "off" and nodes.AUDIO_TEXT == "object"
        out = nodes.sounding_object_extraction({"sample_id": "t", "caption": "a dog barks loudly"})
        assert "acoustic_desc" not in out
        assert out["target_object"] == out["sounding_objects"][0]
    finally:
        _teardown()


def test_audio_side_uses_target_object_by_default():
    nodes = _reload_nodes()
    try:
        st = {"sample_id": "t", "target_object": "dog", "acoustic_desc": "footsteps"}
        assert nodes._audio_text(st) == "dog"
        out = nodes.samaudio_best_of_remove(st)
        assert "without_dog" in out["text_residual_audio_path"]
    finally:
        _teardown()


def test_switch_on_produces_desc_and_audio_side_reads_it():
    nodes = _reload_nodes(ACOUSTIC_DESC="clap", AUDIO_TEXT="acoustic")
    try:
        out = nodes.sounding_object_extraction(
            {"sample_id": "t", "caption": "a dog barks loudly", "mock_acoustic_desc": "footsteps"})
        assert out["acoustic_desc"] == "footsteps" and out["acoustic_desc_source"] == "mock"
        assert out["target_object"] == "dog", "视觉侧目标不能被声学描述改掉"
        st = {"sample_id": "t", **out}
        assert nodes._audio_text(st) == "footsteps"
        out2 = nodes.samaudio_best_of_remove(st)
        assert "without_footsteps" in out2["text_residual_audio_path"]
        out3 = nodes.samaudio_remove_target(st)
        assert "without_footsteps" in out3["mask_residual_audio_path"]
    finally:
        _teardown()


def test_acoustic_text_falls_back_when_no_desc():
    nodes = _reload_nodes(ACOUSTIC_DESC="off", AUDIO_TEXT="acoustic")
    try:
        assert nodes._audio_text({"sample_id": "t", "target_object": "dog"}) == "dog"
    finally:
        _teardown()


def test_invalid_switch_values_rejected():
    for env in ({"ACOUSTIC_DESC": "yes"}, {"AUDIO_TEXT": "caption"}):
        try:
            _reload_nodes(**env)
        except ValueError:
            continue
        finally:
            _teardown()
        raise AssertionError(f"{env} 应该报错")


def test_config_and_record_carry_source():
    nodes = _reload_nodes(ACOUSTIC_DESC="clap", AUDIO_TEXT="acoustic")
    try:
        import utils
        importlib.reload(utils)
        cfg = utils.effective_config()
        assert cfg["acoustic_desc"] == "clap" and cfg["audio_text_source"] == "acoustic"
        rec = utils.build_state_record({"sample_id": "t", "target_object": "man",
                                        "acoustic_desc": "footsteps", "acoustic_desc_score": 0.3,
                                        "acoustic_desc_source": "clap-zeroshot"})
        assert rec["object_name"] == "man" and rec["acoustic_desc"] == "footsteps"
        assert rec["acoustic_desc_source"] == "clap-zeroshot"
    finally:
        _teardown()
        import utils
        importlib.reload(utils)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
