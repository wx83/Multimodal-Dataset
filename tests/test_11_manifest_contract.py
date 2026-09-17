"""The pipeline's self-description must match the pipeline (manifest contract v2).

Why: any tool that renders or audits this graph (the studio) used to keep its
own table of node titles, models and gate→discard_stage mappings. That table
drifted (12 hand-drawn nodes vs 10 real ones, S45) and made the scaffold
unusable on any other pipeline. Now the declarations live next to the code,
and this test keeps them honest: every real node is described, every gate that
routes to discard is declared with the discard_stage the node really writes,
and every declared file exists. Mock only; no models are loaded.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _graph():
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        from main import build_graph
        return build_graph().get_graph()


def test_every_graph_node_has_metadata():
    from manifest import NODE_META
    g = _graph()
    ids = [n for n in g.nodes if n not in ("__start__", "__end__")]
    missing = [n for n in ids if n not in NODE_META]
    assert not missing, f"节点没有 @node_meta：{missing}"
    extra = [n for n in NODE_META if n not in ids]
    assert not extra, f"@node_meta 描述了图里不存在的节点：{extra}"
    for n in ids:
        assert NODE_META[n].get("title"), f"{n} 缺 title"


def test_every_discard_gate_is_declared_once():
    from manifest import GATES
    g = _graph()
    gated = {e.source for e in g.edges if e.target == "discard_sample" and getattr(e, "conditional", False)}
    declared = [x["after"] for x in GATES]
    assert set(declared) == gated, f"GATES 与图上通向 discard 的条件边不一致：declared={sorted(declared)} graph={sorted(gated)}"
    assert len(declared) == len(set(declared)), "同一节点声明了两道闸"


def test_declared_discard_stage_is_what_the_node_writes():
    """discard_stage 是记录里的名字，闸门必须声明它写的那个，不然漏斗对不上。"""
    from manifest import GATES
    src = open(os.path.join(ROOT, "nodes.py"), encoding="utf-8").read()
    written = set(re.findall(r'"discard_stage":\s*"([a-z_]+)"', src))
    for x in GATES:
        assert x["discard_stage"] in written, f"闸 after={x['after']} 声明的 discard_stage={x['discard_stage']} 从未被 nodes.py 写入"


def test_snapshot_is_plain_json_and_live():
    import json
    from manifest import snapshot
    snap = snapshot()
    json.dumps(snap)                                  # 无 callable 残留
    conds = {x["after"]: x["condition"] for x in snap["gates"]}
    import routes
    assert str(routes.MASK_AREA_THRESHOLD) in conds["target_object_segmentation"]
    assert all(isinstance(x["measures"], bool) for x in snap["gates"])
    import nodes
    audio = next(x for x in snap["gates"] if x["after"] == "audio_removal_check")
    assert audio["measures"] is nodes.AUDIO_SCORE_IS_MEASURED


def test_declared_files_exist():
    from manifest import NODE_META, GATES
    for n, m in NODE_META.items():
        for f in m.get("files", []):
            assert os.path.exists(os.path.join(ROOT, f)), f"{n} 声明的文件不存在：{f}"
    for x in GATES:
        for f in x["files"]:
            assert os.path.exists(os.path.join(ROOT, f)), f"闸 {x['after']} 声明的文件不存在：{f}"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
