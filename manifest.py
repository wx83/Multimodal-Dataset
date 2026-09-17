"""The pipeline describes itself: node metadata and gate identity, declared next
to the code they describe and read by any tool that wants to render or audit
the graph (avgraph-strategy-lab's studio reads this; it types nothing itself).

Why (strategy-lab S66, owner 2026-09-16): a scaffold that keeps its own table of
"what each node is" drifts the moment the pipeline changes, and cannot be
pointed at another pipeline without editing the scaffold. LangGraph gives
topology through get_graph(); everything else — title, model, how freely it may
be swapped, whether a gate actually measures, which discard_stage it writes,
which source files back it — is knowledge only the pipeline has.

Usage:
    @node_meta(title="…", model="…", swap="L3", files=[…])
    def some_node(state): …

    GATES.append(gate(after="node_id", discard_stage="mask_check",
                      condition=lambda: f"mask_area_ratio > {MASK_AREA_THRESHOLD}",
                      measures=lambda: True, files=["routes.py"]))

`condition` and `measures` are callables so the manifest always reflects the
live value (env overrides included) rather than an import-time snapshot.
"""
from __future__ import annotations

from typing import Callable

NODE_META: dict[str, dict] = {}
GATES: list[dict] = []


def node_meta(title: str, *, model: str | None = None, swap: str | None = None,
              measures: bool | None = None, flag: str | None = None,
              terminal: str | None = None, files: list[str] | None = None):
    """Attach display/audit metadata to a graph node function. Keyed by the
    function name, which is also the node id LangGraph reports."""
    def deco(fn):
        NODE_META[fn.__name__] = {k: v for k, v in {
            "title": title, "model": model, "swap": swap, "measures": measures,
            "flag": flag, "terminal": terminal, "files": files or [],
        }.items() if v is not None and v != []}
        return fn
    return deco


def gate(*, after: str, discard_stage: str, condition: Callable[[], str],
         measures: Callable[[], bool], files: list[str] | None = None) -> dict:
    return {"after": after, "discard_stage": discard_stage, "condition": condition,
            "measures": measures, "files": files or []}


def snapshot() -> dict:
    """Resolve callables → plain JSON. This is what emit_manifest embeds."""
    return {
        "nodes": {k: dict(v) for k, v in NODE_META.items()},
        "gates": [{"after": g["after"], "discard_stage": g["discard_stage"],
                   "condition": g["condition"](), "measures": bool(g["measures"]()),
                   "files": list(g["files"])} for g in GATES],
    }
