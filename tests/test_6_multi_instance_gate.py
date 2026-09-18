"""Behavior tests for the multi-instance consistency gate.

This gate catches samples where the visual side removed only one instance while
the audio side removed a whole class. On the visual side
`sam3_worker.segment_top_mask` takes only `masks[0]`; on the audio side SAM-Audio
separates every instance matching the text. When one word points at several
objects in the frame, the two sides remove different things, violating the
correctness constraint that the object removed from the audio must be the same
object removed from the video.

Off by default. Turning it on reduces delivered volume (measured at about 8.8%),
and a human should decide whether to pay that cost, so the most important test
here is test_off_by_default_changes_nothing -- with the gate off, behavior must
be bit-for-bit identical to before the gate existed.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _routes(require_single):
    """Reload routes with the given env var (thresholds are module-level constants, so only a reload picks them up)."""
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
    """The mock path and older workers do not report an instance count. Missing data is not the same as multi-instance; we must not drop samples on no evidence."""
    r = _routes(True)
    assert r.route_after_mask_check({**PASSES_AREA, "n_instances": None}) == "continue"
    assert r.route_after_mask_check(PASSES_AREA) == "continue"


def test_area_gate_still_runs_first():
    """A sample failing the area check must be discarded regardless of instance count -- adding a gate must never let something through that used to be rejected."""
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
