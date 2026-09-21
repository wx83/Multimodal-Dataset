"""Whether the effective config is written to disk with every record.

This test encodes the most expensive lesson of 2026-08-30: 8907 historical runs
never recorded which gate setting they ran under, so h0 = 21.84% cannot be
compared across configurations -- it came from a first-frame gate of 0.05, while
the value the code intends is 0.80, and recomputing at the intended value drops
delivered volume from 1945 to 45. The same metric differs 40x between the two
settings, and nobody noticed, because the setting was not in the log.

For the goal of "provide the architecture, let users swap in their own models"
this is a requirement, not a nicety: swapping models is only meaningful as a
comparison, and two runs cannot be compared if neither knows its own config.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import build_state_record, effective_config  # noqa: E402

# Every one of these must be written to disk: drop any one and some class of
# cross-run comparison stops working
REQUIRED = [
    "mask_area_threshold",
    "visual_score_threshold",
    "audio_score_threshold",
    "require_single_instance",
    "sam3_first_frame_threshold",
    "sam3_first_frame_intended",
    "gate_relaxed",
    "gates_measuring",
    "caption_model",
    "extraction_model",
]


def test_all_required_keys_present():
    cfg = effective_config()
    missing = [k for k in REQUIRED if k not in cfg]
    assert not missing, f"missing keys: {missing}"


def test_no_group_failed_to_load():
    """A failed lazy import in any group leaves a *_error key; silently degrading is more dangerous than a missing field."""
    cfg = effective_config()
    errors = {k: v for k, v in cfg.items() if k.endswith("_error")}
    assert not errors, f"some groups failed to load: {errors}"


def test_thresholds_come_from_the_single_source():
    """Thresholds must match routes/nodes bit-for-bit -- a second copy will drift (see commit 3bae3f0)."""
    import nodes
    import routes
    cfg = effective_config()
    assert cfg["mask_area_threshold"] == routes.MASK_AREA_THRESHOLD
    assert cfg["visual_score_threshold"] == routes.VISUAL_SCORE_THRESHOLD
    assert cfg["audio_score_threshold"] == routes.AUDIO_SCORE_THRESHOLD
    assert cfg["require_single_instance"] == routes.REQUIRE_SINGLE_INSTANCE
    assert cfg["sam3_first_frame_threshold"] == nodes.SAM3_FIRST_FRAME_THRESHOLD


def test_relaxed_gate_is_flagged():
    """A relaxed gate must be flagged explicitly, otherwise a dev-setting run looks like a normal run after the fact."""
    import nodes
    cfg = effective_config()
    expected = nodes.SAM3_FIRST_FRAME_THRESHOLD < nodes.SAM3_FIRST_FRAME_INTENDED
    assert cfg["gate_relaxed"] is expected


def test_gates_measuring_is_honest():
    """Which gates actually measure something must be reported honestly.

    A gate that reads a constant and a gate that really checks look identical in
    pass rate -- that is exactly what happened across the 8907 historical runs:
    the audio gate rejected 1 sample, which looked like "almost everything passes
    on the audio side", when in fact the score was never measured at all.
    """
    import nodes
    g = effective_config()["gates_measuring"]
    assert set(g) == {"mask", "visual", "audio", "cross_modal"}
    assert g["audio"] is nodes.AUDIO_SCORE_IS_MEASURED
    # cross_modal is declared in models/__init__.py as the last gate, but it was
    # never wired into the graph. This assertion will fail the day it actually is
    # wired in -- which is exactly when this code needs updating.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wiring = open(os.path.join(root, "main.py")).read() + open(
        os.path.join(root, "nodes.py")).read()
    assert "CrossModalChecker" not in wiring, \
        "the cross_modal gate appears to be wired into the graph; update the False in gates_measuring"


def test_audio_score_carries_measured_flag():
    from utils import build_state_record
    rec = build_state_record({"sample_id": "t", "audio_removal_score": 0.9,
                              "audio_score_measured": False})
    assert rec["metrics"]["audio_score_measured"] is False


def test_record_carries_config():
    rec = build_state_record({"sample_id": "t", "target_object": "dog", "status": "passed"})
    assert "config" in rec, "the record has no config block"
    assert all(k in rec["config"] for k in REQUIRED)


def test_logging_never_breaks_the_pipeline():
    """An empty state must still produce a record -- a logging failure must never take the pipeline down."""
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
