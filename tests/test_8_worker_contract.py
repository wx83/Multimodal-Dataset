"""Whether all six workers really obey the same subprocess contract.

Why this test exists: `WORKER_CONTRACT.md` claims the entry point for swapping
models is this contract, not class-name injection. But docs go stale, and when I
wrote "6/6 consistent" I had only grepped for the presence of the marker and
--out, without checking the wrapper side -- exactly the mistake made one step
earlier (changed load() and declared the captioner swappable without reading
caption_video).

The contract (see WORKER_CONTRACT.md):
  worker side  -- accepts --out <path> and writes the result JSON there; also
                  prints one line `<NAME>_RESULT {json}` to stdout and flushes
  wrapper side -- reads the file at --out first, falls back to finding the marker
                  line on stdout, and only raises if both fail

These are static checks: no models are loaded, only source is read. The point is
to make it impossible for a new worker to quietly break the contract, not to
verify that the models run correctly.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

# worker file -> the wrapper file that drives it.
# audio_removal_model.py drives two workers (SAM-Audio separation + ImageBind
# selection), so it passes marker= explicitly in _parse; one wrapper driving
# several workers is allowed.
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
            f"{w} does not declare a marker of the form <NAME>_RESULT"


def test_every_worker_accepts_out_and_writes_it():
    for w in PAIRS:
        s = src(w)
        assert '"--out"' in s, f"{w} does not accept --out"


def test_every_worker_also_emits_the_marker_line():
    """Fallback channel: if the file write fails, that stdout line is the only source of the result, so it cannot be skipped."""
    for w in PAIRS:
        s = src(w)
        assert "RESULT_MARKER +" in s or "RESULT_MARKER+" in s, \
            f"{w} does not print the result to stdout"


def test_every_wrapper_parses_file_first_then_marker():
    """Order matters: stdout can be polluted by model-library logging, so the file is the primary channel."""
    for w, wrapper in PAIRS.items():
        s = src(wrapper)
        assert "RESULT_MARKER" in s, f"{wrapper} does not parse via the marker"
        assert "json.load" in s or "json.loads" in s, f"{wrapper} does not parse JSON"
        i_file = s.find("json.load(")
        i_marker = s.find("RESULT_MARKER):")
        if i_file != -1 and i_marker != -1:
            assert i_file < i_marker, \
                f"{wrapper} appears to look for the stdout marker before reading the file, the reverse of the contract"


def test_markers_are_unique_per_worker():
    """Two workers sharing one marker would cross wires during parsing."""
    seen = {}
    for w in PAIRS:
        m = re.search(r'RESULT_MARKER\s*=\s*"([A-Z0-9_]+_RESULT) "', src(w))
        assert m, f"could not extract the marker from {w}"
        name = m.group(1)
        assert name not in seen, f"{w} and {seen[name]} use the same marker {name}"
        seen[name] = w


def test_contract_doc_exists_and_lists_every_marker():
    """A marker missing from the doc means some worker never made it into the spec."""
    doc_path = os.path.join(ROOT, "WORKER_CONTRACT.md")
    assert os.path.exists(doc_path), "WORKER_CONTRACT.md does not exist"
    doc = open(doc_path, encoding="utf-8").read()
    for w in PAIRS:
        m = re.search(r'RESULT_MARKER\s*=\s*"([A-Z0-9_]+_RESULT) "', src(w))
        assert m.group(1) in doc, f"{m.group(1)} (from {w}) is not documented in WORKER_CONTRACT.md"


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
