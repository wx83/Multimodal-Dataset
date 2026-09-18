# The Worker Contract: This Pipeline's Real Model-Swap Entry Point

This document answers one concrete question: **what "they can swap the model themselves" actually means
in practice.**

This system is positioned as providing an architecture rather than chasing the peak performance of any one
model, so "how to swap" is the product manual itself. And as of 2026-08-30 this contract existed only in
the six parallel worker implementations; it had never been written down.

## The Extension Point Is the Subprocess Contract, Not a Class Name

Every heavy model lives in its own conda env and is invoked by `models/<name>_model.py` as a **subprocess**
calling `models/<name>_worker.py`. That boundary is where a model gets swapped — and it also isolates
dependency conflicts (SAM3 needs a new transformers, EffectErase needs diffusers 0.30-0.31 +
transformers<5; installed together they fight).

All six workers obey the same contract:

| Convention | Content |
|---|---|
| Input | command-line arguments, including `--out <path>` |
| Output | write the result JSON into the file `--out` points at |
| Fallback | also print `<NAME>_RESULT {json}` to stdout on one line |
| Parsing | the wrapper reads the file first; if that fails it looks for the marker line in stdout; only if both fail does it error out, attaching the last 2000 characters of stdout |

The six existing markers: `QWEN3_OMNI_RESULT`, `SAM3_RESULT`, `SAM_AUDIO_RESULT`,
`EFFECTERASE_RESULT`, `LTX_ENHANCE_RESULT`, `IB_SELECT_RESULT`.

**So swapping a model = writing a new worker that obeys this contract**, plus pointing the wrapper at it
(`python_bin` and `worker` are both `__init__` parameters). No need to touch the graph, the routes, or any
node.

## Three Tiers of Substitutability — and Which Tier Actually Holds Today

| Tier | Meaning | Current state |
|---|---|---|
| **Swap checkpoint** | same architecture, different weights | **fully supported**, via env or parameter (e.g. `SAM3_MODEL_DIR`, `QWEN3_OMNI_PATH`) |
| **Swap within the same architecture family** | e.g. Qwen3-Omni → Qwen2.5-Omni | only the captioner supports it (`--model_class` / `--processor_class`) |
| **Swap across families** | e.g. SAM3 → a different segmenter | **all require a new worker**; this is by design, not a defect |

### How Far the Captioner's Substitutability Goes

On 2026-08-30 I turned `qwen3_omni_worker.py`'s model class into an injected parameter, but **that undid
only one coupling**. `caption_video()` still contains two more things specific to the Qwen-Omni family:

- `process_mm_info` (from `qwen_omni_utils`)
- `generate(..., thinker_return_dict_in_generate=True, return_audio=False)`
  — parameters specific to the dual-head thinker/talker architecture

Qwen2.5-Omni and Qwen3-Omni belong to the same family and share the thinker/talker structure, so it
**will probably run**, but this has not been verified on real hardware (no transformers locally, and I
will not guess class names or guess compatibility).
Switching to a non-Qwen omni model requires writing a new worker.

**Do not assume that `--model_class` makes the captioner freely swappable.**

## What a New Worker Must Satisfy

1. Accept `--out <path>` and write the result JSON into it
2. Also `sys.stdout.write(MARKER + json.dumps(result) + "\n")` and flush
3. The fields of the result JSON must line up with the parsing code in the corresponding wrapper
   (e.g. a segmentation worker must supply `first_frame_ratio` / `passed` / `mask_path` / `n_instances`)
4. Live in its own conda env, pointed at through the wrapper's `python_bin`

## After Swapping, How Do You Know Whether It Got Better or Worse

This is the entire point of swapping a model, so it gets its own section:

Every record written to disk carries a `config` block (added 2026-08-30) recording the thresholds of the
four gates, the actual and intended values of the first-frame gate, the `gate_relaxed` flag,
`gates_measuring` (which gates are actually measuring), and the model identities of the captioner and of
extraction.

**Before comparing two runs, check that their `config` blocks differ only in the one item you meant to
swap.** The cost of not doing so has been measured: the 8,907 historical runs did not record the gate
setting; h₀ = 21.84% was produced with the first-frame gate at 0.05, while the code's intended value is
0.80, and recomputing at the intended value drops deliveries from 1,945 to 45 — the same metric differs
40x between the two settings, and the two look identical in the logs.

Also note that `gates_measuring` is currently `{mask: true, visual: true, audio: false,
cross_modal: false}`: **when you swap the audio model, the audio gate gives no signal at all** (the score
is the hard-coded constant 0.90; 1,945 out of 1,946 are exactly that value). Until the audio gate is
wired up, model comparisons on the audio side can only be done by human listening or by measuring
separately.
