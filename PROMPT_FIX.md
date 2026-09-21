# Root Cause and Fix for the Extraction Node Prompt

## Root Cause (which line)

`models/object_extraction_model.py`, `SYSTEM_PROMPT`:

```
"An object qualifies only if it has a clear physical mechanism for sound production "
"(e.g. vibrating strings, flowing water, a running engine, a speaking person, "
"flapping wings, crackling fire). "
```

**`a running engine` is written out as a qualifying example.** GPT-4o-mini emitting `car engine` is not a
model error — it followed the instruction precisely. The whole prompt defines "produces sound" and
**never requires the output to be visually segmentable**, which is exactly what downstream SAM3 needs.

The two nodes are each correct on their own; **nobody wrote down the assumption at the interface**.

## Consequences (measured, n=8907)

| Target | Samples | Pass rate |
|---|---|---|
| `car engine` | 665 | **0.2%** |
| `vehicle engine` | 140 | 0% |
| `truck engine` | 84 | 0% |
| `motorcycle engine` | 76 | 0% |
| `bus engine` | 61 | 0% |
| `footsteps` | 40 | 0% |
| full-pipeline baseline | 8907 | 21.84% |

The automated contract checker (`experiments/contract_check.py in the auto-graph-research-lab repo`) turned up
**32 such values, covering 2902 samples, yielding only 41 and wasting 2861**.

## Why It Cannot Be Fixed Downstream

Rewriting `car engine` to `car` at runtime **breaks the audio-visual hard consistency constraint**: on the
audio side SAM-Audio's prompt is still `car engine` (remove the engine sound), while the visual side
erases the entire car — the two sides are not removing the same thing.
This constraint is correctness, not preference: zero degrees of freedom.

(`man's voice → person` is even more obvious: besides speaking, a person also produces footsteps and
clothing rustle. Erase the whole person but remove only the speech, and the remaining footsteps become
"an invisible person walking" — and the audio gate is a no-op, so it cannot detect this.)

## The Fix: Change the Prompt So It Produces Qualifying Output in One Shot

```python
SYSTEM_PROMPT = (
    "You are an audio-visual scene analyst. Given a video caption, identify the "
    "objects that are BOTH actively producing sound AND visually segmentable in "
    "the frame.\n"
    "\n"
    "An object qualifies only if BOTH hold:\n"
    "  (a) it has a clear physical mechanism for sound production, and\n"
    "  (b) it is a visible, boundable entity that could be outlined in the image.\n"
    "\n"
    "Critically: if the sound-producing mechanism is hidden inside or is a part of "
    "a larger object (an engine inside a car, a speaker inside a phone), name the "
    "VISIBLE WHOLE, not the hidden part — output 'car', not 'car engine'.\n"
    "If the sound is an event or action rather than an object (footsteps, speech, "
    "a horn honking), name the visible entity producing it — output 'person', not "
    "'footsteps'.\n"
    "Exclude sounds with no boundable visual source (wind, ambient noise, "
    "background music, rain) — these have no object to segment.\n"
    "Exclude static background objects that do not themselves emit sound "
    "(buildings, walls, trees, roads).\n"
    "\n"
    "Output ONLY a comma-separated list of the qualifying objects, most prominent "
    "first. If nothing qualifies, output: none."
)
```

Three key changes:

1. **(b) visual segmentability becomes a hard condition** — this is the half of the contract the original prompt was missing.
2. **Hidden part → visible whole**, using `car engine` directly as the counterexample (the original prompt used it as a positive example).
3. **Sound event → the entity producing it**, `footsteps` → `person`.

Consistency then holds naturally: **there is only one word from end to end**, the audio side and the
visual side use the same one, and "switching targets" simply does not arise.

## To Be Validated (this is what the experiment should actually measure)

**Not** "can the new prompt rescue the engine class" — that is nearly certain (they are at 0% now).
**But whether it hurts the ones that currently pass**: `man` 48%, `woman` 55%, `dog` 30%, `car` 34%.
The new prompt is stricter and may judge some currently passing samples as `none`.

So the A/B **must sample failures and successes at the same time**, comparing:
- Failure class: does the new prompt produce a segmentable word (improvement)
- Success class: does the new prompt still produce the same word (no regression)

**Blocker**: **not a single** historical caption was saved (`data/work/captions` is empty), so
8907 × 10 minutes ≈ 1484 GPU-hours of caption are all lost. An A/B requires regenerating them, and
Qwen3-Omni is a 66GB MoE that does not fit on the idle 46GB card; CPU offload is needed.

## Two Logging Gaps Found Along the Way

- The `sounding_objects` list (the extraction node's full output) is **not written to disk**; only the
  final `object_name` is stored. So the question "did this clip already contain a segmentable alternative
  sound-producing object" cannot be answered — and that might be a better fix than changing the prompt.
- The caption is not written to disk. Both should be added to the disk-write logic in `utils.py`.
