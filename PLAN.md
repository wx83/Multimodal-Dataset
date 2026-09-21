# Long-Run Experiment Plan

Goal: take the assumptions in the [orchestration framework](https://tdoc.dev/d/avgraph-orchestration-theory/v/2)
and knock them down one by one on the real pipeline. **Measure first, optimize later** — not one of the
five numbers that decide everything has been measured yet.

## Current Status (2026-08-29)

| Item | Status |
|---|---|
| Six conda environments | ✅ weihan built them today (`sam3` has torch 2.10.0+cu128, CUDA available) |
| pipeline code | ✅ already at `/scratch/weihan/av_langgraph_pipeline` |
| mock full chain | ✅ runs end-to-end on the server (`test_1` `test_2` `test_5` all pass, langgraph 1.2.5 compatible) |
| input material | ✅ `/scratch/unbalanced_mp4`, 284,678 mp4 files total |
| idle compute | ✅ GPU 1 (A6000 46GB) idle; GPU 0 occupied with 46GB |
| **model weights** | ❌ **the only blocker**, see below |

### Missing Weights (needs weihan to confirm or provide)

| Model | Location the code expects | Current state |
|---|---|---|
| SAM3 | `pretrained_weight/sam3` | missing |
| SAM-Audio | `pretrained_weight/sam_audio` | missing |
| EffectErase LoRA | `pretrained_weight/inpainting` | missing |
| Wan2.1-Fun-1.3B-InP (EffectErase base) | `Wan-AI/Wan2.1-Fun-1.3B-InP` | missing |
| Qwen3-Omni-30B | HF cache | missing |
| LTX-2.3 22B | `ltx-2.3-22b-distilled-1.1.safetensors` | ⚠️ `/scratch/ltx-2-3-22b-dev-Q8_0.gguf` (22G) exists but the **format differs** (GGUF vs safetensors); either convert it or change the code |

## The Key Feasibility Insight: No Need to Wait for All the Weights

**h₀ = the product of the per-gate pass rates, and the gates are sequential**, so every model that lands
unlocks one more segment to measure:

```
SAM3 in place    -> can measure the mask gate pass rate (cheapest segment, a failure costs only 15 seconds)
+ EffectErase    -> can measure the visual removal gate
+ SAM-Audio      -> can measure the audio gate  -> h₀ is complete at this point
+ Qwen3-Omni     -> replaces the pinned target, uses the real caption -> but +10 minutes per sample
+ LTX-2          -> only affects the final enhancement, does not affect h₀
```

**SAM3 alone is enough to start.** It is the lowest-cost, highest-information step.

## Measurement Cost (costed out, not estimated)

Failing samples exit early, so the average per-sample cost is far below 28.5 minutes:

| True h₀ | Avg per sample | 300 samples | 1000 samples |
|---|---|---|---|
| 40% | 16.8 min | 84 GPU-h | 279 GPU-h |
| 20% | 11.6 min | 58 h | 194 h |
| 10% | 8.3 min | 42 h | 139 h |
| 5% | 6.1 min | 31 h | 102 h |
| 2% | 4.2 min | 21 h | 70 h |

**The lower the pass rate, the cheaper the measurement** — and a low pass rate is exactly the case where
you most need this number. On a single A6000 running continuously, 300 samples take roughly 1–3.5 days.

### An Architectural Problem Found Along the Way

The table above **excludes caption**. Qwen3-Omni costs 10 minutes per sample, and it runs **before all
the gates** — 300 samples is 50 GPU-hours of pure caption, more expensive than everything else combined.

**No gate can protect it**, because the mask gate needs a target object, and the target object comes from
the caption. If a cheap coarse filter could be inserted before the caption (for example, a generic detector
deciding "is there a segmentable salient object in the frame at all"), the cost would drop sharply. This
is itself worth stating as an experimental hypothesis.

---

## Phase Plan

### Phase 0 · Unblock the weights (needs a human)
Confirm where the six items above land. The minimum set to start work = **SAM3**.

### Phase 1 · L0 measurement layer (the bulk of the long run)
Measure the five numbers per `AGENTS.md §4`, with a fixed configuration and no search of any kind.

1. **h₀ baseline pass rate** — start with 300 samples, use `decide_from_real_data.py` for the verdict and confidence interval
2. **σ observation noise** — same clip, same configuration, repeated 10 times, look at the variance of the three gate scores
3. **Δq cross-configuration quality gap** — same clip, sweep 5 parameter sets, look at the score spread
4. **κ search cost ratio** — computed directly from the Phase 2 run logs
5. **A near-threshold density** — computed directly from the score distribution of the h₀ batch

**This phase needs no agent autonomy at all.** It is the measurement layer, run with deterministic scripts.

**Hypotheses that get validated along the way**:
- Whether the three gates eliminate samples in balanced proportions (mock data shows 22.0/21.5/21.5%, no single bottleneck)
- Whether the score medians sit above the thresholds (mock shows 0.232/0.842/0.830 vs thresholds 0.15/0.80/0.80;
  if real data behaves the same, the threshold amplification effect holds and tuning is extremely risky)
- The **rank correlation** between local gate scores and final delivery — this alone decides whether per-node decomposition is worth doing (Proposition 2)

### Phase 2 · Meta-seed comparison
Once the five numbers are in, run M1 / M2 / M3 plus a no-search control group per `METASEEDS.md`.
**The control group must be present.** Record the measured (F, A, R) coordinates each round.

### Phase 3 · Only if the Phase 1 numbers support it
Only go to per-sample search if h₀ < 3.5% (Proposition 1's boundary); otherwise a single pass.
Only do multi-agent per-node decomposition if local signals correlate strongly in rank with the final result.

---

## Execution Architecture

One **resident master** holds the state, the meta-seeds, and the cross-round memory, and decides when to
adjust based on the replanning interval X; each round **dispatches independent subagents**, at least one
of them **with adversarial instructions** ("the default conclusion is wrong").

The reason for the layering is **error decorrelation, not parallelism**: in one night the adversarial
review found four real bugs, two of which were enough to invalidate every conclusion, and none of them
were found by a better search algorithm.

## What Must Be Written to Disk Every Round

- `data/logs/state.jsonl`: the gate scores of every sample, **including the discarded ones**
- the research log repo`s LOG.md: the process log, **refuted hypotheses are kept, not deleted**
- Every conclusion labeled: theorem / measured / conjecture, with the non-significant parts stated outright
- Which adapters the agent wrote under M3 and which turned out useful afterwards — the only cumulative asset at position 2

## Stopping Conditions

- All five numbers measured, with confidence intervals that do not straddle a decision boundary → Phase 1 complete
- Or: compute exhausted → report the current confidence intervals and how many more samples are needed
