# Seeds and Meta-Seeds

**Seed** = where this round starts: the initial configuration plus the initial plan.
**Meta-seed** = **the feasible region itself**: what the agent is allowed to change, what it must not
change, and which directions are worth exploring.

The meta-seed is not part of the (F, A, R) triple; it determines the **domain** of those three dimensions.
This is the interface through which human taste is injected into the system.

## Why the Meta-Seed May Be the Most Valuable Knob

Measured: the starting point (just the initial configuration) is worth **+43.4 ±0.8** deliveries, while
probe depth is worth only **−1.1 ±0.1** — a 40x difference. And the meta-seed sits one level above the
starting point: it determines which set the starting point can fall in.

**But it has never been measured as a variable.** The three meta-seeds below are designed for exactly
that: same sample batch, same budget, only the meta-seed changes, compare the output.

---

## M1 · Conservative: parameters only

**Feasible region**: the numeric parameters inside each node.
- `SEGMENT`: dilation amount, tracking window, frame sampling stride
- `INPAINT`: `INPAINT_NUM_FRAMES`, number of steps, guidance strength
- `AUDIO`: seed, best-of N
- `ENHANCE`: `AVENHANCE_DENOISE_STRENGTH`, `AVENHANCE_FINE_STEPS`

**Not allowed**: swapping models, changing prompt wording, touching the LangGraph topology, touching any gate.

**The bet**: h₀ is already not low, the configuration space is small, offline tuning is enough, and
per-sample search is a net loss (Proposition 1).

## M2 · Menu: parameters plus model choice

**Feasible region**: all of M1, **plus** swapping models from a menu whose format compatibility has been
verified in advance.
- `SEGMENT`: SAM3 | SAM2 | (others whose mask format has been verified)
- `AUDIO`: SAM-Audio's visual-prompt path | text-prompt path
- `EXTRACT`: GPT-4o-mini | local LLM

**Not allowed**: adding new entries to the menu (that is M3's job), changing prompt wording, touching gates.

**The bet**: different clips suit different models (sample heterogeneity γ > 0), and model choice is a
coarser dimension than parameters, hence easier to learn.

## M3 · Extensible: allowed to write glue code

**Feasible region**: all of M2, **plus** allowing the agent to **write adapters that bring models outside
the menu in**, and to change prompt wording.

**This is the L3 restructuring layer.** Traditionally the boundary of the configuration space is not
decided by "what is useful", it is decided by "who wrote an adapter in advance" — and nobody writes an
adapter layer for a dimension whose usefulness is still unknown. When the agent can write glue on demand,
that constraint loosens.

**Not allowed**: changing any gate, quality line, or acceptance definition. **Writing adapters expands the
space; changing acceptance changes the reward.**

**The bet**: the payoff at position 2 is cumulative and not bound by sample complexity (write the glue
once and that dimension stays in the space permanently, with no need to learn it from samples).

---

## How to Compare

Three meta-seeds, the same sample sequence, the same compute budget, compared on deliveries per
1000 GPU-minutes.

**What must be recorded at the same time** (otherwise the comparison cannot be attributed):
- Which coordinate in (F, A, R) space each round actually landed on (measured with the definition in `measure_policy_space.py`, not annotated by hand)
- Which adapters the agent wrote under M3 and which proved useful afterwards — the only cumulative asset at position 2
- h₀, σ, Δq under each meta-seed

**Expected failure modes** (written down in advance, so no explanation gets invented after the fact):
- M3 may lose because the agent spends its time writing glue instead of producing data — so **bill them
  separately**: the cost of writing glue goes under "one-time investment" and is not amortized into the
  per-sample cost.
- M2 may show no difference from M1 because the options on the model menu are in fact about equal — if so,
  the real heterogeneity is not along the model dimension, and it should be looked for in M1's parameters.
- All three may lose to the "search nothing" control group. **The control group must be present**
  (AGENTS.md §5, first item).

## Initial Seed (shared by every meta-seed)

Start from the production default configuration, not from a random point. The reason is in AGENTS.md §5,
second item: the starting-point difference is worth 40x the probe depth, and not controlling it means
nothing was measured at all.
