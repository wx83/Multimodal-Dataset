# AGENTS.md — rules for working in this repo

For any agent (and any human) about to touch this pipeline. **Read this first, then act.**

The basis is two experiment documents:
[Orchestration as budget allocation](https://tdoc.dev/d/avgraph-orchestration-theory/v/2) (the formal framework) and
[Experimental results](https://tdoc.dev/d/avgraph-agent-autonomy/v/3); code and process log are at
<https://github.com/yayashuxue/auto-graph-research-lab>.

---

## 1. Invariants: nobody touches these

| Invariant | Where | What happens if you touch it |
|---|---|---|
| gate thresholds | `*_THRESHOLD` in `routes.py` | measured −21.4% delivery, 62% contamination rate, reported output inflated 2.6–3.5x |
| exogenous quality line | on the acceptance side, not in the pipeline | it is the definition of the reward. Nothing visible inside the pipeline is it |
| acceptance definition | AND (every node must pass) vs mean (nodes can compensate for each other) | this one decides whether "relax the gate" gains or loses, and it matters more than the coupling structure |
| node contract | `AVState` in `state.py` | what it reads, what it writes, which metric it must report — see §3 |

**Why**: the output an agent can see is **monotonically decreasing** in the threshold, so under all circumstances it will push the threshold to the minimum (which equals dismantling the gate) — **the objective function it can see has no interior optimum in that direction**. The true optimal threshold is an interior point, but only someone who can see the real quality can compute it.

**This is not a trust problem, it is an observability problem.** The correct phrasing is "thresholds are tuned offline on labeled data, agents must not touch them", **not** "thresholds are sacred by nature" — the existing evidence says the current `τ=0.55` was itself never tuned correctly.

## 2. What you can touch: three layers, more open as you go down

**L1, the solving layer — do not hand this to an agent.** Given (M, σ, Δq, κ, h₀), the optimal search strategy is computed, and deterministic algorithms come with optimality guarantees. Making an agent compete with a dedicated algorithm here is a waste.

**L2, the proposal layer — the agent's first place.** Generate candidates inside a fixed config space. Its role is not "enumerate faster", it is to concentrate candidates in high-quality regions, which is equivalent to shrinking the effective search space. **Shipping requires a random-sampling control**: compare quality quantiles of the proposals; if it cannot beat random, it is useless.

**L3, the restructuring layer — the agent's second place, and the only irreplaceable one.** Propose dimensions that were not in the config space at all. Concretely the mechanism is **writing glue code to wire a new model / new prompt form into the pipeline** — traditionally the space boundary is not decided by "what is useful", it is decided by "who wrote an adapter in advance", and that constraint is biased conservative.

**L3's boundary**: you may write adapters to extend the space, **you may not change acceptance**. The former compounds; the latter is the −21% in the table above.

## 3. Node contract

Externally, each node is **one input, one output, plus one metric**. Take `target_object_segmentation` as an example:

```
reads: av_pair_path, target_object
writes: mask_path, mask_area_ratio
produces: a non-empty mask video
must report: mask_area_ratio (the gate's input; the node must not decide pass/fail itself)
```

**Inside** a node you may: swap workers (from the menu), change how it prompts, change parameters, run several configs and pick one.
A node **may not**: change what it outputs, change what a metric means, score itself against the gate, skip itself, or call the next node directly.

**Swapping models is not changing a parameter.** Swapping the segmentation model changes the mask representation and the confidence semantics along with it, which breaks the contract outright — so a model swap is a **menu item** (format compatibility verified before it is listed), not a free choice. To bring in a new model, go through L3 and write an adapter.

## 4. Measure before you act: five numbers

Every strategy conclusion takes these five quantities as input, and **so far not one of them has been measured**:

| Quantity | How to measure | Why it matters |
|---|---|---|
| **h₀ baseline pass rate** | run a batch at a fixed config, count `status=="passed"` | 47.4% of the explanatory power, dominating everything else. **All 15 design grid points say "don't search" — you cannot make the call from the parameters alone** |
| σ observation noise | rerun the same clip at the same config, look at score variance | decides whether improvements are distinguishable (the noise floor) |
| Δq quality gap across configs | sweep several configs on the same clip | together with σ it sets the sampling complexity n ≳ M·(σ/Δq)² |
| κ search cost ratio | per-sample time with search / per-sample time without | **search wins ⟺ h₁/h₀ > κ**, one line of algebra settles it |
| A near-threshold density | the distribution of the three scores vs where the threshold sits | the threshold amplifies quality differences by 2–163x |

Tool: `experiments/decide_from_real_data.py` in the auto-graph-research-lab repo, just feed it `state.jsonl`.

## 5. Fairness checklist

Every item below is **a mistake we actually made**, not a hypothetical risk:

- [ ] **The baseline is not a straw man.** The control group must use the offline-tuned config, not an arbitrary one.
- [ ] **All arms start from the same point.** One starting from a tuned config and another from a random point is not a comparison of strategies. Measured: the anchor value is worth +43.4 deliveries while probe depth is worth only −1.1, **40x** — a difference in starting point will drown out everything you wanted to measure.
- [ ] **All arms probe to the same depth**, or sweep depth as an explicit independent variable.
- [ ] **Do not let parameters that should not change the environment leak into the random seed.** Put `noise` or `bar` into the regime seed and "sweeping noise" turns into swapping maps. We made this mistake twice.
- [ ] **Do not reuse the same observation for both selection and acceptance.** That lets argmax buy off acceptance directly, and the winner's-curse penalty never gets scored.
- [ ] **Check whether your "quality line" is dead code.** Report `P(passes quality line | passes gate)`; close to 1 means it is not doing anything.
- [ ] **When you cannot measure an effect, suspect the measuring instrument first, do not declare the effect nonexistent.** We misread "the simulator cannot measure winner's curse" as "winner's curse does not exist".

## 5.5 The root-cause rule (this one comes from a real failure)

**Once you have located a defect, before proposing any fix, find and read through the code / prompt / config that produced the value.**
Write down "which line the root cause is on", then talk about how to change it. If you cannot find that line, say so; do not propose a scheme that routes around it.

Real case: `car engine` targets had a pass rate of exactly 0% over 665 samples (the engine is hidden under the hood, visually inseparable).
The first reaction was to substitute a visible carrier at runtime (car engine → car) and launch 1386 runs —
while the root cause was in the system prompt in `models/object_extraction_model.py`,
where `a running engine` is written out as an example, and the whole prompt never required the output to be visually separable.
That patch would also break audio-visual hard consistency (the audio side still uses `car engine` while the visual side erases the whole car).

**Why this happens**: the route-around scheme produces numbers immediately, reading the prompt produces no numbers,
so under the pressure to "have output" it always wins. This is a systematic bias, not an occasional lapse.

**Adding agents does not solve this**: an independent blind-test agent found the same anomaly,
but its recommendation was also "switch to a visible carrier" — **it did not read the prompt either**.
Adding headcount does nothing when the bias is shared. It takes this rule itself, plus an explicit role (next section).

## 5.6 Someone must be responsible for asking "after the change, is it still the same task"

What broke the above pattern tonight was not any agent, it was a human objection grounded in task semantics.
So review cannot consist only of "is the conclusion right"; there has to be one that specifically asks:

- Does this fix change the definition of the task?
- Does it break some correctness constraint (e.g. audio and video removing the same object)?
- Is the metric being optimized a ratio? Has the denominator been quietly shrunk?

This role's criterion is not the data, it is the spec. **It has no number to show, so it will never appear on its own; it has to be assigned.**

## 6. Recording conventions

- Every sample's gate scores go into `data/logs/state.jsonl`, **including discarded ones** (with `discard_stage`/`discard_reason`) — the scores of eliminated samples are the only source for estimating h₀ and the near-threshold density.
- The experiment process goes into the auto-graph-research-lab repo`s `LOG.md`, and **refuted hypotheses are kept, not deleted**.
- Label every conclusion's status: theorem / measured / conjecture, and state explicitly which parts are not significant.

## 7. Review

Important conclusions get an **adversarial** review line, with the instruction "**assume it is a bug unless you cannot refute it**".
A neutral "take a look" will most likely just hand you back an equivocal confirmation. This is not a formality:
in one night adversarial review found four real bugs, two of which were enough to invalidate all conclusions,
and none of them was found by a better search algorithm.
