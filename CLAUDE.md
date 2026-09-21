# CLAUDE.md

## One rule, and why it has to be mechanical

**Before spending compute, read the code that produced the value.**

This rule was already written down once in `AGENTS.md §5.5`. **I wrote it myself, and then I violated it myself**:
on 2026-08-30 I found that engine-class targets had a pass rate of exactly 0% over 665 samples, I proposed a downstream patch and **launched 1386
SAM3 runs**; the root cause was one line of the SYSTEM_PROMPT in `models/object_extraction_model.py`
(`a running engine` was written there explicitly as a qualifying example). An independent blind-test agent, given the same data,
**also proposed the same kind of downstream patch, and also did not read that prompt**. What broke this was the user, not any check.

So: **do not expect that reading this paragraph will prevent it. Under delivery pressure, the plan that produces numbers immediately always wins.**
What actually works are the two mechanical gates below.

## Two gates (both in the auto-graph-research repo (github.com/yayashuxue/auto-graph-research))

**Before launch** — `python3 loop/preflight.py <declare.json>`
Any experiment that spends GPU or API budget has to clear this first. It blocks: large spend without having located the root cause (R1),
fixing downstream when the root cause is already known, with no stated reason (R2), changing task semantics without an **independent** review (R3).
Fail it and you may not launch.

**Before recording** — `python3 loop/step.py <step.json>`
It blocks: a ratio with no absolute counts (C1), a conclusion with no theorem/measured/conjecture label (C2), a proposed fix with no record of which root-cause line it is (C3),
not answering "is it still the same task after the change" (C4), claiming compute savings without deducting sunk cost (C6),
inferring a threshold from data while checking only the side that agrees (C7), declaring an experiment without a `run_ref` config snapshot or
with a hash that does not match the config (C9), `depends_on` pointing at a nonexistent step or claim (C10).
Fail it and no LOG is written, no commit happens.
C9/C10 were added on 2026-09-15 (S46): the config an experiment ran under must be persisted along with the record, it cannot stay in script constants;
dependencies written as a field are what lets a refutation event automatically list its downstream consumers (`python3 loop/provenance.py S24`).

An eighth check was tried (a universal assertion must report the number of items enumerated), and **my own backtest refuted it**:
across 14 historical steps it wrongly blocked 4 and had 0 true positives. The root cause was not a sloppy regex —
a regex can only see "claims to have enumerated", not "actually enumerated", and the two read identically in text.

That yields a boundary worth more than one more gate:
**a mechanical gate can police "properties of the record", not "whether the work behind the record was actually done".**
C1–C7 can all be decided by reading the step file alone (does the ratio come with absolute counts, is the status labeled,
is the root-cause line number recorded); C8 would have to decide whether a fact outside the file happened, which is beyond the capability boundary.
That can only be covered by a process defense: **at the start of every step, go back and verify the previous step's universal assertions.**
On the night of 2026-08-30, that habit caught 4 instances of the same failure within 1 step,
and the 4th one it caught was a factual error in the very step that proposed the habit.

The seven checks plus three checks across the two gates — **each one corresponds to a real failure**, not a hypothetical checklist.
C7 was added on 2026-08-30: from "the boundary 0.15 agrees with the threshold" I inferred the first-frame gate = 0.1,
and checked only the side that agreed; in fact (0.05, 0.10] holds 611 nonzero values, and the gate is 0.05.
One-sided agreement can be satisfied by several candidate thresholds at once, so its discriminating power is 0.
Details are in the header comments of `loop/checks.py` and `loop/preflight.py`.

## Three things nobody may touch

Gate thresholds, the exogenous quality line, the acceptance definition. Rationale and the measured cost are in `AGENTS.md §1`
(when touched: −21.4% delivery, 62% contamination rate, reported output inflated 2.6–3.5x).

The correct phrasing is "thresholds are tuned offline on labeled data, agents must not touch them", **not** "thresholds are sacred by nature" —
the existing evidence says the current `τ=0.55` was itself never tuned correctly.

## One thing an agent review cannot replace

R3 allows an agent to do the semantic review, but remember the record of 2026-08-30: **the independent agent made the same mistake I did**.
What multiple agents buy you is **decorrelated errors** (adversarial review did find 4 real bugs),
**not decorrelated values** — every agent is biased toward "optimize the measurable metric", and none will naturally ask
"after this change, is it still the same task".

**Changes that touch correctness constraints (e.g. audio and video must remove the same object) wait for human confirmation.**

## Status at a glance

- Baseline: 8,907 real runs, h₀ = 21.84% (95% CI 20.99–22.71).
  **But all 8,907 ran in the relaxed setting**: the measured first-frame gate = **0.05** (among 3,432 nonzero values,
  exactly 0 are ≤0.05, lower bound 0.0501), while `SAM3_FIRST_FRAME_INTENDED = 0.80`,
  a 16x difference. Note that the current default in `nodes.py` is 0.1, which differs again from the 0.05 of the historical runs —
  where 0.05 came from is not yet established (env override? an earlier code version?), still to be checked.
  Recomputed at the intended value 0.80, only 45 of the 1,945 delivered samples survive (−97.7%); at the 0.30 setting 1,011 survive.
  That recomputation only counts samples whose true ratio is >0.80, independent of where the gate was set at the time, so it is unaffected by the uncertainty above.
  **h₀ describes the 0.05 setting, not the intended config**, do not use it as the baseline for the intended setting.
  **Verified (S39, measured, 2026-08-31): the denominator is right, the threshold is wrong.**
  Both denominator conventions were computed on the same batch of masks: under the `largest_cc/mask area` convention the delivered group was 28/28
  and the control group (historical mask=0) 6/6, **all >0.80**, discriminating power 0 — that gate would behave like the audio gate,
  passing good and bad alike; whereas under the current `largest_cc/full frame area`, the two groups' medians are 0.3484 vs 0.0320,
  a 10.9x difference, which does discriminate. So `SAM3_FIRST_FRAME_INTENDED = 0.80` needs to be
  **re-tuned** (on labeled data, not these 34 samples), and the denominator does not need to change.
  Incidentally: the `largest_cc` step is nearly a no-op — across the 34 detected samples that ratio has median 1.0000
  and mean 0.988, SAM3's `masks[0]` is almost always a single connected component, and "fragmentation" is not a real problem.
- Bottleneck: the mask gate's conditional pass rate is 28.71%, and 58.3% of masks are **exactly 0**.
  The zeros are SAM3's real output, not an artifact of erasure — the high-frequency words unique to the zero group are
  `vehicle engine` 140, `truck engine` 84, `footsteps` 40, all inseparable.
- The audio gate is a no-op: it rejected 1 sample out of 8,907.
  **The reason is not that the gate is too loose, it is that the score is never measured at all**: `audio_removal_check` is one line,
  `state.get("mock_audio_removal_score", 0.90)`, and never calls any model.
  Among the 1,946 samples that reach that gate, `audio_removal_score` takes only 2 distinct values (1,945 of them exactly 0.90),
  while `visual_removal_score` has 693. **The audio side has never been validated, not on a single sample.**
- Of the five quantities to be measured (h₀ σ Δq κ A), h₀ and A have been measured, and σ on the main bottleneck is measured **= 0** (SAM3 is deterministic)
- **Proposition 1 "do not do per-sample search" has status "conditional", not "confirmed"** — it flips with the gate setting:

  | First-frame gate | Delivered | h₀ | 95%CI | Verdict |
  |---|---|---|---|---|
  | 0.05 (as historically run) | 1,945 | 21.84% | [20.99, 22.71] | don't search (stable) |
  | 0.30 | 1,011 | 11.35% | [10.71, 12.03] | don't search (stable) |
  | 0.50 | 269 | 3.02% | [2.68, 3.40] | inside the decision band · undecidable |
  | 0.80 (what the code intends) | 45 | 0.51% | [0.38, 0.68] | **search (stable)** |

  The decision band is 1%–5% (it shifts with gamma, centered at 3.5%). "Stable" = the whole CI falls on one side of the band.
  The inequality `search wins ⟺ h₁/h₀ > κ` itself was not refuted; what was refuted is treating some particular h₀ value
  as an intrinsic property of the pipeline. **Whenever you record a rate, you must record the config setting that produced it.**
- Any first-frame gate ≤0.15 is **idle**: the lower bound of `mask_area_ratio` across the 1,945 delivered samples
  is 0.1501, and what has actually been binding all along is `MASK_AREA_THRESHOLD = 0.15` in `routes.py`.
  The difference between "dev setting vs intended setting" only starts to show above 0.15.
- **The other two verdicts also need revision** (re-reviewed 2026-08-30; not one of the three verdicts survives as written):
  - Proposition 3 "thresholds amplify by 2–163x" was originally judged **negative**, on the grounds that the near-threshold density A was only
    10.6%/1.7%/0.0%. But the audio figure (0/1946) was computed from the constant 0.90, which is **empty evidence**;
    and the mask figure's denominator includes 4,789 samples erased to 0, which no threshold placement could ever let through.
    Excluding them, A goes from 873/8221 = 10.6% to **873/3432 = 25.4%**.
    A also decreases monotonically with the gate setting: 0.15→25.4%, 0.30→19.9%, 0.50→6.0%, 0.80→1.1%.
    Downgraded to **conditional**, not overturned — A is only a proxy for the amplification factor, it cannot be converted directly.
  - "The K gates are asymmetric" was originally judged **negative**; the direction is right but the reasoning has to be rewritten: it is not that the audio gate is too loose,
    it is that the score was never measured. Rewritten, that verdict is stronger — the two gates cannot be compared, because one of them is not a gate.
- **Denominator rule**: when computing "near-threshold density", the denominator may only include samples whose outcome that threshold could change;
  samples erased upstream or already fixed must be excluded, otherwise you systematically underestimate the threshold's influence.
- Logging gap: neither `sounding_objects` nor the caption **was persisted** (the latter cost about 1,484 GPU-hours)
- **Reproducibility gap: 20.3% of delivered samples cannot be re-run on point.dd.works.**
  The historical data spans two sources: `insave` 7,820 (87.8%) and `javedit` 1,073 (12.0%);
  among the 1,945 delivered samples, javedit accounts for 394 (20.3%). But `data/raw/` only contains `insave/`,
  and for javedit only the intermediate products under `data/work/` remain — **the source videos are not on this machine**.
  Any experiment that needs to re-run source videos will systematically miss that 20.3%; filter by prefix before sampling and say so.
