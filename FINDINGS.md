# FINDINGS

An audit of the 8,907-run baseline, and — more to the point — an account of the
machinery that produced it. Sept 2026.

Everything below came out of an automated research loop run against this repo
over ~75 recorded steps. The findings are the evidence; the loop is the thing
worth keeping. This document is organized that way: what the loop is, then what
it found, then what it could not establish.

Every number here is reproducible from `results/*.jsonl` in the research repo,
and every claim is traceable to a numbered step record (`S27`, `S50`, …). Where
a later step overturned an earlier one, the earlier one is kept and marked
refuted rather than deleted — that history is the most useful part of the record.

---

## Part 1 — What the loop is

### The problem it exists to solve

On 2026-08-30 a run of `engine`-class targets came back at exactly 0% pass rate.
The obvious move was a downstream patch, and 1,386 SAM3 runs were launched to
support it. The actual cause was one line in the object-extraction system
prompt: `a running engine` was written into it as a *positive* example. An
independently-tasked second agent, given the same data and no knowledge of the
first attempt, proposed the same downstream patch and also did not read that
prompt.

The generalizable lesson is not "read the code first." That rule was already
written down, by the same author, in `AGENTS.md §5.5` — and then violated. Under
output pressure, the option that produces a number quickly always wins.

So the loop does not rely on anyone remembering anything. It relies on two
mechanical gates that a step cannot get past.

### Gate 1 — before spending compute

`preflight.py` runs before any experiment that costs GPU time or API budget.
Five rules, of which the load-bearing ones are:

| | blocks |
|---|---|
| R1 | a large spend with no located root cause |
| R2 | a known root cause being patched downstream, with no stated reason |
| R3 | a change to task semantics with no independent review |

If it does not pass, the experiment does not start. R2 is the one that would have
stopped the 1,386 runs.

### Gate 2 — before writing a record

`step.py` runs before anything is committed to the log. Eleven checks. A
representative few:

| | blocks |
|---|---|
| C1 | a ratio with no absolute numbers beside it |
| C2 | a conclusion with no epistemic status (measured / extrapolated / conjecture) |
| C3 | a proposed fix with no root-cause line reference |
| C4 | a change to the pipeline with no answer to "is it still the same task?" |
| C7 | a threshold inferred from data where only the agreeing side was checked |
| C9 | an experiment with no config snapshot, or a snapshot whose hash disagrees |
| C10 | a `depends_on` or `refutes` pointing at a step or claim that does not exist |

Each of these eleven exists because of one specific real mistake, not as a
hypothetical checklist. C7 is the clearest case: the first-frame gate was
inferred to be 0.1 because the observed boundary at 0.15 agreed with it — only
the agreeing side was checked. The disagreeing side had 611 rows in it, and the
gate was actually 0.05 (§2.2). A one-sided agreement can be satisfied by many
candidate values at once; its discriminating power is zero.

### What the gates cannot do, and how we know

A twelfth check was built — "a universal claim must report how many items it
enumerated" — and then killed by its own backtest against 14 historical steps:
4 false positives, 0 true positives.

The failure is not fixable by a better regex. A regex can see *that a step claims
to have enumerated something*; it cannot see *whether the enumeration happened*,
and the two are worded identically.

That gives a boundary worth more than the check would have been:

> **Mechanical gates can enforce properties of the record. They cannot enforce
> that the work behind the record was done.**

C1–C7 are all decidable by reading the step file alone. C8 required deciding a
fact outside the file. The second category needs a process defence instead: each
step opens by re-checking the previous step's universal claims. On one evening
that habit caught four same-shaped errors in a single step — the fourth being an
error in the very step that proposed the habit.

### What a step looks like

One JSON file per step, append-only:

```
hypothesis      what was expected before looking
what_i_did      the actual commands
claims[]        each with status: measured | extrapolated | theorem | conjecture
root_cause      file and line, when a defect is claimed
task_semantics  answer to "is this still the same task"
refuted         which earlier claim this overturns, if any
run_ref         config hash + full config + checkpoint id
depends_on      claim-level references, e.g. S50.claim[4]
default_flips   any default this step proposes changing
```

`depends_on` being claim-level rather than step-level is what makes refutation
tractable. When `S50.claim[4]` fell, `provenance.py` listed exactly the
downstream claims that had consumed *that claim* — not everything that had ever
cited S50. Three findings in this document were retracted that way, and the
retraction took minutes rather than a re-read of the whole log.

### The part a machine cannot do

Some questions are not decidable by any measurement available to the loop.
"Does this clip contain the target's action sound?" is one. "Is speech in scope
for this task?" is another. These get written as **asks** — a file with the
question, why the agent cannot answer it, the claims that depend on it, and what
it blocks.

An ask is not advisory. Check C11 refuses to record any step that flips a default
while an open ask blocks that flag. Work routes around the question instead of
guessing at it.

When an ask needs human perception rather than human judgement, the loop builds a
**pack**: a seeded random sample, rendered to 360p video and loudness-normalized
audio, A/B blinded, with the key held in a git-ignored file that never enters
published data. The reviewer answers in a browser and gets back a result code;
`ask.py ingest` decodes it, unblinds it, and appends per-sample answers to the
record. Two packs were run this way (§2.6, §2.7).

The blinding is not ceremony. In P01 the reviewer's free-text notes contained the
finding — not the tally. The tally was 7:2 and not significant. The notes said
the winning method kept picking whatever removed the *loudest* sound regardless
of which object was the target. No automated metric in the pipeline would have
surfaced that.

### The cost to the pipeline

All of the above lives outside this repo. What it needs *from* a pipeline is in
PR #7: a `checkpointer=None` parameter and a self-describing manifest. About 200
lines, no behavior change. That is the entire integration surface.

---

## Part 2 — What it found

### 2.1 The audio gate has never measured anything

`audio_removal_check` rejected 1 sample out of 8,907.

The reason is not that the threshold is loose. It is that no score is computed.
The node body is one line returning `state.get("mock_audio_removal_score", 0.90)`.
It calls no model.

The signature is visible in the outputs. Of the 1,946 samples that reached the
gate, `audio_removal_score` takes **2 distinct values** and 1,945 of them are
exactly 0.90. The comparable visual score takes **693** distinct values.

This matters beyond the gate itself. It means **the audio half of this pipeline
has never been verified on a single sample**, and any comparison between the two
gates is void — one of them is not a gate.

`models/__init__.py` names three checkers that were declared and never wired:
`VisualChecker`, `AudioChecker`, `CrossModalChecker`. The audio one is the one
with consequences. (S29)

> This is also the finding that motivates the manifest in PR #7. It took a person
> reading `nodes.py` line by line. With a manifest whose `measures` field is a
> live callable, it is one boolean in a JSON snapshot.

### 2.2 The delivery rate is not comparable across runs

This one was stated badly in earlier drafts. Precisely:

- The code's intended first-frame gate is `SAM3_FIRST_FRAME_INTENDED = 0.80`.
- All 8,907 historical runs were produced at **0.05**.
- Those are two configurations **16× apart in threshold**.
- The delivery rates they produce are **43× apart**.

| first-frame gate | delivered | h₀ | 95% CI |
|---|---|---|---|
| 0.05 (what actually ran) | 1,945 | **21.84%** | [20.99, 22.71] |
| 0.30 | 1,011 | 11.35% | [10.71, 12.03] |
| 0.50 | 269 | 3.02% | [2.68, 3.40] |
| 0.80 (what the code intends) | 45 | **0.51%** | [0.38, 0.68] |

So h₀ = 21.84% describes the 0.05 tier and nothing else. Quoting it as *the*
baseline overstates delivery under the intended configuration by 43×.

**How 0.05 was established.** The first attempt (S27) inferred 0.1, because the
observed lower bound at 0.15 was consistent with it. That is the one-sided
inference C7 now blocks. Checking the side that *should be empty*: among 3,432
non-zero values, the interval `(0, 0.05]` contains **exactly 0** rows — lower
bound 0.0501 — while `(0.05, 0.10]` contains **611**. Had the gate been 0.1,
those 611 would have been zeroed. (S27 → S28)

Note that any gate at or below 0.15 is inert anyway: `MASK_AREA_THRESHOLD = 0.15`
in `routes.py` binds first, and the delivered samples' `mask_area_ratio` has a
lower bound of 0.1501. The two configurations only diverge above 0.15.

**One consequence.** The standing recommendation "do not do per-sample search"
flips with this tier. The inequality behind it is unchanged; what was wrong was
treating one h₀ value as an intrinsic property of the pipeline. At 0.05 and 0.30
the recommendation holds with the whole confidence interval on one side. At 0.80
it reverses, also with the whole interval on one side.

**The rule that follows:** a rate recorded without the configuration that
produced it is not a measurement.

### 2.3 The intended threshold is not the fix

The natural response to §2.2 is to set the gate to 0.80. Measured, that is wrong
for a second reason.

`SAM3_FIRST_FRAME_INTENDED = 0.80` was evidently written for a
`largest_cc / mask_area` denominator. Computed both ways on the same masks: under
that denominator, 28/28 delivered samples **and** 6/6 historical-zero controls all
exceed 0.80 — discriminating power zero, a gate that passes good and bad alike,
exactly like the audio gate. Under the current `largest_cc / frame_area`
denominator the two groups' medians are 0.3484 and 0.0320, a 10.9× separation.

So the denominator is right and the threshold needs re-tuning against labelled
data — not these 34 samples. (S39)

Incidentally, `largest_cc` is close to a no-op: across those samples the ratio has
median 1.0000 and mean 0.988. SAM3's `masks[0]` is almost always a single
connected component. Fragmentation was not a real problem.

### 2.4 The real bottleneck, and why it is not a threshold problem

`mask_check` rejects 5,861 of the 8,221 samples that reach it — **71.3%**. It is
where the pipeline's throughput goes.

But **4,789** of those 5,861 rejections — **81.7%** of them, and 58.3% of
everything that reaches the gate — have `mask_area_ratio` **exactly 0**, which is
SAM3's genuine output, not an artifact of the gate. The words that distinguish
the zero group are `vehicle engine` (140), `truck engine` (84), `footsteps` (40)
— sound sources that are not separable objects. No threshold change reaches
these; they are a task-definition question about what counts as a sounding object.

This also corrects an earlier verdict. "Thresholds amplify outcomes 2–163×" was
rejected on the basis that near-threshold density was only 10.6%. That denominator
included 4,789 samples that were zeroed upstream and could never pass at any
threshold. Excluding them: **25.4%** (873/3,432). The verdict becomes conditional
rather than reversed, and a measurement rule falls out of it:

> When computing near-threshold density, the denominator may contain only samples
> whose outcome that threshold could actually change.

### 2.5 20.3% of delivered samples cannot be reproduced

The history spans two sources: `insave` (7,820, 87.8%) and `javedit` (1,073,
12.0%). Of the 1,945 delivered samples, **394 (20.3%)** are `javedit`.

Only `data/raw/insave/` exists on `point.dd.works`. For `javedit` only
intermediate artifacts under `data/work/` remain; the source videos are not on
the host.

Any experiment that re-runs from source silently omits those 394. This is not a
small correction — it is a fifth of the delivered set, and it is not randomly
distributed. Sampling must filter by prefix and say so.

Separately, `sounding_objects` and the captions were never written to disk. The
captions represent roughly 1,484 GPU-hours that cannot be re-used.

### 2.6 Neither audio selector's key correlates with audio quality

Best-of selection currently ranks candidates by ImageBind text-audio similarity
(`ib_ta`). Measured against removal quality on 307 samples × 10 candidates:
within-sample Spearman **+0.02**. The candidate `ib_ta` selects ranks first by
actual quality in **36 of 307 cases (12%)**. Re-keying on other available fields
recovers nothing. (S52, S53)

CLAP was tried as a replacement and does not work either:

- With `object_name` as the text query, AUC **0.509** [0.48, 0.53] — chance. (S47, S48)
- With zero-shot acoustic labels instead of the object name, AUC rises to
  **0.634** (p = 5e-7). Real signal, and not enough to select on. (S56)
- Head-to-head against ImageBind with a human judge (pack P01, 20 samples):
  **7 : 2**, p ≈ 0.18. Not significant.
- Asked directly whether a clip contains an action sound, CLAP agrees with human
  labels **1 time in 6**. (S69)

The reviewer's notes on P01 are the part that matters. CLAP's wins came from
picking whichever candidate removed the **loudest** sound, independent of whether
that sound was the target. The `min_residual_energy = 0.3` floor does not catch
this. It is a metric being optimized against its own intent — and it was found by
a person listening, not by any number in the record.

**Conclusion: CLAP is not the answer here, and the drafts proposing it (#3, #4)
are closed.** Recorded so that nobody spends the same weeks again. The finding
that a *contrastive audio-text* model cannot do this, while a human can do it in
seconds, is itself the more interesting result.

### 2.7 For person-class targets, the audio task is often empty

The task semantics were ambiguous and had to be asked (ask A02). The owner's
answer: for a person, the target sound is their **actions**; speech is
deliberately out of scope because there is no AV alignment for it and robots care
about sound events; non-verbal vocalization does count; removal is joint and
removes both by default; speech-bearing samples are not dropped.

Under that definition, pack P02 (21 of 40 reviewed) gives:

| | n | share | 95% CI |
|---|---|---|---|
| has an action sound | 13 | 62% | [41%, 79%] |
| **no action sound at all** | **6** | **29%** | **[14%, 50%]** |
| speech / vocalization only | 3 | | |
| uncertain | 2 | | |

Extrapolating to the 947 delivered person-class samples: roughly **271** of them
[131, 473] have nothing for the audio side to remove.

This replaces an earlier and wrong version of the same account. S50 had measured
"no speech" instead — 826 of 913, 90.5% — and concluded the audio side was
mostly idle. Under the owner's definition that measurement answers the wrong
question, and it is only off by so much because the definition was assumed rather
than asked. The 90% figure should not be cited.

One boundary case is still open: whether laughter is speech or a sound event.
Inferred as a sound event from the owner's definition, recorded as an open ask
rather than settled in chat.

### 2.8 Two smaller items

**Multi-instance asymmetry.** When the target word names several objects, the
visual side takes `masks[0]` while the audio side separates all instances by
text. The two sides then remove different things, which violates the constraint
that both must remove the same object. Rate: **27/399 = 6.8%** [4.7%, 9.7%] of
delivered samples. A gate exists (`REQUIRE_SINGLE_INSTANCE`) and is **off by
default**, because turning it on reduces delivery and that is a person's call.
Note the comment above it in `routes.py` still cites the older 8.8% estimate from
a 7/80 sample; 6.8% is the current number.

**Determinism.** Run-to-run variance at the main bottleneck is **0**. SAM3 is
deterministic here, so repeated runs of the same configuration buy nothing.

---

## Part 3 — What is not established

Listed because leaving it out would be the same failure this whole document is
about.

1. **No delivery improvement has been proven.** Every finding above is an audit
   result or a refutation. Nothing here has been shown to make the pipeline
   deliver more or better samples. The proposals that would (re-tune the mask
   threshold against labelled data; actually wire an audio checker) are untested.

2. **The audio side remains unmeasured.** `sam-audio` is not installed on
   `point.dd.works` and is not on PyPI; the single visible GPU is held by another
   process. §2.1 is therefore still true today, not just historically.

3. **P02 is interim** — 21 of 40 reviewed, by one reviewer. The 29% interval
   [14%, 50%] is wide enough that the practical conclusion could change.

4. **The scaffold has only ever run against one pipeline.** Its generality is a
   design claim, not a measured one. PR #7 is the smallest test of it: if the
   integration surface really is two hooks, a second pipeline should cost about a
   day. Until that happens, treat "this generalizes" as a conjecture.

5. **Re-tuning the first-frame threshold needs labelled data** that does not
   exist yet. The 34 samples in §2.3 establish which denominator to use and
   nothing more.

---

## Index

| § | claim | steps |
|---|---|---|
| 2.1 | audio gate measures nothing | S26, S29, S34 |
| 2.2 | h₀ is tier-dependent; gate was 0.05 | S27, S28, S31, S33 |
| 2.3 | 0.80 is untuned for the current denominator | S39 |
| 2.4 | mask gate is the bottleneck; density denominator rule | S25, S32 |
| 2.5 | 20.3% not reproducible | S40, S41 |
| 2.6 | neither selector's key correlates | S47, S48, S52, S53, S56, S67, S69 |
| 2.7 | person-class audio task often empty | S50 (refuted), S68, S71 |
| 2.8 | multi-instance 6.8% | S40, S41, S42 |

On the loop itself: C8's rejection is S37 (corrected by S38), the manifest
protocol is S45, and the claim-level provenance fields are S46.

Step records, the gate implementations, and the review interface live in the
research repo ([`auto-graph-research`](https://github.com/yayashuxue/auto-graph-research)). Ask for access if you want to read a
step's raw record or re-run a pack.
