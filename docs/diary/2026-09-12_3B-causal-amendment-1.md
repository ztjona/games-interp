# Phase 3B-causal — amendment 1, PRE-DATA (2026-09-12)

**Status: pre-data.** Written after the pre-registration was committed
(`b64b3d3`, 2026-09-12 10:58) and **before any interchange intervention was
run**: no IIA value existed when any change below was decided. The only new
measurement is competence-audit Test F, a property of the champion, not an
outcome of the experiment. Frozen from the commit that adds it, like the
pre-registration.

Amends: [`2026-09-12_3B-causal-preregistration.md`](2026-09-12_3B-causal-preregistration.md).

**Why this file exists.** During implementation the committed
pre-registration was edited in place, which its own §14 forbids. Those edits
were reverted — the pre-registration is byte-identical to `b64b3d3` — and are
recorded here instead. The pre-registration's own "Amendment log" still reads
*(none)* because it is frozen; the log of record is the 3B-causal chapter in
[`phase-3.md`](phase-3.md). The results entry reports the as-registered and
as-amended analyses side by side wherever they differ (§A6).

## Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| **z** | ℝ⁵¹² | the `s4.fc1` hook value: the **pre-ReLU** output of the `fc1` Linear, and what the SAEs were trained on |
| **h = relu(z)** | ℝ⁵¹², ≥ 0 | what the two heads read (dropout is the identity at inference) |
| **Test F** | [0, 1], ideal 0 | competence audit: how often the unmasked top-1 action is illegal |

All other terms: the pre-registration's §0.

## A1. Correction: the hook is PRE-ReLU

**Registered** (§0, §1, §4, §5.1, §7, §11): the `s4.fc1` value was treated as
**post**-ReLU, so the downstream map was `tanh(W h + b)` with `h` the hook value.

**Fact.** `ActivationStore` hooks the `fc1` `nn.Linear` module, and the model
applies ReLU functionally afterwards (`x = F.relu(self.fc1(x))`). No model in
the repo has an `nn.ReLU` module — every model uses functional ReLU, so
**every hook is a pre-activation**. On champYb `s4.fc1`: min −5.70, zero exact zeros;
`K05`'s checkpoint names this file as its training data.

**Amended.**

- The hook value is `z` (pre-ReLU); the heads read `h = relu(z)`. The downstream
  map is `z ↦ tanh(W·relu(z) + b)`.
- The row-space / null-space exactness of §4 holds **on `h`**. On `z` the map is
  piecewise linear: a perturbation acts only through the units active after it,
  then through W. Changing only units that stay negative changes nothing, exactly.
- §5.1: every patch is made on `z` (the latent-set, direction and full-activation
  formulas unchanged, with `z` in place of `h`), and the decision is the legal
  argmax of `W·relu(z') + b`.
- **Withdrawn**: §1's clause about "negative coordinates fc1 can never produce",
  §5.1's "no clipping in the primary analysis" paragraph, and §11's "share of
  negative coordinates reported; clipped robustness variant". They addressed a
  problem that does not exist. The ReLU after the hook is the network's own and
  is always applied; there is no clipping choice.
- §5.4 R5: the linear probe is fit on the same `z`, so its coefficients are
  already the patch direction.
- **§7 Tier A, replaced:**

  | id | control | required outcome |
  |---|---|---|
  | A1a | perturb `h = relu(z)` along a random null-space direction of W (readout check) | max \|Δlogit\| < 1e-5 on every pair |
  | A1b | perturb `z` only on units negative in the base, keeping them negative | max \|Δlogit\| < 1e-5 on every pair |
  | A2 | full patch `z_b' = z_s` | patched decision = the network's decision on `s`, for 100 % of pairs |
  | A3 | **forward-pass equivalence**: the closed form `W·relu(z') + b` vs the real model run with a forward hook replacing the `fc1` output by `z'` | identical decisions; max \|Δlogit\| < 1e-5 |

  Registered A1 (null-space on the hook value) is not exact once the ReLU sits
  between hook and head; registered A3 (a trivially causal direction) is
  subsumed by A2. The new A3 is also the portability check (§A4).
- §7 B1: random directions are drawn in `z` space.

**Consequence for §3 (disclosure).** The 2026-09-07 screen's r = −0.19 between
a concept's best-latent MCC and `‖W d_j‖`, and that session's PCA-alignment
numbers, were computed treating `z` as the heads' input. They ignore the ReLU
and are **not causal magnitudes at all**; none is carried forward. The
null-space / row-space check of that session was computed on `h` and stands,
as a statement about the readout. H-C4 stays two-sided.

**Motivation:** code inspection during implementation. **Not results-motivated.**

## A2. Training facts, and Test F replaces the scratch measurement

**§2.1, registered:** the training code "is not on this machine"; the illegal
rate was a 30,000-position scratch measurement.

**Amended — training masks illegal actions in every loss.** Reported
2026-09-12 by the training project; the trainer is not in this repo, so these
are cited, not re-verified:

- place TD loss: only on the move played; next-state max over legal moves only
  (`_masked_max`, `QuartoRL/RL_functions.py:1537`);
- select oracle loss: SmoothL1 with illegal pieces masked
  (`RL_functions.py:1569`; lines 645–654 of the Yb training script);
- hot-piece loss: multiplied by the legal mask (`RL_functions.py:1738`);
- place-win hinge: legal cells only (`RL_functions.py:1682`);
- **no legality loss** in the champYb recipe. One existed only in
  `QC_unifiedNoMask` (2026-05-11), rejected because its illegal top-1 stayed at
  **0.32**.

**Amended — Test F**, added to `scripts/model_competence_audit.py`. Artefact:
`data/quarto/audit-champYb-ownpositions.json` (champYb's own position set,
295,029 positions with at least one occupied and one empty cell):

| pieces on board | champYb: raw place top-1 on an occupied cell | untrained champYb | uniform chance |
|---|---:|---:|---:|
| 1–4 | 40.3 % | 30.8 % | 18.5 % |
| 5–8 | **82.8 %** | 54.6 % | 39.7 % |
| 9–12 | 94.7 % | 70.7 % | 61.6 % |
| 13–15 | 93.8 % | 85.8 % | 83.8 % |
| **all** | **67.2 %** | 47.3 % | 34.4 % |

Mean **1.77** engine retries to a legal placement (p90 4). **Select head**
(post-placement states as in Test B, n = 3,834): raw top-1 is a piece not in
storage **28.5 %** (untrained 40.6 %). Test D on this position set: −0.245
(−0.31 on the shared champAa set of `audit-champYb.json`, which the
pre-registration cites). The training project's own figures (~70 % place, ~25 %
select) agree.

**§2.2, amended:** point 4 and the "deferred" legality-trained champion become:
a new champion is being trained **in parallel** from 2026-09-12, following
[`../explanations/new-champion-recommendations.md`](../explanations/new-champion-recommendations.md),
and 3B-causal is built to re-run on it unchanged (§A4). The decision to proceed
on champYb is unchanged.

**Motivation:** new information from the training project; a scratch number
replaced by an artefact. **Not results-motivated.**

## A3. Verdict rule `3B.C1`: the asymmetries get names

**§8, registered:** a five-row table (concept-consistent, context-blind,
anti-consistent, off-target, inert) plus an `(on-only)` qualifier. It did not
cover "installs C, but a powered switch-off fails to remove it", which fell
through to `inert`. That case is the add-vs-remove asymmetry Phase 3D
pre-registers (Engels et al.: installing ~80–100 %, removing ~0 %) — the
expected signature of a concept carried redundantly.

**Amended — rules apply in this order; the first match is the verdict.**
"Installs" = switch-on IIA\* ≥ 0.20 and BH-significant; "removes" = powered
switch-off IIA\* ≥ 0.20 and BH-significant.

| verdict | condition | reading |
|---|---|---|
| **underpowered** | switch-on n < 100 or specificity n < 100 | no verdict |
| **context-blind** | installs, **and** the specificity pull toward *c* is BH-significant | a generic knob, not C |
| **concept-consistent (on-only)** | installs, specificity not significant, switch-off underpowered | sufficient; necessity unmeasured |
| **concept-consistent** | installs, specificity not significant, **removes** | the network's variable for C |
| **install-only** | installs, specificity not significant, powered switch-off **does not** remove | sufficient, not necessary: C is also carried elsewhere |
| **remove-only** | does not install, but **removes** | necessary, not sufficient |
| **anti-consistent** | switch-on IIA\* below the null's 5th percentile | pushes away from the target |
| **off-target** | switch-on not significant, but the flip rate above its null's 95th percentile | does something, not this |
| **inert** | none of the above | — |

Thresholds are unchanged. Implemented as `lib/sae/interchange.py::classify`;
the full truth table is `tests/test_interchange.py::TestVerdictRule`.

**§9 H-C2, amended:** "R1 concept-consistent for ≥ 50 %" becomes "R1
**concept-consistent, `(on-only)` or `install-only`** for ≥ 50 %". Installing C
is causal use; necessity is reported separately. The falsification threshold
(< 20 % → captured atoms are epiphenomenal read-outs) is unchanged.

**Motivation:** a gap found while implementing `classify`. **Not
results-motivated.** The as-registered reading stays recoverable: a result
reported `install-only` here would have been `inert` under the registered text.

## A4. Build plan and portability

**§13, amended:**

- Step 1 ✅ — Test F (§A2).
- Step 2 ✅ — `scripts/linear_probe_baseline.py` now saves the probe directions
  as `<stem>_directions.pt` (coef, intercept, train-selected threshold, a fitted
  mask). The registered text was already correct that the probe is fit on raw
  activations; now it is also known that those activations are `z`. **Open:**
  the hawk and tiger probe reports on Yb fc1 predate this and must be re-run,
  and **no hen probe exists on Yb fc1**.
- Step 3 ✅ — `lib/sae/interchange.py` with two readout paths behind one
  interface: the generic forward-hook path (any architecture, any hook) and the
  closed form where the downstream map is known. Tier A (as amended in §A1)
  runs as unit tests on a toy model — 31 tests — and again as the runner's
  dry-run gate on the real champion.

**Portability — a new champion is a new config file, not new code.**

- **One file per champion**: `configs/3B-causal/champ<Tag>.yaml` names the
  champion config, hook module, each head's logit module, the position,
  activation, label and orbit files, the dictionaries and the probe reports.
  The library and scripts name no champion, hook or file.
- **Readout `auto`**: the forward-hook readout is always built; the closed form
  is built from the head weights where possible; A3 must find them identical on
  the dry run, otherwise the run falls back to the forward-hook path and says so.
- **Guards that travel**: a per-sample-encoder check rejects batch-dependent
  dictionaries (BatchTopK in train mode) from R1–R4, and Tier A runs on every
  champion before any score is computed.
- **Results keyed by `(basis, bsp_id, role)`**, with roles R1–R7 defined by how
  a representation is chosen rather than by which dictionary, so two champions'
  summaries join directly.
- **A new champion's prerequisites** are the standard champion pipeline:
  positions, activations, a canonical per-sample fc1 dictionary with its eval
  and top-K export, probes with directions, and orbit IDs. R2 also needs the
  champion's 3A report (for `knee_k`); without it R2 is reported `not-run`,
  never silently dropped.

## A5. What did not change

The method (interchange; Wave 1 offered-piece swap; pair eligibility §5.3), the
prediction table (§6), every threshold (IIA\* ≥ 0.20, BH q = 0.05, n ≥ 100 / 100
/ 50, the DAS-1 gate at 50 %), hypotheses H-C0, H-C1, H-C3, H-C4, H-C5, and
statistics (§10).

## A6. Deviation log (for the results entry)

| # | section | registered | amended | results-motivated? |
|---|---|---|---|---|
| 1 | §0, §1, §4, §5.1, §5.4, §7, §11 | hook value post-ReLU; clipping choice; Tier A1/A3 as written | hook value pre-ReLU `z`; network's own ReLU always applied; Tier A1a/A1b/A2/A3 of §A1 | no — code inspection |
| 2 | §3 | screen r = −0.19 disclosed | same, plus: computed in the wrong space, not carried forward | no |
| 3 | §2.1–2.2 | trainer unseen; scratch illegal rate; legality champion deferred | training masking cited; Test F artefact; champion trained in parallel | no — new information |
| 4 | §8 | five verdicts + `(on-only)` | nine ordered verdicts incl. `install-only` / `remove-only` | no — implementation gap |
| 5 | §9 H-C2 | "concept-consistent" | "concept-consistent, `(on-only)` or `install-only`" | no — follows from #4 |
| 6 | §13 | build plan | steps 1–3 done; portability contract | no |
