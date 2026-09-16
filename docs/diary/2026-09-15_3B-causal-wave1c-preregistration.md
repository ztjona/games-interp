# Phase 3B-causal — Wave 1c pre-registration: installation as the gate, removal as a hypothesis (2026-09-15)

**Status: FROZEN from the commit that adds it.** Written after Wave 1b and
**before any Wave-1c position, pair or score exists**. Once committed it is
never edited; changes go in `*_3B-causal-wave1c-amendment-N.md`, with the rules
of the Wave-1 pre-registration's §14 (say what changed, why, and whether any
result had been seen).

Waves 1 and 1b stay as recorded. This document defines Wave 1c only.
Wave-1b record: [`2026-09-15_3B-causal-wave1b-results.md`](2026-09-15_3B-causal-wave1b-results.md).
Design it builds on: the [Wave-1b pre-registration](2026-09-14_3B-causal-wave1b-preregistration.md),
which carries the [Wave-1 pre-registration](2026-09-12_3B-causal-preregistration.md) and its
amendments [1](2026-09-12_3B-causal-amendment-1.md) and [2](2026-09-12_3B-causal-amendment-2.md).
Definitions not restated here: [`../methods-reference.md`](../methods-reference.md) §9.

## 0. Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| **D(x)** | a cell | the network's decision on input *x*: legal argmax of the place head |
| **IIA_net\*** | (−∞, 1], ideal 1, 0 = no effect | `(A − A₀) / (1 − A₀)`, A = P(patched decision = D(s)), A₀ = P(D(b) = D(s)) |
| **E, ρ** | E ideal ≤ 0; ρ ≥ 0, ideal 0 | excess specificity flip rate over the null; ρ = E / switch-on IIA_net\* (Wave 1b §6) |
| **installs** | verdict | switch-on IIA_net\* ≥ 0.20, BH-significant, and not context-blind (§9) |
| **removal label** | removes / does not remove / underpowered | the switch-off arm, reported beside the verdict (§9) |
| **DAS-k (R8-k)** | k ∈ {2, 4, 8, 16} | the best *k*-dimensional subspace, trained like DAS-1 (§7) |
| **R7-off** | — | DAS-1 trained on switch-off pairs alone (§7) |
| **k\*** | {2, 4, 8, 16} or none | the smallest *k* at which DAS-k removes for at least half the concepts (H-C6) |
| **gold*k*r2** | — | a second, independent draw of the gold*k* protocol (§4) |

## 1. Why Wave 1c

Wave 1b's gate failed in both sets (gold3 37 %, gold5 2 %) for a reason the
design did not anticipate. DAS-1, the positive control, **installs** concepts
almost perfectly and specifically (pinned switch-on IIA_net\* ≈ 0.97, no concept
context-blind), but **does not remove** them: on the pairs where the network
itself leaves the formerly winning cell, the patch moves it only 20–46 % of
the time. The gate required both, so it measured the add-versus-remove
asymmetry rather than the instrument, and no SAE, probe or anchored verdict was
read.

Wave 1c separates the two questions. **Installation** — the question 3B-causal
exists for: do SAE latents, the probe and the anchored slot carry C causally,
measured against the best direction? — is judged on its own, with DAS-1's
installation as the positive control. **Removal** becomes two registered
hypotheses instead of a gate condition. Both are tested on fresh boards.

## 2. Disclosure: everything seen before this was written

- **Wave 1 (pilot) and Wave 1b**: the gate results and every R7 (DAS-1) number
  — arm scores by family, ceilings, target margins, DAS–probe cosines, the
  informative-pair decomposition (`saes/quarto/analysis/3B-causal_champYb_wave1b_diagnostics.json`).
- **The profile of Wave 1b's gate-passing concepts**, made on request after the
  results entry: gold3's 48 `(on-only)` passes are the concepts with too few
  switch-off pairs there (median 39), and all 48 are install-only in gold5;
  of its 17 concept-consistent, 3 are also so in gold5; switch-off removal is
  highest for the poles *little* and *square* in both sets (medians 0.08–0.13,
  all below 0.20; *without_hole* too in gold5); removal is higher in gold3 than
  in gold5 at every number of pieces on the board where both have pairs; switch-off bases concentrate at the
  random hand-over (3-piece boards in gold3, 5-piece in gold5).
- **In particular: the install gate (§10) was chosen knowing that DAS-1 installs
  on 174 / 176 (gold3) and 170 / 176 (gold5) concepts in Wave 1b.** The gate is
  therefore a positive control for the instrument on fresh data, not a test,
  and is stated as such.
- **R1–R6 have never been examined in any wave** — no verdict and no score
  (§12).
- **A descriptive probe made before this document was frozen** (2,000 games per
  *k*, throwaway seeds 9103 / 9105, self-play and unpatched forward passes only,
  as Wave 1b §4.2): excluding the pilot and both Wave-1b runs costs 0.5 % of
  gold3 positions and 0 % of gold5, so the freshness rule (§4.2) is affordable;
  together with the scaled counts above it set the size in §4.1.
  Artefact: `saes/quarto/analysis/3B-causal_champYb_gold-r2-size-probe.json`.

## 3. What changes, and why

| # | Wave-1b finding | Wave 1c |
|---|---|---|
| 1 | The gate asked DAS-1 to install **and** remove; it installs but does not remove | **Gate = DAS-1 installs** (§10) |
| 2 | Removal failed for the best direction, so no representation could be concept-consistent | **Primary verdict = installs** (rule `3B.C3`, §9); removal reported as a separate label |
| 3 | Three untested explanations for the removal failure: (1) removal needs more than one dimension, (2) the joint objective trades removal for installation, (3) the formerly winning cell stays attractive for other reasons | (1) → **H-C6** with DAS-k; (2) → **H-C7** with R7-off; (3) is a threat to validity (§13) |
| 4 | Feasibility required switch-off power, which the verdict no longer uses | feasibility on the switch-on and specificity arms (§4.4) |
| 5 | Wave 1b's boards now carry seen outcomes | **new draws, every pilot and Wave-1b pair board excluded** (§4) |

Everything else is carried over from Wave 1b unchanged: the 176 concepts, the
offered-piece swap and its eligibility rules, caps of 1,000 pairs per
(concept, kind), the switch-off filter, the network-own targets and rule-3B.C2
metrics (IIA_net\*, F, E, ρ), DAS-1 trained on all three kinds, R1–R7, the B1′
and B2 nulls, n_min (100 / 100 / 50), BH at q = 0.05, the 0.20 floor, the 50 %
gate threshold, and confirmation only when both sets agree.

## 4. Data: two new draws

### 4.1 Protocol

As Wave 1b §4.1 — gold*k*: random decisions through the *k*-th placement and
the hand-over after it, then champYb's legal argmax for both sides; placements
*k*+1 onward recorded — with new seeds:

- **gold3r2**: *k* = 3, **60,000** games, seed **3103**.
- **gold5r2**: *k* = 5, **60,000** games, seed **3105**.

Twice Wave 1b's 30,000. The switch-on and specificity arms are capped at 1,000
pairs and every concept already reached the cap, so a larger draw buys nothing
there; it buys the **switch-off** arm, which H-C6 and H-C7 rest on. Scaling
Wave 1b's own design-stage counts (pairs grow with positions): at 30,000 games
gold3 would leave 48 of its 152 pinned concepts without a testable switch-off
arm and give the rest a median of 67 pairs; at 60,000 that is ~8 of 152 and a
median of ~134, and gold5's median rises from 234 to ~469. Halving the noise
also matters: in Wave 1b, 13 of gold3's 17 concept-consistent verdicts did not
replicate, on 50-130 pairs each.
- Generated on CPU; duplicates removed within each set. Names
  `positions-gold{3,5}r2_yb_unique.pt`; labels `bsp_labels-<basis>YbGold{3,5}r2_<n>.pt`;
  orbit IDs from `scripts/compute_orbit_ids.py`. `r2` marks the second
  independent draw of a protocol.

### 4.2 Freshness

Every recorded position whose board, up to the 8 symmetries (piece ignored),
matches the board of **any pair of the pilot or of either Wave-1b run** — any
concept, any kind — is removed before pairs are built. Board-only canonical
hash as Wave 1b §4.3; the number removed is reported and re-checked at run start.

### 4.3 Guard

The generator guard runs as in Waves 1 and 1b: vectorised concepts must equal
the stored labels exactly on the natural positions, or the run aborts.

### 4.4 Feasibility

A set enters the analysis only if **at least 50 % of the 176 concepts are
powered in the switch-on and specificity arms** (n ≥ 100 each), judged on
design-stage pair counts before any score. Switch-off power is reported; it
enters only H-C6 and H-C7, which count concepts whose switch-off arm is powered.

## 5. Pairs

As Wave 1b §5: the same board, a different piece in hand, the base's piece
being the one actually offered; same three kinds, eligibility, caps, seeds
and switch-off filter.

## 6. Metrics

As Wave 1b §6 (rule 3B.C2 metrics): IIA_net\* toward D(s) on switch-on and
switch-off, F / E / ρ on specificity; the oracle score, the ceilings under both
targets, target margins and flip rates reported alongside.

## 7. Representations

- **R1–R7** as in Wave 1b §7, unchanged (R7 = DAS-1 on all three kinds toward
  the network-own targets; R4 selected by switch-on IIA_net\*).
- **R8-k (DAS-k), k ∈ {2, 4, 8, 16}**: the patch `z_b + Q Qᵀ (z_s − z_b)` with
  Q a *d* × *k* orthonormal basis, trained exactly as R7 — cross-entropy over the
  legal set toward the network-own targets, all three kinds, each kind's loss
  averaged then the kinds averaged; Adam, 400 steps, lr 0.05, seeded per
  (concept, fold, *k*); Q re-orthonormalised at every step; same folds, scored
  held out. (*k* = 1 is R7.)
- **R7-off**: DAS-1 trained on the switch-off training pairs **alone**, toward
  D(s); otherwise as R7. It separates explanation (2) from (1): if a direction
  trained only for removal removes, the joint objective was the obstacle.

Folds: 5, grouped by board orbit, shared by all kinds and representations.

## 8. Nulls

- R1–R7 as Wave 1b §8 (B1′ covariance-matched directions for R5–R7, isotropic
  reported alongside; B2 frequency-matched latent sets for R1–R4).
- **R8-k**: 1,000 random *k*-dimensional subspaces, each spanned by *k* draws
  from N(0, Σ_Δ) (Σ_Δ as for B1′), orthonormalised.
- **R7-off**: B1′.
- Greater-tail p on IIA_net\* and on F; **BH at q = 0.05 within each (set,
  representation, arm)**, each *k* its own representation.

## 9. Verdict rule `3B.C3`

For every representation and concept, ordered, first match:

`underpowered` (switch-on n < 100 or specificity n < 100) → `context-blind`
(installs by the floor and significance, and E BH-significant, and ρ ≥ 0.5) →
**`installs`** (switch-on IIA_net\* ≥ 0.20 and BH-significant) →
`anti-consistent` (switch-on IIA_net\* below the null's 5th percentile) →
`off-target` (switch-on not significant, flip rate above the null's 95th
percentile) → `inert`.

**Removal label**, reported beside every verdict and never changing it:
`removes` (switch-off n ≥ 50, IIA_net\* ≥ 0.20, BH-significant) /
`does not remove` / `underpowered` (switch-off n < 50). The full rule-3B.C2
verdict is also reported, so Waves 1b and 1c can be compared.

A representation **removes specifically** at a concept when it removes and its
3B.C3 verdict is not `context-blind`; for R7-off, whose install arm is not
trained, removal alone counts and its E is reported.

## 10. The gate

**R7 must `install` for ≥ 50 % of powered concepts (§11), in each set
separately.** If it fails in a set, no R1–R6 verdict is read for that set.
Stated plainly (§2): Wave 1b makes this near-certain to pass. It confirms the
instrument installs on fresh boards; the hypotheses are tested by R1–R6, whose
results nobody has seen, and by the removal representations, which are new.

## 11. Hypotheses

"Powered" means powered in the switch-on and specificity arms, except in H-C6
and H-C7, which also need a powered switch-off arm.

| id | hypothesis | prediction | falsified if |
|---|---|---|---|
| H-C0 | the gate | ≥ 50 % | < 50 % in a set → that set's R1–R6 unread |
| H-C1 | the probe direction is the network's variable | R5 `installs` for ≥ 50 % of powered concepts | < 20 % while H-C0 holds |
| H-C2 | 3A's captured atoms are causally used | among hawk/hen pinned concepts 3A classed `captured`, R1 `installs` for ≥ 50 % | < 20 % |
| H-C3 | tiling is causal | (a) R1 `installs` less often on tiger than on pinned; (b) R1's switch-on IIA_net\*, by single completing pole, significant for ≤ 2 poles; (c) R2 IIA_net\* ≥ 0.8 × R7's | R2 does not exceed R1 on tiger |
| H-C4 | decodability vs causal efficacy | two-sided Spearman between best-latent MCC and R1 switch-on IIA_net\* | reported as measured |
| H-C5 | ordering of representations | exploratory | — |
| **H-C6** | **removal is low-dimensional** | among pinned concepts with a powered switch-off arm, DAS-k **removes specifically** for ≥ 50 % at some *k* ∈ {2, 4, 8, 16}; k\* reported | < 20 % at every *k* ≤ 16 |
| **H-C7** | **the joint objective blocked removal** | R7-off removes for ≥ 50 % of those concepts | < 20 % |

Between the prediction and the falsification bound a hypothesis is
**inconclusive**. H-C6 and H-C7 are not exclusive: both, either or neither may
hold. **A hypothesis is confirmed or falsified only if gold3r2 and gold5r2
agree**; otherwise it is reported as prefix-dependent — a live possibility for
removal, which differed between Wave 1b's sets.

## 12. What stays sealed

The R1–R6 results of the pilot and of both Wave-1b sets stay sealed until the
Wave-1c results entry is written. After that they may be reported as
**exploratory** — the pilot's under rule 3B.C1, Wave 1b's under 3B.C2 — and are
never pooled with Wave 1c.

## 13. Threats to validity

| threat | handling |
|---|---|
| the gate was chosen after seeing Wave 1b | disclosed (§2); it is a positive control, and every hypothesis rests on representations whose results are unseen or new |
| on switch-off, D(s) reflects every difference between the two pieces, not only C (explanation 3): a *k*-dimensional subspace can reproduce D(s) by carrying other piece variables | removal counts only when **specific** (§9): the same subspace must not be context-blind on pairs where C does not change |
| switch-off bases concentrate at the random hand-over (Wave 1b) | removal is reported by the number of pieces on the board as well as pooled |
| H-C6 takes the best of four *k* | BH within each *k*; the 50 %-of-concepts threshold is far above what four chances at a 5 % false-discovery rate can reach; k\* is descriptive |
| prefix length | two lengths; confirmation requires both |
| the SAEs were trained on the amalgam | as Wave 1b §13: shortlists stay the amalgam's |
| one champion, one hook | scope, as before |

## 14. Build plan

1. The freshness filter over several runs (the pilot and both Wave-1b runs).
2. Configs `configs/3B-causal/champYb-gold3r2.yaml`, `…gold5r2.yaml`
   (`rule: 3B.C3`).
3. `interchange_3b.py` / `lib/sae/interchange.py`: rule 3B.C3 and the removal
   label; feasibility on the switch-on and specificity arms; DAS-k training, the
   subspace patch and the random-subspace null; R7-off; the Wave-1c documents in
   the freeze stamps. Each with tests; `methods-reference.md` §9 updated.
4. Build the two sets (~25 min of CPU at 60,000 games); dry run through
   `launch.ps1` (guard, freshness, Tier A, power, feasibility).
5. The two runs — longer than Wave 1b's by the removal representations and the
   larger switch-off arms, about 2 h each in parallel — handed over rather than
   launched.

## 15. Amendment protocol

As the Wave-1 pre-registration's §14. Amendments are named
`YYYY-MM-DD_3B-causal-wave1c-amendment-N.md` and state whether any Wave-1c
result had been seen; the results entry reports as-registered and as-amended
analyses side by side wherever they differ.
