# Phase 3B-causal — Wave 1b pre-registration: confirmatory, on fresh data (2026-09-14)

**Status: FROZEN from the commit that adds it.** Written after Wave 1 (the pilot)
and **before any Wave-1b position, pair or score exists**. Once committed it is
never edited; changes go in `*_3B-causal-wave1b-amendment-N.md`, with the same
rules as the Wave-1 pre-registration's §14 (say what changed, why, and whether
any result had been seen).

Wave 1 stays as recorded. This document defines Wave 1b only.
Pilot record: [`2026-09-13_3B-causal-wave1-results.md`](2026-09-13_3B-causal-wave1-results.md).
Wave-1 design it builds on: [pre-registration](2026-09-12_3B-causal-preregistration.md),
[amendment 1](2026-09-12_3B-causal-amendment-1.md), [amendment 2](2026-09-12_3B-causal-amendment-2.md).
Definitions not restated here: [`../methods-reference.md`](../methods-reference.md) §9.

## 0. Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| **D(x)** | a cell | the network's decision on input *x*: legal argmax of the place head |
| **D_R** | a cell | the decision after patching representation R from the source into the base |
| **network-own target** | — | what the *network itself* does: D(s) for switch-on and switch-off, D(b) for specificity |
| **IIA_net\*** | (−∞, 1], ideal 1, 0 = no effect | `(A − A₀) / (1 − A₀)`, with A = P(D_R = D(s)) and A₀ = P(D(b) = D(s)) — classic interchange accuracy, chance-corrected |
| **F** | [0, 1], ideal 0 | specificity flip rate: P(D_R ≠ D(b)) on pairs where C is false in both |
| **E** | [−1, 1], ideal ≤ 0 | excess leak: F minus the median F of the null |
| **ρ** | ≥ 0, ideal 0 | relative leak: E / IIA_net\*(switch-on) — collateral per unit of intended effect |
| **gold*k*** | — | a position set of champYb self-play with a *k*-placement random prefix, then best play (§4) |
| **pilot** | — | Wave 1 on champYb, 2026-09-13 |

## 1. Why Wave 1b

Wave 1 was a pilot. As registered its gate failed (DAS-1 concept-consistent on 11
of 176 concepts), so no SAE, probe or anchored verdict was read. The failure was
not the one the gate was built to catch: DAS-1 installed pinned concepts almost
perfectly (IIA\* ≈ 1.00), but the design had three flaws that the pilot's own
ceiling exposed (results entry §5). Wave 1b fixes them and tests the same
hypotheses **on data the pilot never touched**.

## 2. Disclosure: everything seen before this was written

- **The pilot's gate result and every R7 (DAS-1) number**: arm scores per
  family, the full-activation ceiling, R7's network-own accuracy, R7-vs-probe
  cosines, the specificity leak by threat presence, and the exploratory table
  of gate outcomes against a specificity floor (6 % at 0.00, 31 % at 0.05, 66 %
  at 0.10, 87 % at 0.20).
- **In particular: the context-blind criterion below (§9) was chosen knowing
  that, in the pilot, DAS-1's specificity pull toward the concept's cell was a
  median 6.5 % of its install effect.** That is the reason Wave 1b runs only on
  fresh boards (§4.3): the pilot informed the rule, and the rule is tested where
  the outcome is unknown.
- **The pilot's R1–R6 (SAE top-1, top-`knee_k`, top-16, causally selected
  latent, probe, anchored) have never been examined** — no verdict and no score
  (§12).
- Earlier facts carried from Wave 1: Test F, the pre-ReLU hook, the legality
  filter, the Tier-A controls.

## 3. What changes, and why

| # | pilot finding (results entry) | Wave 1b |
|---|---|---|
| 1 | Scoring against the rational move rewards over-driving: DAS-1 beat the network's own ceiling on switch-on (1.000 vs 0.975) and pushed it off *c* on switch-off where the network itself stays (network-own IIA\* −0.55) (§5.3) | **Primary target = the network's own counterfactual decision** (§6). The rational-move score is reported as secondary |
| 2 | The switch-off target was confounded: when the win disappears, the network itself keeps playing *c* ~70 % of the time (ceiling 0.24–0.31) (§5.2) | switch-off scored against D(s), whatever the network does there |
| 3 | The specificity arm had no effect-size floor, and "pull toward *c*" is only one kind of collateral (§5.1) | specificity = **any** change of decision where C is unchanged (flip rate F), judged by its size **relative to the intended effect** (ρ, §9) |
| 4 | DAS-1 was trained to reach the rational move on switch-on pairs only | DAS-1 trained on **all three kinds** toward the network-own targets (§7) — interchange-intervention training in the sense of Geiger et al. |
| 5 | Random unit directions transfer almost nothing, so they make a weak null for directions | **covariance-matched** random directions (§8) |
| 6 | The ceiling was registered but only computed after the run | the ceiling is computed and reported per concept and kind, under both targets |
| 7 | The pilot's boards now carry seen outcomes | **new position sets, and every pilot board excluded** (§4) |

Everything else is carried over from Wave 1 unchanged: the 176 concepts, the
offered-piece swap and its eligibility rules, the base's piece being the piece
actually offered in the game, caps of 1,000 pairs per (concept, kind), the
switch-off filter (the base network must play the expected cell), the
dictionaries and representations R1–R7, n_min (100 / 100 / 50), BH at q = 0.05,
the effect-size floor for installing (0.20), and the 50 % gate.

## 4. Data: the gold sets

### 4.1 Protocol

A **gold*k*** game is champYb against itself. Every decision is **random** up to
and including the *k*-th placement **and the selection that follows it** — the
random player hands over the (*k*+1)-th piece, then leaves. From the (*k*+1)-th
placement on, both sides play **champYb's legal argmax**, for placements and
selections. **Recorded positions**: every placement decision from the (*k*+1)-th
on — a board with *k* or more pieces and the piece in hand.

- **Two sets, both confirmatory: gold3 and gold5** (*k* = 3 and 5).
- 30,000 games each, seeds **3003** (gold3) and **3005** (gold5); duplicates
  removed within each set after generation.
- Names: `positions-gold3_yb_unique.pt`, `positions-gold5_yb_unique.pt`; labels
  `bsp_labels-<basis>YbGold3_<n>.pt` (and `…Gold5…`); orbit IDs from
  `scripts/compute_orbit_ids.py`. "gold" for best play, the digit for the prefix
  length. `copper` is not used: it already means pure random play in
  `Quarto-specifications.md`.

Why a random prefix: with both sides deterministic, every game from the empty
board is the same game. The prefix provides the sample of games.

### 4.2 Prefix length: fixed here, described by a sweep

*k* is fixed at 3 and 5 by this document. Before the gold sets are generated, a
**descriptive sweep** over *k* ∈ {1, 2, 3, 4, 5, 6, 8} (2,000 games each, seed
4000 + *k*) reports: distinct games, recorded positions per game, the
pieces-on-board distribution, threat prevalence, and design-stage Wave-1b pair
counts. **It cannot change *k*.** It involves self-play only; no interchange is
computed.

### 4.3 Freshness

Every recorded position whose **board**, up to the 8 board symmetries (piece
ignored), matches the board of **any pilot pair** — any concept, any kind — is
removed before pairs are built. Matching uses the seed-0 canonical hash of
`scripts/compute_orbit_ids.py`, computed on the board alone. The number removed
is reported.

### 4.4 Guard and feasibility

- The generator guard runs as in Wave 1: vectorised concepts must equal the
  stored labels exactly on the natural gold positions, or the run aborts.
- **A gold set enters the analysis only if at least 50 % of the 176 concepts
  are powered in every arm**, judged on design-stage pair counts before any
  score. A set that fails is reported and dropped. Whether rare concepts reach
  n_min on gold data is not assumed; this count answers it.

## 5. Pairs

As Wave 1 (pre-registration §5.2–5.3; amendment 2 §B2–B3): the same board, a
different piece in hand, the base's piece being the one actually offered. Same
three kinds, same eligibility, same caps and seeds, same switch-off filter.
Only the scoring targets change (§6).

## 6. Metrics (rule `3B.C2`)

For a pair (*b*, *s*) and representation R:

| arm | target | score |
|---|---|---|
| switch-on | D(s) | IIA_net\* = (P(D_R = D(s)) − P(D(b) = D(s))) / (1 − P(D(b) = D(s))) |
| switch-off | D(s) | same |
| specificity | D(b) | F = P(D_R ≠ D(b)); E = F − median F under the null; ρ = E / IIA_net\*(switch-on), undefined when that is ≤ 0 |

Reported alongside, for every representation and kind:

- the **oracle-target IIA\*** — Wave 1's primary score, kept for continuity;
- the **ceiling** (full-activation patch) under both targets;
- the target margin and flip rates.

## 7. Representations

R1–R7 as in Wave 1 (pre-registration §5.4, amendment 2 §B4), with two changes:

- **R7 (DAS-1)** is trained with cross-entropy over the legal set toward each
  pair's network-own target — D(s) for switch-on and switch-off, D(b) for
  specificity — on the training folds' pairs of **all three kinds**. Each
  kind's loss is averaged separately and the three are averaged, so the larger
  specificity set cannot dominate. Adam, 400 steps, lr 0.05, seeded per
  (concept, fold); scored on held-out folds.
- **R4** selects, within the top-16, the latent with the highest switch-on
  **IIA_net\*** on the training folds.

Folds: 5, grouped by board orbit (the gold set's orbit IDs), shared by all kinds.

## 8. Nulls

- **Directions (R5, R6, R7): B1′** — 1,000 unit directions drawn from
  N(0, Σ_Δ), where Σ_Δ is the covariance of z_s − z_b over the concept's pairs
  of all kinds, so null patches move the network as much as real
  source-to-base differences do. Wave 1's isotropic null (B1) is reported
  alongside.
- **Latent sets (R1–R4): B2** as Wave 1 (firing-frequency matched, ±20 %);
  R4's null drawn per fold and pooled, as in Wave 1.
- **p-values**: greater-tail on IIA_net\* (switch-on, switch-off) and on F
  (specificity). Anti-consistent: switch-on IIA_net\* below the null's 5th
  percentile. Off-target: switch-on not significant, but its flip rate above the
  null's 95th percentile.
- **BH** at q = 0.05 within each (gold set, representation, arm).

## 9. Verdict rule `3B.C2`

"Installs" = switch-on IIA_net\* ≥ 0.20 and BH-significant. "Removes" =
switch-off (powered) IIA_net\* ≥ 0.20 and BH-significant. **"Context-blind" =
installs, and the specificity excess E is BH-significant, and ρ ≥ 0.5.** Rules
apply in this order; the first match is the verdict:

`underpowered` → `context-blind` → `concept-consistent (on-only)` →
`concept-consistent` → `install-only` → `remove-only` → `anti-consistent` →
`off-target` → `inert` (meanings as in amendment 1 §A3).

**Why ρ ≥ 0.5.** A representation is context-blind when its collateral — decision
changes where the concept did not change — is at least half its intended effect:
it is then not predominantly about C. That is a principle, not a value read off
the pilot. The pilot's sensitivity table used a different quantity (pull toward
*c*, against an absolute floor) and plays no part here.

## 10. The gate

**R7 must be concept-consistent or `(on-only)` for ≥ 50 % of powered concepts,
in each gold set separately.** If it fails in a set, no R1–R6 verdict is read for
that set.

Stated plainly: the pilot makes it likely that DAS-1, now trained toward the
network's own counterfactuals, passes. The gate is a **positive control for the
instrument**. The hypotheses are tested by R1–R6, whose results nobody has seen.

## 11. Hypotheses

As registered for Wave 1 (pre-registration §9, amendment 1 §A3), scored with
IIA_net\* and rule `3B.C2`:

| id | hypothesis | prediction | falsified if |
|---|---|---|---|
| H-C0 | the gate | ≥ 50 % | < 50 % in a set → that set's R1–R6 unread |
| H-C1 | the probe direction is the network's variable | R5 concept-consistent for ≥ 50 % of powered concepts | < 20 % while H-C0 holds |
| H-C2 | 3A's captured atoms are causally used | among hawk/hen pinned concepts 3A classed `captured`, R1 concept-consistent, `(on-only)` or `install-only` for ≥ 50 % | < 20 % |
| H-C3 | tiling is causal | (a) R1 consistent less often on tiger than on pinned; (b) R1's switch-on IIA_net\*, by single completing pole, significant for ≤ 2 poles; (c) R2 IIA_net\* ≥ 0.8 × R7's | R2 does not exceed R1 on tiger |
| H-C4 | decodability vs causal efficacy | two-sided Spearman between best-latent MCC and R1 switch-on IIA_net\* | reported as measured |
| H-C5 | ordering of representations | exploratory | — |

**Confirmation requires both sets.** A hypothesis is confirmed or falsified only
if gold3 and gold5 agree. If they disagree, it is reported as **prefix-dependent**
and no claim is made.

## 12. What stays sealed

The pilot's R1–R6 stay sealed until the Wave-1b results entry is written.
After that they may be reported as **exploratory, under rule 3B.C1**, and never
pooled with Wave 1b.

## 13. Threats to validity

| threat | handling |
|---|---|
| the rule was chosen after seeing the pilot | fresh boards only (§4.3); full disclosure (§2) |
| the network-own target could reward a representation that carries *other* variables along with C | the specificity arm scores exactly that |
| greedy play rarely hands over a winning piece, so switch-off bases come mainly from the random handover at the switch point and from greedy blunders | design-stage power count and feasibility rule (§4.4) |
| prefix length | two pre-registered lengths; confirmation requires both |
| the SAEs were trained on the amalgam, not on gold positions | gold boards are legal game states within the amalgam's support, but base rates differ; reported, and SAE shortlists stay the amalgam's |
| one champion, one hook | scope, as in Wave 1 |

## 14. Build plan

1. A random-prefix, greedy-continuation mode in the S4 position generator
   (`scripts/games/quarto_s4.py`, `scripts/generate_positions.py`), recording
   only placement decisions from the (*k*+1)-th on.
2. The descriptive prefix sweep (§4.2).
3. Generate gold3 and gold5; deduplicate; labels (hawk, hen, tiger); orbit IDs;
   the freshness filter (§4.3), with a board-only canonical hash exposed by
   `scripts/compute_orbit_ids.py`.
4. `interchange_3b.py`: network-own targets, the F / E / ρ specificity arm,
   rule `3B.C2`, the B1′ null, DAS-1's all-kinds objective, per-concept
   ceilings, a freshness check. Each with tests.
5. One config per gold set (`configs/3B-causal/champYb-gold3.yaml`, `…gold5…`).
6. Dry run through `launch.ps1` (guard, Tier A, power, feasibility); then the
   two runs — each about an hour, so handed over rather than launched.

## 15. Amendment protocol

As the Wave-1 pre-registration's §14. Amendments are named
`YYYY-MM-DD_3B-causal-wave1b-amendment-N.md` and state whether any Wave-1b
result had been seen; the results entry reports as-registered and as-amended
analyses side by side wherever they differ.
