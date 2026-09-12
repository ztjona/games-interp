# Phase 3B-causal — pre-registration (2026-09-12)

**Status: FROZEN from the commit that adds it.** This file fixes the design,
predictions, verdict rule and controls of 3B-causal *before any interchange
number exists*. Once committed it is never edited. Any change after data is
seen goes in a new dated entry named `*_3B-causal-amendment-N.md`, and the
results entry must list every deviation from this document. See §14.

Parent: [`phase-3.md`](phase-3.md). Handoff from 3A:
[`2026-09-04_3A-close-and-3B-reorder.md`](2026-09-04_3A-close-and-3B-reorder.md) §9.

## 0. Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| **base / source** | — | an interchange pair: the input we intervene on (base, `b`) and the input we copy a representation from (source, `s`) |
| **interchange intervention** | — | run the network on `b`, but overwrite one representation of concept C with its value from `s`. If the representation IS the network's variable for C, the network now acts as if C had `s`'s value (Geiger et al., causal abstraction) |
| **IIA** | [0, 1], ideal 1 | interchange-intervention accuracy: share of pairs whose patched decision lands in the target set |
| **r₀** | [0, 1] | no-patch rate: share of pairs whose *unpatched* base decision already lands in the target set |
| **IIA\*** | (−∞, 1], ideal 1, 0 = no effect | chance-corrected IIA = (IIA − r₀) / (1 − r₀) |
| **legal set** | cells | empty cells of the board. Every score is computed over legal actions only |
| **pole** | one of 8 | an attribute value: TALL/LITTLE, BLACK/WHITE, SQUARE/CIRCLE, WITH_HOLE/WITHOUT_HOLE. hawk = the 4 positive poles, hen = the 4 negative ones |
| **`knee_k`** | integer ≥ 1 | latents a concept needs before restricted R² plateaus (3A; [`../methods-reference.md`](../methods-reference.md) §3.4) |
| **DAS-1** | — | Distributed Alignment Search, 1-D: the single direction at fc1 that maximises IIA on training pairs, evaluated on held-out pairs |
| **G11** | gate | decodability ≠ use. The question this phase exists to answer |

## 1. The question

Every 3A number is decodability. 3A found that attribute-pinned concepts are
carried by single dictionary atoms (80.1 % `captured`, `knee_k` 1) while their
disjunctions are tiled across one atom per disjunct (0.0–2.4 % `captured`,
`knee_k` = the disjunct count). **3B-causal asks whether those atoms are the
network's own variables** — whether overwriting them changes what champYb does,
in the way the concept says it should.

**Why interchange, and not free steering.** Steering `h + α·d_j` needs a dose
α, and every choice is a confound (cross-latent scale, off-distribution drift,
and at this post-ReLU hook, negative coordinates fc1 can never produce).
Interchange removes α altogether: the patched value is **the value the
representation actually takes on a real input where the concept holds**. The
dose comes from the data.

## 2. Subject: champYb, and the legality filter [DIRECT + DECISION]

### 2.1 What was measured (2026-09-12)

**The game engine does not penalise illegal moves.** `QuartoGame`
(`Quartopy/quartopy/game/quarto_game.py`) calls the bot up to
`MAX_TRIES = 16` times with increasing `ith_option` until it returns a valid
move; an invalid choice is simply retried. `S4ModelBot.place_piece` / `.select`
walk the network's full 16-way ranking and return the first valid entry.

**champYb's raw first choice is usually illegal.** Place head, 30,000 positions
sampled (seed 0) from `positions-amalgam_yb_unique.pt`, raw argmax over all 16
cells, counted only on positions with ≥ 1 occupied cell. Scratch measurement —
reproduced as competence-audit Test F before anything depends on it (§13, step 1):

| pieces on board | champYb raw argmax on an occupied cell | untrained champYb (E_0000) | uniform-random chance |
|---|---:|---:|---:|
| 1–4 | 40.7 % | 31.2 % | 18.5 % |
| 5–8 | **82.6 %** | 54.8 % | 39.7 % |
| 9–12 | 94.7 % | 70.0 % | 61.4 % |
| 13–15 | 93.5 % | 87.6 % | 83.8 % |
| **all** | **67.2 %** | 47.4 % | 34.2 % |

The engine therefore needs a mean of **1.76 retries** (p90 4, max 15) to reach a
legal placement. Competence Test D agrees: Q(empty) − Q(occupied) averages
**−0.31** (positive on only 16.9 % of positions; untrained: 0.00).

**The training code is not on this machine** (champYb's checkpoints came from
the sibling `hierarchical-SAE` project), so training-time handling cannot be
read from source. The behaviour rules out one possibility: an illegal-move
*penalty* would have pushed Q(occupied) down, and it is up. The measurement is
consistent with training under the same retry engine or with legal-only
exploration — in both, **Q of an illegal action is never a training target**.

[AI-REASONED PROVISIONAL ANALYSIS] Why the trained net prefers occupied cells
*more* than the untrained one: the place unit for cell *c* is only ever trained
on states where *c* is empty, where high value tracks activity in *c*'s
neighbourhood (a completing move sits next to pieces). Its extrapolation to
states where *c* itself holds a piece reads maximal neighbourhood activity.
Untested; nothing below depends on it.

### 2.2 Decision: proceed on champYb [DECISION]

1. **What the bot plays is the network's own preference among legal moves.**
   `argmax` over a subset does not depend on values outside it. The filter
   removes illegal options; it does not reorder legal ones.
2. **The illegal Q-values could not have shaped fc1.** They were never a loss
   target, so no gradient from them reached the representation. The fc1 we
   analyse was shaped by legal-action values only. The illegal outputs are a
   downstream extrapolation, not a contaminant.
3. **The design removes the issue instead of caveating it**: every
   counterfactual in this pre-registration keeps the **legal set identical**
   between base and source (§5.2). Illegal logits are never scored.
4. Retraining would reset every banked number — positions, activations, the
   43-config sweep, the 114-cell 3A panel, the submitted paper — and the
   trainer is not on this machine.

**Stated scope, carried into every results claim:** conclusions are about
champYb's *preferences among legal actions*. Q-values on illegal actions are not
beliefs and are never interpreted. Occupancy concepts carry no behavioural
prediction (legality is enforced outside the network).

**Deferred, not dropped: a legality-trained comparison champion.** It would
answer a different question — does learning legality change how threats are
represented and used? [AI-REASONED PROVISIONAL ANALYSIS] It may be a *less*
clean subject for threat analysis: to suppress 16 place units on occupied cells
it must spend part of fc1's 32-dimensional causal subspace (§4) on occupancy,
where threats would then compete.

## 3. Disclosure: what has already been seen

Pre-registration is only honest if it states what the authors looked at first.

- **The analytic screen** (2026-09-07, scratch, not committed). At `s4.fc1` the
  decision is exactly `h ↦ tanh(W h + b)`, `W = [fc2_place; fc2_select]` ∈
  ℝ^{32×512}, rank 32. Perturbing along a null-space direction changed q by
  7.9e-07; along a row-space direction it flipped the decision.
- **One correlation**: on K05-champYb/fc1, Pearson r between a concept's
  best-latent MCC and that latent's `‖W d_j‖` was **−0.19** (n = 373). A
  different quantity from anything below, from a screen with three unaddressed
  confounds. **Every hypothesis relating decodability to causal effect is
  therefore tested two-sided** (H-C4).
- The illegal-move measurement of §2.1.

No interchange has been run. No IIA value exists.

## 4. Why fc1, and what the architecture guarantees

From the `s4.fc1` hook the only consumers are `fc2_place` (512→16) and
`fc2_select` (512→16); `fc_hot` is training-only and never read at inference.
So the entire downstream map is `tanh(W h + b)`:

- **All causal influence at fc1 lives in the 32-dim row space of W.** A
  perturbation inside the 480-dim null space changes no logit, exactly, for
  every input.
- **Consequence for implementation**: a patched decision needs only
  `W h' + b` — no trunk forward on the patched activation. The trunk runs once
  per base and once per source.

This exact decomposition exists because fc1 is one severe bottleneck (512 → 32)
from the output. It does **not** hold at conv2 (the downstream map is
nonlinear), which is out of scope here.

## 5. Design

### 5.1 The interchange operation

For base `b` and source `s`, with fc1 activations `h_b`, `h_s` (post-ReLU):

| representation R | patched activation `h_b'` |
|---|---|
| SAE latent set J (error-preserving) | `h_b + Σ_{j∈J} (a_j(s) − a_j(b)) · W_dec[j]`, where `a = SAE.encode(·)` |
| direction w (LP, DAS-1) | `h_b + ((h_s − h_b)·ŵ) ŵ` |
| full activation (ceiling) | `h_s` |

The SAE form patches only the chosen latents' contribution and leaves the rest
of the reconstruction **and the SAE error term** untouched, so reconstruction
error cannot masquerade as an effect.

**Decision**: `argmax` of `(W h_b' + b)` restricted to the legal set of `b`.

**No clipping in the primary analysis.** `h_b'` may carry negative coordinates;
report their share per representation. A clipped variant (`max(h_b', 0)`) is a
robustness check. Clipping is not primary because it breaks the null-space
control (§7, A1).

### 5.2 Counterfactual pairs, Wave 1: the offered-piece swap

`b = (B, p)` and `s = (B, p')`: **the same board**, a different piece in hand,
`p'` drawn from the pieces available at `b`. `_build_aux32` recomputes the
available-pool mask from board and offered piece, so the swap is a change of
`x_piece` alone.

**The legal set is identical by construction** — the board, hence every empty
cell, is unchanged. This is what removes the legality issue of §2.

The swap flips every concept that reads the piece in hand, which is exactly the
agent-relative family:

| basis | categories in Wave 1 | form | n BSPs | 3A on champYb/fc1 |
|---|---|---|---:|---|
| hawk | `reframed_completable`, `reframed_sq_completable` | **pinned** — one positive pole, e.g. "Row 0: threat in TALL AND offered piece is TALL" | 76 | 80.1 % `captured` (pooled with `reframed_count`) |
| hen | `neg_completable`, `neg_sq_completable` | **pinned** — one negative pole | 76 | 59.4 % `captured` (pooled with `neg_count`) |
| tiger | `tiger_line_winnable`, `tiger_square_winnable` | **disjunctive** — OR over all 8 poles (`tiger == OR(hawk ∪ hen)`, zero violations) | 19 | 2.4 % `captured`, `knee_k` mode 9 |
| tiger | `tiger_offered_completing_attr` | OR over lines, fixed pole | 4 | — |
| tiger | `tiger_win_now_exists` | OR over everything | 1 | — |

**The central contrast.** For a line L with one empty cell *c*, the 8 pinned
concepts `L_completable_<pole>` are exactly the 8 disjuncts of
`tiger_L_winnable`, and all nine share the **same target cell *c***. The same
boards, the same swaps and the same move therefore test a disjunct's
representation against the representation of the OR. It is 3A §5.1 made
causal, with every nuisance variable held fixed.

### 5.3 Pair eligibility

For a concept C on line or square L with exactly one empty cell *c* in B:

| pair type | base | source | target set | measures |
|---|---|---|---|---|
| **switch-on** | C = 0, and `p` wins nowhere on B | C = 1, and *c* is the **unique** cell where `p'` wins | {*c*} | does installing C's representation make the network take *c*? |
| **switch-off** | C = 1, *c* the unique winning cell for `p`, **and the network plays *c*** | C = 0, `p'` wins nowhere | legal cells ≠ *c* (primary); the network's own choice on `s` (secondary) | does removing C's representation stop it taking *c*? |
| **specificity** | C = 0, `p` wins nowhere | C = 0, `p' ≠ p`, `p'` wins nowhere | {*c*} | if patching still pulls the network to *c*, the representation is not C: it is a generic knob |

For `tiger_offered_completing_attr` and `tiger_win_now_exists` the target is the
**set** of cells where the source piece wins, and "unique winning cell" is
replaced by "non-empty winning set".

The restriction to a *unique* winning cell keeps oracle and network targets
aligned: on those sources a competent network plays *c* (Test A: 97.2 %).
Switch-off requires the base network to actually play *c*, so that a change is
measurable.

### 5.4 Representations under test

For each concept C, on dictionary `K05-champYb-s42-jumprelu-t64-exp8-s4.fc1`
(canonical, 3A panel member). `F04` is the replicate; `K05` s43/s44 are the seed
check for R1–R3.

| id | representation | source of the choice |
|---|---|---|
| R1 | SAE **top-1** latent by MCC | top-K export (`scripts/export_topk_matches.py`) |
| R2 | SAE **top-`knee_k`** latents | first `knee_k` of the top-16 list, `knee_k` from C's 3A report, clipped to [1, 16] |
| R3 | SAE **top-16** shortlist | top-K export |
| R4 | the single latent **within the top-16 with the highest held-out IIA\*** | cross-fitted: selected on training folds, scored on the held-out fold |
| R5 | **LP direction** for C | `linear_probe_baseline.py` coefficients, mapped to raw-`h` space (§13, step 2) |
| R6 | **anchored slot** for C | `I04-champYb` anchored SAE — tiger concepts only |
| R7 | **DAS-1** direction for C | trained on training folds, scored held-out; positive control and 1-D ceiling |

**Cross-fitting** (R4, R7): 5 folds, grouped by **board orbit** (3A's orbit IDs),
so no board or its symmetric image appears on both sides.

### 5.5 Scoring

Every score is over the **legal set** of the base:

1. **Δlogit** of the target minus the mean Δlogit of the other legal cells —
   continuous, logit units;
2. **Δrank** of the target among legal cells;
3. **decision**: patched legal argmax ∈ target set → **IIA**, and **IIA\*** against r₀;
4. **oracle quality** (switch-off): does the patched move still win?

IIA\* is the headline. Levels 1–2 are companions that show *partial* effects —
the representation moved the target but not enough to flip the decision.

## 6. Prediction table

**Wave 1 — frozen at concept level.** "Switch-on predicts" means: if the
representation carries C, overwriting it with a C = 1 value makes the network
act as if C held.

| concept | switch-on predicts | specificity predicts | 3A precondition |
|---|---|---|---|
| hawk/hen `L_completable_<pole>` (pinned) | take *c* | no pull toward *c* | `captured`: one atom carries it |
| tiger `L_winnable` (disjunctive) | take *c* | no pull toward *c* | `spread`: one atom per pole |
| `tiger_offered_completes_<attr>` | take a cell completing an `<attr>` line | no pull | — |
| `tiger_win_now_exists` | take a winning cell | no pull | — |

**Waves 2 and 3 — frozen at family level** (full design in their own
pre-registration entries, before they run):

| family | wave | switch-on predicts | context where it predicts nothing |
|---|---|---|---|
| state threat, pinned (gorilla; hawk/hen `*_count`) | 2 | place phase: take the empty cell **only if** the piece in hand has the pole. Select phase: stop offering pieces with the pole | the piece in hand lacks the pole |
| `state_any` (`*_any_threat`) | 2 | as for pinned, for whichever pole matches | — |
| pool reasoning (`pool_safe_count`, `pool_winning_count`) | 3 | select phase: shift toward pieces with fewer opponent winning cells | forced-loss positions |
| occupancy, cell attribute, offered attribute alone, game phase | — | **no prediction** — the rules dictate no response. Exploratory only, no verdict | — |

## 7. Controls and the gate

**Tier A — implementation. Must hold exactly, or the run stops.**

| id | control | required outcome |
|---|---|---|
| A1 | patch along a random **null-space** direction of W | every pair's decision unchanged; max \|Δlogit\| < 1e-5 |
| A2 | **full** patch `h_b' = h_s` | patched decision = network's decision on `s`, for 100 % of pairs |
| A3 | patch the component along **W's place-row for the target cell** | IIA\* > 0.9 on switch-on pairs — a trivially causal direction, as a plumbing check |

**Tier B — null distributions.** One per (concept, representation size), each
1,000 draws, scored exactly like the real representations:

- B1: random unit directions (null for R5, R6, R7);
- B2: random sets of |J| **alive** latents, matched on firing frequency (±20 %)
  (null for R1–R4);
- B3: for R1, the frequency-matched single latent with the lowest \|MCC\| with C.

**Tier C — the gate. The other arm must be able to fire.** (Reporting standard 7.)

- **C1**: DAS-1 (R7), held out, must be **concept-consistent** (§8) for
  **≥ 50 %** of Wave-1 concepts with adequate n. **If it is not**, the network
  does not carry these concepts as 1-D linear causal variables at fc1. A null
  for an SAE latent or the LP direction would then be expected rather than
  informative, so **no R1–R6 verdict is read**; the plan moves to DAS-k (k > 1)
  under a new amendment.
- **C2** (reported, not gating): oracle hit rate of the unpatched network on
  switch-on sources — the ceiling any oracle-target IIA can reach (≈ Test A,
  97.2 %).

## 8. Verdict rule `3B.C1`

For each (representation, concept), only when **n ≥ 100** switch-on pairs and
**n ≥ 100** specificity pairs (switch-off: n ≥ 50). Otherwise: `underpowered`,
no verdict.

`p` = empirical p-value of the observed IIA\* against its Tier-B null. A Gaussian
fit to the null supplies the tail when the observed value is beyond every draw
(flagged `parametric_tail`). Benjamini–Hochberg at **q = 0.05** within each
(wave, representation id). **Effect-size floor: IIA\* ≥ 0.20** — added because
the 2026-08-14 review found a verdict with no floor that inverted under a change
of hook.

| verdict | switch-on | specificity | switch-off (if powered) |
|---|---|---|---|
| **concept-consistent** | IIA\* ≥ 0.20 and BH-significant | pull toward *c* **not** BH-significant | IIA\* BH-significant |
| **context-blind** | IIA\* ≥ 0.20 and BH-significant | pull toward *c* **also** BH-significant | — |
| **anti-consistent** | IIA\* below the null's 5th percentile | — | — |
| **off-target** | IIA\* not significant, but the legal-decision **flip rate** above its null's 95th percentile | — | — |
| **inert** | none of the above | — | — |

A "concept-consistent" verdict whose switch-off arm is underpowered is reported
as `concept-consistent (on-only)`.

## 9. Hypotheses

| id | hypothesis | prediction | falsified if |
|---|---|---|---|
| **H-C0** | the gate: DAS-1 finds a 1-D causal variable at fc1 | ≥ 50 % of Wave-1 concepts concept-consistent under R7 | < 50 % → stop, amend to DAS-k |
| **H-C1** | the linear-probe direction is the network's variable | R5 concept-consistent for ≥ 50 % of powered Wave-1 concepts | < 20 % while H-C0 holds → the decodability-optimal direction is **not** causal: the chess-probe epiphenomenality result, reproduced in a CNN/RL agent |
| **H-C2** | 3A's captured atoms are causally used | among hawk/hen pinned concepts that 3A classed `captured` on K05, R1 concept-consistent for ≥ 50 % | < 20 % → captured atoms are epiphenomenal read-outs; H10's "the dictionary knows the answer" is a statement about decodability only |
| **H-C3** | tiling is causal, not just descriptive | (a) R1 concept-consistent **less often** on tiger than on the pinned concepts; (b) switch-on pairs stratified by the pole through which `p'` completes: R1's IIA\* is significant for **≤ 2 poles**; (c) R2 (top-`knee_k`) IIA\* ≥ 0.8 × R7's | R2 does not exceed R1 on tiger → the tiled atoms are not **jointly** the causal variable |
| **H-C4** | decodability and causal efficacy | **two-sided**, no direction (disclosure, §3): Spearman ρ between a concept's best-latent MCC and its R1 IIA\*, over powered Wave-1 concepts | reported as measured |
| **H-C5** | ordering of representations | exploratory: R1–R7 mean IIA\* by family, no directional prediction | — |

For H-C3(b), only sources where exactly **one** pole completes the line are used;
multi-pole completions are reported separately.

## 10. Statistics and reporting

- Every IIA\* comes with a 95 % bootstrap CI over **board orbits** (1,000 resamples).
- Results are reported **per concept family and per category**, never pooled
  across bases, following the CLAUDE.md rule on whole-basis scalars.
- Results open with a glossary (reporting standard 6) and put these numbers
  side by side: IIA\*, IIA, r₀, n, null p95, and the Δlogit / Δrank companions.
- Every figure is regenerated from the stored per-pair records; none is
  transcribed.

## 11. Threats to validity, and how each is handled

| threat | handling |
|---|---|
| legality filter (§2) | legal set identical in base and source; illegal logits never scored |
| SAE reconstruction error | error-preserving patch (§5.1) |
| dose / off-distribution | no free α: patched values are real source values. Share of negative coordinates reported; clipped robustness variant |
| the swap changes several concepts at once | representations transfer only what they carry. Specificity pairs catch generic knobs; the collateral flips of each pair are recorded |
| target ambiguity | unique winning cell (§5.3) |
| selection bias (R4, R7) | cross-fitted, orbit-grouped folds |
| multiple comparisons | BH within (wave, representation) plus an effect-size floor |
| one champion, one hook | stated as scope. conv2 needs a gradient screen (nonlinear downstream map); a second champion is a separate question |
| a null result from a weak instrument | the Tier-C gate must pass before any R1–R6 verdict is read |

## 12. Waves 2 and 3

**Wave 2** (state threats, pinned and `state_any`): the counterfactual is a
**board piece swap** — one piece on the board exchanged with one piece in the
pool. The board keeps the same occupied cells, so the **legal set is still
identical**. Contexts: piece in hand with the pole (prediction) vs without
(specificity). Pairs whose swap flips more than one *other* concept of the same
family are excluded, and the number of collateral flips is recorded for every
pair.

**Wave 3** (select phase): pool reasoning and state threats in selection. It
needs select-phase states, built by applying the network's own placement — the
construction Test B already uses. Its first instrument check is the select
head's raw illegal (unavailable-piece) rate, the analogue of §2.1.

Each wave gets its own frozen entry, with its concept-level table, before it runs.

## 13. Build plan

1. **Competence-audit Test F** in `scripts/model_competence_audit.py`: raw
   first-choice illegal rate and engine retries, on the champion's own amalgam.
   Reproduces §2.1 from a committed script before anything depends on it.
2. **Persist LP coefficients** in `scripts/linear_probe_baseline.py`: `coef_`,
   intercept, `C` and the train-split seed. The probe is fit on **raw**
   activations (verified: no scaler in the script), so `coef_` is already the
   direction in `h` space. **If a scaler is ever added, the stored direction
   must become `coef / scale`** or R5 silently patches the wrong direction.
   Re-run for champYb fc1 (the activations are on disk; no `_h` needed).
3. `lib/sae/interchange.py` — game-agnostic: patch operations, IIA / IIA\*,
   null samplers, verdict rule `3B.C1`. Tests include the Tier-A controls as
   **unit tests** — a null-space patch must be exactly inert — plus a planted
   latent with a known effect.
4. Quarto counterfactual generator: the offered-piece swap, the eligibility
   rules of §5.3, winning-cell sets via the competence audit's `placing_wins`,
   and orbit IDs.
5. DAS-1 trainer: one unit vector per concept, cross-entropy over the legal set
   toward the target, 5 cross-fitted folds.
6. `scripts/interchange_3b.py` + `runners/3B-causal.ps1` with a `-DryRun` that
   runs **Tier A for real** (the CLAUDE.md rule: a dry-run that skips the real
   gate is not a dry-run).

**Compute.** The trunk runs once per base and once per source (a small CNN; a
few million forwards on one GPU). SAE encoding is a 512 × 4096 matmul per input,
and patched decisions are `W h' + b` alone. Wave 1 is expected to take minutes
to an hour, and **needs no `_h` cache**. Once measured, the runtime goes in the
runner; anything over ~30 min is launched detached via `runners/launch.ps1`.

**Where definitions go.** IIA, IIA\* and rule `3B.C1` move into
`docs/methods-reference.md` §9 when step 3 lands — defined there, linked from
here, never duplicated.

## 14. Amendment protocol

- Once committed, this file is never edited.
- A change after any interchange number exists goes in
  `YYYY-MM-DD_3B-causal-amendment-N.md`: what changed, why, and **whether the
  author had seen results** that motivated it.
- A change *before* any number exists may still be made as an amendment, marked
  `pre-data`.
- The results entry lists every amendment and every deviation, and reports the
  as-registered analysis next to any amended one.

### Amendment log

*(none)*
