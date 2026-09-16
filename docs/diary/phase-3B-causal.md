# Phase 3B-causal — interchange interventions (2026-09-12 -> )

Status: **OPEN.** Wave 1 (pilot) and Wave 1b ran and failed their gates as
registered; **Wave 1c is pre-registered** (installation as the gate, removal as
hypotheses H-C6 / H-C7):
[`2026-09-15_3B-causal-wave1c-preregistration.md`](2026-09-15_3B-causal-wave1c-preregistration.md).
R1–R6 (SAE, probe, anchored) have not been read in any wave.

Parent: [`phase-3.md`](phase-3.md). Split out of it on 2026-09-15, when the
combined ledger passed the ~400-line cap -- as 3A was. Chapters moved
verbatim; newest first. Methods and verdict rules:
[`../methods-reference.md`](../methods-reference.md) §9 (3B.C1), §9.1 (3B.C2),
§9.2 (3B.C3).

## Chapters

### 2026-09-15 -- 3B-causal Wave 1c pre-registered; what gold3's passes had in common

Design (frozen on commit):
[`2026-09-15_3B-causal-wave1c-preregistration.md`](2026-09-15_3B-causal-wave1c-preregistration.md).

- **Gold3's 65 gate passes** (R7 only; `3B-causal_champYb_wave1b_diagnostics.json`,
  `passing_profile`): the 48 `(on-only)` share nothing but a scarcity of
  switch-off pairs (median 39 < 50); in gold5, with ~206 each, all 48 are
  install-only. Of the 17 concept-consistent, 3 are also so in gold5; most
  shrink toward 0 there (wide gold3 CIs, n ≈ 50–130). Removal is somewhat
  higher for the poles *little* and *square* in both sets, never above a median
  of 0.13, and higher in gold3 than gold5 at every board size. No
  representational pattern: the passes were mostly power and noise.
- **Wave 1c**: the gate is DAS-1 *installing* (a positive control, near-certain
  after Wave 1b, stated as such); R1–R6 are judged on installation (rule 3B.C3)
  with removal as a label; removal becomes **H-C6** (does some DAS-k, k = 2–16,
  remove specifically?) and **H-C7** (does DAS-1 trained on removal alone?);
  new draws gold3r2 / gold5r2 (60,000 games each, twice Wave 1b) exclude every
  pilot and Wave-1b pair board. The extra games buy only the switch-off arm --
  the other two are at their 1,000-pair cap -- taking gold3's testable pinned
  concepts from 104/152 to ~144/152 and halving the noise that kept 13 of its
  17 Wave-1b consistent verdicts from replicating.
- Found while building: a second draw of a protocol writes the same raw file
  name as the first, so `build_gold_sets.py` now refuses to reuse a file whose
  provenance names another seed; each draw has its own raw directory.


### 2026-09-15 -- 3B-causal Wave 1b: the gate fails in both sets

Record: [`2026-09-15_3B-causal-wave1b-results.md`](2026-09-15_3B-causal-wave1b-results.md).

- **As registered**: DAS-1 concept-consistent or on-only on gold3 65/176 (37 %),
  gold5 4/176 (2 %) → gate FAILED in both; H-C0 falsified (the sets agree).
  R1–R6 unread in both sets; H-C1 to H-C5 untested.
- **Specificity fixed**: no concept reaches ρ ≥ 0.5; the pilot's
  context-blindness does not recur.
- **Removal fails**: DAS-1 installs (pinned IIA_net\* ≈ 0.97) but, on switch-off,
  leaves the formerly winning cell on only 20–46 % of the pairs where the
  network itself would. The full patch always does, so the information is in
  the hook but not along that one direction: the add-vs-remove asymmetry.
- Decision pending: a new wave with an install gate and a DAS-k removal curve,
  or close and take removal to 3D.


### 2026-09-14 -- 3B-causal Wave 1b built; both gold sets feasible

Record: [`2026-09-14_3B-causal-wave1b-build.md`](2026-09-14_3B-causal-wave1b-build.md).
Design stage only; no Wave-1b score exists.

- **Gold sets**: gold3 225,197 positions, gold5 132,274 (30,000 games each, CPU;
  558 / 0 dropped as pilot boards). Generator guard 0 mismatches on both.
- **Feasible**: powered in every arm gold3 128/176 (73 %; the 48 short are
  hawk/hen pinned, switch-off only), gold5 176/176. Both sets enter.
- **Prefix sweep** (descriptive, cannot change *k*): switch-off is the binding
  arm at every *k*; greedy play rarely hands over a winning piece.
- **Rule 3B.C2 implemented**; the 3B.C1 path reproduces the pilot's code exactly
  on the untrained twin.
- **Correction**: the 3B.C1 code omitted "switch-on not significant" from
  off-target. One pilot R7 verdict moves off-target → inert; the gate is
  unaffected.


### 2026-09-14 -- 3B-causal Wave 1b pre-registered

Design (frozen on commit):
[`2026-09-14_3B-causal-wave1b-preregistration.md`](2026-09-14_3B-causal-wave1b-preregistration.md).

- **Fixes the pilot's three flaws**: the primary target is the network's own
  counterfactual decision (classic interchange accuracy), not the rational move;
  switch-off is scored against what the network does; specificity counts any
  decision change where the concept is unchanged, and is context-blind only when
  that collateral is at least half the intended effect (ρ ≥ 0.5).
- **DAS-1 is trained toward the network's own counterfactuals** on all three
  pair kinds; direction nulls are covariance-matched.
- **Fresh data**: two champYb self-play sets, **gold3 / gold5** (3 or 5 random
  placements, then best play for both sides; the random player hands over the
  next piece, then leaves). Every board seen in a pilot pair is excluded. The
  prefix length is fixed at 3 and 5; a descriptive sweep reports how the data
  change with it but cannot choose it.
- **A hypothesis is confirmed or falsified only if gold3 and gold5 agree.**
- Disclosed: the relative-leak rule was chosen after seeing the pilot; that is
  why only fresh boards are used. The pilot's R1–R6 stay sealed.


### 2026-09-13 -- 3B-causal Wave 1 on champYb: gate fails as registered

Full record: [`2026-09-13_3B-causal-wave1-results.md`](2026-09-13_3B-causal-wave1-results.md).

- **Gate C1 FAILED**: DAS-1 concept-consistent on 11/176 (6%). As registered,
  no R1–R6 verdict is read; SAE, probe and anchored results stay sealed.
- **Not the failure the gate targeted.** DAS-1 installs pinned concepts almost
  perfectly (IIA\* ≈ 1.00) but leaks toward the concept's cell where the concept
  is false (median +0.061). The network itself does not (+0.006–0.009), so the
  leak is real, but the verdict rule gives specificity no effect-size floor.
- **Switch-off target confounded**: the network itself keeps playing *c* ~70%
  of the time when the win disappears (ceiling 0.24–0.31; likely blocking, not
  verified); DAS-1 over-drives against it.
- Exploratory: with a 0.10 specificity floor, 66% would pass. Not adopted.
- Next: a confirmatory **Wave 1b** on fresh data (pre-registered 2026-09-14,
  below).


### 2026-09-12 -- 3B-causal pre-registered; champYb's legality filter

Full design (frozen on commit):
[`2026-09-12_3B-causal-preregistration.md`](2026-09-12_3B-causal-preregistration.md).

- **Interchange, not steering.** A dose α is a confound three ways (scale
  across latents, off-distribution drift, and negative coordinates at a
  post-ReLU hook). Interchange copies the value a representation takes on a
  real input where the concept holds, so the dose comes from the data.
- **Wave 1 = offered-piece swap on a fixed board.** It flips the 8 pinned
  `*_completable` concepts and their OR (tiger `*_winnable`) together, with the
  same target cell, so the pinned-vs-disjunctive contrast of 3A §5.1 is tested
  causally with every nuisance fixed. H-C3 predicts the top-1 latent works
  only for its own pole and the top-`knee_k` set recovers the variable.
- **champYb has not learned legality**: raw place argmax on an occupied cell
  67.2% (chance 34.2%, untrained 47.4%); the engine retries invalid moves
  rather than penalising them. Decision: proceed. Counterfactuals keep the
  legal set identical, and illegal logits are never scored.
- Disclosed before any data: the `‖W d‖` screen and its r = −0.19 with MCC,
  hence H-C4 is two-sided. That screen was also computed in the wrong space:
  **every hook is pre-ReLU** (functional ReLU after the module), so SAE
  directions live in `z` while the heads read `relu(z)`. Patches go back
  through the network's own ReLU.
- Training confirmed from source by the training project: every loss masks
  illegal actions, no legality loss. A new champion is being trained in
  parallel; 3B-causal is built to re-run on it unchanged.
- **Amendment log of record** (the pre-registration itself is frozen, so its
  own log stays empty):
  1. [`2026-09-12_3B-causal-amendment-1.md`](2026-09-12_3B-causal-amendment-1.md)
     — **pre-data**: pre-ReLU hook and the corrected Tier-A controls; training
     facts and Test F; ordered verdict rule with `install-only` /
     `remove-only`; portability contract.
  2. [`2026-09-12_3B-causal-amendment-2.md`](2026-09-12_3B-causal-amendment-2.md)
     — **pre-data**: Wave-1 implementation decisions (pair caps, set-valued
     targets, R4/R7 cross-fitting, null construction, gate counting,
     replicates) and freeze verification by SHA-256 stamps.
