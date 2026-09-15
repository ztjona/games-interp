# Phase 3B-causal — Wave 1b results: the gate fails in both sets; DAS-1 installs but does not remove (2026-09-15)

Confirmatory run of the [Wave 1b pre-registration](2026-09-14_3B-causal-wave1b-preregistration.md)
(rule 3B.C2), built as recorded in [the build entry](2026-09-14_3B-causal-wave1b-build.md).
**Reported as registered.** The gate fails in both gold sets, so **no R1–R6
verdict is read in either set** (§10): the SAE, probe and anchored results stay
sealed. Everything below is R7 (DAS-1, the gate's own representation) or
representation-free. Diagnostics: `scripts/diagnose_3b_wave1b.py` →
`saes/quarto/analysis/3B-causal_champYb_wave1b_diagnostics.json`. Definitions:
[`../methods-reference.md`](../methods-reference.md) §9.1.

## Glossary

| term | range / ideal | meaning |
|---|---|---|
| **D(b), D(s)** | a cell | the network's own legal-argmax decision on the base / the source |
| **IIA_net\*** | (−∞, 1], ideal 1, 0 = no effect | `(A − A₀)/(1 − A₀)`, A = P(patched decision = D(s)), A₀ = P(D(b) = D(s)) |
| **A₀** | [0, 1] | how often the network itself decides the same on base and source |
| **F** | [0, 1], ideal 0 | specificity: share of pairs whose decision the patch changes although C is false in both |
| **E** | [−1, 1], ideal ≤ 0 | F minus the median F of random covariance-matched directions |
| **ρ** | ≥ 0, ideal 0 | E / switch-on IIA_net\*; context-blind at ≥ 0.5 |
| **informative pair** | — | a pair on which the network itself changes its decision, D(s) ≠ D(b) |
| **reproduces D(s)** | [0, 1], ideal 1 | on informative pairs, the patched decision is the network's own counterfactual |
| **leaves D(b)** | [0, 1], ideal 1 | on informative pairs, the patched decision is anything but the base's |
| **install-only** | verdict | installs C specifically, but a powered switch-off does not remove it |
| **gate** | ≥ 50 % | R7 concept-consistent or `(on-only)` on at least half the powered concepts, per set |

## 1. Provenance

Runs 2026-09-15 on Deep Brain, both sets in parallel (GPUs 0, 1), 57 min.
Commit **363b58d** (the Wave-1b implementation); all four design documents
committed and unmodified (SHA-256 stamps in each summary). 176 concepts, caps
1,000, 1,000 null draws, seed 0. Before any score: generator guard 0
mismatches, freshness re-check 0 stale boards, Tier A A1a/A1b/A2/A3 pass
(closed-form readout), feasibility passed (gold3 128/176, gold5 176/176).

## 2. As registered

| set | R7 concept-consistent | R7 `(on-only)` | powered | gate | registered outcome |
|---|---:|---:|---:|---:|---|
| gold3 | 17 | 48 | 176 | **36.9 %** | **FAIL** — R1–R6 unread |
| gold5 | 4 | 0 | 176 | **2.3 %** | **FAIL** — R1–R6 unread |

- **H-C0 (the gate) is falsified in both sets; the sets agree, so H-C0 is
  FALSIFIED** (§11).
- **H-C1 to H-C5 are not tested**: they are scored on R1–R6, which §10 leaves
  unread in a set whose gate fails.
- §10 said the pilot made it *likely* that DAS-1 would pass. It did not.

gold3's higher figure is power, not a prefix effect: its 48 `(on-only)` are
exactly the pinned concepts whose switch-off arm is underpowered, so they are
never asked to remove. Among its 128 fully powered concepts, 17 pass (13 %).

## 3. R7 by family

| set | family | n | verdicts | switch-on IIA_net\* | E (median) | ρ ≥ 0.5 | switch-off IIA_net\* | switch-off BH-sig |
|---|---|---:|---|---:|---:|---:|---:|---:|
| gold3 | hawk pinned | 76 | 44 install-only, 24 on-only, 7 CC, 1 off-target | 0.977 | −0.012 | 0 | 0.00 | 8 |
| gold3 | hen pinned | 76 | 44 install-only, 24 on-only, 8 CC | 0.962 | −0.014 | 0 | 0.00 | 11 |
| gold3 | tiger winnable | 19 | 17 install-only, 2 CC | 0.854 | −0.048 | 0 | 0.12 | 8 |
| gold3 | tiger offered_attr | 4 | 4 install-only | 0.307 | −0.039 | 0 | 0.03 | 0 |
| gold3 | tiger win_now | 1 | inert | 0.105 | +0.020 | 0 | 0.02 | 0 |
| gold5 | hawk pinned | 76 | 73 install-only, 2 CC, 1 off-target | 0.979 | −0.018 | 0 | 0.03 | 2 |
| gold5 | hen pinned | 76 | 73 install-only, 1 CC, 2 off-target | 0.962 | −0.019 | 0 | 0.05 | 1 |
| gold5 | tiger winnable | 19 | 18 install-only, 1 CC | 0.787 | −0.022 | 0 | 0.10 | 7 |
| gold5 | tiger offered_attr | 4 | 2 install-only, 2 inert | 0.209 | −0.054 | 0 | 0.04 | 1 |
| gold5 | tiger win_now | 1 | off-target | 0.083 | +0.029 | 0 | −0.01 | 0 |

Concept-consistent in both sets: `col_2_completable_without_hole`,
`row_2_completable_square`, `square_1_2_completable_square`.

## 4. What failed: removal, not specificity

**Specificity is fixed.** No concept in either set reaches ρ ≥ 0.5; the
median excess leak is slightly *below* the covariance-matched null. The
pilot's context-blindness does not recur once DAS-1 is trained on all three
kinds toward the network's own decisions.

**Installing works; removing does not.** Pooled over each family's pairs:

| set | family | kind | informative share | reproduces D(s) | leaves D(b) | keeps D(b) where D(s) = D(b) |
|---|---|---|---:|---:|---:|---:|
| gold3 | hawk pinned | switch-on | 0.34 | **0.98** | 0.99 | 0.99 |
| gold3 | hawk pinned | switch-off | 0.30 | **0.28** | 0.45 | 0.95 |
| gold3 | tiger winnable | switch-on | 0.37 | 0.86 | 0.87 | 0.98 |
| gold3 | tiger winnable | switch-off | 0.32 | 0.17 | 0.30 | 0.98 |
| gold5 | hawk pinned | switch-on | 0.36 | **0.98** | 0.99 | 0.99 |
| gold5 | hawk pinned | switch-off | 0.25 | **0.15** | 0.21 | 0.99 |
| gold5 | tiger winnable | switch-on | 0.37 | 0.82 | 0.83 | 0.99 |
| gold5 | tiger winnable | switch-off | 0.27 | 0.13 | 0.20 | 0.99 |

hen pinned mirrors hawk pinned (switch-off reproduces 0.30 / 0.14). Target
margins agree (pinned): toward the network's own choice, about +2 logits on
switch-on and −0.5 on switch-off, where the patch flips 2–9 % of decisions.
On switch-off the network itself keeps playing the formerly winning cell on
70–78 % of pairs (A₀; the pilot measured ~70 %), and the full-activation patch
reproduces its choice by construction (Tier A A2 = 1.00).

## 5. Interpretation

[AI-REASONED PROVISIONAL ANALYSIS] The pilot's diagnosis was incomplete. It
put the switch-off failure down to a confounded target (the rational move);
with the network's own counterfactual as the target, the single best direction
still cannot make the network leave the cell where its piece used to win.

The information that moves the decision is in the hook — the full patch moves
it every time — but not along the one direction that installs C so cleanly.
Three explanations fit, and none has been tested:

1. **Removal needs more than one dimension.** The win at *c* may be carried
   redundantly (the threat features of the line and of the piece's poles
   separately), so taking one direction away leaves enough to keep *c* on top.
2. **The joint objective trades removal for installation.** Switch-off is a
   third of DAS-1's loss and the hardest part of it.
3. ***c* stays attractive for reasons other than C.** On three quarters of
   switch-off sources the network plays *c* although the piece no longer wins
   there (blocking, line completion); a direction for C does not encode those.

This is the add-versus-remove asymmetry that Phase 3D was going to
pre-register (install ~80–100 %, remove ~0 %), seen here for the best 1-D
direction before 3D exists. The registered gate asked the positive control to
remove as well as install. That is the property a 1-D instrument appears not
to have, so the gate measured the asymmetry rather than the instrument.

## 6. What stays sealed

- **Wave 1b's R1–R6, both sets**: unread (§10). Their numbers are in the
  summaries; no verdict or score has been examined.
- **The pilot's R1–R6**: §12 allows them to be reported as exploratory, under
  3B.C1, now that this entry exists. Not done here: opening them before the
  next design is frozen would inform that design, the same reason the pilot
  was kept out of Wave 1b.

## 7. Decision pending

1. **Close Wave 1b as registered** and take removal to 3D (DAS-k).
2. **A new pre-registered wave on new fresh sets**: the gate is DAS-1
   *installing* (the positive control for installation); R1–R6 are judged on
   the install and specificity arms; removal is a separately registered curve
   over DAS-k (k = 1, 2, 4, 8, 16), which tests explanation 1 directly.
   New gold seeds, with the pilot's and Wave 1b's boards both excluded.
3. **A post-data amendment** reading Wave 1b's R1–R6 as exploratory under an
   install gate — fastest, exploratory only, and it would inform option 2.

[AI-REASONED PROVISIONAL ANALYSIS] Option 2 is the one that keeps a
confirmatory answer to the question 3B-causal exists for (do SAE latents, the
probe and the anchored slot carry C causally, relative to the best direction?)
while making the removal asymmetry a hypothesis rather than a gate. Compute is
another ~1 h; the pipeline reruns from two new config files.
