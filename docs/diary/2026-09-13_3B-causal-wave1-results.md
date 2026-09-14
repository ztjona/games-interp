# Phase 3B-causal — Wave 1 results on champYb: the gate FAILS as registered (2026-09-13)

Design: [`2026-09-12_3B-causal-preregistration.md`](2026-09-12_3B-causal-preregistration.md),
amended pre-data by [amendment 1](2026-09-12_3B-causal-amendment-1.md) and
[amendment 2](2026-09-12_3B-causal-amendment-2.md). Run:
`saes/quarto/analysis/3B-causal_champYb_wave1.json` (commit `b544a99`, 56 min).
Diagnostics: `saes/quarto/analysis/3B-causal_champYb_wave1_diagnostics.json`
(`scripts/diagnose_3b_wave1.py`). Every number below is read from those two files.

## 0. Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| **IIA\*** | (−∞, 1], ideal 1, 0 = no effect | chance-corrected share of pairs whose patched legal decision lands in the target set ([methods-reference §9](../methods-reference.md)) |
| **ceiling** | (−∞, 1] | IIA\* of the network's **own** decision on the source input — the full-activation patch |
| **network-own IIA\*** | (−∞, 1], ideal 1 | agreement of the patched decision with the network's own decision on the source, chance-corrected by the base's agreement with it — classic interchange accuracy |
| **leak** | [−1, 1], ideal 0 | on specificity pairs, the rise in P(play the concept's cell) under a patch |
| **DAS-1 / R7** | — | the single fc1 direction trained, on held-out folds, to install the concept; the gate's representation |
| **gate C1** | pass if ≥ 50 % | share of powered concepts on which R7 is `concept-consistent` or `(on-only)` |
| switch-on / switch-off / specificity | pair kinds | concept false→true / true→false / false in both (pre-registration §5.3) |

## 1. Summary

**As registered, the gate fails: DAS-1 is concept-consistent for 11 of 176
concepts (6.25 %), against a threshold of 50 %.** By the pre-registration's own
rule (§7 C1), no R1–R6 verdict is read, and H-C0 is falsified as registered.
**R1–R6 stay sealed** — neither their verdicts nor their scores were examined
for this entry.

**The failure is not the failure the gate was designed to detect.** The gate
was meant to fail if no single direction carries these concepts causally. DAS-1
installs them almost perfectly (hawk/hen switch-on IIA\* ≈ 1.00). It fails
because the same directions also nudge the network toward the concept's cell
when the concept is false — a small effect (median IIA\* 0.061, about 6.5 % of
the install effect) but significant at n = 1,000, and the verdict rule gives the
specificity arm **no effect-size floor**. The diagnostics expose that gap and
two more flaws in my design (§5). **Wave 1 on champYb does not settle whether
concepts have clean single-direction causal representations**, in either
direction.

## 2. Integrity [DIRECT]

- **Freeze.** All three design files were committed and clean at run start. The
  run's stamp for the pre-registration did not equal `sha256(git show HEAD:…)`;
  the cause is line endings only — the working copy carried CRLF after the
  2026-09-12 restore, the blob LF. Content is identical and unchanged since the
  freeze commit `b64b3d3` (verified). `freeze_stamps` now hashes line-normalised
  content (`sha256_lf`), so this cannot recur.
- **Tier A on the real champion**: A1a, A1b, A2 and A3 all pass; closed-form readout.
- **Power**: no arm underpowered (switch-on ≥ 586, switch-off ≥ 238,
  specificity 1,000 pairs per concept).
- **The diagnostics rebuilt exactly the run's pairs** (asserted, pair by pair).

## 3. The registered result [DIRECT]

| R7 verdict | hawk pinned (76) | hen pinned (76) | tiger winnable (19) | tiger offered_attr (4) | tiger win_now (1) |
|---|---:|---:|---:|---:|---:|
| concept-consistent | 6 | 4 | 1 | 0 | 0 |
| context-blind | 69 | 72 | 17 | 2 | 0 |
| install-only | 1 | 0 | 1 | 2 | 0 |
| off-target | 0 | 0 | 0 | 0 | 1 |

**Gate C1: 11 / 176 = 6.25 % → FAILED.** Consequences, as registered:

- **no R1–R6 verdict is read** — the SAE, probe and anchored representations
  are not assessed in this wave;
- **H-C0 is falsified as registered**; H-C1 to H-C5 are not evaluated;
- the registered next step is DAS-k (k > 1) under a new amendment — but see §5.4.

The 11 that pass are mostly **diagonals** (8 of the 10 pinned ones are
`diag_main` / `diag_anti`), plus `row_0_completable_square`,
`col_3_completable_square` and `tiger_line_col_2_winnable`.

## 4. Why it failed: diagnostics [DIRECT — post-hoc, on registered quantities]

Medians over concepts, per family:

| family | arm | R7 (DAS-1) IIA\* | **ceiling** (the network itself) | R7 network-own IIA\* |
|---|---|---:|---:|---:|
| hawk pinned | switch-on | **+1.000** | +0.975 | +0.975 |
| | switch-off | +0.553 | **+0.236** | **−0.550** |
| | specificity | **+0.059** | +0.009 | −0.032 |
| hen pinned | switch-on | **+0.998** | +0.961 | +0.960 |
| | switch-off | +0.581 | **+0.310** | **−0.427** |
| | specificity | **+0.070** | +0.006 | −0.032 |
| tiger winnable | switch-on | +0.890 | +0.965 | +0.892 |
| | switch-off | +0.154 | +0.304 | +0.208 |
| | specificity | +0.023 | +0.005 | +0.050 |
| tiger offered_attr | switch-on | +0.419 | +0.976 | +0.332 |
| tiger win_now | switch-on | +0.114 | +0.928 | +0.067 |

Read row by row:

1. **DAS-1 installs pinned concepts completely**, and in doing so reproduces the
   network's own switch-on behaviour (network-own IIA\* 0.96–0.98). Its
   switch-on score even exceeds the ceiling (1.000 vs 0.975): it sends the
   network to *c* in the ~2.5 % of cases where the network, holding the winning
   piece, does not go there itself.
2. **The specificity leak is the direction's own, not the network's.** When the
   piece changes and the concept stays false, the network itself barely moves
   toward *c* (ceiling +0.006–0.009). DAS-1 moves it 7–10× more. So the
   `context-blind` verdicts reflect a real property of the learned directions —
   small, but not an artefact of the design.
3. **The leak is larger where the concept's own threat is absent** — hawk
   +0.045 (absent) vs +0.022 (present); hen +0.055 vs 0.000 (medians of raw
   rises in P(play *c*)).
4. **Switch-off: the network itself mostly keeps playing *c*** when the concept
   turns false (ceiling only 0.24–0.31), while DAS-1 pushes it off *c* (0.55 —
   against the network's own behaviour, network-own IIA\* −0.55 / −0.43).
5. **Set-valued concepts are not one-dimensional**: DAS-1 installs
   `tiger_offered_completes_*` at 0.42 and `tiger_win_now_exists` at 0.11,
   against ceilings of 0.93–0.98.
6. **DAS-1 and the probe point substantially the same way** for pinned concepts
   (median |cos| 0.58 hawk, 0.58 hen; ≈ 0.04 for random directions in 512-d),
   much less for tiger winnable (0.30) and the set-valued concepts (0.25, 0.01).

## 5. Three flaws in the design, found by the design's own quantities

### 5.1 The specificity arm has no effect-size floor

Installing needs IIA\* ≥ 0.20 *and* significance; `context-blind` needs only a
significant specificity pull. With 1,000 pairs and a tight null, a pull of a
few percent is significant. The 160 `context-blind` concepts have specificity
IIA\* from +0.008 to +0.242 (median +0.061), a median **6.5 % of their own
install effect**.

**Exploratory sensitivity — NOT the gate's outcome**, reported only to show
that the verdict hinges on a threshold the pre-registration never set:

| specificity floor | concept-consistent | gate at 50 % |
|---:|---:|---|
| 0.00 (as registered) | 11 / 176 (6 %) | fails |
| 0.05 | 55 / 176 (31 %) | fails |
| 0.10 | 117 / 176 (66 %) | would pass |
| 0.20 | 153 / 176 (87 %) | would pass |

No floor is adopted for Wave 1. Choosing one now, knowing which one passes,
would be the forking path the freeze exists to prevent.

### 5.2 The switch-off target is confounded — most likely by blocking

The registered target assumed that once *c* no longer wins, a rational player
leaves *c*. The network does not: it keeps playing *c* about 70 % of the time
(ceiling 0.24–0.31). [AI-REASONED PROVISIONAL ANALYSIS] *c* is the one empty
cell of a line holding three pieces that share an attribute; filling it with a
non-matching piece kills that line, which is often the right move. The
positions were not checked for this directly. Either way, "leave *c*" is not
the rational response, and the oracle target is the wrong one for this arm.

### 5.3 The oracle target rewards over-driving

The primary target scores a patch by whether it produces *rational* play, not
whether it reproduces *the network's* counterfactual. DAS-1, trained on that
target, exceeds the ceiling on switch-on and moves the network off *c* on
switch-off where the network itself would stay. Interchange accuracy in the
causal-abstraction sense is agreement with the network's own counterfactual —
the registered *secondary* target, which the run did not compute
(deviation, §6).

### 5.4 The registered fallback does not fit this failure

The pre-registration's consequence of a failed gate — move to DAS-k (k > 1) —
assumed the failure would mean "no 1-D direction carries C". What happened is
the reverse: a 1-D direction carries C and more. A 2-D DAS subspace has more
room to install and would not obviously leak less.

## 6. Deviations from the registered analysis

- **The full-activation ceiling** (listed in pre-registration §5.1) and **the
  network-own secondary target** (§5.3, switch-off) were registered but not
  implemented in the run; both were computed after it by
  `scripts/diagnose_3b_wave1.py` on exactly the run's pairs. Only R7 and
  representation-free quantities were touched.
- **Freeze stamp**: raw-byte hash, CRLF-sensitive; content verified identical
  (§2). Fixed for future runs.
- Nothing else. No verdict, threshold or pair was changed after the run.

## 7. What it suggests [AI-REASONED PROVISIONAL ANALYSIS]

A single fc1 direction is **causally sufficient** to make champYb take a win it
was not offered (switch-on ≈ 1.00, reproducing its own behaviour), and that
direction is **substantially aligned with the linear probe** (|cos| ≈ 0.58). But
the best such direction is **output-proximal**: it over-drives the decision in
both directions and leaks toward the line's empty cell where the concept is
false — more so without the threat on the board, which suggests a "go to this
line's empty cell" component that the threat does not gate. That is neither
evidence that the concepts lack a causal linear representation nor evidence that
they have a clean one. It is evidence that DAS-1, **scored against an oracle
target**, finds decision-level directions — which is what the oracle target asks
it to find.

## 8. What next — a decision

The pilot has told us how to test the question properly. Three options:

1. **Treat champYb Wave 1 as a pilot and run a confirmatory Wave 1b** under a
   new pre-registration fixing §5: **network-own interchange accuracy as the
   primary target**; a specificity **effect-size floor** set from the
   ceiling's own scale (fixed *before* data); switch-off scored against the
   network's counterfactual, not the oracle. Run it on **fresh data**: new
   champYb positions and/or new champions, never the pilot's boards. The rarest
   pinned concepts used all their switch-on pairs, so whether fresh data give
   them enough is an empirical question for the design-stage power count.
2. **Do the registered fallback (DAS-k)** as amendment 3. Faithful to the
   letter of §7, but §5.4 argues it does not address the failure observed.
3. **Unseal R1–R6 as explicitly exploratory.** Their verdicts would carry the
   same specificity-floor problem. Doing this *before* option 1 is frozen would
   let their results shape the confirmatory design, so it should come after, if
   at all.

Recommendation: **1**, on data the pilot cannot have contaminated. The portable
pipeline re-runs on any position set or champion with one config file. Keep R1–R6
sealed until Wave 1b is frozen. (Chosen 2026-09-14: see
[`2026-09-14_3B-causal-wave1b-preregistration.md`](2026-09-14_3B-causal-wave1b-preregistration.md).)

## 9. Reproducing

```powershell
pwsh -File runners\launch.ps1 3B-causal -Champ Yb          # the run (~56 min)
python scripts/diagnose_3b_wave1.py --config=configs/3B-causal/champYb.yaml   # §2-5 diagnostics
```
