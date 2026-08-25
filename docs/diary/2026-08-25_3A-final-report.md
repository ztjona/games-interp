# Phase 3A — final report: the wall is a DISJUNCTION wall (2026-08-25)

Status: frozen, self-contained. **Phase 3A is CLOSED.**
Parent ledger: [`phase-3A.md`](phase-3A.md).
Method + verdict rule: [`../methods-reference.md`](../methods-reference.md) §3.
Predecessors, in order: the frozen spec
[`2026-07-21_3A-dilution-diagnostic.md`](2026-07-21_3A-dilution-diagnostic.md);
first run and rule 3A.2 [`2026-07-27_3A-dilution-results.md`](2026-07-27_3A-dilution-results.md);
rule 3A.3 and the readable gate [`2026-08-17_3A-gate-readable.md`](2026-08-17_3A-gate-readable.md);
residuals [`2026-08-21_3A-residuals-and-handoff.md`](2026-08-21_3A-residuals-and-handoff.md);
`hen` + rule 3A.4 + the panel runs [`2026-08-21_hen-basis-and-rule-3A4.md`](2026-08-21_hen-basis-and-rule-3A4.md).

Numbers are `[DIRECT]` from `saes/quarto/analysis/` unless tagged otherwise.
This entry adds three measurements not in any predecessor (§3.1, §5, §6) and
one correction to a standing interpretation (§6).

---

## 0. Glossary

Read this before any number below.

| term | range | ideal for "captured" | meaning |
|---|---|---|---|
| `asymptote_r2` | ≤ 1 | high | Held-out R² of the concept regressed on **all `top_k` = 64** association-ranked latents. **Total recoverable information**, independent of how it is spread. |
| `random_asymptote_r2` | ≤ 1 | ≈ 0 | The same quantity from an SAE trained on an **untrained** network's activations. The *learned-signal* floor. |
| `null_r2` | ~0 | 0 | Permutation floor: labels shuffled after candidate selection. |
| **`solo_frac`** | 0–1 | **≥ 0.70** | `solo_r2 / asymptote_r2` — share of all recoverable signal carried by the single best latent. **The primary concentration metric.** |
| `top_phi` | −1…1 | near ±1 | Signed phi (= MCC) of the best single latent. Comparable to `best_mcc_per_bsp` from the eval pipeline. |
| `knee_k` | 1…64 | 1–2 | Smallest k reaching 90 % of `asymptote_r2`. Brittle as a *threshold*; §5 uses its **distribution**, which is not. |
| `intrinsic_dim` | 1…`community_size` | ~1 | PCA participation ratio of the co-firing community's codes on rows where the concept is TRUE. |
| **`absent`** | — | — | No more signal than a floor: the permutation null (*not real*) **or** the random-model control (*real but not learned*). |
| **`captured`** | — | the solved case | Recovered by a small, low-dimensional latent set. Nothing to fix. |
| **`spread`** | — | **the H10 outcome** | Above both floors, but no single latent carries most of it. The *geometric* verdict — information present, flat dictionary not presenting it in one atom. |
| `geometric_frac` | 0–1 | — | `n_spread / n_threat_bsps`. **Gate G-3A**: ≥ 0.50 → 3C proceeds. |
| `verdict_stability` | `confident` / `undecided` | `confident` | Whether the verdict survives ±`band_sds` sd on every quantity `classify` thresholds on (rule 3A.4/3A.5). |
| **pinned** concept | — | — | A BSP fixed to **one location AND one attribute** (`row_2_threat_tall`, `row_0_count_ge3_tall`, `row_0_completable_tall`). |
| **disjunctive** concept | — | — | A BSP that **ORs over the attribute poles or over locations** (`row_0_any_threat`, `tiger_line_row_0_winnable`, `tiger_offered_completes_tall`). |

`top_k = 64` is the scale of the whole diagnostic: `knee_k`, `community_size`
and `n_candidates` are all bounded by it and `asymptote_r2` is measured at
exactly that support. Every number below is at `top_k = 64`.

---

## 1. What was run

**114 cells** = 44 distinct SAE checkpoints × BSP set, over three champions
(Ta / Ve / Yb), two hooks (`s4.conv2`, `s4.fc1`) and four bases
(gorilla / hawk / hen / tiger). **12,151 concept-verdicts.**

| role | cells | what it is |
|---|---:|---|
| `panel` | 54 | Top-3 distinct **conditions** per (champion, hook, basis), 18 cells × 3, ranked by threat-family MCC. |
| `panel-hen` | 18 | hawk's selections mirrored onto `hen` — the matched-polarity arm. |
| `seed-grid` | 38 | 4 conditions (K03/K04/K05/K06, champYb) × 3 seeds × 4 bases. |
| `anchored-positive-control` | 4 | Supervised I03/I04 anchored SAEs — the calibration check. |

Every cell: `random_control_coverage` **1.00**, `gate_is_provisional` **false**,
uniform orbit-aware train/test splits, rule **3A.5**.

Panel construction, the two selection defects fixed on 2026-08-24, and the
conformance decision are recorded in
[`2026-08-21_hen-basis-and-rule-3A4.md`](2026-08-21_hen-basis-and-rule-3A4.md) §7–8.

---

## 2. Validity audit [DIRECT]

Run before reading any result. Five checks; the diagnostic passes four
outright, and the fifth was a bookkeeping defect now fixed.

### 2.1 The candidate selection does NOT inflate the result — measured, not argued

The pipeline ranks candidates by |phi| on rows that overlap the rows the
held-out R² is later measured on. That is textbook double-dipping, and the
permutation null cannot see it: it shuffles labels **after** selection, so it
holds the selected columns fixed and returns ≈ 0 by construction.

The missing control is a **selection-aware null**: permute the labels *first*,
then run the entire pipeline — re-rank, re-select the top 64, refit the curve.
Run here for the first time, 5 draws per concept, on the two dictionaries that
matter most:

| dictionary | alive | concept | real R² | permutation null | **selection-aware null** |
|---|---:|---|---:|---:|---:|
| K06-champYb conv2 | 199 | `tiger_line_diag_main_winnable` | 0.0666 | 0.0003 | **−0.0006 ± 0.0002** |
| K06-champYb conv2 | 199 | `tiger_square_1_1_winnable` | 0.0583 | −0.0009 | **−0.0008 ± 0.0002** |
| K06-champYb conv2 | 199 | `tiger_offered_completes_square` | 0.1227 | −0.0010 | **−0.0008 ± 0.0001** |
| K03-champYb fc1 | 643 | `tiger_line_diag_main_winnable` | 0.1247 | −0.0001 | **−0.0009 ± 0.0002** |
| K03-champYb fc1 | 643 | `tiger_square_1_1_winnable` | 0.6559 | −0.0008 | **−0.0013 ± 0.0003** |
| K03-champYb fc1 | 643 | `tiger_offered_completes_square` | 0.5084 | −0.0007 | **−0.0009 ± 0.0002** |

**Selecting 64 latents out of 199–643 alive buys zero held-out R² from noise.**
The held-out design does its job at these row budgets (40 k ranking rows put the
sd of a phi estimate near 0.005; the alive pool is small enough that the largest
spurious |phi| is ~0.017, worth ~3 × 10⁻⁴ of R² each, and the 30 % held-out
split removes it).

Two consequences: (a) `asymptote_r2` is an honest estimate, not a
selection artefact; (b) the random-model floor's 0.03–0.05 R² is therefore
**real structure a randomly-initialised CNN's activations carry about board
concepts**, not a measurement artefact — which is exactly what rule 3A.3 claims
it is, now demonstrated rather than assumed.

Reproduce: `docs/methods-reference.md` §3.6 records the protocol.

### 2.2 The random-model control is not over-strong

The floor is only fair if the control dictionary is not a *larger* selection
pool than the run it gates. Measured with the diagnostic's own alive definition:

| dictionary | alive | dead | always-on |
|---|---:|---:|---:|
| `R2-champYbrandom …conv2` (control) | **120** | 3976 | 0 |
| `R3-champYbrandom …fc1` (control) | **190** | 3906 | 0 |
| champYb conv2 panel members | 173–369 | — | — |
| champYb fc1 panel members | 303–690 | — | — |

The controls have the *smaller* pool, so if anything the floor is set slightly
low. The direction of any residual bias is toward calling signal "learned",
which makes the `absent` verdicts in §4 conservative and the `spread` verdicts
marginally generous — relevant only for the one cell in §4.3, whose gap sits on
the margin anyway.

### 2.3 Statistical power is unevenly distributed, and the docs overstate the guard

**45.9 %** of concept-instances (5,580 / 12,151) fall below the
`min_positives = 2000` target, because the widening rule is capped at N: at base
rate 0.0023 (`square_1_1_completable_white`) the whole 296,045-row dataset holds
only **671** positives. `methods-reference` §3.3 says such a verdict "is
provisional"; **nothing in the pipeline implements or reports that** —
`gate_is_provisional` keys on random-control coverage alone.

The empirical direction is reassuring:

| `n_curve_positives` | n | % spread | % captured | % absent | med R² | med `solo_frac` |
|---|---:|---:|---:|---:|---:|---:|
| < 1 000 | 3 576 | 61.4 % | 21.8 % | **16.8 %** | 0.051 | 0.364 |
| 1 000–2 000 | 2 004 | 75.1 % | 19.2 % | 5.7 % | 0.129 | 0.345 |
| 2 000–5 000 | 6 432 | 81.5 % | 17.0 % | 1.6 % | 0.187 | 0.299 |
| ≥ 5 000 | 139 | 89.2 % | 6.5 % | 4.3 % | 0.291 | 0.295 |

Low power pushes concepts toward **`absent`**, not toward `spread`. So the
under-powered tail *deflates* `geometric_frac` — conservative for a gate that
passes on `geometric_frac ≥ 0.50`. It is **not** conservative for the `absent`
verdicts, and every `absent`-driven deprioritization in §4.3 must be read with
that in mind. `methods-reference` §3.3 is corrected in this pass to say what is
actually enforced.

### 2.4 Panel selection is conservative for the conclusion

The panel takes the **top-3 conditions by threat-family MCC**. That selects the
*best* dictionaries, and a better dictionary is more likely to score `captured`
— which counts **against** G-3A. Selecting on quality therefore biases
`geometric_frac` **down**. The conclusion "the concepts are spread" survives a
panel drawn from the strongest available dictionaries, which is the population
the question is about ("can *any* flat SAE present this in one atom?").

### 2.5 DEFECT FOUND AND FIXED: every report was stamped 3A.5 and banded at 3A.4

`DilutionConfig.rule_version` read `3A.5`, whose entire content is the band
calibration (`band_sds` 3.0 → 1.0, `solo_frac_seed_sd` 0.0523 → 0.0407, plus a
measured seed floor on `asymptote_r2`). But `dilution_diagnostic.py` hard-coded
the *old* values as its docopt defaults, and `_config_from_args` overrode the
dataclass with them. Result: **all 114 reports carried `rule_version: 3A.5`
while their bands, `n_undecided`, `undecided_frac`, `geometric_frac_worst_*` and
`gate_verdict_is_stable` were computed at 3A.4's width** — a stored verdict that
could not be traced to the rule that produced it, which is the one thing
`rule_version` exists to prevent.

Effect of the correction (point verdicts are **unchanged** — 0 verdict changes
across all 129 reports including the archive):

| | at 3.0 sd / 0.0523 (on disk) | at **3A.5**: 1.0 sd / 0.0407 | measured cross-seed flip rate |
|---|---:|---:|---:|
| concept-verdicts `undecided` | 2 828 / 12 151 = **23.3 %** | 1 393 / 12 151 = **11.5 %** | **11.4 %** (16 cells, §3.1) |
| cells whose band straddles the 0.50 gate | 15 / 114 | **6 / 114** | — |

The corrected band lands within 0.1 pp of the directly measured flip rate. The
numbers that were on disk overstated the uncertainty by ~2×.

Two sibling drifts found in the same sweep and fixed:

- `summarize_3a_gate.py --expect-rule` defaulted to the literal `3A.3`, so the
  "mixed or stale verdict rules" warning fired on every correct run and had
  become noise.
- `build_3a_panel_runlist.py` decided report freshness on `rule_version` +
  `top_k` only — blind to precisely the constants that had changed, so it
  reported all 114 stale-banded reports `current` and `-SkipExisting` would
  have skipped the repair.

**Fixes.** Every threshold flag now defaults to `auto` = the `DilutionConfig`
value, so the dataclass is the single place a rule constant is written;
`--expect-rule` reads `DilutionConfig().rule_version`; the freshness key covers
every constant `classify`/`classify_with_stability` reads. Three new tests pin
all three (`tests/test_dilution.py::test_cli_defaults_match_dilution_config`,
`::test_cli_overrides_still_apply`,
`::test_panel_freshness_keys_cover_every_banded_constant`). All 129 reports were
reclassified; suite is 961 passed / 11 skipped.

`[AI-REASONED PROVISIONAL ANALYSIS]` This is the **third** recurrence of one
failure mode: a quantity with two homes drifts (2026-08-21, alive-latent
estimate vs `RankingCache`; rule 3A.3, `knee_k` scope vs `community_size`
scope; now the band constants). The general rule the repo should hold to is
that a constant lives in exactly one object and every other site says `auto`.

---

## 3. Stability [DIRECT]

### 3.1 Measured seed replication — the ground truth

4 conditions × 3 seeds × 4 bases, 1,764 concepts. `verdict_stability.py`.

| condition | gorilla | hawk | hen | tiger |
|---|---:|---:|---:|---:|
| K03 champYb fc1 topk-k32-exp8 | 11.8 % | 12.3 % | 7.6 % | **0.0 %** |
| K04 champYb conv2 batchtopk-k32-exp8 | **0.0 %** | 17.5 % | 12.9 % | 17.4 % |
| K05 champYb fc1 jumprelu-t64-exp8 | 9.2 % | 15.2 % | 15.8 % | **0.0 %** |
| K06 champYb conv2 topk-k32-exp8 | **0.0 %** | 20.5 % | 12.3 % | **30.4 %** |

Mean **11.4 %**. Per-concept `solo_frac` cross-seed sd over the 1,764 concepts:
median 0.0191, **mean 0.0407** (the value rule 3A.5 uses), p90 0.1145,
max 0.5103 — heavily right-skewed, which is why the mean and not the median
bands the tail.

`geometric_frac` across the three seeds moves ≤ 0.05 in 13 of the 16 cells. The
exceptions are conv2 on champYb: hawk [0.64, 0.71], hen [0.80, 0.88], and
**tiger [0.39, 0.65] — which spans the gate**.

### 3.2 Robustness across conditions — the top-3 panel

24 cells (18 panel + 6 hen), each with three distinct recipes:

**23 of 24 cells agree on their gate verdict.** Median within-cell
`geometric_frac` range **0.02**; 16 cells at ≤ 0.06.

The single disagreement is **Yb / s4.conv2 / tiger** — 0.87 / 0.65 / 0.22 across
K04 / K06 / K07. That is the same cell the seed grid independently flags, and
the same cell that supplies all 6 of the runs whose band straddles 0.50. Three
instruments, three axes (recipe, seed, band), one cell.

### 3.3 What the band is and is not

`geometric_frac_worst_lo/hi` resolves every in-play concept the same way at
once. It is a **union bound, not a confidence interval**: measured 3–10× wider
than the cross-seed range, and vacuous at small n (tiger has 23 concepts, so
one concept is 0.043 of the fraction). Its one legitimate use is
`gate_verdict_is_stable`. Quote the **counts**
(`n_confident_spread / _captured / _absent / n_undecided`), and where seeds
exist quote the **measured** flip rate instead of the band.

---

## 4. The gate [DIRECT]

**114 cells · 100 % control coverage · 0 provisional · rule 3A.5**

| outcome | cells |
|---|---:|
| **3C-proceeds** | **88** |
| 3C-deprioritized **(captured)** | 23 |
| 3C-deprioritized **(absent)** | 3 |
| band straddles 0.50 (verdict not quotable) | 6 |

Concept level: **9,065 spread (74.6 %) · 2,265 captured (18.6 %) · 821 absent
(6.8 %)**; 11.5 % `undecided`. Unsupervised only (excluding the 4 anchored
controls): 8,952 / 2,237 / 817 of 12,006 — identical to one decimal.

### 4.1 G-3A PASSES, and it passes where it matters

The 3C target is agent-relative threat structure on the strongest champion at
the hook that carries it. **champYb / `s4.fc1` / tiger, every dictionary on
disk:**

| run | geom | spread/cap/abs | undecided | `solo_frac` | `top_phi` | R² | rand R² |
|---|---:|---|---:|---:|---:|---:|---:|
| F02 topk-k64-exp8 | 1.00 | 23/0/0 | 1 | 0.18 | 0.229 | 0.505 | 0.036 |
| K03 topk-k32-exp8 s42/s43/s44 | 1.00 | 23/0/0 | **0** | 0.17 | 0.220 | 0.525 | 0.036 |
| K05 jumprelu-t64-exp8 s42/s43/s44 | 0.96 | 22/1/0 | **0** | 0.19 | 0.225 | 0.524 | 0.036 |
| K11 jumprelu-t64-exp32 | 0.96 | 22/1/0 | **0** | 0.19 | 0.242 | 0.519 | 0.036 |
| K12 jumprelu-t64-exp64 | 0.96 | 22/1/0 | **0** | 0.20 | 0.227 | 0.525 | 0.036 |
| *I04 anchored (supervised control)* | *0.30* | *7/16/0* | *4* | *0.76* | *0.771* | *0.534* | *0.036* |

Nine unsupervised dictionaries — three architectures, four expansions, three
seeds — all at `geometric_frac` 0.96–1.00 with **zero undecided concepts**, on a
target whose information content is **14.6× the learned-signal floor**. This is
the least ambiguous result in the phase.

### 4.2 The calibration check passes — the `captured` arm demonstrably fires

The failure that voided the 2026-07-27 reading was that `captured` fired once in
488 verdicts. It now fires **2,265 times in 12,151**, and specifically:

- **I04-champYb / tigerYb**, the supervised positive control on the champion
  where anchoring actually worked: **16 of 23 captured**, `solo_frac` 0.76,
  `intrinsic_dim` 1.01, `top_phi` 0.771 — against the same dictionary family's
  unsupervised 0/23. The rule separates supervised concentration from
  unsupervised spread on the same champion, hook and rows.
- **champYb / fc1 / gorilla**: 85.5 % captured (`solo_frac` 0.979, `knee_k` 1).
  An entire basis comes out `captured` from an unsupervised dictionary.
- The synthetic planted-concept controls in `tests/test_dilution.py`
  (`captured_single_latent`, `absent_independent`, `disjoint_latents_is_geometric`)
  pin all three arms on data with known ground truth.

**Caveat to state whenever the positive control is cited:** I04 on champTa and
champVe comes out 23/23 `spread`. That is not an instrument failure — anchoring
demonstrably did *not* concentrate tiger conjunctions on those champions
(banked I04 line 0.26 / square 0.41 on Ta). But it means the anchored control
validates the rule's **sensitivity** only where the supervision took. The
instrument's specificity rests on the synthetic controls, not on I04.

### 4.3 The one cell that does not determine its gate

**champYb / s4.conv2 / tiger.** Per-concept, the real R² is 0.05–0.07 against a
random-model floor of 0.031–0.046 — a gap of 0.015–0.03 sitting on top of
`random_margin = 0.02`. Which side of the margin a concept lands on is then a
coin flip:

| run | geom | spread/cap/abs | gate |
|---|---:|---|---|
| K04 batchtopk-k32 s42 / s43 / s44 | 1.00 / 0.87 / 0.87 | 23/0/0, 20/0/3, 20/0/3 | proceeds (s43, s44 straddle) |
| K06 topk-k32 s42 / s43 / s44 | **0.39 / 0.65 / 0.39** | 9/0/14, 15/0/8, 9/0/14 | **absent / proceeds / absent** |
| K07 topk-k16 s42 | 0.22 | 5/0/18 | absent (straddles) |

**Do not quote a gate verdict for this cell.** The quotable statement is the
underlying measurement, which all seven runs agree on: *champYb's conv2
dictionary carries agent-relative tiger conjunctions at 1.6× the
random-model floor, `solo_frac` 0.10–0.34, `top_phi` ≈ 0.10.* Whether that is
labelled `absent` or `spread` is a threshold artefact; either way the
information is not usefully present. Contrast fc1 on the same champion and the
same concepts at **14.6×**.

---

## 5. The result: the wall is a DISJUNCTION wall [DIRECT]

This is the phase's finding, and it is new. It confirms — on data already on
disk, without the new BSP set that entry proposed — the falsifiable prediction
made in [`2026-08-17_3A-gate-readable.md`](2026-08-17_3A-gate-readable.md) §3.

### 5.1 The controlled comparison

Every predecessor compared *bases*. A basis confounds three things at once
(logical form, agent-relativity, prevalence). The comparison that isolates one
of them is **within a single basis**, because hawk contains both shapes and is
never agent-relative:

| basis | category | form | agent-relative? | n | **captured** | `solo_frac` | `knee_k` | R² |
|---|---|---|---|---:|---:|---:|---:|---:|
| gorilla | `threat_line`, `threat_square_2x2` | pinned | no | 608 | **85.5 %** | 0.974 | **1** | 0.763 |
| hawk | `reframed_count`, `reframed_completable` | pinned | count no, completable **yes** | 1064 | **80.1 %** | 0.928 | **1** | 0.530 |
| hen | `neg_count`, `neg_completable` | pinned | same | 1064 | **59.4 %** | 0.889 | 2 | 0.482 |
| **hawk** | **`reframed_any_threat`, `…sq_any_threat`** | **OR over 4 attrs** | **no** | **133** | **0.0 %** | **0.311** | **4** | **0.741** |
| **hen** | **`neg_any_threat`, `…sq_any_threat`** | **OR over 4 attrs** | **no** | **133** | **0.0 %** | **0.320** | **4** | **0.621** |
| **tiger** | **`*_winnable`, `offered_completing_attr`** | **OR over 8 poles** | **yes** | **207** | **2.4 %** | **0.191** | **12** | **0.523** |

champYb, `s4.fc1`, every unsupervised dictionary on disk (9 checkpoints:
F02, K03 s42/s43/s44, K05 s42/s43/s44, K11, K12 — 7–9 per basis), 3,209
concept-verdicts.

Reorganised by the two candidate causes:

| `family_role` | concept shape | n | % spread | % captured | med `solo_frac` | med `knee_k` | med R² |
|---|---|---:|---:|---:|---:|---:|---:|
| `state` | pinned | 1 672 | 20.8 % | **76.7 %** | 0.921 | 1 | 0.555 |
| `agent_relative` | pinned | 1 064 | 29.9 % | **67.8 %** | 0.955 | 1 | 0.484 |
| `state_any` | OR over attrs | 266 | **100.0 %** | **0.0 %** | 0.316 | 4 | **0.680** |
| `agent_relative` | OR over attrs | 207 | **97.6 %** | 2.4 % | 0.191 | 12 | 0.523 |

- **Agent-relativity costs ~9 pp of capture** (76.7 % → 67.8 %). `reframed_completable`
  is agent-relative — it reads the offered piece — is stamped `agent_relative`
  in `CONCEPT_FAMILIES`, has base rate **0.0029**, and is still 80 % captured.
- **Disjunction costs ~70 pp** (76.7 % / 67.8 % → 0.0 % / 2.4 %).
- The disjunctive concepts are **not** information-poor. `state_any` has the
  **highest** recoverable information of any group (R² 0.680). The dictionary
  knows the answer; it does not have an atom for it.
- Prevalence is not the driver and runs the wrong way: the *most* captured group
  (`agent_relative`, pinned, 0.0029) is the **rarest**, and the least captured
  (`state_any`, 0.0316) is the most common. Restricting both arms to base rate
  ∈ [0.015, 0.050] leaves the effect intact — champYb 26.0 % vs 95.8 % spread,
  `solo_frac` 0.903 vs 0.302.

### 5.2 The mechanism, read directly off `knee_k`

`knee_k` is normally not readable alone — noise creep in the R² tail pushes the
crossing right. Here it is not being thresholded; its **distribution** is being
compared against a disjunct count known *by construction*:

| concept shape | disjuncts by construction | n | mode `knee_k` | median | share at ≤ the predicted count |
|---|---:|---:|---:|---:|---:|
| pinned | **1** | 1 168 | **1** | 1 | 60.4 % |
| `*_any_threat` (hawk + hen) | **4** | 266 | **4** | 4 | 57.5 % (≤ 5: 79.3 %) |
| tiger `*_winnable` | **8** (`tiger == OR(hawk ∪ hen)`, verified with zero violations, 2026-08-21) | 171 | **9** | 11 | mode at 9 vs predicted 8 |

**The dictionary reconstructs a 4-way disjunction from exactly its four
disjuncts.** That is a direct observation of the tiling mechanism H10 predicts,
not an inference from a threshold. The tiger prediction — 8 disjuncts, because
`hen` established that tiger's winnable spans both attribute poles — comes out
at mode 9 / median 11.

**Precondition, stated so this is not over-read.** The signature appears
**only** on champYb / fc1, the one (champion, hook) where the dictionary has
resolved anything into atoms at all (pinned `knee_k` mode = 1). Everywhere else
pinned concepts already need 6–30 latents, so there is no atom structure for a
disjunction to be built out of, and `knee_k` is 15–44 and unstructured. The
finding is conditional on atoms existing — and its precondition is exactly what
conv2, champTa and champVe lack.

### 5.3 Champion and hook, for context

Top-3 panel, unsupervised, pooled over the three conditions per cell:

| ch | hook | basis | n | % spread | % captured | % absent | med `solo_frac` | med R² | med rand R² | **R²/rand** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ta | conv2 | gorilla / hawk / hen / tiger | 228/513/513/69 | 100 / 97.7 / 95.5 / 100 | ~0 | 0–4.5 | 0.23–0.35 | 0.08–0.14 | 0.03–0.05 | 1.9–3.2× |
| Ta | fc1 | gorilla / hawk / hen / tiger | 228/513/513/69 | 90.8 / 99.2 / 97.7 / 100 | ~0 | 0–9.2 | 0.28–0.44 | 0.14–0.21 | 0.03–0.04 | 4.1–6.0× |
| Ve | conv2 | " | 228/513/513/69 | 95.9–100 | 0 | 0–4.1 | 0.22–0.25 | 0.08–0.13 | 0.03–0.05 | 1.9–3.5× |
| Ve | fc1 | gorilla / hawk / hen / tiger | 228/513/513/69 | 71.5 / 80.3 / 81.1 / 94.2 | 5.8–19.3 | 0–9.2 | 0.30–0.61 | 0.25–0.28 | 0.03–0.04 | 6.7–10.0× |
| Yb | conv2 | gorilla / hawk / hen / tiger | 228/513/513/69 | 100 / 77.2 / 89.1 / 58.0 | ~0 | 0 / 22.8 / 10.7 / **42.0** | 0.13–0.16 | 0.06–0.13 | 0.03–0.05 | **1.6–2.7×** |
| **Yb** | **fc1** | **gorilla / hawk / hen / tiger** | 228/513/513/69 | **11.4 / 27.3 / 45.8 / 97.1** | **85.5 / 70.0 / 52.8 / 2.9** | 0–3.1 | 0.19–0.98 | 0.49–0.77 | 0.04 | **12.1–19.5×** |

- **conv2 is a floor for every champion** (1.6–3.5× the random-model control,
  `solo_frac` 0.13–0.35, essentially nothing captured). The 2026-08-17 reading
  that the wall "is hook-independent" needs this qualification: conv2's tiger
  verdict is 1.00 because conv2 carries almost nothing, not because it carries
  the information unfactored.
- **fc1's information content grades with champion competence**: 4–6× (Ta) →
  7–10× (Ve) → **12–20×** (Yb) over the same floor. The strongest champion's
  bottleneck carries the most threat information — and factors the most of it
  into atoms.
- **The wall is what is left over after that**: the one thing champYb's fc1
  dictionary does *not* factor is the disjunction.

### 5.4 Polarity: real, small, and mostly a threshold effect

`hen` mirrors hawk category-for-category on the four negative attribute poles.
Paired by mirror BSP (same location, same category shape, opposite pole), 0
unmatched:

| ch | hook | pairs | hawk captured | hen captured | med Δ`solo_frac` | med ΔR² | med Δbase rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Yb | fc1 | 1 197 | **71.2 %** | **52.8 %** | +0.006 | **+0.047** | +0.0004 |
| Yb | conv2 | 1 368 | 0.0 % | 0.1 % | −0.017 | −0.004 | +0.0004 |
| Ve | fc1 | 513 | 17.5 % | 16.4 % | +0.009 | +0.009 | +0.0001 |
| Ta | fc1 | 513 | 0.0 % | 0.2 % | −0.006 | +0.010 | +0.0002 |

`[AI-REASONED PROVISIONAL ANALYSIS]` On champYb fc1 the dictionary carries ~9 %
more information about positive-pole threats (median ΔR² +0.047) — real, and
matched on base rate to 4 × 10⁻⁴. But the concentration shift is tiny
(median Δ`solo_frac` +0.006; hawk beats hen on only 57.3 % of pairs), so **the
71 % → 53 % capture gap is that small shift crossing the 0.70 threshold, not a
large representational difference.** Do not headline it as "positive poles are
privileged". One candidate explanation is ruled out: the trunk takes a
**16-channel one-hot over pieces**, not 4 attribute channels, so both poles are
equally an OR over 8 piece channels — the asymmetry is not an input-encoding
artefact. It is unexplained, small, and champYb/fc1-specific.

`hen`'s real contribution to this phase is §5.2: it is what makes tiger's
disjunct count **8** rather than 4, and therefore what turns the `knee_k`
distribution into a quantitative prediction.

---

## 6. Correction to a standing interpretation

**Superseded:** "The residual wall is specific to **agent-relative** (tiger)
conjunctions on the strongest champion, unsupervised."
([`2026-07-27_3A-dilution-results.md`](2026-07-27_3A-dilution-results.md),
carried into `RESEARCH-STATUS.md`.)

**Replaced by:** the residual wall is specific to concepts that are a
**disjunction over attribute poles** (or over locations). Evidence: hawk's own
`reframed_any_threat` is not agent-relative, and is 100 % spread with
`solo_frac` 0.311 on the same dictionary where hawk's agent-relative
`reframed_completable` is 80 % captured. tiger is spread **because every tiger
threat BSP is a disjunction**, not because it is agent-relative — agent-relativity
costs 9 pp, the disjunction costs 70 pp.

The 2026-08-17 entry already reasoned its way to "a disjunction over attributes
of a joint board × piece condition" from the BSP definitions and set the
prediction that a per-attribute decomposition of tiger should be captured like
hawk. That prediction is now **confirmed from the other direction**: hawk's own
attribute-quantified category behaves like tiger, and hen's does too. The
proposed new BSP set is no longer needed to test it.

`[AI-REASONED PROVISIONAL ANALYSIS]` What this changes downstream: 3C's target
is a readout that can **aggregate a known small set of sibling atoms**, not one
that can represent agent-relativity. The atoms already exist and are clean
(`knee_k` = 1, `solo_frac` ≈ 0.93 for the pinned concepts); what is missing is
an OR over them. That is a much more specific brief than "geometry-aware SAE",
and it favours the hierarchical/aggregating arm (Matryoshka, H-SAE, MP-SAE) over
the manifold/bilinear arm.

---

## 7. What 3A establishes — and what it does not

**Establishes:**

1. **G-3A passes.** 88 / 114 cells `3C-proceeds`; 74.6 % of 12,151
   concept-verdicts `spread`; 0 provisional; 100 % random-control coverage. On
   the 3C target cell (champYb / fc1 / tiger) nine unsupervised dictionaries
   agree at 0.96–1.00 with **zero undecided concepts**.
2. **The gate is a real test.** Both other arms fire — `captured` 2,265 times
   (incl. the supervised positive control at 16/23 and an entire unsupervised
   basis at 85.5 %), `absent` 821 times.
3. **Capacity is not the binding constraint** — now a sixth independent line:
   doubling the dictionary K11 exp32 → K12 exp64 (16,384 → 32,768 slots) moves
   `geometric_frac` by **exactly 0.00** on both bases it was measured on.
4. **The mechanism is tiling by disjunct** (§5.2), observed directly.
5. **The diagnostic's headline number is honest** (§2.1) — selection buys zero
   held-out R², so the learned-signal floor measures a real architectural prior.

**Does not establish:**

1. **Which geometry, in the `diluted` vs `tiled` sense.** Formally still
   DECLINED (2026-08-24). §5.2 is a partial answer for the *disjunctive*
   concepts specifically, and it points at tiling — but it is a distribution
   comparison on one (champion, hook), not the validated planted-control
   statistic the decline asked for.
2. **Causality.** Every number here is decodability. G11 stands untouched;
   3B-causal is what tests it.
3. **champTa / champVe conformance.** 12 of 18 cells contain a member trained
   without its architecture's own specified machinery. All 12 agree on their
   verdict, so no conclusion changes; what is lost is strength of evidence.
   DECLINED 2026-08-24 (needs ~20 h retraining, not re-selection).
4. **A gate verdict for champYb / conv2 / tiger** (§4.3).
5. **A power-flagged provisional marker** for the 45.9 % of concept-instances
   below `min_positives` (§2.3). Direction measured and conservative for the
   gate; not implemented as a flag.

---

## 8. Reproducing

```bash
# verdicts and bands are pure functions of the stored metrics -- no _h caches
python scripts/dilution_diagnostic.py reclassify saes/quarto/analysis/*_dilution-*.json --dry-run
python scripts/summarize_3a_gate.py                 # -> 3A_gate_summary.json + the table
python scripts/verdict_stability.py                 # -> measured flip rates + solo_frac_seed_sd
python -m pytest tests/test_dilution.py -q

# a full re-measurement (needs the _h caches; ~16 h)
pwsh -File runners\3A-dilution.ps1 -Panel -WithHen -DryRun
pwsh -File runners\launch.ps1 3A-dilution -Panel -WithHen
```

Artefacts: `saes/quarto/analysis/{run_id}_dilution-{bsp_set}.json` (114 live,
16 in `superseded/`), `3A_gate_summary.json`, `3A_panel.json`,
`3A_panel_runlist.json`, `3A_verdict_stability.json`.

## 9. Handoff

- **3C proceeds**, with the brief sharpened by §5/§6: an **aggregating readout
  over existing sibling atoms**, targeted at disjunctive concepts, on champYb /
  `s4.fc1`. G-3C is unchanged.
- **3B-causal runs first** (G11). §5 makes the causal question sharper: are the
  clean pinned atoms the ones the policy uses, or is the disjunction computed
  somewhere the dictionary never sees?
- **Open, carried forward:** the `diluted`/`tiled` statistic with a planted
  control; champTa/Ve conformance; a power flag for under-powered concepts; the
  `instrument/metrics` track (effect-size floor, `mcc_at_pref` guard,
  cross-fitting, stale caches, `per_family` backfill).
