# Phase 3A: the gate becomes readable, and the wall is a *framing* effect (2026-08-17)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).
Metric definitions and the verdict rule: [`../methods-reference.md`](../methods-reference.md)
§3, §6. Numbers are `[DIRECT]` from `saes/quarto/analysis/`; interpretation is
`[AI-REASONED PROVISIONAL ANALYSIS]`.

On 2026-08-14 a review found that G-3A "passed" on a criterion an **untrained
network also passes**. This entry records what it took to make the gate a test,
what it then said, and — the part that matters scientifically — what the
concept-family rollup shows that the whole-basis number was hiding.

## 0. Glossary

| term | range | ideal | meaning |
|---|---|---|---|
| **`absent`** | — | — | The dictionary carries no more signal than a floor: the permutation null (*not real*) **or** the random-model SAE control (*real but not learned*). |
| **`captured`** | — | for a solved concept | Recovered by a small, low-dimensional set of latents. |
| **`spread`** | — | the H10 outcome | Above both floors but not concentrated. The geometric verdict. Replaces `diluted`/`tiled` at rule 3A.3. |
| **`geometric_frac`** | 0–1 | — | `spread / n`. **Gate G-3A**: ≥ 0.50 → 3C proceeds. |
| **`random_control_coverage`** | 0–1 | 1.0 | Share of concepts actually tested against the learned-signal floor. Below 1.0 the gate is `provisional`. |
| **concept family** | — | — | The cross-basis unit (`line_threat`, `square_threat`, …), read from the schema's `concept_family` stamp. **The only unit comparable across bases.** |

## 1. What had to be fixed first [DIRECT]

| defect | fix |
|---|---|
| `classify` computed `random_asymptote_r2` and never read it. An SAE trained on an **untrained** network returns `diluted` on every real gorilla threat (R² 0.025–0.034, above `absent_floor`). | **Rule 3A.3** adds the learned-signal floor. |
| The fc1 control (`R1-champ*random`) was degenerate — 4032/4096 latents never fire, the 64 survivors fire on >99.98% of rows, so **zero** are alive and no control number was produced. 13 of 17 cells named a control and were never tested. | **`R3-champ*random`**: the *same recipe* plus `W_enc = W_dec^T`. 87–190 alive. The collapse was the missing tied init, not JumpReLU. |
| `tiled` fired 42/1100 times and **every** firing was a 2-latent community with one negative edge — a Bernoulli on the sign of one partial correlation. Rescoping to the candidate list does not discriminate (Jaccard ≈ 0.10 for *captured* and *spread* alike). | `diluted` + `tiled` → **`spread`**; the bar for restoring the split is recorded in `classify`. |
| The conv2 panel member `E05-champYb` was a collapsed dictionary: 41 alive latents — **fewer than `top_k = 64`**, so its candidate list could not be filled. | Swapped to **`K04`** (canonical, 163 alive, FVU 0.0066). `E05`'s report archived under `analysis/superseded/`. |
| champVe used row-wise train/test splits while champTa/champYb used orbit-aware ones. | `orbit_ids-amalgam_ve_unique.pt` generated; all 7 cells re-run. Effect was **small** — mean orbit size is 1.049–1.051 for *all three* champions (7.9–8.1% of rows share an orbit), so the bias was minor and symmetric. |
| Nothing checked that an SAE could actually answer the question. A control that *trained* was assumed to *work*; a panel member with 41 alive latents passed silently. | **`scripts/check_sae_usable.py`** — applies the diagnostic's own alive definition to **panel members and controls alike**, and the runner throws on failure. |

## 2. The gate [DIRECT]

**18 cells · 100% control coverage · uniform orbit-aware splits · 0 provisional**

| outcome | cells |
|---|---:|
| **3C-proceeds** | **15** |
| 3C-deprioritized **(captured)** | 3 |
| 3C-deprioritized **(absent)** | **0** |

**G-3A passes; 3C proceeds.** Two things make this readable where the
2026-07-27 version was not:

1. **The other arm demonstrably fires.** In the one cell whose dictionary was
   genuinely unlearned (`E05-champYb`, collapsed), the learned-signal floor sent
   **19 of 23** concepts to `absent`. That is the calibration check the previous
   rule failed.
2. **No cell is a dead end.** Once `E05` was replaced by `K04`, the panel's only
   `absent` verdict disappeared — `K04` returns 1.00 on both bases. The three
   deprioritized cells are deprioritized for the *good* reason: the concepts are
   already captured (champYb gorilla 68/76, hawk 133/171, and the anchored
   positive control I04 tiger 16/23).

## 3. The result that only the family rollup shows [DIRECT]

`geometric_frac` per **concept family**, so the same game fact can be compared
across its three framings:

| champion | run | family | gorilla | hawk | tiger |
|---|---|---|---:|---:|---:|
| champTa | F04 fc1 | line_threat | 0.82 | 0.99 | 1.00 |
| champTa | F04 fc1 | square_threat | 1.00 | 1.00 | 1.00 |
| champVe | F04 fc1 | line_threat | 0.65 | 0.87 | 1.00 |
| champVe | F04 fc1 | square_threat | 0.78 | 0.75 | 1.00 |
| **champYb** | **F04 fc1** | **line_threat** | **0.05** | **0.21** | **1.00** |
| **champYb** | **F04 fc1** | **square_threat** | **0.00** | **0.12** | **1.00** |
| champYb | I04 fc1 (anchored) | line_threat | — | — | 0.60 |
| champYb | I04 fc1 (anchored) | square_threat | — | — | **0.11** |
| champYb | K04 conv2 | square_threat | 1.00 | — | 1.00 |

### Reading [AI-REASONED PROVISIONAL ANALYSIS]

**The residual wall is a framing effect, not a concept effect.** On champYb's
fc1 dictionary, `square_threat` is **fully captured under gorilla (0.00) and
almost fully under hawk (0.12), and fully geometric under tiger (1.00)** — the
*same 2x2-square game fact*, in the same dictionary, on the same positions.

**What separates them is LOGICAL FORM, not the offered piece.** This entry
originally said the wall was "the conjunction of a board fact with the agent's
relation to it". That is wrong, and the table refutes it: **hawk uses the
offered piece too**, and hawk is captured. Reading the definitions in
`scripts/games/quarto.py`:

| basis | example | inputs | logical form |
|---|---|---|---|
| gorilla | `row_2_threat_tall` | board only | 3 cells match `tall` AND 1 empty |
| hawk | `row_0_completable_tall` | board + offered | (threat in `tall`) AND (offered is `tall`) |
| tiger | `tiger_line_row_0_winnable` | board + offered | exactly 1 empty AND **EXISTS attr** in {size, colour, shape, hole}: all 4 (3 placed + offered in the gap) share it |

- **hawk is attribute-INDEXED.** Fix `tall`; both conjuncts are separately
  detectable, so a dictionary can spend one latent per conjunct and the
  conjunction is close to linear in them. Factorable.
- **tiger is attribute-QUANTIFIED.** An existential over four attribute-specific
  joint conditions, each depending on *which* cell is empty. There is no fixed
  attribute to key a latent on.

So the residual wall is **a disjunction over attributes of a joint
board x piece condition** -- a form that does not factor into a product of
simple detectors. Prediction this makes: a per-attribute decomposition of tiger
(`tiger_line_row_0_winnable_tall`, ...) should be CAPTURED like hawk. If it is
not, the disjunction is not the mechanism and this reading is wrong.

Three supporting observations:

- **Supervision moves it.** The anchored I04 on the same champion and hook takes
  tiger `square_threat` from 1.00 to **0.11** — so the information is present
  and extractable; the unsupervised dictionary just does not factor it that way.
- **It is champion-dependent in the state framings but not the agent-relative
  one.** gorilla `square_threat` runs 1.00 (Ta) → 0.78 (Ve) → 0.00 (Yb) along
  the competence/objective axis, while tiger stays pinned at 1.00 for all three.
- **It is hook-independent.** champYb conv2 (`K04`) is 1.00 on gorilla *and*
  tiger, consistent with the 2026-08-14 finding that conv2 carries the threat
  information but does not factor it into atoms.

## 4. HP-canonical: CLOSED [DIRECT]

The gate opened on 2026-08-16 because the swept ranges were measured on
pre-conformance recipes. Both axes now have **interior optima under canonical
recipes**, so the ranges are wide enough:

| conv2 sparsity (TopK) | k=16 | k=32 | k=48 | k=64 |
|---|---:|---:|---:|---:|
| legacy | 0.333 | 0.330 | 0.329 | 0.306 |
| **canonical** | 0.269 | **0.312** | 0.293 | 0.290 |

| fc1 expansion (JumpReLU) | exp8 | exp16 | exp32 | exp64 |
|---|---:|---:|---:|---:|
| legacy | 0.424 | 0.407 | 0.385 | 0.327 |
| **canonical** | 0.459 | 0.455 | **0.475** | 0.461 |

Legacy declined monotonically on **both** axes, implying an optimum outside the
range — which is why the gate existed. Canonical peaks **inside**: k=32 (already
the panel setting) and exp32. Reasoning from the legacy curves would have been
wrong in both cases.

**Capacity is not the binding constraint** — now five independent ways: K03
(+56 alive fc1 latents → 0.000 coverage change); K07 vs C01 (+137 alive →
−0.064); the conv2 sparsity curve (k=16 has the *most* alive latents and the
*worst* coverage); the expansion curve above; and the effective-expansion figure
— across an **8× larger dictionary** (4,096 → 32,768 slots) alive latents move
350 → 400 and effective expansion 0.68× → 0.78×, never reaching 1×. The
residual wall cannot be attacked by making dictionaries bigger.

## 5. What is NOT done

3A is **not** complete in totality. Recorded so the gate reading is not
over-quoted:

1. **Single-member panel.** 3A runs one SAE per cell. Panel choice is fragile —
   `F04` ranks 1/8 on hawk and **8/8 on tiger** among champYb's fc1 runs — so a
   top-3 panel was agreed and is not yet built.
2. **No seed replication of verdicts.** Every verdict is `s42`. The natural
   stability metric is the verdict-flip rate across seeds; none exists.
3. **`dead_window` never exercised.** All runs at 0, so how much of the 85–99%
   deadness is a per-batch measurement artefact is still unquantified.

Deferred to the `instrument/metrics` track and still open: the effect-size floor
in `sae_lp_efficiency`, the `mcc_at_pref` prevalence guard, cross-fitted feature
selection, 242 stale matching caches, metric-argmax consistency, and the
`per_family` backfill for existing registry rows.

## 6. Reproducing

```bash
pwsh -File runners/3A-dilution.ps1 -DryRun          # pre-flight, incl. the usability gate
pwsh -File runners/3A-dilution.ps1 -Only champVe    # partial re-run, per-cell
python scripts/dilution_diagnostic.py reclassify saes/quarto/analysis/*_dilution-*.json
python scripts/summarize_3a_gate.py
python scripts/check_sae_usable.py saes/quarto/<ckpt>.pt
```

Rule changes never need the multi-GB `_h` caches: the verdict is a pure function
of the stored metrics, so `reclassify` re-verdicts all 18 reports on CPU.
