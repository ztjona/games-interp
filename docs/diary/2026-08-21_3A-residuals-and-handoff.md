# 3A residuals measured; bimodality retracted; hen specified (2026-08-21)

Status: frozen record + **handoff for the next session**. Parent ledger:
[`phase-3A.md`](phase-3A.md). Metric definitions: [`../methods-reference.md`](../methods-reference.md).
Numbers are `[DIRECT]`; interpretation is `[AI-REASONED PROVISIONAL ANALYSIS]`.

Closes the three residuals left open by
[`2026-08-17_3A-gate-readable.md`](2026-08-17_3A-gate-readable.md) §5, and
records two corrections plus one new BSP basis to build.

## 0. Terminology settled

| term | means |
|---|---|
| **report** | one `analysis/{run_id}_dilution-{bsp_set}.json` file |
| **panel entry** | the `(run_id, bsp_set)` pair a report covers |
| **cell** | RESERVED for the 16 board squares (`cell_occupancy`, `cell_attribute`). Earlier entries misuse "cell" for a panel entry. |

**Every table states its metric and units.** `solo_frac` is dimensionless 0-1:
the share of a concept's recoverable held-out R2 carried by its single
best-associated latent.

## 1. Residual (a): the top-3 panel - SELECTED, NOT YET RUN [DIRECT]

`scripts/select_3a_panel.py` -> `analysis/3A_panel.json`. **33 conditions.**
Two design decisions, both learned from earlier errors:

1. **Ranked on threat-family MCC, not whole-basis `coverage_mcc`.** 3A only
   diagnoses threat categories; a gorilla whole-basis mean is 39%
   `cell_attribute`. Ranking by family picks materially different runs --
   `Ve/conv2/gorilla` goes E01/E05/E02 -> E06/E02/E03.
2. **Deduplicated by CONDITION, not run.** Seeds of one recipe are not three
   panel members; otherwise a well-replicated condition monopolises the panel
   and the robustness question goes unanswered.

Health-gated on `alive >= top_k` (`scripts/check_sae_usable.py`).
**All 15 missing `_h` encodes are DONE** - the run is CPU-only from here.

## 2. Residual (b): verdict stability [DIRECT]

`analysis/3A_verdict_stability.json`. Metrics: `geometric_frac` (dimensionless
0-1) and per-concept verdict agreement, over seeds 42/43/44.

| condition | `geometric_frac` per seed | range | verdict flips | flip type |
|---|---|---:|---|---|
| K03 TopK fc1 / gorillaYb | 0.16 / 0.13 / 0.18 | 0.05 | **9/76 = 11.8%** | `captured <-> spread` |
| K04 BatchTopK conv2 / tigerYb | 1.00 / 0.87 / 0.87 | 0.13 | **4/23 = 17.4%** | `absent <-> spread` |

**Per-concept verdicts flip 12-17% on seed alone**, against coverage-metric
seed sd of 0.001-0.006 - thresholding amplifies seed noise ~10x. The two
conditions flip at *different* boundaries, so there is no single fragile
threshold to tighten.

Consequences: the aggregate G-3A tally (15 proceed / 3 captured / 0 absent) is
robust; **near-boundary entries are not** (`I04-champYb tigerYb` at 0.30 and
`champVe gorilla line_threat` at 0.65 are each within one seed-range of the
0.50 gate); and **single-seed per-concept verdicts must not be quoted** -
including "68/76 captured", which carries a +/-9 band.

## 3. Residual (c): `dead_features_pct` is wrong for TopK [DIRECT]

`scripts/dead_window_audit.py` -> `analysis/dead_window_audit.json`. Measured
on cached codes; no retraining. Units: % of `d_dict`.

| run | W=1 (reported) | W=50 | never fires | artefact |
|---|---:|---:|---:|---:|
| F04 JumpReLU fc1 | 93.5 | 93.1 | 93.1 | 0.4 pp |
| **K03 TopK fc1** | **85.2** | 56.8 | **54.5** | **30.7 pp** |
| K04 BatchTopK conv2 | 95.8 | 95.7 | 95.7 | 0.1 pp |

TopK forces exactly `k` winners per row, so a rare-but-real latent loses the
competition and reads dead; threshold architectures do not have this failure
mode. **K03 has 1,864 live latents, not 606 - TopK dictionaries ARE
overcomplete (3.64x); JumpReLU and BatchTopK are not.** So
`dead_features_pct` is NOT comparable across architectures, and the
`check_sae_usable.py` health gate under-counts alive latents for TopK
(conservative, therefore safe, but wrong).

## 4. RETRACTION: `solo_frac` is not bimodal [DIRECT]

Earlier this session I claimed the 0.70 `captured` threshold "sits in a valley,
so it is discovered rather than imposed". **That is wrong**, on two levels:

- The pooled histogram's apparent bimodality is a **mixture artefact**. Per
  basis, champYb/hawk pools `reframed_completable` (median `solo_frac` 0.97)
  with `reframed_any_threat` (0.32).
- At CATEGORY level there is still no two-cluster structure: of 69 category
  points, **13 have median > 0.8, 36 < 0.4, and 20 sit in between** - a
  continuum. Within-category sd is tight (median 0.090), so categories are
  internally coherent; their medians are simply spread across the range.

**The threshold discretises a continuum; it does not discover a gap.** That
makes the uncertainty band *required*, not optional. Tri-state at +/-3 seed sd
(sd = 0.0110): 61.0% confidently spread, 35.7% confidently captured, **3.4%
undecided** - but concentrated: 0% in several entries and 21.7% in
`I04-champYb tigerYb`.

## 5. The framing ladder [DIRECT]

Median-of-category-median `solo_frac` (dimensionless 0-1), pooled over all
reports, within each concept family:

| concept family | gorilla | hawk | tiger | spread |
|---|---:|---:|---:|---:|
| `line_threat` | 0.62 | 0.41 | 0.25 | 0.37 |
| `square_threat` | 0.64 | 0.53 | 0.23 | 0.41 |

Monotone `gorilla > hawk > tiger` in **both** families - matching the
logical-form ladder (board-only > attribute-indexed > attribute-quantified).
But ranges overlap heavily (gorilla [0.13, 0.95], tiger [0.09, 0.70]): an
ordering of central tendency, **not** separated clusters.

Figures (PDF+PNG in `figures/`, regenerate with
`scripts/plot_geometry_distributions.py`): `fig_solo_frac_by_category` and
`fig_family_ordering`. The latter's y axis is **categorical** (family x basis);
vertical scatter within a row is jitter for legibility and carries no meaning.

## 6. `hen` - SPECIFIED, NOT BUILT

**Measured identity [DIRECT]**, champYb, 10 lines, 296,045 positions:

| | tiger true | OR(hawk) true | tiger AND NOT OR(hawk) | OR(hawk) AND NOT tiger |
|---|---:|---:|---:|---:|
| total | 68,041 | 35,633 | **32,408** | **0** |

`hawk => tiger` with **zero** counterexamples, and **47.6% of tiger's positives
are wins hawk cannot express**. Cause: `BINARY_ATTRS` probes only the POSITIVE
pole (`completable_tall`, never `completable_little`), while tiger's
`_line_would_be_complete` tests `len(set(values)) == 1` and so fires for
all-LITTLE / all-WHITE / all-CIRCLE / all-SOLID too.

**Build `hen`** = the 4 negative poles, as a NEW basis - do NOT extend hawk,
which would silently invalidate every banked hawk number. Then
`tiger = OR(hawk UNION hen)` becomes a verifiable identity, and the polarity
hypothesis becomes testable: if tiger's uncaptured half is specifically the
negative-pole wins, `hen` will be markedly less captured than `hawk` on the
same dictionary. Labels are CPU (`compute_bsp_labels.py`); evaluation reuses
the `_h` caches the prune deliberately kept.

## 7. Cache prune [DIRECT]

`analysis/h_prune_plan.json`. Freed **475 GB** (91 -> 36 `_h` caches; 599 GB
free). Kept: the 33-condition panel, the 15 newly-encoded runs, all six
random-model controls, and all three seeds of the two stability conditions.
Everything pruned regenerates with
`sae_eval.py evaluate <ckpt> --bsps=<sets> --force`; checkpoints and registry
rows are untouched.

## 8. NEXT SESSION - start here

1. **Build `hen`** (§6). Verify `tiger == OR(hawk UNION hen)` exactly, then
   evaluate and run 3A on it.
2. **Run 3A over the 33-condition panel** (§1). ~40+ panel entries at ~5 min
   each = HOURS, so extend `runners/3A-dilution.ps1` with a `-Panel` mode
   reading `analysis/3A_panel.json`, and launch detached via
   `runners/launch.ps1`. Do NOT run it in the foreground.
3. **Implement the tri-state natively** in `dilution_diagnostic.summarize()`
   (confidently-spread / confidently-captured / undecided at +/-3 seed sd), and
   report `geometric_frac` with a seed band wherever seeds exist. Sections 2
   and 4 make this a requirement, not a nicety.
4. **Then report** - and correct "cell" -> "panel entry" across older diary
   entries while doing it.

**Standing corrections not to re-break:** never compare runs on a whole-basis
scalar (compare per concept family); `dead_features_pct` is
architecture-dependent (§3); tiger is *mover-relative*, not owner-relative -
Quarto has no owned pieces; and every table states its metric and units.
