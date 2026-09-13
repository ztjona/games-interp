# Phase 3B-causal — amendment 2, PRE-DATA: Wave-1 implementation decisions (2026-09-12)

**Status: pre-data.** Written while building Wave 1, **before any interchange
intervention was run on champYb**: no IIA value for champYb existed when any
decision below was made. The only runs made so far touch no champYb outcome:
the generator's label check and pair counts (no model), the Tier-A controls, and
a pipeline smoke run on champYb's **untrained twin** (§B8). Frozen from the
commit that adds it.

Amends: [`2026-09-12_3B-causal-preregistration.md`](2026-09-12_3B-causal-preregistration.md),
as already amended by [`2026-09-12_3B-causal-amendment-1.md`](2026-09-12_3B-causal-amendment-1.md).

**Why.** The pre-registration fixes the design; building it forced decisions the
design does not fix. Each one below changes what a number means, so it is fixed
here, before the numbers exist, and not in code comments afterwards.

**Freeze verification.** Every Wave-1 artefact embeds the SHA-256 of the
pre-registration and of every amendment, read at run start before any score is
computed. The results entry checks those hashes against the committed files, so
an edit made after results exist would be detectable.

## Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| **cap** | pairs | maximum pairs per (concept, pair kind), sampled deterministically |
| **set-valued concept** | — | `tiger_offered_completes_<attr>` and `tiger_win_now_exists`: the target is a set of cells, not one cell |
| **fold** | 1–5 | cross-fitting split, grouped by board orbit |

All other terms: the pre-registration's §0 and amendment 1.

## B1. The generator and its guard

Pairs come from `scripts/games/quarto_counterfactuals.py`. Concepts are
recomputed in vectorised form for every (board, piece) combination.
**Guard**: on the natural positions, the vectorised value of every Wave-1
concept must equal its stored BSP label **exactly**, or the run aborts. Measured
at design time: **0 mismatches over 176 concepts × 296,045 positions**
(`positions-amalgam_yb_unique.pt`). Wave 1 is 176 concepts: hawk 76 +
hen 76 pinned `*_completable`, tiger 19 `*_winnable` + 4
`tiger_offered_completes_<attr>` + `tiger_win_now_exists`.

## B2. Pair sampling

- **Cap: 1,000 pairs** per (concept, pair kind), a uniform sample with a seed
  derived from `(bsp_id, kind)` — so a concept's sample never depends on which
  other concepts ran. At n = 1,000, the standard error of an IIA is at most
  0.016, against the pre-registered minimum of n = 100.
- **Specificity pre-cap.** Specificity pairs number 27k–49k per concept; a
  deterministic pre-sample of 200,000 bounds memory before the cap applies.
  (Every concept is below it, so it never binds today.)
- **Switch-off**: candidates are filtered first (the base network must play the
  expected cell, pre-registration §5.3), then capped.

Design-time counts (before caps, switch-off before the network filter): switch-on
min **586** pairs / 190 boards (`square_1_1_completable_circle`), median 1,849;
switch-off candidates min **244** / 82 boards, median 634; specificity min
**27,437**.

## B3. Set-valued concepts: targets the pre-registration did not specify

The pre-registration defines specificity for line and square concepts only. For
the two set-valued tiger concepts:

| concept | pair kind | eligible when | target set |
|---|---|---|---|
| `tiger_offered_completes_<X>` | switch-on | an X-threat exists; base piece lacks X and wins nowhere; source piece has X | cells where the **source** piece wins (as registered) |
| | switch-off | base piece has X and wins; the base network plays one of its winning cells; source piece wins nowhere | legal cells **outside** the base piece's winning set |
| | specificity | an X-threat exists; base and source pieces both win nowhere | cells completing an **X-threat** |
| `tiger_win_now_exists` | switch-on | base piece wins nowhere; source piece wins somewhere | cells where the source piece wins (as registered) |
| | switch-off | base piece wins; the base network plays a winning cell; source piece wins nowhere | legal cells outside the base piece's winning set |
| | specificity | some threat exists; both pieces win nowhere | cells completing **any** threat |

The specificity target asks the same question as for a line: does the patch
pull the network toward the cells the concept is *about*, in a context where the
concept is false?

**Pinned specificity is implemented literally** (the group has exactly one empty
cell; base and source pieces win nowhere). Whether that group's own pole-threat
is on the board is recorded per pair as a covariate, never used to filter.

## B4. Representations

- **R1–R3** come from the top-16 MCC list of the dictionary for C's basis. hawk
  and tiger use the committed `{run_id}_topk-*.json` exports. **hen has none**:
  `sae_eval` was never run on hen. So its list is computed with the same
  functions (`match_features_to_bsps` → `top_k_features_per_bsp`, MCC) from the
  dictionary's codes over all positions, and written as `{run_id}_topk-henYb.json`.
- **R2** uses `knee_k` from C's 3A report for the same dictionary, clipped to
  [1, 16].
- **R4** (causal selection within the top-16): in each fold, the latent with the
  highest IIA\* on the **training folds' switch-on pairs** is selected, then
  scored on the held-out fold's pairs of every kind. Scores pool over the five
  held-out folds.
- **R5**: the probe direction from `linear_probe_*_directions.pt`. A concept
  whose probe was not fitted (constant label) gets R5 `not-run`.
- **R6** (tiger only): anchored `I04-champYb-lh100-s42`, latent index = tiger
  BSP index — the checkpoint's `anchor_feature_idx` equals its `anchor_bsp_idx`
  (verified).
- **R7 (DAS-1)**: trained on the training folds' switch-on pairs — Adam,
  **400 steps, lr 0.05**, full-batch, random initialisation seeded per
  (concept, fold) — and scored on the held-out fold's pairs of every kind.
- **Folds**: 5, grouped by board orbit (`orbit_ids-amalgam_yb_unique.pt`),
  seeded per concept, shared by all pair kinds.
- **Hook values** `z` for bases and sources are recomputed by the model itself,
  not read from the stored activation file, so base and source come from exactly
  the same computation.

## B5. Nulls

- **B1** (random unit directions in `z` space, 1,000 draws) is the null for
  R5, R6 and R7, as registered.
- **B2** (random latent sets, 1,000 draws) is the null for R1–R4. Latents are
  "alive" if their firing frequency over all positions is > 0; each member is
  matched to a member of J within ±20 % firing frequency, falling back to the 8
  nearest-frequency alive latents when that band is empty. For R4, draws are
  matched to each fold's selected latent, scored on that fold's held-out pairs,
  and pooled draw-by-draw across folds.
- **B3** (for R1: the frequency-matched latent with the lowest |MCC| with C) is
  reported as a control, not used for p-values.
- **p-values** are greater-tail for IIA\* on each arm. "Anti-consistent" is
  IIA\* below the null's 5th percentile; "off-target" is the flip rate above
  its null's 95th percentile. Both are per-concept percentile tests, as
  registered, without BH.
- **BH** (q = 0.05) runs within each (representation, arm) across the powered
  concepts.
- Nulls are computed in float32, like the real representations.

## B6. The gate

Gate C1 counts R7 verdicts `concept-consistent` and `concept-consistent
(on-only)` — both are "concept-consistent" in the verdict rule's own words —
over concepts with adequate n. `install-only` is **not** counted: it is not
concept-consistent. If the gate fails, R1–R6 verdicts are written with
`readable: false` and are not reported as findings (pre-registration §7).

## B7. Replicates

F04-s42 and K05-s43/s44 get **point estimates with orbit-bootstrap CIs**, no
nulls and no verdicts: they measure stability. R1 and R3 for all three; R2 only
where that dictionary has its own 3A report for `knee_k` — K05-s43/s44 do, F04
does not, so F04 carries R1 and R3. Verdicts are read on the primary dictionary,
K05-s42, only.

## B8. Verification without seeing a champYb outcome

The full run is long (an estimate is in the runner). Before handing it over,
the whole code path was exercised end-to-end on **champYb's untrained twin**
(`E_0000`), with champYb's own dictionaries. Its numbers are meaningless by
construction and are not interpreted; the run exists to prove the pipeline runs
and to time it. **No champYb interchange number was computed before this
amendment.**

## B9. Deviation log (for the results entry)

| # | topic | registered | fixed here | results-motivated? |
|---|---|---|---|---|
| 1 | pair sample size | unspecified | cap 1,000 per (concept, kind), seeded per concept | no |
| 2 | set-valued specificity / switch-off targets | unspecified | §B3 table | no |
| 3 | hen top-16 lists | "top-K export" | computed with the export's own functions, same metric | no |
| 4 | R4 / R7 cross-fitting details | "cross-fitted, 5 folds" | §B4 | no |
| 5 | null construction details | "1,000 draws, ±20 %" | §B5, incl. R4's per-fold null | no |
| 6 | gate counting | "concept-consistent" | counts `(on-only)`, not `install-only` | no |
| 7 | replicate dictionaries | "stability check" | point estimates + CIs, no verdicts | no |
