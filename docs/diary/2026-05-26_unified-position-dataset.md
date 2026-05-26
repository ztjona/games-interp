# Unified Cross-Champion Position Dataset

**Date:** 2026-05-26

## Problem

Each champion (Aa, S4, Ta, Ve) generates its own position distribution via
self-play. SAEs trained on champion X are evaluated against BSP labels computed
on champion X's positions. This creates an evaluation bias: features that score
well on one champion's visited states may not generalize, and cross-champion
comparisons are not apples-to-apples because the underlying boards differ.

The champion suffix on BSP sets (e.g. `gorillaTa` vs `gorillaVe`) marks which
position distribution was used to compute the labels. The BSP *definitions*
(schema) are identical across champions -- only the label tensor changes.

## Design Decision

Merge all per-champion amalgam files into a single deduplicated position pool.
Evaluate all SAEs against BSP labels computed on this unified pool when
reporting cross-champion results.

### Naming convention

The unified BSP set suffix encodes the position count in thousands (e.g.
`gorilla156k`). This makes the evaluation basis immediately visible and ensures
that when the pool grows (new champion added), the suffix changes and stale
evaluations are never confused with current ones.

- Unified positions: `positions-amalgam_all_unique.pt`
- Unified BSP labels: `bsp_labels-gorilla{N}k_164.pt`, etc.
- Unified activations: `<hook>_amalgam_all_<tag>_activations.pt`

### Workflow

- **Day-to-day**: sweeps and quick evals continue on champion-specific data
  (`gorillaVe`, `gorillaTa`, etc.). No change to the existing workflow.
- **Reporting**: run `scripts/unify_positions.py` to build the unified pool,
  then re-evaluate checkpoints with `--bsps=gorilla{N}k`.
- **Incremental**: when a new champion arrives, re-run `unify_positions.py` --
  it discovers the new amalgam automatically and rebuilds the pool.

### Scope

All four champions (Aa, S4, Ta, Ve) are included. Cross-family merging is safe:
`quarto_s4` delegates BSP computation to `quarto`, tensor shapes and metadata
structures are identical.

## Implementation

New script: `scripts/unify_positions.py` (standalone CLI, docopt).

Steps:
1. Discover all `positions-amalgam*_unique.pt` in `data/quarto/` (excludes `_all_`)
2. Merge + deduplicate via `deduplicate_positions.py` logic
3. Compute BSP labels for gorilla/hawk/tiger with `{N}k` suffix
4. Collect activations for each champion model on the unified positions (4 champions x 2 hooks x 2 models = 16 activation files)

Flags: `--dry-run`, `--force`, `--skip-labels`, `--skip-activations`.
Idempotent: checks provenance to skip rebuild if sources unchanged.

No changes to `sae_eval.py` -- it already supports `--data=<path>` to override
activation resolution, and BSP labels auto-resolve via `--bsps=gorilla{N}k`.

[AI-REASONED PROVISIONAL ANALYSIS]

The bias introduced by champion-specific evaluation is likely modest for
within-champion analysis (the SAE sees the same distribution at train and eval
time), but becomes a confound when comparing SAEs across champions. The unified
pool also increases board diversity: different champions visit different game
states, so the union covers more of the game tree -- particularly important for
rare states like near-wins that weaker champions rarely reach.
