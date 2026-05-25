# Anchored SAE sweep I/J — results

Parent plan: [2026-05-22_anchored-sae-champta-tiger.md](2026-05-22_anchored-sae-champta-tiger.md).
Registry: `saes/quarto/eval_registry.json` (keys `I0*-champTa-*:tigerTa`, `J0*-champTa-*:tigerTa`, etc.).
Anchor analysis JSON: `saes/quarto/analysis/I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1_anchor-tigerTa.json`.

---

## Sweep design

24 configs on champTa fc1, anchored against tigerTa (36 BSPs):

- **I-series** (anchored-jumprelu, t=64, exp=8): lambda_high in {0.03, 0.10, 0.30, 1.0} x 3 seeds
- **J-series** (anchored-batchtopk, k=16, exp=8): same lambda x seed grid

Unanchored baseline: F04-champTa-s42-jumprelu-t64-exp8-s4.fc1.

All 24 runs evaluated against tigerTa, gorillaTa, and hawkTa (72 evals total).

## Headline results

[DIRECT -- from `scripts/registry_query.py top --bsps=tigerTa`]

### Mean across seeds, by condition

| Condition | tigerTa lift | tigerTa MCC | gorillaTa lift | hawkTa lift | FVU |
|---|---:|---:|---:|---:|---:|
| F04 baseline (unanchored) | 0.158 | 0.333 | 0.177 | 0.172 | 0.005 |
| I anch-jrelu lh=0.03 | 0.145 | 0.288 | 0.168 | 0.172 | 0.006 |
| I anch-jrelu lh=0.10 | 0.142 | 0.276 | 0.165 | 0.174 | 0.006 |
| I anch-jrelu lh=0.30 | 0.194 | 0.378 | 0.165 | 0.182 | 0.006 |
| **I anch-jrelu lh=1.00** | **0.255** | **0.494** | 0.166 | **0.184** | 0.006 |
| J anch-btopk lh=0.03 | 0.149 | 0.314 | 0.169 | 0.154 | 0.041 |
| J anch-btopk lh=0.10 | 0.138 | 0.326 | 0.170 | 0.161 | 0.043 |
| J anch-btopk lh=0.30 | 0.148 | 0.338 | 0.150 | 0.152 | 0.044 |
| J anch-btopk lh=1.00 | 0.162 | 0.390 | 0.152 | 0.146 | 0.050 |

### Best single run: I04-lh100-s43

[DIRECT -- from `scripts/registry_query.py compare`]

| Metric | I04-lh100-s43 | F04 baseline | Delta |
|---|---:|---:|---:|
| tigerTa F1-lift | 0.257 | 0.158 | **+0.099** (+62%) |
| tigerTa MCC | 0.496 | 0.333 | **+0.163** (+49%) |
| gorillaTa F1-lift | 0.167 | 0.177 | -0.010 (-6%) |
| hawkTa F1-lift | 0.186 | 0.172 | +0.014 (+8%) |
| FVU | 0.006 | 0.005 | +0.001 |
| Dead features | 87.9% | 87.6% | +0.3% |

## Decision gate evaluation

Referencing the pre-registered gates from the [plan](2026-05-22_anchored-sae-champta-tiger.md):

| Gate | Threshold | Result | Verdict |
|---|---|---|---|
| High-tier F1-lift >= 0.25 | 0.25 on 23 high-tier BSPs | **0.257 overall, high-tier mean lift = 0.315** | **PASS -- lock the recipe** |
| Unsupervised coverage drop < 0.05 | gorillaTa drop < 0.05 | gorillaTa drop = 0.010 | **PASS -- no structural damage** |

The first gate fires: anchored loss recovers >= 60% of the LP gap on the rare concepts. Per the pre-registered rule, the recipe is locked and transfer experiments (conv2/Ta, S4) are next.

## Anchor slot analysis

[DIRECT -- from `scripts/anchor_analysis.py`, output at `saes/quarto/analysis/I04-*_anchor-tigerTa.json`]

### Summary

- **36/36 anchor slots alive** (vs 19.6% alive rate in free slots)
- **25/36 anchor slots are the best feature for their BSP**
- Anchor mean firing rate: 16.9% vs 1.5% for free features (11x)
- 12/36 polysemantic, but semantically coherent (correlated threat concepts)
- 3/36 BSPs have a free feature with higher F1 than the anchor (all marginal except diag_main)

### Per-category breakdown

[DIRECT -- from `scripts/anchor_analysis.py`]

| Category | Tier | n | Alive | Best | Mean F1 | Mean MCC |
|---|---|---:|---:|---:|---:|---:|
| tiger_offered_completing_attr | high | 4 | 4 | 4 | **0.590** | **0.578** |
| tiger_square_winnable | high | 9 | 9 | 9 | **0.405** | **0.410** |
| tiger_line_winnable | high | 10 | 10 | 6 | 0.220 | 0.228 |
| tiger_pool_safe_count | med | 4 | 4 | 3 | 0.867 | 0.696 |
| tiger_pool_winning_count | med | 4 | 4 | 1 | 0.817 | 0.687 |
| tiger_decision_global | med | 5 | 5 | 2 | 0.659 | 0.584 |

### Cross-BSP coverage

Anchor features (trained on tigerTa) incidentally cover:
- **53.7% of gorillaTa BSPs** (88/164) at F1 > 0.3
- **1.2% of hawkTa BSPs** (2/173) at F1 > 0.3

[AI-REASONED PROVISIONAL ANALYSIS]

The high gorillaTa cross-coverage is expected: gorilla includes cell-level occupancy and attribute concepts that correlate with tiger's threat predicates. The near-zero hawkTa cross-coverage is surprising given that hawkTa F1-lift *improved* with anchoring -- the improvement must come from free (non-anchor) features being better allocated when anchor slots handle tiger, not from anchor slots directly encoding hawk concepts.

### Failure mode: diagonal lines

[DIRECT -- from per-slot analysis]

`tiger_line_diag_main_winnable` (slot 17): anchor F1 = 0.040, free feat 537 F1 = 0.223.
`tiger_line_diag_anti_winnable` (slot 18): anchor F1 = 0.024, free feat via slot 31 F1 = 0.129.

These are the only genuine anchor failures. Both diagonal BSPs have very low base rates (~1.3%) and require spatial reasoning that the fc1 128-dim bottleneck cannot represent well -- consistent with Phase 1F finding that line-level threat info is mostly lost by fc1.

## Anchored-batchtopk (J-series): negative result

[AI-REASONED PROVISIONAL ANALYSIS]

J-series shows weak, noisy gains on tigerTa (best J lh=1.0: 0.162 lift vs baseline 0.158) and **hurts** gorillaTa and hawkTa at higher lambdas. The rigid top-k constraint fights with anchor pressure: BatchTopK allocates exactly k features per sample, so forcing 36 anchor slots to fire on their BSP targets means fewer free slots are available, degrading unsupervised coverage. JumpReLU's threshold-based activation is more flexible -- features fire whenever they exceed the threshold, so anchor slots can fire *in addition to* whatever the free slots learn.

Recommendation: **do not use anchored-batchtopk** for future guided SAE work.

## Low-lambda counterproductivity

[AI-REASONED PROVISIONAL ANALYSIS]

I-series lh=0.03 and lh=0.10 perform *below* the unanchored baseline on tigerTa (0.145, 0.142 vs 0.158). The anchor loss is too weak to steer features onto their targets but strong enough to perturb the optimization landscape. There is a threshold effect: lambda must be >= 0.30 for the anchor signal to dominate the SAE's natural feature allocation. Below that, the anchor adds noise.

## Infrastructure notes

Three bugs in `run_sweep.py` and `sae_eval.py` were found and fixed during this sweep:

1. `get_output_path()` didn't handle anchored architecture naming (missing `t{l0_target}` / `k{k}`) -- eval was silently skipped.
2. `--skip-existing` + `--eval` skipped both training and eval instead of just training.
3. Parallel registry writes (3 GPUs) corrupted `eval_registry.json` -- fixed with `msvcrt.locking`.

## Reporting standard for anchored SAE results

Every anchored SAE experiment should produce:

1. **Sweep table** (mean across seeds by condition) against the anchor BSP set + at least one cross-BSP set.
2. **Decision gate evaluation** against pre-registered thresholds.
3. **Anchor slot analysis JSON** via `scripts/anchor_analysis.py` covering: per-slot alive/F1/P/R/MCC, per-category summary, polysemanticity, cross-BSP coverage.
4. **Failure mode identification**: which anchor slots failed (F1 < 0.1) and why.
