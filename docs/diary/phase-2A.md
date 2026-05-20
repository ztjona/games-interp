# Phase 2A — Conv2 architecture sweep on champAa (2026-05-11)

Status: COMPLETED on Deep Brain (3× A6000) after the timeout fix. 33 runs
across Campaigns C / D / F / G; 79 total entries in `eval_registry.json`
at end of phase. The original 2026-05-04 failure was a `run_sweep.py`
timeout, not a methodological issue — the same configs re-ran cleanly
with `--timeout=86400`.

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md). Predecessor:
[`phase-1.md`](phase-1.md) (Anakin fc1 sweep, conv2 LP, competence audit).
Successor: [`phase-2B.md`](phase-2B.md).

## Headline [DIRECT — `eval_registry.json`]

**Winner:** `C01-c2arch-s42-topk-k16-exp8-conv2` — gorilla coverage = **0.353**, cov > 50 = **0.396**. First time any run cracks 0.35.

## Reporting standard adopted here (REQUIRED for every winner claim from 2026-05-11 forward)

Three rules, enforced as policy:

1. **Always report three coverage metrics side-by-side** — F1 (literature standard, base-rate-sensitive), MCC (base-rate-invariant; 0 for any constant predictor), F1-lift (best_F1 minus the trivial `2p/(1+p)` "always-positive" baseline, clipped at 0). MCC and F1-lift add ~0 cost (same TP/FP/FN/TN counts), and they collapse the trivial-feature artifact that gives `offered_piece` an apparent F1 = 0.667. The eval pipeline writes all three to `eval_registry.json`; `scripts/backfill_eval_metrics.py` retro-fills older entries.
2. **Always compare against linear-probe ceiling AND random-network control.** The "learned-gap fraction" `(coverage_SAE − coverage_rand_SAE) / (LP_trained − LP_random)` is the cleanest way to claim an SAE recovers *learned* structure rather than the architectural prior (especially important at conv2 where random conv filters already give LP ≈ 0.78 on cell_attribute).
3. **Always present the per-category breakdown count-weighted by N.** Headline mean drags down through high-N low-F1 categories (76 of 164 gorilla BSPs are threats at ~0.08).

### Template (fill cells with `F1 / MCC / lift`)

| BSP category | N | LP (trained) | LP (random) | baseline | prev best | **NEW** | random-net control | learned-gap frac |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cell_attribute | 64 | 0.971 | 0.782 | … | … | … | … | … |
| cell_occupancy | 16 | 1.000 | 0.829 | … | … | … | … | … |
| game_phase | 3 | 0.936 | 0.685 | … | … | … | … | … |
| global | 1 | 0.694 | 0.634 | … | … | … | … | … |
| offered_piece | 4 | 0.789 | 0.718 | … | … | … | … | … |
| threat_line | 40 | 0.502 | 0.019 | … | … | … | … | … |
| threat_square_2x2 | 36 | 0.680 | 0.016 | … | … | … | … | … |
| **overall coverage** | 164 | 0.789 | 0.428 | … | … | … | … | … |
| **non-threat subset** | 88 | — | — | … | … | … | … | … |
| **threat-only subset** | 76 | — | — | … | … | … | … | … |

Hawk uses the 7 reframed categories: `reframed_count` × 40, `reframed_completable` × 40, `reframed_any_threat` × 10, `reframed_sq_count` × 36, `reframed_sq_completable` × 36, `reframed_sq_any_threat` × 9, `reframed_global` × 2.

### Why MCC / F1-lift matter — 2026-05-11 backfill

F1 inflates trivial features: a feature that fires on every sample scores F1 = 2p/(1+p) for base rate p. The first backfilled entries make this concrete:

| Run | F1 | MCC | F1-lift | trained / random F1 ratio | trained / random F1-lift ratio |
|---|:---:|:---:|:---:|:---:|:---:|
| anakin-batchtopk-k16-exp8-fc1 | 0.338 | 0.307 | 0.141 | — | — |
| anakin-topk-k64-exp8-conv2 | 0.332 | 0.291 | 0.133 | — | — |
| A02 random-net control (conv2) | 0.224 | 0.177 | 0.025 | **1.51×** | **5.32×** |

The F1-lift gap between trained and random networks is ~3.5× more discriminating than the raw-F1 gap. **F1-lift is the recommended *headline* metric for any SAE-vs-random comparison;** raw F1 is kept for literature comparability; MCC adds an orthogonal information-theoretic view.

## Winners table [DIRECT — 2026-05-11]

| BSP category | N | baseline topk-k32-fc1 | Phase 1E fc1 (batchtopk-k16) | Phase 1E conv2 (topk-k64-exp8) | **Phase 2A (C01 topk-k16-conv2)** | C07 jumprelu-t32-conv2 | D02 topk-k64-exp16-conv2 | random conv2 (A02) | random C01 (G01) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cell_attribute | 64 | 0.423 | 0.504 | 0.501 | **0.537** | 0.523 | 0.506 | 0.297 | 0.240 |
| cell_occupancy | 16 | 0.622 | 0.737 | 0.668 | **0.802** | 0.810 | 0.673 | 0.505 | 0.246 |
| game_phase | 3 | 0.608 | 0.577 | 0.621 | 0.557 | 0.506 | 0.567 | 0.459 | 0.217 |
| global | 1 | 0.692 | 0.635 | 0.581 | 0.494 | 0.564 | 0.551 | 0.501 | 0.188 |
| offered_piece | 4 | 0.650 | 0.598 | 0.666 | 0.631 | 0.667* | 0.666* | 0.667* | 0.313 |
| threat_line | 40 | 0.074 | 0.074 | 0.079 | 0.076 | 0.053 | 0.078 | 0.069 | 0.071 |
| threat_square_2x2 | 36 | 0.101 | 0.103 | 0.094 | 0.084 | 0.059 | 0.094 | 0.068 | 0.070 |
| **overall coverage** | 164 | 0.297 | 0.338 | 0.332 | **0.353** | 0.338 | 0.333 | 0.224 | 0.163 |
| **non-threat (88)** | 88 | 0.479 | 0.555 | 0.544 | **0.589** | 0.582 | 0.546 | 0.359 | 0.243 |
| **threat-only (76)** | 76 | 0.087 | 0.087 | 0.086 | 0.080 | 0.056 | 0.086 | 0.068 | 0.070 |

*`offered_piece` ≈ 0.667 is the trivial baseline; only runs strictly below this found real signal.

## Conclusions [INFERENTIAL — 2026-05-11]

1. **Three pre-registered Phase 2A questions are answered.**
   - *Does BatchTopK-k16 dominate at conv2 like at fc1?* **No.** At conv2 TopK-k16 wins (0.353); BatchTopK-k16 trails at 0.322–0.333 across 3 seeds. First clean architecture × hook interaction observed.
   - *Does exp16 help on conv2?* **No.** D01–D04 are within ± 0.005 of their exp8 counterparts. With 80–99 % dead features at exp8, the dictionary is not the bottleneck.
   - *Are results seed-stable?* **Yes.** σ ≈ 0.005 across the 11 F-campaign runs, matching fc1's σ = 0.004. C01's +0.015 lead over the next conv2 arch is ~3σ and treated as real.

2. **Non-threat gains, no threat gains.** C01 lifts non-threat coverage to 0.589 (= 61 % of conv2 linear-probe ceiling 0.971). Threat-only coverage is 0.080 — indistinguishable from a random network (0.070). **The 33-run sweep moves overall coverage entirely through non-threat categories.** Unsupervised SAEs hit a hard ceiling on threats at this hook — *on champAa*. (Phase 2B re-tested this premise on champS4 / champTa; see [`phase-2B.md`](phase-2B.md).)

3. **Trained-vs-random gap is real and large.** G-campaign trained-minus-random gaps: topk-k16 +0.190, batchtopk-k16 +0.092, topk-k64-exp16 +0.100. C01's gap is double the others, consistent with a tight L0 budget forcing the SAE to exploit trained-network structure rather than the conv-architecture prior.

4. **TopK's deterministic per-sample budget is load-bearing on conv2.** At k = 16 conv2, TopK keeps ~19 % live features; BatchTopK / JumpReLU collapse to 1–2 %. Opposite of the fc1 pattern, where BatchTopK's adaptive allocation was the differentiator.

## Notes carried forward

- Random-net baseline of 0.78 on `cell_attribute` was rationalised in Phase 1F (random-features / extreme-learning-machine regime); see [`phase-1.md`](phase-1.md) § "Why the random-net baseline is 0.78 on cell_attribute".
- Hawk-set re-evaluation of C/D/F winners was a follow-up: the C/D/F/G checkpoints live only on Deep Brain (`saes/**/*.pt` is gitignored). Procedure: pull repo on Deep Brain, run the hawk-eval loop there, push `eval_registry.json` + new `cache/*_matching-hawk.pt`. Cheap (~1 h CPU).
- Feature reuse / absorption diagnostic on C01 was queued (uses `best_feature_per_bsp` from the matching cache); not yet executed.
