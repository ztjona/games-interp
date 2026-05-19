# Quarto SAE Research — Quick Reference

## Project State (2026-05-18)

**Active phase:** Phase 2B — champS4 follow-up sweep + diagnostic. Open question:
why do the four top-Aa configs underperform their champAa twins when re-trained
on champS4 activations? See "Phase 2B" below for results and the proposed
LP-diagnostic-first plan.

## Project State (2026-05-04)

### Completed Phases
- **Phase 0:** Data pipeline rebuilt with mode_2x2=True. Amalgam dataset: 275,916 unique positions.
- **Phase 1A:** Linear probes on gorilla (164 BSPs). Trained F1=0.402, Random F1=0.194. H1 confirmed (threats non-linear), H7 confirmed (offered piece not learned).
- **Phase 1B:** Per-category coverage integrated into sae_eval.py.
- **Phase 1C:** Conv2 TopK SAE trained and evaluated. Coverage=0.335 (vs fc1=0.297). Both have high dead features.
- **Phase 1D:** Hawk BSPs (lines-only, 92). Count reframing 14x better than boolean threats (F1=0.315 vs 0.022). Completability near-zero (F1=0.003). H1b partial.
- **Phase 1D update:** Hawk expanded with 2x2 squares (92 -> 173 BSPs). Labels were recomputed and fc1 probes were rerun on hawk_173: trained F1=0.200, random F1=0.005, with strong square-count signal (F1=0.455) but still weak square completability (F1=0.027).
- **Phase 1E:** Anakin sweep completed (2026-04-01). 28/33 configs trained+evaluated, 5 timed out (converged but no checkpoint). See analysis below.
- **Phase 1F:** Conv2 gorilla linear probes completed. Trained conv2 coverage=0.789 vs random conv2 coverage=0.428. Threat BSPs jump from near-zero at fc1 to clearly linear at conv2 (`threat_line` 0.502, `threat_square_2x2` 0.680), confirming H8.

### Phase 1E: Anakin Sweep Results (2026-04-06)

**Sweep integrity:** 28 configs completed 25k steps; 5 timed out on GPU 0 (all had converged FVU by 70–94% of training). Tagged as partial — no rerun needed.

**Statistical validity:** Seed variability σ=0.004 (range 0.008 across 3 seeds). Only differences >0.012 (~3σ) should be considered meaningful. Three statistically distinguishable performance tiers emerged:

| Tier | Coverage range | Runs | Key members |
|------|:---:|:---:|---|
| 1 (best) | 0.326–0.338 | 3 | BatchTopK-k16-fc1, TopK-k64-conv2 (exp4 & exp8) |
| 2 (bulk) | 0.275–0.314 | 20 | Most fc1 configs (TopK, BatchTopK, JumpReLU, P-Annealing) |
| 3 (low) | 0.227–0.263 | 5 | Gated, Vanilla, TopK-k128-fc1 |

**Key findings:**
1. **Most architectures are interchangeable** — 20/28 runs cluster in Tier 2 (coverage ~0.29 ± 0.01). Architecture choice matters less than hook point and sparsity level.
2. **conv2 has a real edge** — Tier 1 includes 2 conv2 runs and only 1 fc1 run. Conv2 excels at spatial BSPs (cell_attribute F1=0.50 vs 0.44 at fc1).
3. **BatchTopK-k16-fc1 is the fc1 winner** — coverage=0.338, cov>50=36.6%. Achieves best results with extreme sparsity (L0=16), likely because BatchTopK's adaptive per-batch allocation is efficient.
4. **High L0 hurts** — Gated (L0=340–360), Vanilla (L0=609–633), and TopK-k128 (L0=128) all fall in Tier 3. Low FVU ≠ interpretability.
5. **Expansion factor 4→8 helps, 8→16 doesn't** — TopK-k32 fc1: exp4=0.286, exp8=0.301, exp16=0.303.
6. **Seed stability is good** — coverage 0.293–0.301 across 3 seeds (range=0.008).

**Critical BSP-category insights (all confirmed vs linear probe baseline):**

| Category | LP Baseline | Best SAE | SAE/LP | Interpretation |
|---|:---:|:---:|:---:|---|
| cell_occupancy | 0.998 | 0.737 | 0.74 | SAE recovers ~74% of available signal |
| cell_attribute | 0.611 | 0.504 | 0.82 | SAE nearly matches probe |
| game_phase | 0.908 | 0.651 | 0.72 | SAE captures majority of phase info |
| offered_piece | 0.602 | 0.637* | ~1.0 | *Both at noise floor — see note below |
| global | 0.698 | 0.678 | 0.97 | SAE matches probe |
| threat_line | 0.022 | 0.087 | 3.9× | Both near zero — info not linearly present |
| threat_square | 0.113 | 0.107 | 0.95 | Both near zero — info not linearly present |

*Note on offered_piece:* F1=0.667 is the trivial all-positive baseline (P=0.5, R=1.0 when base_rate=0.5). The 18 SAE runs at F1≈0.667 found **no** discriminative features for offered_piece — those are just features that always fire. Runs scoring F1<0.667 (0.595–0.637) found real but weak signal. The LP also scores below the trivial baseline (0.602). This is architecturally expected: the offered piece enters via fc_in_piece→ReLU→(1,4,4)→conv1→conv2→fc1, entangled through two conv layers.

**Partial runs (converged, no checkpoint):**

| Config | Steps | FVU@last | L0 | Status |
|---|:---:|:---:|:---:|---|
| conv2-batchtopk-k64-exp4 | 19.5k/25k | 0.0067 | 64 | partial (converged) |
| conv2-panneal-exp4 | 17.5k/25k | 0.0006 | 649 | partial (converged) |
| conv2-topk-k64-exp2 | 23.5k/25k | 0.0062 | 64 | partial (converged) |
| conv2-vanilla-l1_001-exp4 | 21.5k/25k | 0.0003 | 1065 | partial (converged) |
| fc1-gated-l1_002-exp16 | 20.5k/25k | 0.0001 | 499 | partial (not fully converged) |

### Key Metrics Summary
| Probe / SAE | BSP Set | Coverage / F1 |
|-------|---------|:---:|
| Linear (fc1) | gorilla (164) | 0.402 |
| Linear (fc1, random net) | gorilla (164) | 0.194 |
| Linear (conv2) | gorilla (164) | 0.789 |
| Linear (conv2, random net) | gorilla (164) | 0.428 |
| Linear (fc1) | hawk (173, full reframed) | 0.200 |
| Linear (fc1, random net) | hawk (173, full reframed) | 0.005 |
| Best SAE (BatchTopK-k16-fc1) | gorilla | 0.338 |
| Best SAE (TopK-k64-conv2-exp8) | gorilla | 0.332 |
| **Best SAE overall (C01 TopK-k16-conv2-exp8, 2026-05-11)** | gorilla | **0.353** |
| Pre-anakin baseline (TopK-k32-fc1) | gorilla | 0.297 |

### Phase 1F: Conv2 Linear Probe Results (2026-04-24)

**Result:** H8 is confirmed. Threat information is linearly present in conv2 and mostly absent by fc1.

| Category | fc1 probe | conv2 probe | conv2 random | Interpretation |
|---|:---:|:---:|:---:|---|
| cell_occupancy | 0.998 | 1.000 | 0.829 | Easy to decode even from random conv filters, but still perfect when trained |
| cell_attribute | 0.611 | 0.971 | 0.782 | Strong learned spatial attribute representation in conv2 |
| threat_line | 0.022 | 0.502 | 0.019 | Learned threat-line information is present in conv2 and collapses by fc1 |
| threat_square_2x2 | 0.113 | 0.680 | 0.016 | Square threats are even more linearly accessible in conv2 |
| offered_piece | 0.602 | 0.789 | 0.718 | High random control means this remains a weak interpretability signal |
| global | 0.698 | 0.694 | 0.634 | Similar across hooks; not a discriminating category |
| game_phase | 0.909 | 0.936 | 0.685 | Phase info is present throughout, with some architecture prior |

**Implication:** The dominant open problem is no longer whether the network contains threat information. It does. The open problem is why the current unsupervised SAEs fail to recover more of the conv2 threat signal.

**Threat-focused re-rank:** if we weight threat recovery more heavily than raw gorilla coverage, the best conv2 follow-up checkpoint is `hook-sweep-conv2-topk-k32-exp8-conv2`, with `anakin-topk-k64-exp8-conv2` close behind. The overall-coverage winner remains `anakin-batchtopk-k16-exp8-fc1`, so the next phase should compare the strongest fc1 and conv2 candidates on hawk_173 rather than assuming the same winner for every objective.

### Hypotheses Status Update (post-Anakin)
| # | Hypothesis | Status |
|---|-----------|--------|
| H1 | fc1 encodes cell-level BSPs linearly but not threats | ✅ CONFIRMED — consistent across 28 SAE configs |
| H1b | Threats may be accessible in reframed basis | ⚠️ PARTIAL — count encoding 14× better but still F1=0.315 |
| H2 | Feature absorption | OPEN |
| H3 | Concept heterogeneity → architecture-dependent failures | ❌ WEAKENED — most architectures perform similarly |
| H5 | Feature shrinkage (L1/ReLU) | ⚠️ PARTIAL — Gated/Vanilla do score lower (Tier 3), but P-Annealing doesn't dominate Tier 1 |
| H6 | Wrong hook point → conv2 may be better | ✅ SUPPORTED — conv2 runs in Tier 1 with fc1 runs in Tier 2 |
| H7 | Offered piece is not learned | ✅ CONFIRMED — F1=0.667 is trivial baseline, not real signal |
| H8 | **NEW:** Threat info is spatially encoded, lost at fc1 bottleneck | ✅ CONFIRMED — conv2 threat probes are high while fc1 stays near-zero |

### BSP Sets
- **gorilla_164:** 7 categories (cell_occupancy, cell_attribute, threat_line, threat_square_2x2, offered_piece, global, game_phase)
- **hawk_173:** 7 reframed categories (count×40, completable×40, any_threat×10, sq_count×36, sq_completable×36, sq_any_threat×9, global×2)
- See `BSP-schema-summary.md` for full details

### Recovery — A/B Follow-up Panel (interrupted 2026-04-24)

A power loss between 19:21 and 19:29 on 2026-04-24 left the panel partially complete:

| Run | Trained | Evaluated | Action |
|---|:---:|:---:|---|
| `A01-random-control-s42-batchtopk-k16-exp8-fc1` | ✅ | ❌ | eval on gorilla |
| `A02-random-control-s42-topk-k32-exp8-conv2` | ✅ (5k steps per config) | ❌ | eval on gorilla |
| `B01-conv2-completion-s42-batchtopk-k64-exp4-conv2` | ✅ | ❌ | eval on gorilla |
| `B02-conv2-completion-s42-p-annealing-exp4-conv2` | ✅ | ❌ | eval on gorilla |
| `B03-conv2-completion-s42-topk-k64-exp2-conv2` | partial (4k/25k, no .pt) | — | **rerun training** |
| `B04-conv2-completion-s42-vanilla-l1_001-exp4-conv2` | not started | — | **train from scratch** |

### Phase 2A: Conv2 Architecture Sweep — COMPLETED (re-run on Deep Brain 3×GPU, 2026-05-11)

**Status:** All 33 runs (Campaigns C/D/F/G) completed successfully on Deep Brain after the timeout fix. 79 total entries now in `eval_registry.json`. The original 2026-05-04 failure was a `run_sweep.py` timeout, not a methodological issue — the same configs re-ran cleanly with `--timeout=86400`.

**New winner:** `C01-c2arch-s42-topk-k16-exp8-conv2` — gorilla coverage = **0.353**, cov>50 = **0.396**. First time any run cracks 0.35.

#### Reporting standard (REQUIRED for every winner claim)

Three rules, enforced as of 2026-05-11:

1. **Always report three coverage metrics side-by-side** — F1 (literature standard, base-rate-sensitive), MCC (base-rate-invariant; 0 for any constant predictor), F1-lift (best_F1 minus the trivial 2p/(1+p) "always-positive" baseline, clipped at 0). MCC and F1-lift add ~0 cost (same TP/FP/FN/TN counts), and they collapse the trivial-feature artifact that gives `offered_piece` an apparent F1 = 0.667. The eval pipeline writes all three to `eval_registry.json`; `scripts/backfill_eval_metrics.py` retro-fills older entries.
2. **Always compare against linear-probe ceiling AND random-network control.** The "learned-gap fraction" `(coverage_SAE − coverage_rand_SAE) / (LP_trained − LP_random)` is the cleanest way to claim a SAE recovers *learned* structure rather than the architectural prior (especially important at conv2 where random conv filters already give LP ≈ 0.78 on cell-attribute).
3. **Always present the per-category breakdown count-weighted by N.** Headline mean drags down through high-N low-F1 categories (76 of 164 gorilla BSPs are threats at ~0.08).

Template (fill cells with `F1 / MCC / lift`):

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

(Hawk uses the 7 reframed categories: `reframed_count` ×40, `reframed_completable` ×40, `reframed_any_threat` ×10, `reframed_sq_count` ×36, `reframed_sq_completable` ×36, `reframed_sq_any_threat` ×9, `reframed_global` ×2.)

##### Why MCC / F1-lift matter (2026-05-11 backfill)

F1 inflates trivial features: a feature that fires on every sample scores F1 = 2p/(1+p) for base rate p. The first backfilled entries make this concrete:

| Run | F1 | MCC | F1-lift | trained / random F1 ratio | trained / random F1-lift ratio |
|---|:---:|:---:|:---:|:---:|:---:|
| anakin-batchtopk-k16-exp8-fc1 | 0.338 | 0.307 | 0.141 | — | — |
| anakin-topk-k64-exp8-conv2 | 0.332 | 0.291 | 0.133 | — | — |
| A02 random-net control (conv2) | 0.224 | 0.177 | 0.025 | **1.51×** | **5.32×** |

The F1-lift gap between trained and random networks is ~3.5× more discriminating than the raw-F1 gap. **F1-lift is now the recommended *headline* metric for any SAE-vs-random comparison;** raw F1 is kept for literature comparability; MCC adds an orthogonal information-theoretic view. The trio is cheap to compute simultaneously.

#### Winners table (2026-05-11)

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

#### Key conclusions

1. **The three pre-registered Phase 2A questions are answered:**
   - *Does BatchTopK-k16 dominate at conv2 like at fc1?* **No.** At conv2 TopK-k16 wins (0.353); BatchTopK-k16 trails at 0.322–0.333 across 3 seeds. First clean architecture × hook interaction observed.
   - *Does exp16 help on conv2?* **No.** D01–D04 are within ±0.005 of their exp8 counterparts. With 80–99 % dead features at exp8, the dictionary is not the bottleneck.
   - *Are results seed-stable?* **Yes.** σ ≈ 0.005 across the 11 F-campaign runs, matching fc1's σ = 0.004. C01's +0.015 lead over the next conv2 arch is ~3σ and treated as real.

2. **Non-threat gains, no threat gains.** C01 lifts non-threat coverage to 0.589 (= 61 % of conv2 linear-probe ceiling 0.971). Threat-only coverage is 0.080 — indistinguishable from a random network (0.070). **The 33-run sweep moves overall coverage entirely through non-threat categories.** Unsupervised SAEs have a hard ceiling on threats at this hook.

3. **Trained-vs-random gap is real and large.** G-campaign trained-minus-random gaps: topk-k16 +0.190, batchtopk-k16 +0.092, topk-k64-exp16 +0.100. C01's gap is double the others, consistent with a tight L0 budget forcing the SAE to exploit trained-network structure rather than the conv-architecture prior.

4. **TopK's deterministic per-sample budget is load-bearing on conv2.** At k=16-conv2, TopK keeps ~19 % live features; BatchTopK/JumpReLU collapse to 1–2 %. Opposite of the fc1 pattern, where BatchTopK's adaptive allocation was the differentiator.

### Phase 1G: Model Competence Audit (RESULTS — executed 2026-05-11, decisive)

**Run.** `scripts/model_competence_audit.py` with N=5,000 for A/B/C and the full 275,916 positions for D/E; trained and random conv2 models compared. Output: `data/quarto/model_competence_audit.json`.

| Test | Trained | Random | Diagnosis |
|---|---:|---:|---|
| A: winning-placement acc | **0.700** | 0.223 | Strong — trained model recognises immediate wins ~3× random |
| B: losing-piece avoidance | 0.405 | 0.433 | **Indistinguishable from random** — model does NOT reason about what it hands the opponent |
| B: positions excluded (forced loss) | 307 | 607 | — |
| C: piece-sensitivity (mean distinct frac, 1.0 = fully sensitive) | 0.144 | 0.113 | Placement is almost piece-invariant — H7 reconfirmed behaviorally |
| D: Q(empty) − Q(occupied) | **−0.472** | −0.001 | Trained head assigns *lower* raw Q to empty cells (only 25.3% positive). Legality is supplied entirely by the external mask in `predict()` |
| E: entropy / uniform-cap by phase | 2.51/2.57 (early), 1.99/2.21 (mid), 1.31/1.43 (late) | 2.57/2.57, 2.19/2.21, 1.39/1.43 | Modest concentration only at mid-game; late-game still within 8% of uniform |

**Interpretation.** The trained agent has learned a single asymmetric heuristic — "does this position give me an immediate win?" — and almost nothing else. It does not reason about what piece it gives, does not internalise legality, and barely concentrates probability mass even when one move is forced. This **fully explains the hawk SAE results**: threat-count features exist at moderate F1-lift (~0.12) because the model uses them to detect *its own* winning chances; threat-completability features sit at noise floor (~0.03) because the model never computes "is this piece losing for me to give." There is nothing for an SAE to find on the completability axis.

**Implications for the research program:**
1. **Phase 3A E2E SAE is no longer purely architectural.** The legitimate question becomes *"once we project conv2 onto Q-head-relevant directions, what fraction of variance is the immediate-win check vs everything else?"* — E2E is now diagnostic for the asymmetry itself.
2. **Matryoshka is deprioritised** — it solves feature absorption, but the missing concepts aren't being absorbed, they're never represented.
3. **Anchored / guided SAEs against `reframed_completable`** would also fail by construction; do not attempt.
4. **Phase 4 (architectural fix) is promoted from "branch" to "primary path":** the auxiliary "predict-threat-from-fc1" head during DQN training is now the most direct route to a model whose threat representations a generic SAE can recover. The current model is the bottleneck, not the SAE family.
5. **Re-narrate the dissertation contribution** from "find threat features in Quarto via SAEs" to "characterise the attack-recognition / defence-blindness asymmetry of a DQN-trained Quarto agent through behavioural + SAE evidence." This is a stronger negative result with cleaner experimental closure.

**What the tests were** (kept for reproducibility): A: positions where the offered piece can win at least one cell, measure argmax-legal placement; B: positions with ≥1 safe and ≥1 losing piece in storage, measure argmax-legal selection after applying A's placement, exclude forced-loss positions; C: replay each position with all 16 candidate offered pieces, count distinct argmax-legal cells; D: mean `Q[empty] − Q[occupied]` unmasked across all positions; E: softmax entropy of legal-masked Q, bucketed by piece count.

### Phase 3A — New Architectures (PLANNED, scope reshaped by Phase 1G)

Implementation order, motivated by Phase 2A conclusions:

1. **End-to-end SAE** (Braun et al., 2024). Trains the SAE to preserve downstream Q-head loss rather than raw activation MSE. Directly diagnostic: if threats are present in conv2 but the Q-head ignores them, E2E will *deprioritize* threat features relative to vanilla — quantifying how load-bearing threats actually are. Complements Phase 1G at the SAE level. Implementation cost: moderate (loss + forward hook to `fc2_board`/`fc2_piece`).

2. **Matryoshka SAE** (Bussmann et al., 2024). Nested dictionaries d ⊂ 2d ⊂ 4d ⊂ 8d, sparse at each level. Directly attacks feature absorption / splitting — the suspected cause of the BatchTopK/JumpReLU 99 %-dead pattern at conv2. Best bet for closing the SAE/LP gap on non-threat categories.

3. *Lower priority:* crosscoders (conv2 ↔ fc1 transfer), transcoders (replace fc layer with sparse map). Worth doing after E2E + Matryoshka.

### Other near-term work

- **Hawk_173 evaluation on the C/D/F winners (must run on Deep Brain).** The C/D/F/G checkpoints (33 runs) live only on Deep Brain — `saes/**/*.pt` is gitignored, so local hawk eval fails with FileNotFoundError. Procedure: pull repo on Deep Brain, run the hawk-eval loop there, push the updated `eval_registry.json` and the new `cache/*_matching-hawk.pt` files (the latter are not gitignored). Cheap to add — pure evaluation, ~1 h CPU.
- **Feature reuse / absorption diagnostic on C01** — uses `best_feature_per_bsp` from the matching cache. Quantifies how often the same feature wins for multiple BSPs at conv2.

### Why the random-net baseline is 0.78 on cell_attribute (added 2026-05-11)

Random conv filters are a fixed nonlinear projection of the input; a downstream linear probe can usually decode any linearly-accessible input attribute through such a projection (this is the *random features / extreme learning machine* regime). Since piece attributes enter as one-hot inputs, they survive random projections almost intact. Consequence: **for cell-level BSPs the relevant baseline is not 0, it is the random-conv2 probe**, and the right "did the SAE learn something" metric is the *learned-gap fraction*

  (cov_SAE − cov_rand_SAE) / (LP_trained − LP_random)

For C01 this is 1.57 on `cell_attribute` (the SAE recovers more than the LP-gap because LP exploits projections the SAE's sparse code cannot) — strong evidence C01 captures learned attribute structure, not the architectural prior. For threat categories the same fraction is ≈ 0.01: nothing learned-specific is recovered.

This reframing changes how we interpret the threat probes: `threat_line` LP F1 = 0.502 is **fully learned** (random conv2 = 0.019), so the trained `conv1+conv2` stack does compute threats; the bottleneck is fc1 (LP F1 = 0.022). This motivates an **architectural fix**: adding a small auxiliary threat-prediction head on fc1 during DQN training (λ ≈ 0.1 weight, 76-dim BCE) would force the bottleneck to preserve threat information. Parked under Phase 4 (interpretability-by-architecture); execute only after Phase 1G confirms the model uses threats.

### Phase 2B: champS4 follow-up sweep (2026-05-18)

A new champion (`champS4`, unified-aux autoregressive CNN, fc1=512) was trained
on Deep Brain. `commands.sh` re-ran the full pipeline on the new model:
competence audit, S4-self-play position regeneration, BSP labels against the new
distribution (`gorillaS4`/`hawkS4`), and the four top-Aa configs as a
mini-sweep (`A01`, `C01`, `C07`, `D02`).

#### Model competence audit (champS4 plays strictly better than champAa)

| Test | champS4 | champAa | Δ |
|---|---:|---:|---:|
| A. Winning-placement acc | **0.787** | 0.700 | +0.087 |
| B. Losing-piece avoidance | **0.620** | 0.405 | **+0.215** |
| C. Offered-piece sensitivity (mean distinct frac) | **0.228** | 0.144 | +0.084 |
| D. Q(empty) − Q(occupied), fraction positive | 0.119 | 0.253 | **−0.134** |
| E. Late-game (11–15) mean entropy | 1.249 | 1.312 | small decrease |

The unified-aux model finally reasons about *what piece it hands the opponent*
(test B doubles from 0.40 → 0.62), the first time any Quarto agent in this
project rises above random on the defense axis. Test D regresses: the unmasked
board head is even less legality-aware than champAa.

#### SAE sweep results — champS4 SAEs underperform champAa twins

Top by F1-lift on **gorillaS4** (only 4 runs trained; same ranking as champAa
mini-set: D02 ≈ A01 > C07 > C01):

| run | F1-lift | cov | MCC | FVU | L0 | dead % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| D02 topk-k64-exp16 conv2 | 0.128 | 0.327 | 0.282 | 0.012 | 64 | 93.1 |
| A01 batchtopk-k16-exp2 fc1 | 0.125 | 0.324 | 0.292 | 0.058 | 16 | **96.6** |
| C07 jumprelu-t32-exp8 conv2 | 0.106 | 0.305 | 0.247 | 0.075 | 45 | 95.3 |
| C01 topk-k16-exp8 conv2 | 0.039 | 0.225 | 0.226 | 0.061 | 16 | 58.0 |

Pairwise vs the corresponding champAa twin (S4 minus Aa, gorilla{S4 vs *}):

| pair | Δ F1-lift | Δ FVU | comment |
|---|:---:|:---:|---|
| A01 fc1 batchtopk | −0.016 | +0.049 | small coverage drop, much higher FVU |
| C01 conv2 topk k16 | **−0.116** | +0.036 | severe collapse (cov>0.5 falls 0.40→0.04) |
| C07 conv2 jumprelu | −0.033 | +0.049 | L0 drifted up; metrics jsonl has only 7 rows (early-stop) |
| D02 conv2 topk k64 | −0.006 | +0.008 | essentially matches Aa; best reconstruction (91% reconstructable) |

On **hawkS4** all four cluster at lift 0.058–0.074 vs the champAa anakin-batchtopk
peak of 0.105 — threats are uniformly harder to recover on the new champion.

#### Open question and diagnostic plan

The headline puzzle: **champS4 plays better but its activations are harder to
interpret** with the recipes that won on champAa. FVU is 3–6× higher across the
board, and one config (C01 k=16) collapsed entirely. Possible causes, in
decreasing order of likelihood:

1. **The S4 activations themselves are less linearly separable.** The four-run
   spread is consistent with a representation-level shift, not an SAE-recipe
   issue. Without an LP ceiling on S4 activations we cannot tell whether the
   gap is "model" or "SAE".
2. **A01 expansion mismatch.** `A01-champS4` uses `expansion=2` to match the
   *dictionary size* of the Aa twin (d_dict=1024 from d_act=128×8 = d_act=512×2);
   but the S4 fc1 has 4× more activation dimensions to compress, which the
   96.6% dead-features rate is consistent with.
3. **C07 truncated by patience.** `*_metrics.jsonl` has 7 rows vs the 51 of the
   other three runs — likely `patience=20 + min_improvement=0.02` triggered on
   jumprelu's θ warm-up plateau.
4. **C01 k=16 is too tight for S4 conv2.** Same hyperparameters as the Aa
   winner but cov>50 collapsed to 4%. Dead-features fell from 81% → 58%, so
   features stayed alive but carry less concept information.

**Plan (ordered by cost):**

1. **LP baseline on champS4** (cheap; runs on Deep Brain via the LP block in
   the user's `commands.sh` — see CLAUDE.md note on that file's status).
   All three coverage metrics computed for `{s4.fc1, s4.conv2} ×
   {trained, random} × {gorillaS4, hawkS4}` = 8 results. The script's outputs are directly comparable to `eval_registry`
   entries because the LP script was extended (2026-05-18) to emit
   `coverage_mcc` and `coverage_f1_lift` alongside `coverage` (F1). If LP on
   S4 falls in line with LP on Aa, the gap is the SAE recipes; if LP itself
   drops, the model's representations changed and no SAE recipe will close it.
2. **Fix C07 patience** (one-line config edit; lower `min_improvement` or
   raise `patience` for jumprelu only).
3. **Rerun A01 at expansion=8** (d_dict=4096 — matches the conv2 runs in
   feature budget; addresses the dead-features rate).
4. **Only if (1)–(3) leave a residual gap, launch a scoped conv2-S4 sweep**
   focused on the C01 k=16 collapse region (k ∈ {16, 24, 32}, exp=8;
   batchtopk variants for cross-check). fc1-S4 explicitly deprioritized
   beyond step 3, consistent with the existing fc1 deprioritization rule.

Once LP results land, this section gets a head-to-head table mirroring the
2026-05-11 winners table, with `S4-LP / S4-LP-random / S4-best-SAE / Aa twin`
columns.

### Deprioritized (2026-04-27)

- **Broad unsupervised arch sweeps on fc1** — Anakin (28 configs, σ=0.004 across seeds) showed this is second-order for fc1. Scope clarified 2026-04-29: this deprioritization is fc1-specific. Conv2 has never had a complete architecture sweep (only topk/jumprelu/gated tested; batchtopk/vanilla/panneal missing) — Campaign C–G fills this gap.
- **fc1-only threat investigation** — Phase 1F decisively redirected this work to conv2. fc1 is retained only as a bottleneck-comparison reference.
- **`offered_piece` as a coverage signal** — F1 ≈ 0.667 is the trivial all-positive baseline (P=0.5, R=1.0). Keep it in the per-category breakdown for transparency, but **exclude it from any threat-focused or headline ranking**. Anakin's overall coverage means are inflated by this category for many runs.

### Follow-up Run Naming (from 2026-04-24 onward)

- Use experiment IDs of the form `{Major}{Minor}-{tag}-s{seed}` so checkpoints, eval caches, and registry keys stay unique.
- Current follow-up panel uses `A01`/`A02` for random-model controls and `B01`–`B04` for the conv2 completion panel.
- Keep the seed in the experiment ID itself, because the trainer names checkpoints from `experiment + architecture suffix + hook`.

### Known Issues
- docopt: avoid `--` prefix collisions and line continuations in docstrings
- GPU 0 significantly slower than GPUs 1/2 — use `--split` to avoid assigning heavy conv2 configs to GPU 0
- offered_piece F1=0.667 is a metrics artifact (trivial baseline), not real coverage
- conv2 overall coverage mixes strongly learned categories with architecture-easy categories; always compare against the random conv2 control
