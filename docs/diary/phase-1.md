# Phase 1 — SAE family, hooks, and the competence wall (2026-Q1/Q2)

Phase 1 ran on `champAa` (legacy `CNN_uncoupled`) and answered three
empirical questions:

1. Which SAE architecture works best at this hook? (Phase 1E — Anakin sweep)
2. Where in the network do threats actually live? (Phase 1F — conv2 LP)
3. Does the model itself use threats, or do we have nothing to find? (Phase 1G — competence audit)

The answer pattern that emerged — "threats live in conv2, the model only
uses them asymmetrically, and unsupervised SAEs hit a hard ceiling on
threat categories" — set the agenda for Phase 2 (architecture-on-conv2,
then new champions).

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md).

## Completed sub-phases — at a glance

| Sub-phase | Date | Output |
|---|---|---|
| 0  | early Q1 | Data pipeline rebuilt with `mode_2x2=True`. Amalgam dataset: 275,916 unique positions. |
| 1A | early Q1 | Linear probes on gorilla (164 BSPs). Trained F1 = 0.402, random F1 = 0.194. H1 confirmed (threats non-linear at fc1); H7 confirmed (offered piece not learned). |
| 1B | early Q1 | Per-category coverage integrated into `sae_eval.py`. |
| 1C | early Q1 | Conv2 TopK SAE trained and evaluated. Coverage = 0.335 (vs fc1 = 0.297). Both have high dead features. |
| 1D | early Q1 | Hawk BSPs (lines-only, 92). Count reframing 14× better than boolean threats (F1 = 0.315 vs 0.022). Completability near-zero (F1 = 0.003). H1b partial. |
| 1D update | early Q1 | Hawk expanded with 2×2 squares (92 → 173 BSPs). Trained F1 = 0.200, random F1 = 0.005; strong square-count signal (F1 = 0.455), weak square-completability (F1 = 0.027). |
| 1E | 2026-04-06 | Anakin sweep (28 configs). See § Anakin sweep below. |
| 1F | 2026-04-24 | Conv2 gorilla linear probes. See § Conv2 LP below. |
| 1G | 2026-05-11 | Model competence audit (test A/B/C/D/E). See § Competence audit below. |

## Phase 1E — Anakin sweep results [DIRECT from `eval_registry.json`]

**Sweep integrity:** 28 configs completed 25k steps; 5 timed out on GPU 0 (all had converged FVU by 70–94 % of training). Tagged as partial — no rerun needed.

**Statistical validity:** Seed variability σ = 0.004 (range 0.008 across 3 seeds). Only differences > 0.012 (~3σ) should be considered meaningful. Three statistically distinguishable performance tiers:

| Tier | Coverage range | Runs | Key members |
|---|---|---|---|
| 1 (best) | 0.326–0.338 | 3 | BatchTopK-k16-fc1, TopK-k64-conv2 (exp4 & exp8) |
| 2 (bulk) | 0.275–0.314 | 20 | Most fc1 configs (TopK, BatchTopK, JumpReLU, P-Annealing) |
| 3 (low)  | 0.227–0.263 | 5  | Gated, Vanilla, TopK-k128-fc1 |

### Conclusions [INFERENTIAL — original analysis 2026-04-06]

1. **Most architectures are interchangeable** — 20/28 runs cluster in Tier 2 (coverage ~0.29 ± 0.01). Architecture choice matters less than hook point and sparsity level.
2. **conv2 has a real edge** — Tier 1 includes 2 conv2 runs and only 1 fc1 run. Conv2 excels at spatial BSPs (cell_attribute F1 = 0.50 vs 0.44 at fc1).
3. **BatchTopK-k16-fc1 is the fc1 winner** — coverage = 0.338, cov > 50 = 36.6 %. Achieves best results with extreme sparsity (L0 = 16), likely because BatchTopK's adaptive per-batch allocation is efficient.
4. **High L0 hurts** — Gated (L0 = 340–360), Vanilla (L0 = 609–633), and TopK-k128 (L0 = 128) all fall in Tier 3. Low FVU ≠ interpretability.
5. **Expansion factor 4→8 helps, 8→16 doesn't** — TopK-k32 fc1: exp4 = 0.286, exp8 = 0.301, exp16 = 0.303.
6. **Seed stability is good** — coverage 0.293–0.301 across 3 seeds (range = 0.008).

### Critical BSP-category insights [DIRECT — vs linear probe baseline]

| Category | LP Baseline | Best SAE | SAE/LP | Interpretation |
|---|:---:|:---:|:---:|---|
| cell_occupancy | 0.998 | 0.737 | 0.74 | SAE recovers ~74 % of available signal |
| cell_attribute | 0.611 | 0.504 | 0.82 | SAE nearly matches probe |
| game_phase | 0.908 | 0.651 | 0.72 | SAE captures majority of phase info |
| offered_piece | 0.602 | 0.637* | ~1.0 | *Both at noise floor — see note below |
| global | 0.698 | 0.678 | 0.97 | SAE matches probe |
| threat_line | 0.022 | 0.087 | 3.9× | Both near zero — info not linearly present |
| threat_square | 0.113 | 0.107 | 0.95 | Both near zero — info not linearly present |

*Note on `offered_piece`:* F1 = 0.667 is the trivial all-positive baseline (P = 0.5, R = 1.0 when base_rate = 0.5). The 18 SAE runs at F1 ≈ 0.667 found **no** discriminative features — they are features that always fire. Runs scoring F1 < 0.667 (0.595–0.637) found real but weak signal. The LP also scores below the trivial baseline (0.602). This is architecturally expected: the offered piece enters via `fc_in_piece → ReLU → (1,4,4) → conv1 → conv2 → fc1`, entangled through two conv layers.

### Partial runs (converged, no checkpoint)

| Config | Steps | FVU @ last | L0 | Status |
|---|:---:|:---:|:---:|---|
| conv2-batchtopk-k64-exp4 | 19.5k / 25k | 0.0067 | 64 | partial (converged) |
| conv2-panneal-exp4 | 17.5k / 25k | 0.0006 | 649 | partial (converged) |
| conv2-topk-k64-exp2 | 23.5k / 25k | 0.0062 | 64 | partial (converged) |
| conv2-vanilla-l1_001-exp4 | 21.5k / 25k | 0.0003 | 1065 | partial (converged) |
| fc1-gated-l1_002-exp16 | 20.5k / 25k | 0.0001 | 499 | partial (not fully converged) |

## Phase 1F — Conv2 LP results [DIRECT, 2026-04-24]

**Result:** H8 is confirmed (on champAa). Threat information is linearly present in conv2 and mostly absent by fc1.

| Category | fc1 probe | conv2 probe | conv2 random | Interpretation |
|---|:---:|:---:|:---:|---|
| cell_occupancy | 0.998 | 1.000 | 0.829 | Easy to decode even from random conv filters, but still perfect when trained |
| cell_attribute | 0.611 | 0.971 | 0.782 | Strong learned spatial attribute representation in conv2 |
| threat_line | 0.022 | 0.502 | 0.019 | Learned threat-line information is present in conv2 and collapses by fc1 |
| threat_square_2x2 | 0.113 | 0.680 | 0.016 | Square threats even more linearly accessible in conv2 |
| offered_piece | 0.602 | 0.789 | 0.718 | High random control means weak interpretability signal |
| global | 0.698 | 0.694 | 0.634 | Similar across hooks; not discriminating |
| game_phase | 0.909 | 0.936 | 0.685 | Phase info present throughout, with some architecture prior |

### Conclusions [INFERENTIAL — 2026-04-24]

**Implication:** The dominant open problem is no longer whether the network contains threat information. It does. The open problem is why current unsupervised SAEs fail to recover more of the conv2 threat signal.

**Threat-focused re-rank:** if we weight threat recovery more heavily than raw gorilla coverage, the best conv2 follow-up checkpoint is `hook-sweep-conv2-topk-k32-exp8-conv2`, with `anakin-topk-k64-exp8-conv2` close behind. The overall-coverage winner remains `anakin-batchtopk-k16-exp8-fc1`, so the next phase should compare the strongest fc1 and conv2 candidates on `hawk_173` rather than assuming the same winner for every objective.

**Why the random-net baseline is 0.78 on cell_attribute** *(added 2026-05-11)*: random conv filters are a fixed nonlinear projection of the input; a downstream linear probe can usually decode any linearly-accessible input attribute through such a projection (the *random features / extreme learning machine* regime). Since piece attributes enter as one-hot inputs, they survive random projections almost intact. Consequence: **for cell-level BSPs the relevant baseline is not 0, it is the random-conv2 probe**, and the right "did the SAE learn something" metric is the *learned-gap fraction*

  `(cov_SAE − cov_rand_SAE) / (LP_trained − LP_random)`

For C01 (Phase 2A winner) this is 1.57 on `cell_attribute` (the SAE recovers more than the LP-gap because LP exploits projections the SAE's sparse code cannot) — strong evidence C01 captures learned attribute structure, not the architectural prior. For threat categories the same fraction is ≈ 0.01: nothing learned-specific is recovered.

## Phase 1G — Model competence audit [DIRECT — `data/quarto/model_competence_audit.json`, 2026-05-11]

`scripts/model_competence_audit.py` with N = 5,000 for A/B/C and the full 275,916 positions for D/E; trained and random conv2 models compared.

| Test | Trained | Random | Diagnosis |
|---|---:|---:|---|
| A: winning-placement acc | **0.700** | 0.223 | Strong — trained model recognises immediate wins ~3× random |
| B: losing-piece avoidance | 0.405 | 0.433 | **Indistinguishable from random** — model does NOT reason about what it hands the opponent |
| B: positions excluded (forced loss) | 307 | 607 | — |
| C: piece-sensitivity (mean distinct frac, 1.0 = fully sensitive) | 0.144 | 0.113 | Placement almost piece-invariant — H7 reconfirmed behaviorally |
| D: Q(empty) − Q(occupied) | **−0.472** | −0.001 | Trained head assigns *lower* raw Q to empty cells (only 25.3 % positive). Legality is supplied entirely by the external mask in `predict()` |
| E: entropy / uniform-cap by phase | 2.51 / 2.57 (early), 1.99 / 2.21 (mid), 1.31 / 1.43 (late) | 2.57 / 2.57, 2.19 / 2.21, 1.39 / 1.43 | Modest concentration only at mid-game; late-game still within 8 % of uniform |

### Conclusions [INFERENTIAL — 2026-05-11]

The trained champAa has learned a single asymmetric heuristic — "does this position give me an immediate win?" — and almost nothing else. It does not reason about what piece it gives, does not internalise legality, and barely concentrates probability mass even when one move is forced. This **fully explains the hawk SAE results**: threat-count features exist at moderate F1-lift (~0.12) because the model uses them to detect *its own* winning chances; threat-completability features sit at noise floor (~0.03) because the model never computes "is this piece losing for me to give." There is nothing for an SAE to find on the completability axis.

**Implications for the research program (as written 2026-05-11):**

1. **Phase 3A E2E SAE is no longer purely architectural.** The legitimate question becomes *"once we project conv2 onto Q-head-relevant directions, what fraction of variance is the immediate-win check vs everything else?"* — E2E is now diagnostic for the asymmetry itself.
2. **Matryoshka is deprioritised** — it solves feature absorption, but the missing concepts aren't being absorbed, they're never represented.
3. **Anchored / guided SAEs against `reframed_completable`** would also fail by construction; do not attempt.
4. **Phase 4 (architectural fix) is promoted from "branch" to "primary path":** the auxiliary "predict-threat-from-fc1" head during DQN training is now the most direct route to a model whose threat representations a generic SAE can recover. The current model is the bottleneck, not the SAE family.
5. **Re-narrate the dissertation contribution** from "find threat features in Quarto via SAEs" to "characterise the attack-recognition / defence-blindness asymmetry of a DQN-trained Quarto agent through behavioural + SAE evidence."

**These implications were specifically about champAa.** Phase 2B retired (1) (no E2E started yet), (4) (champS4's unified-aux objective already solved the bottleneck without an architectural fix at training time), and softened (3) (anchored SAEs against threat categories on champTa now have a real LP target — see `phase-2B.md`).

### Test definitions (kept for reproducibility)

- **A.** Positions where the offered piece can win at least one cell; measure argmax-legal placement.
- **B.** Positions with ≥ 1 safe and ≥ 1 losing piece in storage; measure argmax-legal selection after applying A's placement, exclude forced-loss positions.
- **C.** Replay each position with all 16 candidate offered pieces; count distinct argmax-legal cells.
- **D.** Mean `Q[empty] − Q[occupied]` unmasked across all positions.
- **E.** Softmax entropy of legal-masked Q, bucketed by piece count.

## Hypothesis ledger at end of Phase 1

| # | Hypothesis | Status at end of Phase 1 |
|---|---|---|
| H1 | fc1 encodes cell-level BSPs linearly but not threats | ✅ CONFIRMED — consistent across 28 SAE configs |
| H1b | Threats may be accessible in reframed basis | ⚠️ PARTIAL — count encoding 14× better but still F1 = 0.315 |
| H2 | Feature absorption | OPEN |
| H3 | Concept heterogeneity → architecture-dependent failures | ❌ WEAKENED — most architectures perform similarly |
| H5 | Feature shrinkage (L1 / ReLU) | ⚠️ PARTIAL — Gated/Vanilla score lower (Tier 3), but P-Annealing doesn't dominate Tier 1 |
| H6 | Wrong hook point — conv2 may be better | ✅ SUPPORTED — conv2 in Tier 1 with fc1 in Tier 2 |
| H7 | Offered piece is not learned | ✅ CONFIRMED — F1 = 0.667 is trivial baseline, not real signal |
| H8 | **NEW:** Threat info is spatially encoded, lost at fc1 bottleneck | ⚠️ CHAMPION-SPECIFIC — confirmed for champAa; overturned for champS4 / champTa (see `phase-2B.md`) |

## Recovery panel — interrupted by power loss (2026-04-24)

A power loss between 19:21 and 19:29 on 2026-04-24 left the A/B follow-up panel partially complete:

| Run | Trained | Evaluated | Action |
|---|:---:|:---:|---|
| `A01-random-control-s42-batchtopk-k16-exp8-fc1` | ✅ | ❌ | eval on gorilla |
| `A02-random-control-s42-topk-k32-exp8-conv2` | ✅ (5k steps per config) | ❌ | eval on gorilla |
| `B01-conv2-completion-s42-batchtopk-k64-exp4-conv2` | ✅ | ❌ | eval on gorilla |
| `B02-conv2-completion-s42-p-annealing-exp4-conv2` | ✅ | ❌ | eval on gorilla |
| `B03-conv2-completion-s42-topk-k64-exp2-conv2` | partial (4k/25k, no .pt) | — | **rerun training** |
| `B04-conv2-completion-s42-vanilla-l1_001-exp4-conv2` | not started | — | **train from scratch** |

These were folded into Phase 2A on Deep Brain (2026-05-11) without separate retry; the Phase 2A registry has the final entries.
