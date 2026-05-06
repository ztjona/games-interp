# Quarto SAE Research — Quick Reference

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

### Phase 2A: Conv2 Architecture Sweep (launched 2026-04-29, failed 2026-05-04)

**Motivation:** The "architecture doesn't matter" conclusion from Anakin was valid only for fc1. On conv2: only 3 of 6 architectures have ever been tested (topk, jumprelu, gated); only 2 runs were at exp8; the fc1 winner (BatchTopK-k16) has never been run on conv2; and SAE/LP efficiency at conv2 is only 42% vs 84% at fc1. A systematic sweep is warranted.

**Planned campaigns (all failed — no checkpoints saved):**

| Campaign | Purpose | Configs | Status |
|----------|---------|---------|--------|
| C (C01–C14) | conv2-512 all 6 architectures at exp8, seed=42 | 14 | ❌ No checkpoint saved |
| D (D01–D04) | conv2-512 TopK+BatchTopK at exp16 (larger dicts) | 4 | ❌ No checkpoint saved |
| F (F01–F10) | Seed stability: seeds 43+44 for 5 leading conv2 archs | 10 | ❌ No checkpoint saved |
| G (G01–G05) | Random-model controls on conv2-512 | 5 | ❌ No checkpoint saved |

**Execution post-mortem (2026-05-04):** All 33 runs were killed by a hard `timeout=3600s` (1h) in `run_sweep.py`. CPU training on `conv2_512_amalgam_activations.pt` takes 2–7+ hours per run (confirmed from B-series: B01=2.03h, B03=6.08h, B04=7.37h). Partial JSONL metrics exist; no `.pt` checkpoints; no eval results. **Fix applied:** timeout now defaults to 86400s (24h) and is configurable via `--timeout=<sec>`.

**Key conclusions from partial logs:**

- **Gated (C10/C11) — remove from re-run queue.** L0 ≈ 1050 from step 500 and never moves. This mirrors the completed fc1 gated trajectory: L0 stabilised by step 500 and changed by <3 units over 25,000 steps for both l1_002 and l1_0005 variants. The architecture reaches a fixed equilibrium (~25% activation density) regardless of L1 strength in this regime. More training cannot fix this; a 2-order-of-magnitude stronger penalty would likely destroy coverage instead.

- **G series (random controls) — deprioritise but keep.** FVU=0.62 (G02) and 0.34 (G03) after 500–1500 steps reflects lower variance in random-model activations — expected. The trained-vs-random coverage comparison at fc1 is already established via A01/A02. G-series confirms the same sanity-check at conv2 but is lower priority than completing C/D/F.

**Re-run commands (with fixed timeout):**
```bash
# GPU 1 — Campaign C (skip C10/C11 — gated, known failure)
python run_sweep.py --configs=configs/followup --tier=C --gpu=1 --eval --skip-existing
# GPU 2 — Campaign F
python run_sweep.py --configs=configs/followup --tier=F --gpu=2 --eval --skip-existing
# GPU 0 — Campaigns D and G
python run_sweep.py --configs=configs/followup --tier=D,G --gpu=0 --eval --skip-existing
```

**Key questions still open:**
1. Does BatchTopK-k16 dominate at conv2 the way it does at fc1?
2. Does exp16 (d_dict=8192) improve coverage for 512d inputs?
3. Are threat-recovery results stable across seeds?

### Next Steps After Phase 2A Completes

1. **Hawk_173 SAE evaluation** on the best conv2 checkpoint from Phase 2A. Tests whether threat recovery improves with richer dictionaries.
2. **Feature reuse / absorption diagnostic** on the best conv2 SAE — uses `best_feature_per_bsp` from the matching cache. Quantifies whether conv2 SAE coverage stalls at ~0.33 due to absorption vs. true information ceiling.
3. **Conditional:** if unsupervised SAEs still leave large conv2 probe gap, run Guided/anchored SAE pilot at conv2 (Phase 3A/3B in EXPERIMENT-PLAN).

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
