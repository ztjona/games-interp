# Experiment Plan — 2026-03-27

## Current State Assessment

### Clean Slate (2026-03-27)

**All prior experiments are invalidated.** The 17 runs in the legacy registry were trained on data generated with `mode_2x2=False` (no 2×2 square wins), while the model Aa_replay was trained with `mode_2x2=True`. This mismatch means the positions, activations, BSP labels, and SAE features from those runs do not reflect the game the model actually learned. All legacy artifacts have been moved to `data/quarto/legacy_mode2x2_false/`, `saes/quarto/legacy_mode2x2_false/`, and `configs/legacy_mode2x2_false/`.

**Current data:** Empty. No positions, activations, BSP labels, or SAE runs exist yet.

**Available models:**
| Model | Path | Role |
|-------|------|------|
| Aa_replay (trained, epoch 5000) | `models/quarto/20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt` | Primary — trained agent (~82% win rate vs. random) |
| Aa_replay (untrained, epoch 0) | `models/quarto/20260226_1420-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_0000.pt` | Random baseline — same architecture, no training |

**Code fixes applied this session:**
- `mode_2x2=True` in position generation
- Bot classes refactored to proper `BotAI` subclasses (compatible with quartopy `play_games()`)
- Diagonal BSP naming: `diag_main` / `diag_anti`
- Function signature fixes (`_load_shared_model`, `_make_bots`)

### Hypotheses (carried forward from legacy, to be re-tested)

These hypotheses were formulated from the legacy runs. They remain plausible but **unvalidated** — the corrected data pipeline may change results significantly (2×2 square wins add 36 BSPs and change the game's strategic landscape).

| # | Hypothesis | Status | Test |
|---|-----------|--------|------|
| H1 | **fc1 encodes cell-level BSPs linearly but not threats** — occupancy/attributes are linearly accessible (F1≈0.6-1.0); threats are non-linear conjunctions (F1≈0.02-0.11) | ✅ CONFIRMED (Phase 1A, 2026-03-31) | Linear probe baseline |
| H1b | **Wrong feature basis** — following Nanda (emergent-linear-representations-world-models), threats may be linearly accessible under a reframed basis (e.g. relative to offered piece, or threat-count encodings) | ⚠️ PARTIAL (Phase 1D, 2026-03-31) — Count basis 14× better (F1=0.315 vs 0.022), but completability near-zero. Threats are partially linearizable, not fully. | Phase 1D: Reframed linear probes |
| H2 | **Feature absorption** — hierarchical BSPs cause child features to absorb parent directions | OPEN | Feature anchoring (Phase 3), absorption analysis |
| H3 | **Concept heterogeneity** — BSP complexity varies enormously; TopK assumes uniform intrinsic dimensionality | OPEN | SpaDE (Phase 3), per-category coverage breakdown |
| H4 | ~~**Training is too short**~~ | DROPPED — non-linearity, not training length, is the bottleneck | ~~Longer training~~ |
| H5 | **Feature shrinkage** — L1/ReLU-based architectures shrink feature magnitudes | OPEN | p-annealing (Phase 2) |
| H6 | **Wrong hook point** — other layers may encode BSPs more cleanly | OPEN | Multi-hook sweep (Phase 1) |
| H7 | **Offered piece is not learned** — trained model ≈ random model on offered piece attributes (F1 0.60 vs 0.65); fc1 representations don't reflect piece selection strategy | ✅ CONFIRMED (Phase 1A, 2026-03-31) | Linear probe baseline |

---

## Proposed Phases

### Phase 0: Data Regeneration & Random Baseline
*Goal: Rebuild the data pipeline with correct `mode_2x2=True` and establish a random-network control*

All data, activations, BSP labels, and SAE runs prior to this plan were generated with `mode_2x2=False` (see legacy data in `data/quarto/legacy_mode2x2_false/`). This phase regenerates everything from scratch.

#### 0A. Position Generation (4 modes × 10k games)
- `random_v_random`, `model_v_random`, `random_v_model`, `model_v_model`
- Model: `Aa_replay` checkpoint

#### 0B. Deduplication → amalgam dataset
- Combine all raw position files into `positions-amalgam_unique.pt`

#### 0C. Activation Collection (Aa_replay)
- Hook: `fc1` (128-dim bottleneck)
- Positions: amalgam_unique

#### 0D. BSP Label Computation
- Full gorilla set (164 BSPs) from amalgam_unique

#### 0E. Random Network Baseline
**Rationale:** To distinguish SAE features that reflect learned game knowledge from artifacts of network architecture, train SAEs on a random-init model's activations using the same positions and hyperparameters. If coverage is comparable between trained and random models, the SAE is capturing geometric artifacts, not learned concepts.

- **Model:** Initial checkpoint of Aa_replay at epoch 0: `models/quarto/20260226_1420-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_0000.pt`
- **Activations:** Collect fc1 activations from the random-init model on the same amalgam_unique positions.
- **Parallel training:** For each SAE variant, train on both Aa_replay and random-init activations with identical hyperparameters. Compare coverage, FVU, dead features, and board reconstruction.
- **Decision gate:** If random-init coverage ≥ 80% of Aa_replay coverage → the SAE features are mostly architectural, not learned → investigate alternative hook points or evaluation methods.

### Phase 1: Diagnostics (low cost, high information value)
*Goal: Determine the theoretical ceiling and identify which hypotheses matter*

#### 1A. Linear Probe Baseline
**Rationale:** If logistic regression probes on raw fc1 activations can't classify BSPs well, then no SAE can either — the information simply isn't there in a linearly accessible form.

- Train L2-regularized logistic regression probes on raw fc1 activations for all 164 BSPs
- Use the same amalgam dataset and train/test split
- Report per-BSP F1 and per-category mean F1
- Compare: if probe coverage >> 0.20, the SAE is the bottleneck; if probe coverage ≈ 0.20, fc1 is the bottleneck

**Literature basis:** Karvonen et al. (sae-evaluation-metrics) compare SAEs vs linear probes on board game models and find SAEs don't close the gap. Chanin et al. (absortion-features) use probes as performance ceiling for absorption detection.

**Implementation:** Write a simple `scripts/linear_probe_baseline.py` script. ~1 hour to implement and run.

**Results (2026-03-31):**

| Category (n) | Trained F1 | Random F1 | Δ |
|---|---:|---:|---:|
| cell_occupancy (16) | 0.998 | 0.564 | +0.434 |
| cell_attribute (64) | 0.611 | 0.281 | +0.330 |
| game_phase (3) | 0.909 | 0.545 | +0.364 |
| offered_piece (4) | 0.602 | 0.648 | -0.046 |
| global (1) | 0.698 | 0.564 | +0.134 |
| threat_line (40) | 0.022 | 0.000 | +0.022 |
| threat_square_2x2 (36) | 0.113 | 0.000 | +0.113 |
| **Overall** | **0.402** | **0.194** | **+0.208** |

**Key findings:**
- **Cell-level concepts are linearly accessible** — occupancy near-perfect, attributes solid
- **Threats are NOT linearly accessible** — despite the model needing them to win 82% of games
- **Offered piece is NOT learned** — random model matches trained (H7 confirmed)
- The model likely encodes threats non-linearly OR in a different basis (motivates Phase 1D)

#### 1B. Per-Category Coverage Breakdown
**Rationale:** The 0.20 mean coverage might hide extreme variance across BSP categories. Threats (complex conjunctions) may have near-zero coverage while occupancy (simple binary) may be well-covered.

- Recompute coverage from existing eval results, broken down by BSP category:
  - cell_occupancy (16 BSPs)
  - cell_attribute (64 BSPs)
  - threat_line (40 BSPs)
  - threat_square_2x2 (36 BSPs)
  - offered_piece (4 BSPs)
  - game_phase (3 BSPs)
  - global (1 BSP)
- This requires no new training — just reanalyzing existing eval caches

**Literature basis:** Hindupur et al. (projecting-assumptions-duality-sparse-autoencoders) show that concept heterogeneity causes architecture-dependent failures.

#### 1C. Multi-Hook Quick Sweep
**Rationale:** fc1 is the "recommended" hook but hasn't been compared. conv2 (32×4×4=512 dim) or other layers might encode BSPs more cleanly.

- Train a quick TopK-k32-exp8 SAE on conv2 activations (need to first collect activations)
- Evaluate with gorilla BSP set
- If coverage is dramatically different, redirect all future experiments to the better hook

**Literature basis:** Residual stream analysis (residual-stream-analysis-multi-layer) shows features are layer-specific per-token.

#### 1D. Reframed Linear Probes (Nanda-inspired)
**Rationale:** Nanda et al. (emergent-linear-representations-world-models) showed that OthelloGPT's board state appeared non-linear under BLACK/WHITE basis but was perfectly linear under the model's natural MINE/YOURS basis. Our fc1 threat probes score near-zero (F1≈0.02), but the model wins 82% of games — the information must be present. The apparent non-linearity may be an artifact of the wrong feature basis.

**Approach:** Design alternative probe targets that reframe threat BSPs to match fc1's likely encoding:
1. **Attribute-count encoding:** Instead of binary `row_0_threat_tall`, probe for the *count* of TALL pieces in row 0 (0-4). The model may represent "3 TALL in a row" as a count, not as a boolean conjunction.
2. **Offered-piece-relative threats:** A threat only matters if the offered piece can complete it. Reframe: `row_0_completable_tall` = 1 iff row 0 has 3 TALL + 1 empty AND the offered piece is TALL.
3. **Any-attribute threats per line:** Instead of per-attribute threats, probe for `row_0_has_any_threat` (OR across all 4 attributes). The model may collapse attribute-specific threats into a single "this line is dangerous" signal.
4. **Distance-to-win encoding:** Number of threats on the board (0, 1, 2, ...).

**Implementation:** Extend `compute_bsp_labels.py` to generate a "reframed" label set alongside gorilla. Then re-run linear probes on these new targets.

**Expected outcome:** If reframed probes achieve F1 > 0.5 on threats, the model *does* linearly encode threat information — just not in our original basis. This would make SAE feature recovery feasible and redirect the interpretation of SAE features.

**Cost:** Low — label computation + probe training (~2h).
**Information value:** Very high — could completely change our understanding of what fc1 encodes.

**Results (2026-03-31):**

BSP set: **hawk** (92 BSPs across 4 reframed categories). Probes run on exactly the same fc1 activations as gorilla.

| Category (n) | Trained F1 | Random F1 | Δ |
|---|---:|---:|---:|
| reframed_count (40) | 0.315 | 0.000 | +0.315 |
| reframed_completable (40) | 0.003 | 0.000 | +0.003 |
| reframed_any_threat (10) | 0.086 | 0.000 | +0.086 |
| reframed_global (2) | 0.418 | 0.231 | +0.187 |
| **Overall (hawk)** | **0.157** | **0.005** | **+0.152** |

**Cross-comparison with gorilla threat categories on the same activations:**

| Probe Target | Gorilla threat_line F1 | Hawk reframed_count F1 | Ratio |
|---|---:|---:|---:|
| Trained | 0.022 | 0.315 | **14× improvement** |
| Random | 0.000 | 0.000 | — |

**Key findings:**
- **Reframed counts are partially linearly accessible** — F1=0.315 (vs 0.022 for gorilla threat_line). The model encodes "how many matching pieces in a line" more linearly than the boolean "is there a threat?"
- **Completability is NOT linearly accessible** — F1=0.003, near zero. The conjunction of "threat exists AND offered piece completes it" is genuinely non-linear in fc1. This is an AND over two different representation subspaces (board state × offered piece), confirming the model doesn't fuse these in fc1.
- **Any-threat (OR reduction)** — F1=0.086, modest improvement over per-attribute threats (0.022). OR-aggregation across attributes helps but doesn't fully linearize.
- **Global threat existence** — F1=0.418, best in hawk. The scalar "is there any threat on the board?" is moderately accessible.
- **Random baseline near zero across all hawk categories** — confirms these are genuinely learned representations, not architectural artifacts.

**Decision gate assessment:** Reframed probes did NOT reach the F1 > 0.5 threshold to declare threats "linearly accessible in a different basis." Result is **intermediate**: count-based reframing reveals partial linear structure (14× improvement), but threats remain fundamentally harder than cell-level properties. This suggests:
1. fc1 encodes partial threat information (attribute counts per line) but not the full conjunction
2. The conjunction (3-of-4 matching + one empty) likely requires non-linear computation
3. SAE features may capture the count-based intermediate better than final threat booleans
4. **Implication for SAE evaluation:** The hawk reframed_count BSPs (F1=0.315) represent a realistic SAE target — non-trivial but achievable. Evaluating SAEs on hawk alongside gorilla will give a more nuanced picture.

**Update (2026-04-06):** Hawk was expanded to 173 BSPs by adding 2×2 square reframings, and the fc1 probes were rerun on the updated set. The trained model reaches overall F1=0.200 vs 0.005 for the random control; the new square-count category is substantially linearly accessible (F1=0.455), while square completability remains low (F1=0.027). This preserves the original conclusion: fc1 contains partial count-like threat structure but not the full conjunction.

---

### Phase 1E: Anakin Sweep — Systematic SAE Architecture Comparison (experiment "anakin")
**Created:** 2026-04-01  
**Status:** ✅ DONE (2026-04-01 to 2026-04-02). 28/33 configs completed, 5 timed out on GPU 0.

**Execution summary:** Run across 3 GPUs on 2026-04-01. GPUs 1 & 2 completed all 22 configs in ~90 min each. GPU 0 processed 6/11 configs (~9 hours) — 5 conv2 and exp=16 configs timed out at the 1-hour-per-config limit due to GPU 0 being significantly slower. All 5 partials had converged FVU (relative change <3.5% in last 3 logged steps) at 70–94% of training, so they are informative without re-running.

**Results:** See RESEARCH-STATUS.md for full analysis. Key conclusion: **architecture choice matters less than hook point (conv2 > fc1) and staying in the right sparsity regime (L0=16–64 for best coverage).**

**Rationale:** Phases 1A–1D established the linear probe ceiling and the fc1 representation structure. Now we need to systematically test whether any SAE architecture can *match or exceed* probe performance on the accessible BSPs, and *discover structure* in the less-accessible ones. The existing SAE baseline (topk-k32-exp8-fc1) has coverage=0.30 — below the probe ceiling of 0.40 for gorilla. A comprehensive sweep over architectures, sparsity levels, expansion factors, and hooks is needed before investing in novel approaches (Phase 3).

**Design:** 33 configs across 11 tiers, covering:
- **6 architectures:** vanilla, topk, batchtopk, gated, jumprelu, p-annealing
- **2 hooks:** fc1 (128-dim, 23 configs) and conv2 (512-dim, 10 configs)
- **3 expansion factors:** 4, 8, 16 for fc1; 2, 4, 8 for conv2
- **Sparsity sweep:** k ∈ {16, 32, 64, 128} for topk; k ∈ {16, 32, 64} for batchtopk; l0 ∈ {32, 64, 128} for jumprelu
- **Seed stability:** 3 seeds (42, 43, 44) on topk-k32-exp8-fc1
- All configs: 25000 training steps, batch_size=4096, lr=3e-4, gorilla BSP evaluation

**Key conv2 note:** All conv2 configs use the 512-dim full-spatial representation (`conv2_512_amalgam_activations.pt`, 275K × 512), not the per-cell representation. This means d_dict at exp=4 is 2048 — sparsity ratios are matched to fc1 (k=64/2048 ≈ k=32/1024 ≈ 3.12%).

**Infrastructure:**
- Config generator: `scripts/generate_anakin_configs.py`
- Sweep orchestrator: `run_sweep.py` (supports --split for multi-GPU)
- Pre-flight validator: `validate_sweep.py --smoke-test`
- Configs: `configs/anakin/*.yaml` (33 files)

**Execution (3 GPUs in parallel):**
```bash
python validate_sweep.py --smoke-test
# Then in 3 separate terminals:
python run_sweep.py --configs=configs/anakin --gpu=0 --split=1/3 --eval --skip-existing
python run_sweep.py --configs=configs/anakin --gpu=1 --split=2/3 --eval --skip-existing
python run_sweep.py --configs=configs/anakin --gpu=2 --split=3/3 --eval --skip-existing
```

**Decision gates after sweep:**
- ~~If max coverage > 0.35 → identify best architecture, sweep hyperparams further~~ → **max coverage = 0.338 (close). BatchTopK-k16 and conv2-TopK-k64 are best. No further sweeping needed — architecture is not the bottleneck.**
- ~~If max coverage < 0.25 across all archs → SAE approach may be limited~~ → **Not triggered. SAEs work, just not for threats.**
- ~~If conv2 >> fc1 → redirect all future work to conv2 hook~~ → **conv2 has a real edge (Tier 1 vs Tier 2) but the gap is modest (~0.03 coverage). conv2 experiments should be prioritized but fc1 is not worthless.**
- ~~If p-annealing dominates → confirms Karvonen et al.~~ → **P-annealing is in Tier 2 (bulk), not Tier 1. Does not dominate for Quarto, contradicting Karvonen et al.'s Othello results.**
- ~~If seed stability < 0.6 cosine → consider Archetypal SAE~~ → **Coverage stability is good (σ=0.004). Decoder cosine stability still un-measured.**
- ~~Compare per-category: which architecture best recovers threats vs cell properties~~ → **No architecture recovers threats. The bottleneck is the network's non-linear encoding of threats, not the SAE architecture.**

**Decision gate outcomes — new directions prompted:**
1. **H8 (spatial threat encoding):** The model processes offered piece + board through conv layers. Threats are spatial patterns (3-in-a-line). Conv2 outperforms fc1 for spatial BSPs. Linear probe on conv2 activations would reveal whether threat info exists spatially before the fc1 bottleneck destroys it. → **Phase 1F**
2. **Offered_piece F1=0.667 is a trivial baseline artifact.** The offered piece is entangled through 2 conv layers by fc1. Not a useful evaluation signal. → **Drop offered_piece from focus; probe fc_in_piece hook if piece identity matters.**
3. **Coverage metric has a noise floor.** With 76 threat BSPs at ~0.07 F1, they drag the mean. Per-category coverage is the only meaningful lens.

### Phase 1F: Conv2 Linear Probe — Test Spatial Threat Encoding (H8)
**Created:** 2026-04-06
**Status:** ✅ DONE (2026-04-24)

**Rationale:** The Anakin sweep confirms threats are undetectable at fc1 across ALL architectures. The linear probe on fc1 also gets F1≈0.02 on threat_line. But the model wins 82% of games, so it MUST represent threats somewhere. Conv2 (32×4×4=512d) retains spatial structure — and threats are inherently spatial (3 aligned pieces + 1 empty cell). If threats are linearly accessible at conv2 but not fc1, this proves the fc1 bottleneck (128-d) collapses spatial threat information into nonlinear representations.

**Method:**
- Ran `linear_probe_baseline.py` on `data/quarto/conv2_512_amalgam_activations.pt` with gorilla BSP labels
- Compared threat_line and threat_square F1 between conv2 and fc1 probes
- Collected random-model conv2 activations and ran the same probe as a control

**Results:**

| Category | fc1 probe | conv2 probe | conv2 random |
|---|:---:|:---:|:---:|
| cell_occupancy | 0.998 | 1.000 | 0.829 |
| cell_attribute | 0.611 | 0.971 | 0.782 |
| threat_line | 0.022 | 0.502 | 0.019 |
| threat_square_2x2 | 0.113 | 0.680 | 0.016 |
| offered_piece | 0.602 | 0.789 | 0.718 |
| global | 0.698 | 0.694 | 0.634 |
| game_phase | 0.909 | 0.936 | 0.685 |
| **Overall** | **0.402** | **0.789** | **0.428** |

**Interpretation:**
- **H8 confirmed.** Threat information is linearly accessible in conv2 and largely gone by fc1.
- The trained-vs-random gap is decisive for threat BSPs (`threat_line`: 0.502 vs 0.019, `threat_square_2x2`: 0.680 vs 0.016), so this is learned spatial structure rather than a generic convolutional prior.
- Some categories remain high even in the random control (`cell_occupancy`, `cell_attribute`, `offered_piece`, `global`), so raw overall coverage is no longer enough on conv2. Category-wise comparisons against the random control are required.
- **Research consequence:** do not spend the next cycle on another broad sweep of the same unsupervised SAE families. The main gap is now between the conv2 probe ceiling and the current SAE recoverability.

**Cost:** Low (~1h, completed).
**Information value:** Critical — it decisively redirects future work toward conv2-based SAE recovery and away from fc1 threat analysis.

**Literature basis:** Karvonen et al. (sae-evaluation-metrics) found p-annealing best for board games; Bricken et al. and Gao et al. showed TopK outperforms vanilla on LLMs; Hindupur et al. (projecting-assumptions-duality) predict architecture-dependent failures on heterogeneous concept sets.

---

### Phase 2: Training Improvements (medium cost)
*Goal: Address feature shrinkage, training length, and the one untested architecture*

#### 2A. p-Annealing (experiment "darwin")
**Rationale:** p-annealing is the technique most directly validated on board game SAEs (Karvonen et al. achieve best coverage with it). It's already implemented but never run. The Lp norm with p→0.2 approximates L0, reducing feature shrinkage that plagues vanilla/ReLU SAEs.

- Config already exists: `connor-p-annealing.yaml`
- **Modify:** increase `num_batches` to 15000 (p needs at least `p_anneal_steps` to complete annealing, currently set to 15000 in the code)
- Run with expansion 8 and expansion 16
- Test multiple `lp_weight` values: 1e-3, 5e-3, 1e-2

**Key concern:** Current implementation uses fixed `lp_weight`. Karvonen et al. use adaptive coefficient annealing (rescaling λ at each step to keep penalty magnitude constant as p changes). Check if this is implemented; if not, it may need to be added.

#### ~~2B. Longer Training~~ — DROPPED
**Dropped (2026-03-31):** Phase 1A results show the bottleneck is non-linear encoding of threats, not insufficient training. Longer training would only improve already-high categories (occupancy, attributes) with diminishing returns.

#### 2C. Higher Expansion with Dead Feature Mitigation
**Rationale:** Expansion 16 failed because of 95%+ dead features. The aux loss weight or mechanism may be insufficient. Before giving up on scaling width, try stronger dead feature revival.

- TopK-k32 with expansion 16, aux_loss_weight increased from 0.01 to 0.1
- BatchTopK-k32 with expansion 16 (BatchTopK's built-in threshold adjustment may handle dead features better)
- If dead features remain >80%, the issue is architectural, not hyperparameter

---

### Phase 3: Novel Approaches (higher cost, higher novelty)
*Goal: Address fundamental limitations identified in Phase 1*

#### 3A. Guided SAE (G-SAE) — Semi-Supervised Training
**Rationale:** Härle et al. (Feature-Monosemanticity-Score-GSAE) show that conditioning a subset of SAE features on known concept labels nearly doubles monosemanticity (FMS: 0.27→0.52). We have 164 BSP labels — this is a natural fit.

- **Architecture:** Standard TopK SAE with an additional BCE loss on the first 164 features, each supervised against a BSP
- **Training loss:** L_total = L_reconstruction + α·L_bce where L_bce matches features[0:164] against BSP labels
- **Expected benefit:** Forces 164 features to align with BSPs directly; remaining features discover unsupervised concepts
- **Risk:** May overfit to BSP supervision; need to validate that unsupervised features still capture useful information
- **Novelty for thesis:** Applying G-SAE to board games (originally designed for LLMs) is novel and directly addresses the known/unknown concept distinction from Peng et al. (use-sparse-autoencoders-discover-unknown)

**Implementation:** Extend `BaseSAE` with a `GuidedSAE` class that takes BSP labels as auxiliary input during training.

#### 3B. Feature Anchoring (from Unified Theory)
**Rationale:** Tang et al. (unified-theory-sparse-dictionary-learning) prove that SDL optimization is fundamentally underdetermined and propose feature anchoring — constraining encoder/decoder directions toward known semantic directions — as an architecture-agnostic fix.

- **Method:** Add an L2 penalty pulling k encoder rows and decoder columns toward BSP directions (obtained from linear probes in Phase 1A)
- **Advantage over G-SAE:** Softer constraint; doesn't require BSP labels at every training step, only anchor directions
- **Requires:** Phase 1A results (probe weight vectors as anchors)

#### 3C. SpaDE — Distance-Based SAE for Heterogeneous Concepts
**Rationale:** Hindupur et al. (projecting-assumptions-duality-sparse-autoencoders) show that TopK assumes angular separability with uniform intrinsic dimensionality — but Quarto BSPs have wildly varying complexity (occupancy vs. threats). SpaDE uses Euclidean distances to prototypes + sparsemax, enabling adaptive sparsity per concept.

- **Implementation:** Replace encoder with distance computation to learned prototypes; apply sparsemax
- **Expected benefit:** Features for simple concepts (occupancy) use few active features; features for complex concepts (threats) use more
- **Risk:** SpaDE is only validated on synthetic/vision tasks; scaling to board game activations is untested
- **Novelty:** First application of SpaDE to board game interpretability

#### 3D. End-to-End SAE Training
**Rationale:** Braun et al. (identifying-functionally-important-features-end) show that local SAEs learn features important for dataset statistics, not network computation. E2E training against the model's output distribution (KL divergence) finds functionally important features with 2× fewer active features.

- **Method:** Replace MSE loss with KL divergence between the model's Q-value outputs with and without SAE activations
- **Advantage:** Features are guaranteed to be functionally relevant to the model's decisions
- **Risk:** 2–3.5× compute cost; requires differentiable path from SAE output through the rest of the network

---

### Phase 4: Evaluation Improvements
*Goal: Better metrics and deeper understanding*

#### 4A. Absorption Detection
**Rationale:** Chanin et al. (absortion-features) show feature absorption is universal in SAEs and worsens with sparsity. Our hierarchical BSPs (occupancy → attributes → threats) are a prime target.

- For each BSP, identify false-negative cases where the best-matching SAE feature fails to fire but a linear probe succeeds
- Use integrated gradients to find which *other* feature absorbed the BSP direction
- Report absorption rate per BSP category

#### 4B. Feature Stability Across Seeds
**Rationale:** Fel et al. (archetypal-sae-adaptive) show SAEs have ~0.5 cosine stability across seeds. We've only trained single seeds. If Quarto SAE features are unstable, all feature-level claims are unreliable.

- Train 3 seeds of the best config
- Compute pairwise cosine stability of decoder dictionaries (Hungarian alignment)
- If stability < 0.6, consider Archetypal SAE constraints

#### 4C. Discovery Evaluation
**Rationale:** Peng et al. (use-sparse-autoencoders-discover-unknown) argue SAEs excel at discovering unknown concepts, not acting on known ones. Our BSP evaluation is entirely in the "known concept" regime.

- Examine top-activating features that don't correspond to any BSP
- Characterize what game properties they encode (piece combinations, strategic patterns, temporal features)
- This is qualitative but could reveal genuinely novel game concepts

---

## Execution Priority

| Priority | Experiment | Cost | Information Value | Dependency | Status |
|----------|-----------|------|-------------------|------------|--------|
| **0** | 0. Data regeneration + random baseline | Low-Medium | **Prerequisite** | None | ✅ DONE |
| **1** | 1A. Linear probe baseline | Low (1h) | **Critical** | Phase 0 | ✅ DONE |
| **2** | 1B. Per-category coverage | Negligible | **High** | Phase 0 | ✅ DONE (integrated into sae_eval.py) |
| **3** | 1C. Multi-hook sweep (conv2) | Medium | **High** — may redirect all work | Phase 0 | SAE trained, needs re-eval |
| **4** | **1D. Reframed linear probes** | **Low (2h)** | **Very High** — Nanda-inspired, could unlock threats | Phase 1A | ✅ DONE — partial success (count F1=0.315, completable≈0) |
| **5** | **1E. Anakin sweep** | **Medium (4 days GPU)** | **Very High** — systematic arch comparison, hook comparison, sparsity/expansion sweep | Phase 0, 1A | ✅ DONE — 28/33 completed, 5 partial (converged). See RESEARCH-STATUS.md for full analysis |
| **5.1** | **1F. Conv2 linear probe** | **Low (1h)** | **Very High** — tests H8, determines if threats are spatially encoded before fc1 collapses them | Phase 0 | ✅ DONE — H8 confirmed |
| **5.1b** | **A/B follow-up panel recovery (B03+B04 retrain, A01/A02/B01–B04 gorilla eval)** | **Low (3–4h)** | **Required** — completes the conv2-completion + random-control panel that the 2026-04-24 power loss interrupted | A/B configs in `configs/followup/` | ✅ DONE (2026-04-28) |
| **5.4** | **Phase 2A: Conv2 arch sweep (C/D/F/G campaigns, 33 configs)** | **High (100–250 CPU-h)** | **Very High** — first full architecture comparison on conv2-512; establishes whether BatchTopK-k16 advantage holds at conv2 and whether exp16 helps | C10/C11 excluded (gated L0 locked) | **IN QUEUE — re-run with fixed timeout (2026-05-04)** |
| **5.2** | **1G. Hawk 173 SAE evaluation** | **Low (2h)** | **Very High** — evaluate best SAEs on reframed line+square threat BSPs now that conv2 threat accessibility is known | Phase 1D, 1F, 5.4 | gated on 5.4 |
| **5.2b** | **1G'. Conv2 linear probe on hawk_173** | **Low (1h)** | **Very High** — establishes the conv2 ceiling for reframed threats (currently only fc1-hawk is known) | Phase 1F | **NEXT** |
| **5.3** | **1H. Conv2 feature reuse / absorption analysis** | **Low-Medium** | **High** — determines whether multiple BSPs collapse onto shared SAE features despite strong probe accessibility | 1E, 1F, 5.4 | gated on 5.4 |
| **6** | 2A. p-Annealing | Medium (2-3h) | **High** — directly validated on board games | Phase 0 | ⊂ Anakin sweep (included in Tier A) |
| **7** | 3A. Guided SAE (G-SAE) | High | **Very High** — novel, exploits BSP labels | 1A, 1E, 1F, 1G, 1H | gated on 1G+1H |
| **8** | 2C. Higher expansion + aux | Medium | Medium | None | ⊂ Anakin sweep (Tiers E, F) |
| **9** | 3B. Feature anchoring | High | High | 1A, 1G, 1H | gated on 1G+1H |
| **10** | 4A. Absorption detection | Medium | High — explains failure mode | 1E | superseded by 1H |
| ~~11~~ | ~~2B. Longer training~~ | ~~Medium-High~~ | ~~Medium~~ | | DROPPED |
| **12** | 3C. SpaDE | Very High | High but risky | 1B | |
| **13** | 3D. End-to-end SAE | Very High | High | Model architecture access | |
| **14** | 4C. Discovery evaluation | Low (qualitative) | Medium | Good SAE from 1E | |
| **15** | 4B. Stability across seeds | Medium (3× train) | Medium | Best config known | ⊂ Anakin sweep (Tier H) |
| ~~D1~~ | ~~Broad unsupervised architecture sweep (re-run)~~ | ~~Medium~~ | ~~Low~~ | — | **DEPRIORITIZED 2026-04-27** — Anakin showed σ=0.004 across seeds; the gap is not the architecture |
| ~~D2~~ | ~~fc1 deep-dive on threats~~ | ~~Low~~ | ~~Low~~ | — | **DEPRIORITIZED 2026-04-27** — Phase 1F redirected threat work to conv2 |
| ~~D3~~ | ~~`offered_piece` as a coverage signal~~ | — | — | — | **DEPRIORITIZED 2026-04-27** — F1=0.667 is the trivial all-positive baseline (P=0.5, R=1.0); keep in per-category breakdowns for transparency, exclude from headline rankings |

**Naming note (2026-04-24):** All new follow-up runs should use experiment IDs of the form `{Major}{Minor}-{tag}-s{seed}`. For the current queue, `A01`/`A02` are random-model controls and `B01`–`B04` are conv2 completion reruns. Do not rely on YAML filenames alone; the checkpoint stem is built from the config's `experiment:` field.

---

## Key Literature Cross-References

| Paper Tag | Key Insight for This Plan |
|-----------|--------------------------|
| sae-evaluation-metrics | Coverage + board reconstruction metrics; p-annealing; board game SAE benchmark (our gold standard comparison) |
| emergent-linear-representations-world-models | **Linear representations emerge under the right basis.** OthelloGPT appeared non-linear under BLACK/WHITE but was linear under MINE/YOURS. Motivates Phase 1D: reframing threat BSPs may reveal linear structure in fc1 |
| Feature-Monosemanticity-Score-GSAE | G-SAE nearly doubles monosemanticity with supervised conditioning; FMS metric |
| unified-theory-sparse-dictionary-learning | SDL is underdetermined; feature anchoring restores identifiability; explains dead features and absorption theoretically |
| projecting-assumptions-duality-sparse-autoencoders | Architecture↔geometry duality; TopK fails on heterogeneous concepts; SpaDE proposal |
| decomposing-dark-matter-sparse-autoencoders | Nonlinear error floor is irreducible by scaling; motivates architectural innovation |
| absortion-features | Feature absorption is universal; hierarchical concepts are worst affected; absorption rate increases with sparsity |
| archetypal-sae-adaptive | SAE features unstable across seeds; Archetypal constraint improves stability + concept recovery |
| measuring-sensitivity | Feature sensitivity declines with width; higher L0 → better sensitivity |
| use-sparse-autoencoders-discover-unknown | SAEs are better at discovery than acting on known concepts; our BSP eval is all "known concept" regime |
| resurrecting-salmon-rethinking-mechanistic-interpr | Domain-specific SAEs explain 15-20% more variance; dark matter framework as diagnostic |
| identifying-functionally-important-features-end | E2E SAEs find functionally important features; 2× fewer active features needed |
| have-covered-all-bases-here | ReasonScore for identifying behavior-linked features; model diffing methodology |

---

## Decision Gates

- **After Phase 1A:** ✅ RESOLVED — Linear probe coverage = 0.40 on fc1. Cell-level BSPs are linearly accessible (F1 0.6-1.0). Threats are not (F1 ≈ 0). fc1 is viable for cell-level concepts; threat coverage requires non-linear disentangling or basis reframing.
- **After Phase 1C:** If conv2 coverage on threats >> fc1 → spatial layer may encode threats more explicitly → redirect threat-focused work to conv2
- **After Phase 1D:** ⚠️ RESOLVED PARTIAL (2026-03-31) — Reframed count probes reached F1=0.315 (14× improvement) but did not cross 0.5 threshold. Completability near-zero. Threats are **partially linearizable** via count-based reframing. Implication: SAEs should be evaluated on both gorilla (hard booleans) and hawk (reframed counts) to capture the intermediate structure. G-SAE / E2E approaches remain valuable for capturing the non-linear conjunction residual.
- **After Phase 2A:** If p-annealing coverage < 0.25 → feature shrinkage alone doesn't explain the gap → prioritize Phase 3
- **After Phase 2A:** If p-annealing coverage > 0.35 → p-annealing is the answer → sweep hyperparameters extensively
- **After Phase 3A:** If G-SAE coverage > 0.40 → supervised conditioning works → investigate how much supervision is needed (partial labels, fewer BSPs)
