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

| # | Hypothesis | Test |
|---|-----------|------|
| H1 | **fc1 doesn't encode BSPs linearly** — the 128-dim bottleneck may lack capacity or encode information nonlinearly | Linear probe baseline (Phase 1) |
| H2 | **Feature absorption** — hierarchical BSPs cause child features to absorb parent directions | Feature anchoring (Phase 3), absorption analysis |
| H3 | **Concept heterogeneity** — BSP complexity varies enormously; TopK assumes uniform intrinsic dimensionality | SpaDE (Phase 3), per-category coverage breakdown |
| H4 | **Training is too short** — 5000 batches may be insufficient for feature crystallization | Longer training (Phase 2) |
| H5 | **Feature shrinkage** — L1/ReLU-based architectures shrink feature magnitudes | p-annealing (Phase 2) |
| H6 | **Wrong hook point** — other layers may encode BSPs more cleanly | Multi-hook sweep (Phase 1) |

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

#### 2B. Longer Training (experiment "edison")
**Rationale:** 5000 batches may be insufficient. Feature crystallization in SAEs often requires extended training, especially for TopK where dead feature revival depends on aux loss.

- Retrain the best-performing config (arnold_beta-jumprelu-t32, which has best coverage) with:
  - 25,000 batches (5× current)
  - 50,000 batches (10× current)
- Monitor metrics at checkpoints to detect early saturation vs. continued improvement
- If improvement plateaus early, training length is not the bottleneck

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

| Priority | Experiment | Cost | Information Value | Dependency |
|----------|-----------|------|-------------------|------------|
| **0** | 0. Data regeneration + random baseline | Low-Medium | **Prerequisite** — all subsequent work depends on this | None |
| **1** | 1A. Linear probe baseline | Low (1h) | **Critical** — determines if fc1 is viable | Phase 0 |
| **2** | 1B. Per-category coverage | Negligible | **High** — identifies which BSPs fail | Phase 0 |
| **3** | 2A. p-Annealing | Medium (2-3h train) | **High** — directly validated on board games | Phase 0 |
| **4** | 1C. Multi-hook sweep | Medium (collect + train) | **High** — may redirect all work | None |
| **5** | 2B. Longer training | Medium-High | Medium — may just confirm saturation | None |
| **6** | 3A. Guided SAE (G-SAE) | High (implement + train) | **Very High** — novel, exploits BSP labels | 1A |
| **7** | 2C. Higher expansion + aux | Medium | Medium | None |
| **8** | 3B. Feature anchoring | High (implement) | High | 1A |
| **9** | 4A. Absorption detection | Medium (implement) | High — explains failure mode | 1A |
| **10** | 4B. Stability across seeds | Medium (3× train) | Medium | Best config known |
| **11** | 3C. SpaDE | Very High (new arch) | High but risky | 1B confirms heterogeneity |
| **12** | 3D. End-to-end SAE | Very High (new training loop) | High | Model architecture access |
| **13** | 4C. Discovery evaluation | Low (qualitative) | Medium | Good SAE available |

---

## Key Literature Cross-References

| Paper Tag | Key Insight for This Plan |
|-----------|--------------------------|
| sae-evaluation-metrics | Coverage + board reconstruction metrics; p-annealing; board game SAE benchmark (our gold standard comparison) |
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

- **After Phase 1A:** If linear probe coverage < 0.30 on fc1 → abandon fc1, pivot to conv2 or multi-hook approach
- **After Phase 1A:** If linear probe coverage > 0.50 on fc1 → fc1 is viable, the SAE is the bottleneck → proceed with Phase 2+3
- **After Phase 2A:** If p-annealing coverage < 0.25 → feature shrinkage alone doesn't explain the gap → prioritize Phase 3
- **After Phase 2A:** If p-annealing coverage > 0.35 → p-annealing is the answer → sweep hyperparameters extensively
- **After Phase 3A:** If G-SAE coverage > 0.40 → supervised conditioning works → investigate how much supervision is needed (partial labels, fewer BSPs)
