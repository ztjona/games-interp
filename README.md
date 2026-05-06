# Games-Interp: Mechanistic Interpretability of Board Game Neural Networks

PhD research project using Sparse Autoencoders (SAEs) to extract interpretable features
from board game neural networks (Quarto, Othello, Tic-tac-toe).

## Status (April 2026)

**Phase:** Post-regeneration Quarto SAE work is active. Phase 1F confirmed that threat information is linearly accessible in conv2 and largely lost by fc1. The next step is to test whether the current SAEs recover that structure, especially on hawk_173.

| Milestone | Status |
|-----------|--------|
| Quarto CNN trained (Aa_replay) | Done |
| Position collection (4 opponent modes) | Done — 275,916 unique positions |
| BSP computation | Done — gorilla_164 + hawk_173 on disk |
| SAE library (6 architectures) | Done — vanilla, topk, batchtopk, gated, jumprelu, p-annealing |
| fc1 linear probes | Done — gorilla + hawk_173 + random controls |
| conv2 hook comparison | Done — conv2 TopK baseline evaluated |
| Anakin sweep | Done — 28 checkpoints + 5 converged partials |
| Conv2 linear probe (Phase 1F) | Done — gorilla conv2 coverage 0.789 vs 0.428 random control |
| Next decision point | Hawk_173 SAE eval + conv2 feature-reuse analysis |

**Current source-of-truth docs:**
- `RESEARCH-STATUS.md` — active status, winners, open hypotheses
- `EXPERIMENT-PLAN-2026-03-27.md` — active plan and decision gates
- `Quarto-specifications.md` — Quarto model, data, hooks, BSP sets
- `BSP-schema-summary.md` — gorilla/hawk BSP semantics and probe status

**Historical note:** The Arnold and Arnold Beta sections below are pre-regeneration results from the legacy `mode_2x2=False` pipeline. They are kept for provenance only and should not be used to plan new work.

### Historical Coverage Results — Arnold + Arnold Beta (2026-03-07, legacy_mode2x2_false)

11 valid experiments evaluated on fc1, gorilla BSP set (164 BSPs), amalgam dataset (240,845 positions).

| SAE | L0 | FVU | Coverage | Cov>50% | Cov>75% | Board Rec | Reconstruct |
|-----|----|-----|----------|---------|---------|-----------|-------------|
| **arnold_beta-jumprelu-t64** | 62.8 | 0.062% | 0.206 | 0.195 | 0.024 | **0.826** | **51.8%** |
| **arnold_beta-jumprelu-t32** | 33.9 | 0.264% | **0.214** | **0.213** | **0.055** | 0.822 | 50.6% |
| arnold_beta-topk-k128 | 128.0 | 0.014% | 0.182 | 0.165 | 0.024 | 0.818 | 49.4% |
| arnold-batchtopk-k32 | 32.0 | 0.199% | 0.209 | 0.226 | 0.031 | 0.817 | 49.4% |
| arnold_beta-gated-l1_005 | 330.9 | 0.030% | 0.191 | 0.134 | 0.018 | 0.806 | 46.3% |
| arnold_beta-gated-l1_01 | 274.0 | 0.037% | 0.205 | 0.201 | 0.037 | 0.801 | 45.1% |
| arnold-topk-k64 | 64.0 | 0.047% | 0.200 | 0.177 | 0.018 | 0.809 | 47.0% |
| arnold-topk-k32 | 32.0 | 0.242% | 0.201 | 0.159 | 0.012 | 0.799 | 44.5% |
| arnold-vanilla-l1_01 | 393.3 | 0.077% | 0.205 | 0.207 | 0.043 | 0.678 | 18.3% |
| arnold-vanilla-l1_005 | 493.1 | 0.071% | 0.195 | 0.165 | 0.037 | 0.719 | 28.1% |
| arnold-vanilla-l1_05 | 233.8 | 0.134% | 0.204 | 0.189 | 0.037 | 0.620 | 12.8% |

**Key findings (2026-03-07):**
- Coverage is narrow across all architectures (~0.18–0.21 mean F1 on 164 BSPs). The signal is in **board reconstruction** (does any single feature reach ≥90% precision?).
- JumpReLU (t64) achieves the best board reconstruction (82.6%) and fraction of reconstructable BSPs (51.8%) — best overall.
- Vanilla SAEs have comparable coverage but much weaker board reconstruction (62–72%), suggesting dense features with low per-feature precision.
- Sparse architectures (TopK, BatchTopK, JumpReLU with correct L0) reconstruct far more BSPs despite similar mean F1.
- BatchTopK eval must be run in **train mode** (batch-level sparsity), not inference mode (calibrated thresholds collapse L0 at eval time).

### Arnold Sweep Results (2026-03-06)

**Valid runs** (no implementation bugs, results are interpretable):

| SAE | FVU | L0 | L0σ | Dead% | Med.Freq | Notes |
|-----|-----|-----|------|-------|---------|-------|
| vanilla l1=0.005 | 0.071% | 494 | 22.6 | 3.4% | 0.49 | Baseline |
| vanilla l1=0.01 | 0.077% | 394 | 28.5 | 2.2% | 0.41 | **Best overall** |
| vanilla l1=0.05 | 0.135% | 235 | 26.2 | 3.1% | 0.27 | Sparsest vanilla |
| topk k=32 | 0.243% | 32 | 0.0 | 61.2% | 0.0 | High dead (geometry mismatch) |
| topk k=64 | 0.048% | 64 | 0.0 | 72.5% | 0.0 | High dead (geometry mismatch) |
| batchtopk k=32 | 0.202% | 32 | 9.4 | 82.5% | 0.0 | High dead (geometry mismatch) |

**Invalid runs** (bugs in implementation — do not use for comparisons):

| SAE | Outcome | Bug | Fix |
|-----|---------|-----|-----|
| gated l1=0.005 | FVU=1.0, L0=0, dead=99% | W_gate got zero recon gradient | Via-gate aux loss added |
| gated l1=0.01 | FVU=1.0, L0=0, dead=99% | Same | Via-gate aux loss added |
| jumprelu t=32 | L0=595 ≠ 32 | L0 penalty had ∂/∂θ=0 | Sigmoid STE for L0 |
| jumprelu t=64 | L0=604 ≠ 64 | Same | Sigmoid STE for L0 |
| gated (default) | FVU=1.0 | Key collision + via-gate bug | Fixed |
| jumprelu (default) | Crashed step 1800 | Not in arnold configs (manual test) | N/A |
| vanilla (default) | Crashed step 1100 | Not in arnold configs (manual test) | N/A |

**Key findings:**
- Natural L0 for fc1 is ~400/1024 (39%). TopK k=32/64 forces 3–6% active, causing geometry mismatch and high dead features. k=128 (12.5%) tested in arnold_beta.
- Vanilla l1=0.01 is the current best: FVU=0.077%, 2.2% dead, but L0=394 is too high for per-feature interpretability.
- Both gated and jumprelu had zero-gradient bugs in their sparsity mechanisms. Fixed 2026-03-06.

### Pilot Results Summary

| SAE | FVU | L0 | Dead % | Verdict |
|-----|-----|----|--------|---------|
| Vanilla (exp8, l1=0.001) | 0.018% | 574 | 28.5% | Near-identity: L1 too weak (valid data point) |
| TopK k=16 (exp8) | 0.78% | 16 | 88.7% | **INVALID** — aux loss had zero-gradient bug |

### Bug Fixes Log

**2026-03-04:**
- **TopK aux loss**: Replaced step-function dead count (zero gradient) with residual reconstruction through dead features (Gao et al. 2024)
- **Metrics**: Added `l0_std` (sparsity variation across positions) and `median_feat_freq` (typical feature utilization)
- **Constructor kwargs**: Fixed BatchTopK and JumpReLU constructor argument names
- **Test suite**: 51 tests covering all 6 architectures

**2026-03-06:**
- **GatedSAE via-gate**: `(gate_pre > 0).float()` blocks all reconstruction gradient to W_gate. Added via-gate auxiliary loss `||x − ReLU(gate_pre)@W_dec.detach()||²` (Rajamanoharan et al. 2024a)
- **JumpReLU L0 penalty**: `(h>0).float()` has zero gradient w.r.t. theta. Replaced with sigmoid kernel estimator `σ((z−θ)/ε)` for differentiable L0 (Rajamanoharan et al. 2024b)

## Project Structure

```
games-interp/
├── models/                  # Game model checkpoints (.pt)
│   ├── quarto/              #   CNN_uncoupled DQN-trained models
│   │   └── CNN_uncoupled.py #   Model class definition
│   ├── othello/             #   GPT-style transformer (planned)
│   └── tictactoe/           #   CNN models (planned)
├── lib/                     # Core SAE training library
│   └── sae/                 #   SAE architectures, training loop, registry
│       ├── __init__.py      #   Public API exports
│       ├── architectures.py #   6 SAE variants (vanilla, topk, etc.)
│       ├── train.py         #   train_sae(), metrics, checkpoints
│       ├── registry.py      #   Experiment tracking
│       └── hooks.py         #   Model hook utilities
├── saes/                    # Trained SAE checkpoints (per game)
│   ├── quarto/              #   {experiment}-{arch}-{hook}-*.pt
│   │   ├── training_registry.json  # Experiment metadata
│   │   ├── eval_registry.json      # Coverage/reconstruction results (per run)
│   │   ├── {run_id}_h.pt           # Cached (N×d_dict) activations [gitignored]
│   │   └── {run_id}_matching-{animal}.pt  # Cached F1 matrices [gitignored]
│   ├── othello/
│   └── tictactoe/
├── data/                    # Activation datasets & game states (.pt)
│   ├── quarto/              #   {hook}_{opponents}_activations.pt + _meta.pt
│   │   ├── bsp_labels-*.pt  #   BSP label tensors
│   │   └── bsp_schema-*.json #   BSP definitions
│   ├── othello/
│   └── tictactoe/
├── bsps/                    # [DEPRECATED] Legacy BSP JSON schemas
├── eval_registry/           # Experiment registry (JSON per run)
├── logs/                    # Training logs
├── scripts/
│   ├── collect_activations.py    # CLI: collect activations via self-play
│   ├── deduplicate_positions.py  # CLI: combine and deduplicate position files
│   ├── compute_bsp_labels.py     # CLI: compute BSP labels from metadata
│   ├── generate_bsps.py          # [Can deprecate] BSP schema generator
│   ├── plot_training.py          # Live training metrics visualization
│   └── games/                     # Game-specific modules
│       ├── __init__.py            #   Registry (get_game_module)
│       └── quarto.py              #   Quarto: bots, positions, BSP computation
├── configs/                 # YAML training configurations
│   ├── pilot-vanilla.yaml   #   Example: vanilla SAE config
│   └── pilot-topk.yaml      #   Example: topk SAE config
├── train_vanilla.py         # [DELETE after pilot finishes] Direct training: vanilla SAE
├── train_topk.py            # [DELETE after pilot finishes] Direct training: topk SAE
├── sae_train.py             # Unified training CLI (YAML config support)
├── Quarto-specifications.md      # Game-specific documentation
├── SELF-IMPROVEMENT.md           # Agent/skill update recommendations
└── .gitignore
```

## Setup

```bash
pip install torch numpy tqdm quartopy pyyaml docopt plotly dash
```

**Stack:** PyTorch, NumPy, tqdm, Plotly, Dash, docopt, YAML, quartopy

### Current Quarto Datasets (March 2026)

**Position datasets** (metals theme):
- `positions-amalgam_unique.pt` (240,845 positions) — Combined all 4 opponent modes, deduplicated

**Activations:**
- `fc1_amalgam_activations.pt` (240845×128) — fc1 hook on amalgam positions

**BSP labels** (animals theme):
- `bsp_labels-gorilla_164.pt` — All 164 BSPs
- `bsp_labels-fox_87.pt` — Cell properties only (87 BSPs: cell_occupancy, cell_attribute, offered_piece, game_phase)

See [Quarto-specifications.md](Quarto-specifications.md) Dataset Catalog for complete provenance.

## Scripts

### Workflow: Data Collection → SAE Training

**1. Generate positions** (multiple opponent modes for diverse coverage):
```bash
# Collect positions from different opponent modes
python scripts/collect_activations.py <model> --hook fc1 --game quarto \
    --opponents random_v_random --num-games 10 --output data/quarto/positions-random_v_random_raw.pt
python scripts/collect_activations.py <model> --hook fc1 --game quarto \
    --opponents model_v_random --num-games 10000 --output data/quarto/positions-model_v_random-{model_name}_raw.pt
# ... repeat for random_v_model, model_v_model
```

**2. Combine and deduplicate** (amalgam = all modes combined):
```bash
python scripts/deduplicate_positions.py \
    data/quarto/positions-*_raw.pt \
    --output data/quarto/positions-amalgam_unique.pt
```

**3. Collect activations** (run model on amalgam positions):
```bash
python scripts/collect_activations.py <model> --hook fc1 --game quarto \
    --positions-file data/quarto/positions-amalgam_unique.pt \
    --output data/quarto/fc1_amalgam_activations.pt
```

**4. Compute BSP labels** (for evaluation):
```bash
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt \
    --game quarto --name gorilla --output data/quarto/bsp_labels-gorilla_164.pt
```

Output files include full provenance (source files, model checkpoint, deduplication stats).

### Compute BSP Labels

Compute binary BSP (Board State Property) labels from position metadata:

```bash
python scripts/compute_bsp_labels.py <positions_file> --game {game} --name {animal} --output <path>
```

**Arguments:**
- `--name` — BSP set name (e.g., 'gorilla', 'fox'). Auto-detected from `--output` if omitted.
- `--output` — Output path (e.g., `bsp_labels-gorilla_164.pt`)

**Output:**
- `bsp_labels-{animal}_{count}.pt` — (N, num_bsps) binary tensor
- `bsp_schema-{animal}_{count}.json` — BSP definitions

**Filtering options:**
- `--categories cell_occupancy,cell_attribute` — Include only specific categories
- `--exclude-categories threat_square_2x2` — Exclude categories
- `--list-categories` — Show available categories

### Deduplicate Positions

Combine multiple position files and remove duplicates (critical: deduplicate AFTER combining, not before):

```bash
python scripts/deduplicate_positions.py file1.pt file2.pt file3.pt --output combined_unique.pt
```

Output includes provenance metadata tracking all source files and deduplication statistics.

### SAE Training & Evaluation

#### Unified Training with YAML Configs

The project uses a unified CLI script (`sae_train.py`) with YAML configuration files for reproducible SAE training:

```bash
# Train from config file (recommended)
python sae_train.py --config=configs/pilot-vanilla.yaml
python sae_train.py --config=configs/pilot-topk.yaml

# Or use command line arguments
python sae_train.py pilot vanilla --expansion=8 --lr=3e-4
python sae_train.py pilot topk --k=16 --expansion=8

# Parallel experiments (in separate terminals)
python sae_train.py sweep-lr-1 vanilla --lr=1e-4 &
python sae_train.py sweep-lr-2 vanilla --lr=3e-4 &
python sae_train.py sweep-lr-3 vanilla --lr=1e-3 &

# Monitor training live in another terminal
python scripts/plot_training.py saes/quarto/pilot-*_metrics.jsonl --live
```

**Outputs:**
- Checkpoints: `saes/{game}/{experiment}-{arch}-{hook}-*.pt`
- Metrics (JSONL): `saes/{game}/{experiment}-{arch}-{hook}-*_metrics.jsonl`
- Registry: `saes/{game}/training_registry.json` (experiment tracking)

**YAML Config Example** (`configs/pilot-vanilla.yaml`):
```yaml
experiment: pilot
architecture: vanilla
game: quarto
hook: fc1
data: data/quarto/fc1_amalgam_activations.pt
expansion: 8
batch_size: 4096
num_batches: 1000    # Quick test
lr: 3e-4
seed: 42
log_every: 50        # Frequent logs for monitoring
l1_weight: 1e-3
```

#### Live Training Monitoring with Plotly Dash

Monitor training progress in real-time or compare multiple runs:

```bash
# One-time static plot (opens in browser)
python scripts/plot_training.py saes/quarto/pilot-vanilla-exp8-fc1_metrics.jsonl

# Live monitoring server (updates every 10s automatically)
python scripts/plot_training.py saes/quarto/pilot-*_metrics.jsonl --live
# Then open http://localhost:8050 in your browser
# Leave the tab open - plot refreshes automatically, no need to reload

# Compare multiple architectures side-by-side (live)
python scripts/plot_training.py saes/quarto/pilot-vanilla-exp8-fc1_metrics.jsonl \
                                saes/quarto/pilot-topk-k16-exp8-fc1_metrics.jsonl --live
```

**Metrics plotted:**
- **Loss**: Reconstruction loss (lower is better)
- **L0**: Average active features per sample (sparsity measure)
- **FVU**: Fraction of variance unexplained (closer to 0 is better)
- **Dead Features %**: Percentage of features that never activate (track feature utilization)

**Live mode** runs a local web server (Dash). Open the URL once in your browser and leave 
the tab open. The plot refreshes automatically every 10 seconds - no need to reload or reopen tabs.
Each training run appears as a separate colored trace on shared subplots for easy comparison.

**JSONL format:** Metrics are logged as JSON Lines (one JSON object per line), enabling
streaming writes during training and easy incremental parsing. Each line contains:
`{"step": int, "loss": float, "l0": float, "fvu": float, "dead_features_pct": float}`

#### SAE Evaluation (Coverage + Board Reconstruction)

```bash
# Evaluate a single checkpoint (auto-resolves data and BSP paths from metadata)
python sae_eval.py evaluate saes/quarto/arnold_beta-jumprelu-t64-exp8-fc1.pt

# Force re-evaluation (also regenerates h and matching caches)
python sae_eval.py evaluate saes/quarto/arnold_beta-jumprelu-t64-exp8-fc1.pt --force

# Evaluate all valid experiments at once
for f in saes/quarto/arnold-batchtopk-k32-exp8-fc1.pt \
          saes/quarto/arnold-topk-k32-exp8-fc1.pt \
          saes/quarto/arnold-topk-k64-exp8-fc1.pt \
          saes/quarto/arnold-vanilla-l1_005-exp8-fc1.pt \
          saes/quarto/arnold-vanilla-l1_01-exp8-fc1.pt \
          saes/quarto/arnold-vanilla-l1_05-exp8-fc1.pt \
          saes/quarto/arnold_beta-gated-l1_005-exp8-fc1.pt \
          saes/quarto/arnold_beta-gated-l1_01-exp8-fc1.pt \
          saes/quarto/arnold_beta-jumprelu-t32-exp8-fc1.pt \
          saes/quarto/arnold_beta-jumprelu-t64-exp8-fc1.pt \
          saes/quarto/arnold_beta-topk-k128-exp8-fc1.pt; do
  python sae_eval.py evaluate "$f"
done

# Browse results
python sae_eval.py history
python sae_eval.py history --arch jumprelu

# Side-by-side comparison
python sae_eval.py compare arnold_beta-jumprelu-t64-exp8-fc1 arnold_beta-topk-k128-exp8-fc1
```

**Evaluation caches** (written to `saes/{game}/`, gitignored via `*.pt`):
- `{run_id}_h.pt` — full `(N × d_dict)` activation matrix; reused by downstream analysis
- `{run_id}_matching-{animal}.pt` — `(d_dict × num_bsps)` precision / recall / F1 tensors

The notebook `notebooks/sae_feature_analysis.ipynb` loads these caches directly — no re-encoding needed when switching models.

#### Visualization data export (for boardSAE-atlas)

The sibling [`boardSAE-atlas`](../boardSAE-atlas/) static site loads JSON bundles
and ONNX-exported encoders from `boardSAE-atlas/public/{data,models}/<game>/`.
Both files are produced by scripts in this repo:

```bash
# 1. JSON bundles (registry, BSP catalogue, feature pages, top-board galleries).
#    Auto-picks the top-N SAEs by coverage on the chosen BSP set.
python scripts/export_viz_data.py --game quarto

# 2. ONNX encoders (game net + each shipped SAE).
python scripts/export_onnx.py --game quarto
```

`export_onnx.py` refuses to ship a BatchTopK SAE whose `_threshold_estimate`
buffer is all zeros — that should never happen for SAEs produced by
`sae_train.py` (calibration is the last step of `train_sae()`), but if you
hit it (training crashed between the SGD loop and the calibration call),
either re-train or rerun with `--allow-uncalibrated`. ONNX export uses the
legacy TorchScript backend (its deprecation warning is suppressed); revisit
`dynamo=True` once it's the default in PyTorch 2.9.

#### Advanced Training via Skills (For Custom Architectures)

Training scripts live in the agent skill directories:

```bash
python ~/.agents/skills/sae-implementation/scripts/sae_train.py train <arch> \
    --model <path> --hook <layer> --data <path> [options]
```

## Games

| Game | Model Type | Model Class | BSPs | Status |
|------|-----------|-------------|------|--------|
| Quarto | CNN (DQN) | `QuartoCNN` | 164 binary | **Active** — See [Quarto-specifications.md](Quarto-specifications.md) |
| Othello | GPT-style transformer | TBD | — | Planned |
| Tic-tac-toe | CNN | TBD | — | Planned |

### Quarto Model Architecture

```
x_piece (16,) → FC(16→16, ReLU) → reshape(1,4,4)
                                        ↓
x_board (16,4,4) ─── concat ──→ (17,4,4)
                                        ↓
                    Conv2d(17→16, 3×3, pad=1, ReLU)   ← "conv1"  (B,16,4,4)
                                        ↓
                    Conv2d(16→32, 3×3, pad=1, ReLU)   ← "conv2"  (B,32,4,4)
                                        ↓
                              Flatten → (B,512)
                                        ↓
                    FC(512→128, ReLU)                  ← "fc1"   (B,128)  ★ best SAE target
                                        ↓
                               Dropout(0.5)
                           ┌────────┴────────┐
              fc2_board(128→16, tanh)   fc2_piece(128→16, tanh)
                 Q-values board            Q-values piece
```

## Key Concepts

- **SAE Architectures:** Vanilla, TopK, BatchTopK, Gated, JumpReLU, p-Annealing
- **BSPs (Board State Properties):** Ground-truth game concepts for evaluation. All BSPs are binary (0 or 1). Computed from game-specific functions in `scripts/games/{game}.py`.
- **Naming Conventions:** 
  - Position datasets use **metals theme** (copper, bronze, iron, steel, amalgam)
  - BSP sets use **animals theme** (gorilla, fox, etc.) — user chooses names based on scope
    - SAE follow-up runs now use prefixed experiment IDs: `{Major}{Minor}-{tag}-s{seed}`
    - Put the campaign ID in `experiment:` and let the trainer append architecture + hook to the checkpoint stem
    - See [Quarto-specifications.md](Quarto-specifications.md) for the active Quarto naming policy
- **Coverage:** Fraction of BSPs with a matching SAE feature (high F1)
- **Causal Verification:** Clamp/ablation tests to distinguish correlation from causation
- **Provenance:** Each dataset tracks source files, model checkpoint, deduplication stats, and creation date

## Documentation Structure

- **README.md** (this file) — Project overview, quick start
- **{Game}-specifications.md** — Per-game details (BSPs, model architecture, data formats)
- **SELF-IMPROVEMENT.md** — Recommendations for agent/skill updates, research directions
- **~/.agents/skills/** — SAE implementation and evaluation skills (training/eval scripts)
