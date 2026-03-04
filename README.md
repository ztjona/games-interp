# Games-Interp: Mechanistic Interpretability of Board Game Neural Networks

PhD research project using Sparse Autoencoders (SAEs) to extract interpretable features
from board game neural networks (Quarto, Othello, Tic-tac-toe).

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
│   │   └── training_registry.json  # Experiment metadata
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

#### Advanced Training via Skills (For Custom Architectures)

Training and evaluation scripts live in the agent skill directories and are invoked
from the project root:

```bash
# Training
python ~/.agents/skills/sae-implementation/scripts/sae_train.py train <arch> \
    --model <path> --hook <layer> --data <path> [options]

# Evaluation
python ~/.agents/skills/sae-board-bench/scripts/sae_eval.py evaluate <checkpoint> \
    --model <path> --game <name> --hook <layer> --data <path>
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
  - See [SELF-IMPROVEMENT.md](SELF-IMPROVEMENT.md) for detailed naming strategy
- **Coverage:** Fraction of BSPs with a matching SAE feature (high F1)
- **Causal Verification:** Clamp/ablation tests to distinguish correlation from causation
- **Provenance:** Each dataset tracks source files, model checkpoint, deduplication stats, and creation date

## Documentation Structure

- **README.md** (this file) — Project overview, quick start
- **{Game}-specifications.md** — Per-game details (BSPs, model architecture, data formats)
- **SELF-IMPROVEMENT.md** — Recommendations for agent/skill updates, research directions
- **~/.agents/skills/** — SAE implementation and evaluation skills (training/eval scripts)
