# Quarto Game Specifications

## Game Overview

**Board:** 4×4 grid  
**Pieces:** 16 unique pieces, each with 4 binary attributes  
**Win conditions:**
1. **Line mode (default):** 4 pieces in a row/column/diagonal sharing any attribute
2. **2×2 mode:** 4 pieces in a 2×2 square sharing any attribute

## Piece Attributes

Each piece has 4 binary attributes:
- **Size:** TALL or SHORT
- **Coloration:** DARK or LIGHT
- **Shape:** SQUARE or ROUND
- **Hole:** HOLLOW or SOLID

Total unique pieces: 2^4 = 16

## Model Architecture

**Current model:** `CNN_uncoupled` (in `models/quarto/CNN_uncoupled.py`)

**Hookable layers:**
- `fc_in_piece` — (B,16) Piece input embedding
- `conv1` — (B,16,4,4) Early spatial features
- `conv2` — (B,32,4,4) Higher-level spatial
- `fc1` — (B,128) **Recommended hook point** — shared bottleneck before dual heads
- `fc2_board` — (B,16) Board placement Q-values
- `fc2_piece` — (B,16) Piece selection Q-values

**Note:** Always verify current architecture with:
```bash
python scripts/sae_train.py list-hooks models/quarto/{model}.pt --model-class models.quarto.CNN_uncoupled.QuartoCNN
```

## Network Inputs

**Board encoding:** (16, 4, 4) one-hot tensor
- 16 channels (one per piece type)
- 4×4 spatial grid
- Channel i is 1 where piece i is placed, 0 elsewhere

**Offered piece encoding:** (16,) one-hot vector
- One channel per piece type
- 1 at index of offered piece, 0 elsewhere

### Piece Index Mapping

Index is a 4-bit binary encoding: `[size][coloration][shape][hole]`

| Index | Binary | Size | Coloration | Shape | Hole |
|-------|--------|------|------------|-------|------|
| 0 | 0000 | SHORT | DARK | ROUND | SOLID |
| 1 | 0001 | SHORT | DARK | ROUND | HOLLOW |
| 2 | 0010 | SHORT | DARK | SQUARE | SOLID |
| 3 | 0011 | SHORT | DARK | SQUARE | HOLLOW |
| 4 | 0100 | SHORT | LIGHT | ROUND | SOLID |
| 5 | 0101 | SHORT | LIGHT | ROUND | HOLLOW |
| 6 | 0110 | SHORT | LIGHT | SQUARE | SOLID |
| 7 | 0111 | SHORT | LIGHT | SQUARE | HOLLOW |
| 8 | 1000 | TALL | DARK | ROUND | SOLID |
| 9 | 1001 | TALL | DARK | ROUND | HOLLOW |
| 10 | 1010 | TALL | DARK | SQUARE | SOLID |
| 11 | 1011 | TALL | DARK | SQUARE | HOLLOW |
| 12 | 1100 | TALL | LIGHT | ROUND | SOLID |
| 13 | 1101 | TALL | LIGHT | ROUND | HOLLOW |
| 14 | 1110 | TALL | LIGHT | SQUARE | SOLID |
| 15 | 1111 | TALL | LIGHT | SQUARE | HOLLOW |

Bit mapping: bit 3 = size_tall, bit 2 = coloration_light, bit 1 = shape_square, bit 0 = hole_hollow

## BSP (Board State Property) Definitions

**Total:** 164 binary BSPs across 7 categories

### BSP Categories

| Category | Count | Description |
|----------|-------|-------------|
| `cell_occupancy` | 16 | Is each cell (r,c) occupied? |
| `cell_attribute` | 64 | Binary attributes of piece at each cell (4 per cell) |
| `threat_line` | 40 | 3 of 4 pieces in a line share attribute (rows, cols, diagonals) |
| `threat_square_2x2` | 36 | 3 of 4 pieces in a 2×2 square share attribute |
| `offered_piece` | 4 | Binary attributes of currently offered piece |
| `game_phase` | 3 | early (0-5 pieces), mid (6-11), late (12-16) |
| `global` | 1 | `winning_move_exists` |

### Binary Encoding Convention

**Cell and offered piece attributes use explicit positive naming:**
- `*_size_tall`: 1 = TALL, 0 = SHORT
- `*_coloration_dark`: 1 = DARK, 0 = LIGHT
- `*_shape_square`: 1 = SQUARE, 0 = ROUND
- `*_hole_hollow`: 1 = HOLLOW, 0 = SOLID

**Example BSP IDs:**
- `cell_0_0_occupied` — Is top-left cell occupied?
- `cell_0_0_size_tall` — Is piece at (0,0) TALL?
- `cell_2_3_coloration_dark` — Is piece at (2,3) DARK?
- `row_0_threat_size_tall` — Does row 0 have 3 TALL pieces + 1 empty cell?
- `square_1_1_threat_shape_square` — Does 2×2 at (1,1) have 3 SQUARE pieces?
- `offered_hole_hollow` — Is the offered piece HOLLOW?
- `game_phase_mid` — Are there 6-11 pieces on board?

### Threat Detection Rules

**Line threat** (rows, columns, diagonals):
- Exactly 3 pieces present in the line
- All 3 share the specified binary attribute (e.g., all TALL)
- Exactly 1 cell is empty
- The fourth piece (if placed with same attribute) would win

**Square threat** (2×2 patterns):
- Exactly 3 of 4 cells in the square are occupied
- All 3 pieces share the specified binary attribute
- Exactly 1 cell is empty
- The fourth piece (if placed with same attribute) would win

## BSP Computation

**Implementation:** `scripts/games/quarto.py`

**Functions:**
- `get_all_bsp_definitions()` — Returns metadata for all 164 BSPs
- `compute_bsp_vector(metadata, bsp_ids)` — Computes binary labels from game state

**Usage:** Pass a positions file directly (standard workflow). The `*_meta.pt` files from `collect_activations.py` are only created when using `--opponents` mode (on-the-fly position generation) and are not needed in the standard pipeline.

```bash
# Compute all BSPs from a positions file
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt \
    --game quarto --name gorilla

# Compute subset (cell properties only)
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt \
    --game quarto --name fox \
    --categories cell_occupancy,cell_attribute,offered_piece,game_phase
```

## Common BSP Sets

| Animal Name | Count | Categories Included |
|-------------|-------|---------------------|
| `gorilla` | 164 | ALL (complete set) |
| `elephant` | 80 | cell_occupancy + cell_attribute (no threats) |
| `cheetah` | 56 | cell_occupancy + threat_line (no attributes) |
| `penguin` | 128 | ALL except threat_square_2x2 |

*Note: Animal names are chosen by user when creating BSP sets.*

## Data Collection

**Opponent modes:**
- `random_v_random` — Uniform game tree coverage (baseline)
- `model_v_random` — Model (P1) vs random (P2) — model places first
- `random_v_model` — Random (P1) vs model (P2) — model reacts to random choices
- `model_v_model` — Self-play with temperature=0.1

**Important:** All modes collect activations from YOUR trained model. The opponent
mode controls position distribution, not which model is interpreted.

**Strategy:** Collect all 4 modes separately, then:
1. Train SAEs on combined dataset (broad feature discovery)
2. Test each SAE on individual opponent modes to study distribution shift
3. Compare feature usage: Does the model use different features vs random vs itself?

**Typical dataset sizes:**
- 10K games → ~100K positions for Quarto
- Each position: board state + offered piece + metadata

## Naming Conventions

**Datasets (Metals theme):**
Position datasets are named after metals to indicate opponent strength progression:
- `copper` — random_v_random (pure baseline, weakest opponents)
- `bronze` — model_v_random (first alloy, model vs weak)
- `iron` — random_v_model (stronger, weak vs model)
- `steel` — model_v_model (hardened self-play, strongest)
- `amalgam` — combined all modes (ultimate mixture)

**BSPs (Animals theme):**
BSP label sets are named after animals to indicate set size:
- `gorilla` — Full set (164 BSPs, all categories)
- `fox` — Cell properties only (87 BSPs: cell_occupancy, cell_attribute, offered_piece, game_phase)
- Custom animal names for targeted subsets (user-defined during computation)

**Filename format:**
- Positions: `positions-{metal_name}_unique.pt`
- BSP labels: `bsp_labels-{animal_name}_{count}.pt`
- BSP schema: `bsp_schema-{animal_name}_{count}.json`

## Dataset Catalog

**Current datasets** (as of March 2026):

### Position Datasets

| Name | Description | Source Files | Model | N Positions | Generation Date |
|------|-------------|--------------|-------|-------------|-----------------|
| `amalgam` | Combined all opponent modes, deduplicated | `positions-random_v_random_raw.pt`<br>`positions-model_v_random-Aa_replay_raw.pt`<br>`positions-random_v_model-Aa_replay_raw.pt`<br>`positions-model_v_model-Aa_replay_raw.pt` | Aa_replay (20260227_1103) | TBD | 2026-03-03 |
| `copper` | random_v_random only (not yet created) | `positions-random_v_random_raw.pt` | N/A | ~121 | 2026-03-03 |
| `bronze` | model_v_random only (not yet created) | `positions-model_v_random-Aa_replay_raw.pt` | Aa_replay | ~102K | 2026-03-03 |
| `iron` | random_v_model only (not yet created) | `positions-random_v_model-Aa_replay_raw.pt` | Aa_replay | ~101K | 2026-03-03 |
| `steel` | model_v_model only (not yet created) | `positions-model_v_model-Aa_replay_raw.pt` | Aa_replay | ~94K | 2026-03-03 |

### BSP Label Sets

| Name | BSP Count | Categories Included | Source Dataset | Generation Date | Purpose |
|------|-----------|---------------------|----------------|-----------------|---------|
| `gorilla` | 164 | ALL (cell_occupancy, cell_attribute, threat_line, threat_square_2x2, offered_piece, game_phase, global) | amalgam | TBD | Full coverage evaluation |
| `fox` | 87 | cell_occupancy, cell_attribute, offered_piece, game_phase | (extractable from gorilla) | N/A | Positional properties only |

**Notes:**
- All position datasets use model **Aa_replay** checkpoint: `20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt`
- Random baseline model (epoch 0, untrained): `20260226_1420-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_0000.pt`
- Raw files preserved for reference; unique files created via deduplication
- Position counts are approximate before deduplication
- BSP labels computed from position metadata, independent of boards/pieces tensors

## Deduplication Workflow

**Correct workflow:**
1. Generate raw positions (may contain natural duplicates)
2. Aggregate multiple raw files if needed
3. Deduplicate AFTER aggregation using `deduplicate_positions.py`

**Why:** Early-game positions naturally repeat across datasets. Deduplicating per-dataset wastes computation, and aggregating pre-deduped datasets still contains duplicates.

**Example:**
```bash
# Combine and deduplicate multiple opponent modes
python scripts/deduplicate_positions.py \
    data/quarto/positions-random_v_random_raw.pt \
    data/quarto/positions-model_v_random-Aa_replay_raw.pt \
    --output data/quarto/positions-combined_unique.pt
```

## Validation Status

**BSP computation:** ✅ Fully tested and complete (March 2026)
- Empty board, binary attributes, cell occupancy
- Row, column, diagonal threats (3 of 4 detection)
- 2×2 square threats (3 of 4 detection)
- Offered piece attributes, game phases
- **Winning move detection** (line and 2×2 completions)
- All 164 BSPs validated with known game states

**Deduplication utility:** ✅ Fully tested
- Single-file deduplication (75% unique in test)
- Multi-file aggregation (66.7% unique with cross-file duplicates)
- Stats output verified

**Position file format:** ✅ Confirmed
```python
{
    "boards": torch.Tensor,      # (N, 16, 4, 4)
    "pieces": torch.Tensor,      # (N, 16)
    "metadata": list[dict],      # N metadata dicts
    "provenance": dict,          # Source info
}
```

## Known Issues / TODOs

*(None currently)*

## Resolved Issues

1. **Diagonal BSP naming** ✅ RESOLVED (2026-03-27) — Renamed `diag_0` → `diag_main`, `diag_1` → `diag_anti`

1. **Piece encoding documentation** ✅ RESOLVED (2026-03-27) — See Piece Index Mapping table above.

2. **mode_2x2 in position generation** ✅ RESOLVED (2026-03-27)
   - `generate_positions` had `mode_2x2=False`; fixed to `True`
   - Legacy data moved to `data/quarto/legacy_mode2x2_false/`
   - Legacy SAEs moved to `saes/quarto/legacy_mode2x2_false/`
