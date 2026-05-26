# Quarto Game Specifications

## Game Overview

**Board:** 4×4 grid  
**Pieces:** 16 unique pieces, each with 4 binary attributes  
**Win conditions:**
1. **Line mode (default):** 4 pieces in a row/column/diagonal sharing any attribute
2. **2×2 mode:** 4 pieces in a 2×2 square sharing any attribute

## Piece Attributes

Each piece has 4 binary attributes (quartopy naming):
- **Size:** TALL or LITTLE
- **Coloration:** BLACK or WHITE
- **Shape:** SQUARE or CIRCLE
- **Hole:** WITH_HOLE or WITHOUT_HOLE

Total unique pieces: 2^4 = 16

## Model Architecture

**Current models (multi-champion era from 2026-05-14):**
- `CNN_uncoupled` (champAa baseline) — `models/quarto/CNN_uncoupled.py`
- `Sa_S4` unified-aux autoregressive CNN (champS4) — `models/quarto/Sa_S4.py`,
  registered via `configs/models/champS4.yaml` and the `quarto_s4` game module.

**Hookable layers — champAa (`CNN_uncoupled`):**
- `fc_in_piece` — (B,16) Piece input embedding
- `conv1` — (B,16,4,4) Early spatial features
- `conv2` — (B,32,4,4) Higher-level spatial. **Recommended hook point for threat-focused probes/SAEs after Phase 1F**
- `fc1` — (B,128) Shared bottleneck before dual heads. Still useful for bottleneck comparisons, but no longer the preferred hook for threat recovery.
- `fc2_board` — (B,16) Board placement Q-values
- `fc2_piece` — (B,16) Piece selection Q-values

**Hookable layers — champS4 (`Sa_S4`), accessed via the `quarto_s4` game module:**
- `s4.conv1` — (B,16,4,4) Early spatial features (same shape as champAa).
- `s4.conv2` — (B,32,4,4) Higher-level spatial — flatten to `(B, 512)` for SAE
  training (`collect_activations.py --flatten-position`).
- `s4.fc1` — (B,**512**) Shared bottleneck (4× wider than champAa's fc1=128).
  Note: A01-style fc1 SAEs need expansion ≥ 8 to match the per-feature budget
  of the Aa equivalent at d_act=128, exp=8 (= d_dict=1024); the original
  champS4 A01 config used exp=2 to match dictionary size and saw 96.6% dead
  features as a result.

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
| 0 | 0000 | LITTLE | BLACK | CIRCLE | WITHOUT_HOLE |
| 1 | 0001 | LITTLE | BLACK | CIRCLE | WITH_HOLE |
| 2 | 0010 | LITTLE | BLACK | SQUARE | WITHOUT_HOLE |
| 3 | 0011 | LITTLE | BLACK | SQUARE | WITH_HOLE |
| 4 | 0100 | LITTLE | WHITE | CIRCLE | WITHOUT_HOLE |
| 5 | 0101 | LITTLE | WHITE | CIRCLE | WITH_HOLE |
| 6 | 0110 | LITTLE | WHITE | SQUARE | WITHOUT_HOLE |
| 7 | 0111 | LITTLE | WHITE | SQUARE | WITH_HOLE |
| 8 | 1000 | TALL | BLACK | CIRCLE | WITHOUT_HOLE |
| 9 | 1001 | TALL | BLACK | CIRCLE | WITH_HOLE |
| 10 | 1010 | TALL | BLACK | SQUARE | WITHOUT_HOLE |
| 11 | 1011 | TALL | BLACK | SQUARE | WITH_HOLE |
| 12 | 1100 | TALL | WHITE | CIRCLE | WITHOUT_HOLE |
| 13 | 1101 | TALL | WHITE | CIRCLE | WITH_HOLE |
| 14 | 1110 | TALL | WHITE | SQUARE | WITHOUT_HOLE |
| 15 | 1111 | TALL | WHITE | SQUARE | WITH_HOLE |

Bit mapping: bit 3 = TALL, bit 2 = WHITE, bit 1 = SQUARE, bit 0 = WITH_HOLE

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

**Cell and offered piece attributes use quartopy enum naming directly:**
- `*_tall`: 1 = TALL, 0 = LITTLE
- `*_black`: 1 = BLACK, 0 = WHITE
- `*_square`: 1 = SQUARE, 0 = CIRCLE
- `*_with_hole`: 1 = WITH_HOLE, 0 = WITHOUT_HOLE

**Example BSP IDs:**
- `cell_0_0_occupied` — Is top-left cell occupied?
- `cell_0_0_tall` — Is piece at (0,0) TALL?
- `cell_2_3_black` — Is piece at (2,3) BLACK?
- `row_0_threat_tall` — Does row 0 have 3 TALL pieces + 1 empty cell?
- `square_1_1_threat_square` — Does 2×2 at (1,1) have 3 SQUARE pieces?
- `offered_with_hole` — Is the offered piece WITH_HOLE?
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

## Bootstrap and regeneration

Dataset and activation files are gitignored — regenerate from positions. The current pipeline recipe lives in [`commands.sh`](commands.sh) (transient, rewritten per phase). For the historical champAa bootstrap, see [`docs/diary/phase-1.md`](docs/diary/phase-1.md). For a new champion onboarding checklist, see [`configs/models/README.md`](configs/models/README.md).

Linear-probe baseline (upper bound on what any SAE can recover) is `scripts/linear_probe_baseline.py`; outputs land at `data/quarto/linear_probe_<bsp_set>_<act_stem>_results.json` with the same F1 / MCC / F1-lift schema as `sae_eval`.

---

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

**SAE follow-up run naming (from 2026-04-24 onward):**
- Use experiment IDs of the form `{Major}{Minor}-{tag}-s{seed}`.
- `Major` is one uppercase letter for the campaign family (`A`, `B`, `C`, ...).
- `Minor` is a zero-padded two-digit condition index inside that campaign (`01`, `02`, ...).
- `tag` is a short semantic label such as `random-control`, `conv2-completion`, or `seedpanel`.
- `s{seed}` is mandatory for any fixed-seed run, even when the config filename already implies the seed.
- The trainer already appends architecture and hook, so put the campaign ID in `experiment:` rather than duplicating the full config name.

**Examples:**
- `A01-random-control-s42` → `A01-random-control-s42-batchtopk-k16-exp8-fc1.pt`
- `A02-random-control-s42` → `A02-random-control-s42-topk-k32-exp8-conv2.pt`
- `B03-conv2-completion-s42` → `B03-conv2-completion-s42-topk-k64-exp2-conv2.pt`

**Rule of use:**
- Reuse the same `{Major}{Minor}-{tag}` only for the same experimental condition; vary `s{seed}` for replicated seeds.
- Allocate a new `{Major}{Minor}` whenever the condition itself changes (hook, architecture family, data source, or research purpose).

**Deterministic run-id rule (from 2026-05-14 onward):**

The full run_id (= checkpoint stem) is built by the trainer as:

```
{experiment}-{arch}-{sparsity}-exp{E}-{hook}
```

where `{sparsity}` is:

| architecture        | sparsity token              |
|---------------------|-----------------------------|
| `topk`, `batchtopk` | `k{k}`                      |
| `vanilla`, `gated`  | `l1_{l1_weight as digits}`  |
| `jumprelu`          | `t{l0_target as int}`       |
| `panneal`           | *(omitted)*                 |

This rule lives in `sae_train.build_filename_suffix()`; it is the single
source of truth. **Do not reconstruct run_ids in shell scripts or notebooks
by hand** — use the helper:

```bash
python scripts/run_id.py <config.yaml>               # run_id stem
python scripts/run_id.py <config.yaml> --checkpoint  # full .pt path
python scripts/run_id.py <config.yaml> --metrics     # metrics jsonl path
```

The `experiment:` field is the only free string in the YAML; everything after
it is derived. Keep `experiment:` short — it is the run_id prefix and gets
re-used verbatim as the eval-registry key.

**Champion tag (multi-model era, from 2026-05-14):**
- `champAa` — original Aa_replay uncoupled CNN (fc1=128).
- `champS4` — unified-aux autoreg CNN (fc1=512). Wins ~60% head-to-head vs champAa.
- Champion model paths and game modules are registered in `configs/models/champ*.yaml`
  and consumed by `scripts/model_competence_audit.py --model-config=...`.
- New artifact suffixes when working with a non-baseline champion:
  - positions → `positions-amalgam_<tag>_unique.pt` (e.g. `_s4`)
  - activations → `<hook>_amalgam_<tag>_activations.pt`
  - SAE `experiment:` field includes `champ<Tag>` (e.g. `C01-champS4-s42`).
- Aa_replay artifacts are *not* renamed; absence of a `_<tag>` suffix implies champAa.

**One registry, distinct BSP-set names (from 2026-05-14):**
- All champion SAEs write to a single registry: `saes/quarto/eval_registry.json`.
  Set `game: quarto` in the SAE YAML even for non-baseline champions; the
  `data:` field encodes the champion-specific activation source.
- BSP labels for a new distribution use the original animal name with the
  champion tag appended in CamelCase (no underscore), e.g. `gorillaS4`,
  `hawkS4`. This keeps glob resolution unambiguous (`bsp_labels-gorilla_[0-9]*.pt`
  does not match `bsp_labels-gorillaS4_*.pt`).
- Registry keys are `run_id:bsp_set`, so the same run_id can appear under
  `gorilla` and `gorillaS4` without collision — but in practice each run is
  evaluated only against the BSP set matching its training distribution.
- Cross-champion comparison via `registry_query.py compare A B --bsps=<setA> --bsps-b=<setB>`.
- `sae_eval` glob is `bsp_labels-{animal}_[0-9]*.pt` (numeric count suffix
  required); this prevents `gorilla` from accidentally matching `gorillaS4`
  or any other future champion-tagged variant.

## Dataset and artifact catalog

**Authoritative sources, not duplicated here.** Per-champion specs live in [`configs/models/champ*.yaml`](configs/models/); per-instance files are on the filesystem:

- Position datasets: `data/quarto/positions-amalgam_<tag>_unique.pt` (current champions: `s4`, `ta`, `ve`; champAa is un-tagged: `positions-amalgam_unique.pt`).
- BSP labels: `data/quarto/bsp_labels-<animal>_<count>.pt` (e.g. `gorillaVe_164`, `hawkTa_173`, `tigerS4_36`). Schema: `data/quarto/bsp_schema-<basis>_<count>.json` (basis-only — same schema for all champions in a basis).
- Activations: `data/quarto/<hook>_amalgam_<tag>{,_random}_activations.pt`. Trained-vs-random pairs are required for any headline gap claim.

List what's currently on disk with `ls data/quarto/positions-amalgam_*_unique.pt`, `ls data/quarto/bsp_labels-*.pt`, `ls data/quarto/*_activations.pt`. To add a new champion, follow [`configs/models/README.md`](configs/models/README.md).

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

## Position file format

```python
{
    "boards": torch.Tensor,      # (N, 16, 4, 4)
    "pieces": torch.Tensor,      # (N, 16)
    "metadata": list[dict],      # N metadata dicts
    "provenance": dict,          # Source info (seed, model, mode, date)
}
```

BSP computation and the dedup utility are covered by `tests/test_bsp_logic.py` and `tests/test_sae_eval.py`.
