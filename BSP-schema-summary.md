# BSP Schema Summary — Quarto

**Source:** `scripts/games/quarto.py` → `get_all_bsp_definitions()` (337 total BSPs)

## Gorilla Set (164 BSPs) — Raw Game Properties

Probes for the ground-truth game state as defined by the rules.

| Category | Count | Formula | Description | Linear Probe F1 (trained / random) |
|----------|------:|---------|-------------|-------------------------------------|
| `cell_occupancy` | 16 | 4×4 cells | Is cell (r,c) occupied? | 0.998 / 0.564 |
| `cell_attribute` | 64 | 16 cells × 4 attrs | Binary attribute of piece at (r,c) | 0.611 / 0.281 |
| `threat_line` | 40 | 10 lines × 4 attrs | 3-of-4 pieces in line share attribute + 1 empty | 0.022 / 0.000 |
| `threat_square_2x2` | 36 | 9 squares × 4 attrs | 3-of-4 pieces in 2×2 square share attribute + 1 empty | 0.113 / 0.000 |
| `offered_piece` | 4 | 4 attrs | Binary attribute of currently offered piece | 0.602 / 0.648 |
| `game_phase` | 3 | 3 bins | early (0-5 pieces), mid (6-11), late (12-16) | 0.909 / 0.545 |
| `global` | 1 | 1 | `winning_move_exists` — can offered piece be placed to win? | 0.698 / 0.564 |

### Phase 1F Update — Conv2 vs fc1 on Gorilla

Phase 1F established that the gorilla threat BSPs are linearly accessible in `conv2` even though they appear almost absent at `fc1`.

| Category | fc1 probe | conv2 probe | conv2 random |
|----------|:---------:|:-----------:|:------------:|
| `cell_occupancy` | 0.998 | 1.000 | 0.829 |
| `cell_attribute` | 0.611 | 0.971 | 0.782 |
| `threat_line` | 0.022 | 0.502 | 0.019 |
| `threat_square_2x2` | 0.113 | 0.680 | 0.016 |
| `offered_piece` | 0.602 | 0.789 | 0.718 |
| `global` | 0.698 | 0.694 | 0.634 |
| `game_phase` | 0.909 | 0.936 | 0.685 |

**Interpretation:** threat BSPs are genuine learned spatial features in conv2, not an artifact of untrained conv filters. This makes conv2 the correct hook for threat-focused SAE follow-up.

### Lines (10 total)
- 4 rows: `row_0` through `row_3`
- 4 columns: `col_0` through `col_3`
- 2 diagonals: `diag_main` (top-left to bottom-right), `diag_anti` (top-right to bottom-left)

### 2×2 Squares (9 total)
- `square_{top_r}_{left_c}` where `top_r ∈ {0,1,2}`, `left_c ∈ {0,1,2}`
- Each covers cells: `(top_r, left_c)`, `(top_r, left_c+1)`, `(top_r+1, left_c)`, `(top_r+1, left_c+1)`

### Binary Attributes (4 total)
| Suffix | Metadata Key | Positive (=1) | Negative (=0) |
|--------|-------------|---------------|---------------|
| `_tall` | size | TALL | LITTLE |
| `_black` | coloration | BLACK | WHITE |
| `_square` | shape | SQUARE | CIRCLE |
| `_with_hole` | hole | WITH_HOLE | WITHOUT_HOLE |

**Note:** Only positive-value variants are probed. Negative variants (e.g., `_little`, `_white`) are the complement for cell attributes, but for threats they represent distinct board patterns (3 LITTLE in a row ≠ not having 3 TALL in a row). This asymmetry is a known limitation — see "Known Gaps" below.

---

## Hawk Set (173 BSPs) — Nanda-Inspired Reframed Properties

Reframes threats under alternative bases that may match the model's internal representation more closely. Motivated by Nanda et al. (emergent-linear-representations-world-models): OthelloGPT appeared non-linear under BLACK/WHITE but was linear under MINE/YOURS.

### Line Reframing (90 BSPs)

| Category | Count | Formula | Description | Linear Probe F1 (trained / random) |
|----------|------:|---------|-------------|-------------------------------------|
| `reframed_count` | 40 | 10 lines × 4 attrs | ≥3 occupied cells share attribute (ignores empty cells) | 0.315 / 0.000 |
| `reframed_completable` | 40 | 10 lines × 4 attrs | Threat exists AND offered piece has matching attribute | 0.003 / 0.000 |
| `reframed_any_threat` | 10 | 10 lines × 1 | Any attribute creates a threat in this line (OR across attrs) | 0.086 / 0.000 |

### Square Reframing (81 BSPs) — NEW

| Category | Count | Formula | Description | Linear Probe F1 |
|----------|------:|---------|-------------|-----------------|
| `reframed_sq_count` | 36 | 9 squares × 4 attrs | ≥3 occupied cells in 2×2 share attribute | 0.455 / 0.000 |
| `reframed_sq_completable` | 36 | 9 squares × 4 attrs | Threat in 2×2 AND offered piece matches | 0.027 / 0.000 |
| `reframed_sq_any_threat` | 9 | 9 squares × 1 | Any attribute creates a threat in this 2×2 | 0.276 / 0.000 |

### Global Reframing (2 BSPs)

| Category | Count | Description | Linear Probe F1 (trained / random) |
|----------|------:|-------------|-------------------------------------|
| `reframed_global` | 2 | `board_threat_exists` (any line OR square has threat) + `board_completable_exists` (any threat completable with offered piece) | 0.597 / 0.408 |

**Note:** Global BSPs now include both lines AND 2×2 squares in their computation (updated 2026-03-31).

---

## Semantic Relationships

### Hierarchy
```
cell_occupancy  ←  foundation for everything else
    ↓
cell_attribute  ←  requires occupancy
    ↓
threat_line / threat_square_2x2  ←  conjunction: 3 matching + 1 empty
    ↓
winning_move_exists  ←  threat + placement + offered piece compatibility
```

### Gorilla ↔ Hawk Correspondence
| Gorilla | Hawk (reframed) | Relationship |
|---------|----------------|-------------|
| `threat_line` | `reframed_count` | count_ge3 is a **superset** of threats (fires on 3-of-3, 3-of-4, and 4-of-4; threat only fires on 3-matching + 1-empty) |
| `threat_line` | `reframed_completable` | completable is a **strict subset** of threats (threat AND offered matches) |
| `threat_line` | `reframed_any_threat` | any_threat is an **OR reduction** across 4 attributes per line |
| `threat_square_2x2` | `reframed_sq_count` | Same superset relationship as above |
| `threat_square_2x2` | `reframed_sq_completable` | Same strict subset relationship |
| `threat_square_2x2` | `reframed_sq_any_threat` | Same OR reduction |
| `global` (`winning_move_exists`) | `reframed_global` (`board_completable_exists`) | Similar but NOT identical — `winning_move_exists` simulates placement; `board_completable_exists` checks attribute match only |

### Key Semantic Differences
1. **`count_ge3` vs `threat`**: `count_ge3` fires when ≥3 occupied cells match, regardless of whether there's an empty cell. A full line of 4 where 3 match will fire `count_ge3` but NOT `threat` (no empty cell = no threat).
2. **`winning_move_exists` vs `board_completable_exists`**: `winning_move_exists` simulates placing the offered piece in each empty cell and checks if it completes a win. `board_completable_exists` only checks if any threat's attribute matches the offered piece. In rare edge cases these could differ.
3. **Negative attributes omitted**: A line of 3 LITTLE + 1 empty is a valid Quarto threat, but is NOT detected by either set. Both gorilla and hawk only probe positive attribute values (TALL, BLACK, SQUARE, WITH_HOLE).

---

## Known Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| **Negative attribute threats** (3 LITTLE, 3 WHITE, etc.) | Missing half of threat patterns. Would double threat/count BSP counts. Not critical for probes since model encodes these symmetrically. | Low — flag for future |
| **Distance-to-win encoding** (count of threats on board) | Multi-class, not binary. Would require binarization (num_threats ≥ 1, ≥ 2, etc.). Experiment plan Phase 1D mentioned this. | Medium — future work |
| **Turn/player information** | "Whose turn is it?" not in metadata. | Low |
| **Piece interaction BSPs** | E.g., "offered piece is same size as piece at (r,c)". Combinatorial explosion. | Low |

---

## BSP ID Format Examples

### Gorilla
```
cell_0_0_occupied            →  cell_{r}_{c}_occupied
cell_2_3_tall                →  cell_{r}_{c}_{attr_suffix}
row_0_threat_tall            →  {line_type}_{idx}_threat_{attr_suffix}
square_1_2_threat_black      →  square_{tr}_{lc}_threat_{attr_suffix}
offered_with_hole            →  offered_{attr_suffix}
game_phase_mid               →  game_phase_{phase}
winning_move_exists          →  (singleton)
```

### Hawk — Lines
```
row_0_count_ge3_tall         →  {line_type}_{idx}_count_ge3_{attr_suffix}
diag_main_completable_black  →  {line_type}_{idx}_completable_{attr_suffix}
col_2_any_threat             →  {line_type}_{idx}_any_threat
```

### Hawk — Squares
```
square_0_0_count_ge3_tall    →  square_{tr}_{lc}_count_ge3_{attr_suffix}
square_1_2_completable_black →  square_{tr}_{lc}_completable_{attr_suffix}
square_2_1_any_threat        →  square_{tr}_{lc}_any_threat
```

### Hawk — Global
```
board_threat_exists          →  (singleton, includes lines + squares)
board_completable_exists     →  (singleton, includes lines + squares)
```

---

## Verification Status (2026-03-31)

- [x] 26/26 unit tests passing (`tests/test_bsp_logic.py`)
- [x] Gorilla: 164 BSPs, 7 categories — fully tested
- [x] Hawk lines: 90 BSPs, 3 categories — fully tested
- [x] Hawk squares: 81 BSPs, 3 categories — fully tested (NEW)
- [x] Hawk global: 2 BSPs — updated to include 2×2 squares, tested
- [x] Gorilla square threat dispatch not broken by hawk additions (regression test)
- [x] Hawk labels recomputed (`bsp_labels-hawk_173.pt` on disk)
- [x] fc1 linear probes re-run on updated hawk_173
- [ ] Conv2 linear probes not yet run on hawk_173
