"""Generate Board State Property (BSP) definitions for board games.

Creates the BSP JSON files used by sae_eval.py to evaluate SAE features
against ground-truth game concepts.

Usage:
    generate_bsps.py <game> [--output <path>]
    generate_bsps.py (-h | --help)

Arguments:
    <game>    Game name: quarto, othello, tictactoe

Options:
    -h --help         Show this help message
    --output <path>   Output JSON path [default: auto]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# Quarto BSPs (164 total — all binary)
# ──────────────────────────────────────────────────────────────────────

# Binary encoding convention:
#   cell_r_c_size_tall:         1 = TALL,   0 = SHORT
#   cell_r_c_coloration_dark:   1 = DARK,   0 = LIGHT
#   cell_r_c_shape_square:      1 = SQUARE, 0 = ROUND
#   cell_r_c_hole_hollow:       1 = HOLLOW, 0 = SOLID

QUARTO_BINARY_ATTRS = {
    "size_tall": ("TALL", "SHORT"),
    "coloration_dark": ("DARK", "LIGHT"),
    "shape_square": ("SQUARE", "ROUND"),
    "hole_hollow": ("HOLLOW", "SOLID"),
}


def generate_quarto_bsps() -> list[dict]:
    """Generate all binary BSPs for Quarto (4x4 board).

    Categories:
    - cell_occupancy: 16 BSPs (is each cell occupied?)
    - cell_attribute: 64 BSPs (16 cells × 4 binary attributes)
    - threat_line: 40 BSPs (10 lines × 4 attributes; 3 of 4 sharing)
    - threat_square_2x2: 36 BSPs (9 squares × 4 attributes; 3 of 4 sharing)
    - offered_piece: 4 BSPs (offered piece's 4 binary attributes)
    - game_phase: 3 BSPs (early, mid, late game)
    - global: 1 BSP (winning_move_exists)

    Total: 164 binary BSPs
    """
    bsps = []

    # ── Cell occupancy (16) ───────────────────────────────────────────
    for r in range(4):
        for c in range(4):
            bsps.append(
                {
                    "id": f"cell_{r}_{c}_occupied",
                    "description": f"Is cell ({r},{c}) occupied?",
                    "type": "binary",
                    "category": "cell_occupancy",
                }
            )

    # ── Cell attributes (64) ──────────────────────────────────────────
    for r in range(4):
        for c in range(4):
            for attr_name, (pos_val, _) in QUARTO_BINARY_ATTRS.items():
                attr_clean = attr_name.split("_")[0]  # "size" from "size_tall"
                bsps.append(
                    {
                        "id": f"cell_{r}_{c}_{attr_name}",
                        "description": f"Piece at ({r},{c}) is {pos_val} (1) or not (0)",
                        "type": "binary",
                        "category": "cell_attribute",
                    }
                )

    # ── Line threats: 3 of 4 pieces sharing attribute (40) ────────────
    lines = []
    # Rows
    for r in range(4):
        lines.append(("row", r, [(r, c) for c in range(4)]))
    # Columns
    for c in range(4):
        lines.append(("col", c, [(r, c) for r in range(4)]))
    # Diagonals
    lines.append(("diag", 0, [(i, i) for i in range(4)]))
    lines.append(("diag", 1, [(i, 3 - i) for i in range(4)]))

    for line_type, line_idx, cells in lines:
        for attr_name in QUARTO_BINARY_ATTRS.keys():
            attr_clean = attr_name.split("_")[0]
            bsps.append(
                {
                    "id": f"{line_type}_{line_idx}_threat_{attr_name}",
                    "description": f"{line_type.capitalize()} {line_idx} has 3 of 4 pieces sharing {attr_clean}, 4th empty",
                    "type": "binary",
                    "category": "threat_line",
                }
            )

    # ── 2x2 square threats: 3 of 4 pieces sharing attribute (36) ──────
    # All 2x2 squares on the 4x4 board (9 total)
    squares = []
    for top_r in range(3):  # 0, 1, 2
        for left_c in range(3):  # 0, 1, 2
            cells = [
                (top_r, left_c),
                (top_r, left_c + 1),
                (top_r + 1, left_c),
                (top_r + 1, left_c + 1),
            ]
            squares.append((top_r, left_c, cells))

    for sq_r, sq_c, cells in squares:
        for attr_name in QUARTO_BINARY_ATTRS.keys():
            attr_clean = attr_name.split("_")[0]
            bsps.append(
                {
                    "id": f"square_{sq_r}_{sq_c}_threat_{attr_name}",
                    "description": f"2x2 square at ({sq_r},{sq_c}) has 3 of 4 pieces sharing {attr_clean}, 4th empty",
                    "type": "binary",
                    "category": "threat_square_2x2",
                }
            )

    # ── Offered piece attributes (4) ──────────────────────────────────
    for attr_name, (pos_val, _) in QUARTO_BINARY_ATTRS.items():
        attr_clean = attr_name.split("_")[0]
        bsps.append(
            {
                "id": f"offered_{attr_name}",
                "description": f"Offered piece is {pos_val} (1) or not (0)",
                "type": "binary",
                "category": "offered_piece",
            }
        )

    # ── Game phase (3) ────────────────────────────────────────────────
    bsps.extend(
        [
            {
                "id": "game_phase_early",
                "description": "Early game (0-5 pieces on board)",
                "type": "binary",
                "category": "game_phase",
            },
            {
                "id": "game_phase_mid",
                "description": "Mid game (6-11 pieces on board)",
                "type": "binary",
                "category": "game_phase",
            },
            {
                "id": "game_phase_late",
                "description": "Late game (12-16 pieces on board)",
                "type": "binary",
                "category": "game_phase",
            },
        ]
    )

    # ── Global: winning move exists (1) ───────────────────────────────
    bsps.append(
        {
            "id": "winning_move_exists",
            "description": "There is an immediate winning placement",
            "type": "binary",
            "category": "global",
        }
    )

    return bsps


# ──────────────────────────────────────────────────────────────────────
# Othello BSPs
# ──────────────────────────────────────────────────────────────────────


def generate_othello_bsps() -> list[dict]:
    bsps = []

    for r in range(8):
        for c in range(8):
            bsps.append(
                {
                    "id": f"cell_{r}_{c}_state",
                    "description": f"State of cell ({r},{c}): empty/mine/yours",
                    "type": "categorical",
                    "category": "cell_state",
                    "values": ["empty", "mine", "yours"],
                }
            )
            bsps.append(
                {
                    "id": f"cell_{r}_{c}_legal",
                    "description": f"Is ({r},{c}) a legal move?",
                    "type": "binary",
                    "category": "legal_moves",
                }
            )

    bsps.append(
        {
            "id": "num_legal_moves",
            "description": "Number of legal moves available",
            "type": "integer",
            "category": "global",
            "range": [0, 64],
        }
    )
    bsps.append(
        {
            "id": "disc_difference",
            "description": "Difference in disc count (mine - yours)",
            "type": "integer",
            "category": "global",
            "range": [-64, 64],
        }
    )

    return bsps


# ──────────────────────────────────────────────────────────────────────
# Tic-tac-toe BSPs
# ──────────────────────────────────────────────────────────────────────


def generate_tictactoe_bsps() -> list[dict]:
    bsps = []

    for r in range(3):
        for c in range(3):
            bsps.append(
                {
                    "id": f"cell_{r}_{c}_state",
                    "description": f"State of cell ({r},{c}): empty/X/O",
                    "type": "categorical",
                    "category": "cell_state",
                    "values": ["empty", "X", "O"],
                }
            )

    # Line threats
    lines = []
    for r in range(3):
        lines.append(("row", r, [(r, c) for c in range(3)]))
    for c in range(3):
        lines.append(("col", c, [(r, c) for r in range(3)]))
    lines.append(("diag", 0, [(i, i) for i in range(3)]))
    lines.append(("diag", 1, [(i, 2 - i) for i in range(3)]))

    for line_type, line_idx, cells in lines:
        for player in ["X", "O"]:
            bsps.append(
                {
                    "id": f"{line_type}_{line_idx}_threat_{player}",
                    "description": f"{player} has 2 in {line_type} {line_idx}, 3rd empty",
                    "type": "binary",
                    "category": "threat",
                }
            )

    bsps.append(
        {
            "id": "winning_move_exists",
            "description": "Current player can win immediately",
            "type": "binary",
            "category": "global",
        }
    )

    return bsps


# ──────────────────────────────────────────────────────────────────────

GENERATORS = {
    "quarto": generate_quarto_bsps,
    "othello": generate_othello_bsps,
    "tictactoe": generate_tictactoe_bsps,
}


def main():
    args = docopt(__doc__)
    game = args["<game>"]

    if game not in GENERATORS:
        print(
            f"Unknown game '{game}'. Options: {list(GENERATORS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args["--output"] == "auto":
        output_path = PROJECT_DIR / "bsps" / game / "bsps.json"
    else:
        output_path = Path(args["--output"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bsps = GENERATORS[game]()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bsps, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "game": game,
                "num_bsps": len(bsps),
                "output": str(output_path),
                "categories": list(set(b.get("category", "unknown") for b in bsps)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
