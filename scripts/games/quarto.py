"""Quarto-specific position generation and model loading.

Depends on the ``quartopy`` library for game logic.

Opponent modes
--------------
- ``random_v_random``  – broad uniform coverage of the game tree
- ``model_v_random``   – trained model (P1) vs random (P2)
- ``random_v_model``   – random (P1) vs trained model (P2)
- ``model_v_model``    – strategic positions from competent self-play

Note: All modes collect activations from YOUR trained model on the generated
positions. The opponent mode controls position distribution, not which model
is interpreted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

OPPONENT_MODES = (
    "random_v_random",
    "model_v_random",
    "random_v_model",
    "model_v_model",
)


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────


def load_model(model_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load a QuartoCNN model from a weights file."""
    sys.path.insert(0, str(PROJECT_DIR))
    from models.quarto.CNN_uncoupled import QuartoCNN

    model = QuartoCNN.from_file(str(model_path))
    model.eval()
    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────
# Bots
# ──────────────────────────────────────────────────────────────────────


class _RandomBot:
    """Minimal random bot for self-play (duck-types BotAI)."""

    @property
    def name(self):
        return "RandomBot"

    def __init__(self, **kw):
        pass

    def select(self, game, ith_option=0, *a, **kw):
        from random import choice

        r, c = choice(game.storage_board.get_valid_moves())
        return game.storage_board.get_piece(r, c)

    def place_piece(self, game, piece, ith_option=0, *a, **kw):
        from random import choice

        return choice(game.game_board.get_valid_moves())


class _ModelBot:
    """Bot that plays using a trained QuartoCNN checkpoint.

    Mirrors the caching logic of the project's ``CNN_bot``:
    - A single forward pass produces *both* board-position and piece-selection
      rankings, cached in ``_board_ranking`` / ``_piece_ranking``.
    - ``place_piece(ith_option=0)`` triggers a fresh forward pass; retries
      (ith_option > 0) walk down the cached ranking.
    - ``select()`` reuses the ranking from the most recent forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        deterministic: bool = False,
        temperature: float = 0.1,
        **kw,
    ):
        self.model = model
        self.deterministic = deterministic
        self.temperature = temperature
        self._recalculate = True
        self._board_ranking: torch.Tensor | None = None  # (1, 16)
        self._piece_ranking: torch.Tensor | None = None  # (1, 16)

    @property
    def name(self):
        return f"ModelBot({getattr(self.model, 'name', 'unknown')})"

    # ── internal forward pass ────────────────────────────────────────

    def _calculate(self, game):
        from quartopy.game.piece import Piece

        if self._recalculate:
            board_t = torch.tensor(
                game.game_board.encode(), dtype=torch.float32
            )  # (1, 16, 4, 4)

            offered = game.selected_piece
            if isinstance(offered, Piece):
                piece_oh = offered.vectorize_onehot().reshape(1, -1)
            else:
                piece_oh = np.zeros((1, 16), dtype=np.float32)

            piece_t = torch.tensor(piece_oh, dtype=torch.float32)

            self._board_ranking, self._piece_ranking = self.model.predict(
                board_t,
                piece_t,
                TEMPERATURE=self.temperature,
                DETERMINISTIC=self.deterministic,
            )
            self._recalculate = False

    # ── placement ────────────────────────────────────────────────────

    def place_piece(self, game, piece, ith_option=0, *a, **kw):
        self._recalculate = True
        self._calculate(game)

        # Walk board ranking, skip occupied cells
        valid_count = 0
        for rank_pos in range(self._board_ranking.shape[1]):
            idx = int(self._board_ranking[0, rank_pos].item())
            r, c = divmod(idx, 4)
            if game.game_board.is_empty(r, c):
                if valid_count == ith_option:
                    return (r, c)
                valid_count += 1

        # Fallback (should never happen)
        from random import choice

        return choice(game.game_board.get_valid_moves())

    # ── selection ────────────────────────────────────────────────────

    def select(self, game, ith_option=0, *a, **kw):
        from quartopy.game.piece import Piece

        # On the very first selection of a game, no placement has happened
        # yet, so we need a forward pass with the current (empty) board.
        if self._piece_ranking is None:
            self._recalculate = True
            self._calculate(game)

        # Walk piece ranking, skip pieces no longer in storage
        valid_count = 0
        for rank_pos in range(self._piece_ranking.shape[1]):
            piece_idx = int(self._piece_ranking[0, rank_pos].item())
            piece = Piece.from_index(piece_idx)
            if game.storage_board.find_piece(piece) is not None:
                if valid_count == ith_option:
                    return piece
                valid_count += 1

        # Fallback (should never happen)
        from random import choice

        r, c = choice(game.storage_board.get_valid_moves())
        return game.storage_board.get_piece(r, c)


# ──────────────────────────────────────────────────────────────────────
# Position generation
# ──────────────────────────────────────────────────────────────────────


def _load_shared_model(
    opponents: str,
    model_path: str | Path | None,
    model2_path: str | Path | None,
    device: str,
) -> tuple[nn.Module | None, nn.Module | None]:
    """Load models once (shared across all games).

    Returns (model1, model2) where:
    - random_v_random: (None, None)
    - model_v_random: (model, None)
    - random_v_model: (None, model)
    - model_v_model (same): (model, model) - same instance
    - model_v_model (diff): (model1, model2) - different instances
    """
    if opponents == "random_v_random":
        return None, None

    if model_path is None:
        raise ValueError(
            f"opponents='{opponents}' requires a model, but no model_path was given"
        )

    model1 = load_model(model_path, device=device)

    if opponents == "model_v_random":
        return model1, None
    elif opponents == "random_v_model":
        return None, model1
    elif opponents == "model_v_model":
        if model2_path:
            # Different models
            model2 = load_model(model2_path, device=device)
            return model1, model2
        else:
            # Same model (self-play)
            return model1, model1

    return model1, None


def _make_bots(
    opponents: str,
    model1: nn.Module | None,
    model2: nn.Module | None,
) -> tuple[object, object]:
    """Instantiate fresh bots for a single game (no model reloading)."""
    if opponents == "random_v_random":
        return _RandomBot(), _RandomBot()
    if opponents == "model_v_random":
        return (
            _ModelBot(model1, deterministic=False, temperature=0.1),
            _RandomBot(),
        )
    if opponents == "random_v_model":
        return (
            _RandomBot(),
            _ModelBot(model2, deterministic=False, temperature=0.1),
        )
    if opponents == "model_v_model":
        return (
            _ModelBot(model1, deterministic=False, temperature=0.1),
            _ModelBot(model2, deterministic=False, temperature=0.1),
        )
    raise ValueError(
        f"Unknown opponent mode '{opponents}'. Choose from: {OPPONENT_MODES}"
    )


def generate_positions(
    num_games: int,
    seed: int = 42,
    opponents: str = "random_v_random",
    model_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Generate board positions from Quarto games.

    For each *placement* turn we record:
      - board encoding  (16, 4, 4)  one-hot
      - offered piece   (16,)       one-hot
      - metadata dict with ground-truth info for BSP labelling

    Parameters
    ----------
    num_games : int
        Number of games to play.
    seed : int
        Random seed for reproducibility.
    opponents : str
        One of ``random_v_random``, ``model_v_random``, ``random_v_model``,
        ``model_v_model``.
    model_path : str | Path | None
        Checkpoint used by model bots. Required for all modes (even
        random_v_random, as YOUR model processes all positions afterward).
    device : str
        Torch device for model bots.

    Returns
    -------
    boards   : list[np.ndarray]  (16, 4, 4)
    pieces   : list[np.ndarray]  (16,)
    metadata : list[dict]
    """
    from random import seed as rseed

    from quartopy import QuartoGame
    from quartopy.game.piece import Piece
    from tqdm.auto import tqdm

    rseed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model once, share across all games
    shared_model = _load_shared_model(opponents, model_path, device)

    boards: list[np.ndarray] = []
    pieces: list[np.ndarray] = []
    metadata: list[dict] = []

    desc = f"Quarto [{opponents}]"

    for game_idx in tqdm(range(num_games), desc=desc):
        bot1, bot2 = _make_bots(opponents, shared_model)
        game = QuartoGame(player1=bot1, player2=bot2, mode_2x2=False)

        turn_count = 0
        while not game.player_won and not game.game_board.is_full():
            game.play_turn()

            # After a placement turn, record the resulting board state
            if not game.pick:
                # Selection turn → skip (no board change)
                pass
            else:
                # Placement turn (pick just toggled to True)
                board_enc = game.game_board.encode()[0]  # (16, 4, 4)

                offered = game.selected_piece
                if isinstance(offered, Piece):
                    piece_vec = offered.vectorize_onehot()  # (16,)
                else:
                    piece_vec = np.zeros(16, dtype=np.float32)

                # Piece count
                n_pieces = sum(
                    1
                    for r in range(4)
                    for c in range(4)
                    if not game.game_board.is_empty(r, c)
                )

                # Cell-level metadata
                cell_info: dict[str, Any] = {}
                for r in range(4):
                    for c in range(4):
                        occupied = not game.game_board.is_empty(r, c)
                        cell_info[f"{r}_{c}_occupied"] = occupied
                        if occupied:
                            p = game.game_board.get_piece(r, c)
                            cell_info[f"{r}_{c}_size"] = p.size.value
                            cell_info[f"{r}_{c}_coloration"] = p.coloration.value
                            cell_info[f"{r}_{c}_shape"] = p.shape.value
                            cell_info[f"{r}_{c}_hole"] = p.hole.value

                # Offered piece attributes
                offered_attrs: dict[str, str] = {}
                if isinstance(offered, Piece):
                    offered_attrs["size"] = offered.size.value
                    offered_attrs["coloration"] = offered.coloration.value
                    offered_attrs["shape"] = offered.shape.value
                    offered_attrs["hole"] = offered.hole.value

                meta = {
                    "game_idx": game_idx,
                    "turn": turn_count,
                    "n_pieces": n_pieces,
                    "player_won": game.player_won,
                    "cells": cell_info,
                    "offered_piece": offered_attrs,
                }

                boards.append(board_enc.astype(np.float32))
                pieces.append(piece_vec.astype(np.float32))
                metadata.append(meta)

                turn_count += 1

            game.cambiar_turno()

    return boards, pieces, metadata


# ──────────────────────────────────────────────────────────────────────
# BSP (Board State Property) Computation
# ──────────────────────────────────────────────────────────────────────

# Binary encoding convention:
#   cell_r_c_size_tall:         1 = TALL,   0 = SHORT
#   cell_r_c_coloration_dark:   1 = DARK,   0 = LIGHT
#   cell_r_c_shape_square:      1 = SQUARE, 0 = ROUND
#   cell_r_c_hole_hollow:       1 = HOLLOW, 0 = SOLID

BINARY_ATTRS = {
    "size_tall": ("TALL", "SHORT"),
    "coloration_dark": ("DARK", "LIGHT"),
    "shape_square": ("SQUARE", "ROUND"),
    "hole_hollow": ("HOLLOW", "SOLID"),
}


def get_all_bsp_definitions() -> list[dict]:
    """Return metadata for all available BSPs (164 total).

    All BSPs are binary (1 or 0). Categories:
    - cell_occupancy: 16 BSPs
    - cell_attribute: 64 BSPs
    - threat_line: 40 BSPs (rows, cols, diagonals)
    - threat_square_2x2: 36 BSPs (2x2 square patterns)
    - offered_piece: 4 BSPs
    - game_phase: 3 BSPs
    - global: 1 BSP
    """
    bsps = []

    # Cell occupancy (16)
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

    # Cell attributes (64)
    for r in range(4):
        for c in range(4):
            for attr_name, (pos_val, neg_val) in BINARY_ATTRS.items():
                bsps.append(
                    {
                        "id": f"cell_{r}_{c}_{attr_name}",
                        "description": f"Piece at ({r},{c}) is {pos_val} (1) else {neg_val} (0)",
                        "type": "binary",
                        "category": "cell_attribute",
                    }
                )

    # Line threats (40)
    lines = []
    for r in range(4):
        lines.append(("row", r, [(r, c) for c in range(4)]))
    for c in range(4):
        lines.append(("col", c, [(r, c) for r in range(4)]))
    lines.append(("diag", 0, [(i, i) for i in range(4)]))
    lines.append(("diag", 1, [(i, 3 - i) for i in range(4)]))

    for line_type, line_idx, cells in lines:
        for attr_name in BINARY_ATTRS.keys():
            attr_clean = attr_name.split("_")[0]
            bsps.append(
                {
                    "id": f"{line_type}_{line_idx}_threat_{attr_name}",
                    "description": f"{line_type.capitalize()} {line_idx} has 3 of 4 sharing {attr_clean}",
                    "type": "binary",
                    "category": "threat_line",
                }
            )

    # 2x2 square threats (36)
    for top_r in range(3):
        for left_c in range(3):
            for attr_name in BINARY_ATTRS.keys():
                attr_clean = attr_name.split("_")[0]
                bsps.append(
                    {
                        "id": f"square_{top_r}_{left_c}_threat_{attr_name}",
                        "description": f"2x2 square at ({top_r},{left_c}) has 3 of 4 sharing {attr_clean}",
                        "type": "binary",
                        "category": "threat_square_2x2",
                    }
                )

    # Offered piece (4)
    for attr_name, (pos_val, neg_val) in BINARY_ATTRS.items():
        bsps.append(
            {
                "id": f"offered_{attr_name}",
                "description": f"Offered piece is {pos_val} (1) else {neg_val} (0)",
                "type": "binary",
                "category": "offered_piece",
            }
        )

    # Game phase (3)
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

    # Global (1)
    bsps.append(
        {
            "id": "winning_move_exists",
            "description": "There is an immediate winning placement",
            "type": "binary",
            "category": "global",
        }
    )

    return bsps


def compute_bsp_vector(
    metadata: dict,
    bsp_ids: list[str] | None = None,
) -> np.ndarray:
    """Compute binary BSP vector for a single game state.

    Parameters
    ----------
    metadata : dict
        Metadata dict from generate_positions() with keys:
        - 'cells': dict with '{r}_{c}_occupied', '{r}_{c}_{attr}' keys
        - 'offered_piece': dict with 'size', 'coloration', 'shape', 'hole'
        - 'n_pieces': int
        - 'player_won': bool (for determining winning_move_exists)
    bsp_ids : list[str] | None
        BSP IDs to compute. If None, computes all 164 BSPs.

    Returns
    -------
    np.ndarray
        Binary vector of shape (len(bsp_ids),) with 0s and 1s
    """
    if bsp_ids is None:
        bsp_ids = [b["id"] for b in get_all_bsp_definitions()]

    cells = metadata.get("cells", {})
    offered = metadata.get("offered_piece", {})
    n_pieces = metadata.get("n_pieces", 0)

    values = []

    for bsp_id in bsp_ids:
        values.append(_compute_single_bsp(bsp_id, cells, offered, n_pieces))

    return np.array(values, dtype=np.float32)


def _compute_single_bsp(
    bsp_id: str,
    cells: dict,
    offered: dict,
    n_pieces: int,
) -> float:
    """Compute a single BSP value (0.0 or 1.0)."""

    # Cell occupancy
    if "occupied" in bsp_id and not bsp_id.startswith("offered"):
        parts = bsp_id.split("_")  # ['cell', r, c, 'occupied']
        key = f"{parts[1]}_{parts[2]}_occupied"
        return 1.0 if cells.get(key, False) else 0.0

    # Cell attributes
    if bsp_id.startswith("cell_") and any(a in bsp_id for a in BINARY_ATTRS):
        parts = bsp_id.split("_")  # ['cell', r, c, attr, value]
        r, c = parts[1], parts[2]
        attr_key = f"{r}_{c}_{parts[3]}"  # e.g., "0_0_size"

        # Check if cell is occupied
        if not cells.get(f"{r}_{c}_occupied", False):
            return 0.0

        # Get the attribute value from metadata
        attr_raw = parts[3]  # "size", "coloration", "shape", "hole"
        attr_value = cells.get(attr_key, "")

        # Determine binary convention
        attr_name = "_".join(parts[3:])  # "size_tall", "coloration_dark", etc.
        if attr_name in BINARY_ATTRS:
            pos_val, _ = BINARY_ATTRS[attr_name]
            return 1.0 if attr_value == pos_val else 0.0

        return 0.0

    # Line threats
    if bsp_id.startswith(("row_", "col_", "diag_")) and "threat" in bsp_id:
        return _compute_line_threat(bsp_id, cells)

    # Square threats
    if bsp_id.startswith("square_") and "threat" in bsp_id:
        return _compute_square_threat(bsp_id, cells)

    # Offered piece
    if bsp_id.startswith("offered_"):
        attr_name = bsp_id.replace("offered_", "")  # "size_tall", etc.
        if attr_name in BINARY_ATTRS:
            pos_val, _ = BINARY_ATTRS[attr_name]
            attr_raw = attr_name.split("_")[0]  # "size", "coloration", etc.
            return 1.0 if offered.get(attr_raw, "") == pos_val else 0.0
        return 0.0

    # Game phase
    if bsp_id.startswith("game_phase_"):
        if "early" in bsp_id:
            return 1.0 if n_pieces <= 5 else 0.0
        elif "mid" in bsp_id:
            return 1.0 if 6 <= n_pieces <= 11 else 0.0
        elif "late" in bsp_id:
            return 1.0 if n_pieces >= 12 else 0.0

    # Winning move exists
    if bsp_id == "winning_move_exists":
        return _compute_winning_move_exists(cells, offered)

    return 0.0


def _compute_line_threat(bsp_id: str, cells: dict) -> float:
    """Check if a line has 3 of 4 pieces sharing an attribute."""
    # Parse: "row_2_threat_size_tall" → line_type='row', idx=2, attr='size_tall'
    parts = bsp_id.split("_")
    line_type = parts[0]  # "row", "col", "diag"
    line_idx = int(parts[1])
    attr_name = "_".join(parts[3:])  # "size_tall", "coloration_dark", etc.

    # Get cell coordinates
    if line_type == "row":
        coords = [(line_idx, c) for c in range(4)]
    elif line_type == "col":
        coords = [(r, line_idx) for r in range(4)]
    elif line_type == "diag" and line_idx == 0:
        coords = [(i, i) for i in range(4)]
    elif line_type == "diag" and line_idx == 1:
        coords = [(i, 3 - i) for i in range(4)]
    else:
        return 0.0

    # Count pieces with attribute and empty cells
    attr_raw = attr_name.split("_")[0]
    pos_val, _ = BINARY_ATTRS[attr_name]

    matching = 0
    empty = 0

    for r, c in coords:
        key_occupied = f"{r}_{c}_occupied"
        if not cells.get(key_occupied, False):
            empty += 1
        else:
            key_attr = f"{r}_{c}_{attr_raw}"
            if cells.get(key_attr, "") == pos_val:
                matching += 1

    # Threat: exactly 3 matching, 1 empty
    return 1.0 if (matching == 3 and empty == 1) else 0.0


def _compute_square_threat(bsp_id: str, cells: dict) -> float:
    """Check if a 2x2 square has 3 of 4 pieces sharing an attribute."""
    # Parse: "square_1_2_threat_shape_square" → top_r=1, left_c=2, attr='shape_square'
    parts = bsp_id.split("_")
    top_r = int(parts[1])
    left_c = int(parts[2])
    attr_name = "_".join(parts[4:])  # "size_tall", etc.

    coords = [
        (top_r, left_c),
        (top_r, left_c + 1),
        (top_r + 1, left_c),
        (top_r + 1, left_c + 1),
    ]

    attr_raw = attr_name.split("_")[0]
    pos_val, _ = BINARY_ATTRS[attr_name]

    matching = 0
    empty = 0

    for r, c in coords:
        key_occupied = f"{r}_{c}_occupied"
        if not cells.get(key_occupied, False):
            empty += 1
        else:
            key_attr = f"{r}_{c}_{attr_raw}"
            if cells.get(key_attr, "") == pos_val:
                matching += 1

    # Threat: exactly 3 matching, 1 empty
    return 1.0 if (matching == 3 and empty == 1) else 0.0


def _compute_winning_move_exists(cells: dict, offered: dict) -> float:
    """Check if placing the offered piece anywhere would complete a line or 2x2 square.

    A winning move exists if the offered piece can be placed in an empty cell such that
    it completes a line (row/col/diagonal) or 2x2 square where all 4 pieces share at
    least one attribute.
    """
    if not offered:  # No piece offered yet
        return 0.0

    # Get offered piece attributes
    offered_attrs = {
        "size": offered.get("size", ""),
        "coloration": offered.get("coloration", ""),
        "shape": offered.get("shape", ""),
        "hole": offered.get("hole", ""),
    }

    # Find all empty cells
    empty_cells = []
    for r in range(4):
        for c in range(4):
            if not cells.get(f"{r}_{c}_occupied", False):
                empty_cells.append((r, c))

    if not empty_cells:
        return 0.0

    # For each empty cell, check if placing offered piece would win
    for r, c in empty_cells:
        if _would_win_at_position(r, c, offered_attrs, cells):
            return 1.0

    return 0.0


def _would_win_at_position(r: int, c: int, offered_attrs: dict, cells: dict) -> bool:
    """Check if placing a piece with offered_attrs at (r,c) would complete a winning line/square."""

    # Check all lines containing (r, c)
    lines_to_check = [
        [(r, i) for i in range(4)],  # row
        [(i, c) for i in range(4)],  # column
    ]

    # Add diagonals if on diagonal
    if r == c:  # main diagonal
        lines_to_check.append([(i, i) for i in range(4)])
    if r + c == 3:  # anti-diagonal
        lines_to_check.append([(i, 3 - i) for i in range(4)])

    # Check each line
    for line in lines_to_check:
        if _line_would_be_complete(line, r, c, offered_attrs, cells):
            return True

    # Check all 2x2 squares containing (r, c)
    # A cell can be part of up to 4 different 2x2 squares
    for top_r in range(max(0, r - 1), min(3, r + 1)):
        for left_c in range(max(0, c - 1), min(3, c + 1)):
            square = [
                (top_r, left_c),
                (top_r, left_c + 1),
                (top_r + 1, left_c),
                (top_r + 1, left_c + 1),
            ]
            if (r, c) in square:
                if _square_would_be_complete(square, r, c, offered_attrs, cells):
                    return True

    return False


def _line_would_be_complete(
    line: list, new_r: int, new_c: int, offered_attrs: dict, cells: dict
) -> bool:
    """Check if a line would be complete with the new piece."""
    # Collect attributes of all pieces in the line (including the hypothetical new one)
    pieces = []

    for r, c in line:
        if r == new_r and c == new_c:
            # This is the position where we're placing the offered piece
            pieces.append(offered_attrs)
        else:
            # Get existing piece attributes (if occupied)
            if cells.get(f"{r}_{c}_occupied", False):
                pieces.append(
                    {
                        "size": cells.get(f"{r}_{c}_size", ""),
                        "coloration": cells.get(f"{r}_{c}_coloration", ""),
                        "shape": cells.get(f"{r}_{c}_shape", ""),
                        "hole": cells.get(f"{r}_{c}_hole", ""),
                    }
                )
            else:
                # Empty cell (line not yet full)
                return False

    # Check if line is full (should be 4 pieces)
    if len(pieces) != 4:
        return False

    # Check if all 4 pieces share at least one attribute
    for attr in ["size", "coloration", "shape", "hole"]:
        values = [p[attr] for p in pieces if p[attr]]
        if len(values) == 4 and len(set(values)) == 1:
            return True

    return False


def _square_would_be_complete(
    square: list, new_r: int, new_c: int, offered_attrs: dict, cells: dict
) -> bool:
    """Check if a 2x2 square would be complete with the new piece."""
    pieces = []

    for r, c in square:
        if r == new_r and c == new_c:
            pieces.append(offered_attrs)
        else:
            if cells.get(f"{r}_{c}_occupied", False):
                pieces.append(
                    {
                        "size": cells.get(f"{r}_{c}_size", ""),
                        "coloration": cells.get(f"{r}_{c}_coloration", ""),
                        "shape": cells.get(f"{r}_{c}_shape", ""),
                        "hole": cells.get(f"{r}_{c}_hole", ""),
                    }
                )
            else:
                return False

    if len(pieces) != 4:
        return False

    # Check if all 4 pieces share at least one attribute
    for attr in ["size", "coloration", "shape", "hole"]:
        values = [p[attr] for p in pieces if p[attr]]
        if len(values) == 4 and len(set(values)) == 1:
            return True

    return False
