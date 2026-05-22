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
# Bots (proper BotAI subclasses for quartopy compatibility)
# ──────────────────────────────────────────────────────────────────────

from quartopy import BotAI, Piece


class RandomBot(BotAI):
    """Random bot that extends BotAI for proper quartopy integration."""

    @property
    def name(self):
        return "RandomBot"

    def __init__(self, **kw):
        super().__init__()

    def select(self, game, ith_option=0, *a, **kw):
        from random import choice

        r, c = choice(game.storage_board.get_valid_moves())
        return game.storage_board.get_piece(r, c)

    def place_piece(self, game, piece, ith_option=0, *a, **kw):
        from random import choice

        return choice(game.game_board.get_valid_moves())


class ModelBot(BotAI):
    """Bot that plays using a trained QuartoCNN checkpoint.

    Extends BotAI for proper quartopy integration.
    Mirrors the caching logic of hierarchical-SAE's ``CNN_bot``:
    - A single forward pass produces *both* board-position and piece-selection
      rankings, cached in ``_board_ranking`` / ``_piece_ranking``.
    - ``place_piece(ith_option=0)`` triggers a fresh forward pass; retries
      (ith_option > 0) walk down the cached ranking.
    - ``select()`` reuses the ranking from the most recent forward pass.
    """

    @property
    def name(self):
        return f"ModelBot({self._label})"

    def __init__(
        self,
        model: nn.Module,
        deterministic: bool = False,
        temperature: float = 0.1,
        label: str = "unknown",
        **kw,
    ):
        super().__init__()
        self.model = model
        self.DETERMINISTIC = deterministic
        self.TEMPERATURE = temperature
        self._label = label
        self._recalculate = True
        self._board_ranking: torch.Tensor | None = None
        self._piece_ranking: torch.Tensor | None = None

    # ── internal forward pass ────────────────────────────────────────

    def _calculate(self, game):
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
                TEMPERATURE=self.TEMPERATURE,
                DETERMINISTIC=self.DETERMINISTIC,
            )
            self._recalculate = False

    # ── placement ────────────────────────────────────────────────────

    def place_piece(self, game, piece, ith_option=0, *a, **kw):
        if ith_option == 0:
            self._recalculate = True
        self._calculate(game)

        idx_board = int(self._board_ranking[0, ith_option].item())
        return game.game_board.get_position_index(idx_board)

    # ── selection ────────────────────────────────────────────────────

    def select(self, game, ith_option=0, *a, **kw):
        # On the very first selection of a game, no placement has happened
        # yet, so we need a forward pass with the current (empty) board.
        if self._piece_ranking is None:
            self._recalculate = True
            self._calculate(game)

        idx_piece = int(self._piece_ranking[0, ith_option].item())
        return Piece.from_index(idx_piece)


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
        return RandomBot(), RandomBot()
    if opponents == "model_v_random":
        return (
            ModelBot(model1, deterministic=False, temperature=0.1),
            RandomBot(),
        )
    if opponents == "random_v_model":
        return (
            RandomBot(),
            ModelBot(model2, deterministic=False, temperature=0.1),
        )
    if opponents == "model_v_model":
        return (
            ModelBot(model1, deterministic=False, temperature=0.1),
            ModelBot(model2, deterministic=False, temperature=0.1),
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
    from tqdm.auto import tqdm

    rseed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model once, share across all games
    shared_model = _load_shared_model(opponents, model_path, None, device)

    boards: list[np.ndarray] = []
    pieces: list[np.ndarray] = []
    metadata: list[dict] = []

    desc = f"Quarto [{opponents}]"

    for game_idx in tqdm(range(num_games), desc=desc):
        bot1, bot2 = _make_bots(opponents, *shared_model)
        game = QuartoGame(player1=bot1, player2=bot2, mode_2x2=True)

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

# Binary encoding convention — using quartopy's own naming:
#   cell_r_c_tall:        1 = TALL,        0 = LITTLE
#   cell_r_c_black:       1 = BLACK,       0 = WHITE
#   cell_r_c_square:      1 = SQUARE,      0 = CIRCLE
#   cell_r_c_with_hole:   1 = WITH_HOLE,   0 = WITHOUT_HOLE
#
# BINARY_ATTRS: bsp_suffix → (metadata_key, positive_value, negative_value)

BINARY_ATTRS = {
    "tall": ("size", "TALL", "LITTLE"),
    "black": ("coloration", "BLACK", "WHITE"),
    "square": ("shape", "SQUARE", "CIRCLE"),
    "with_hole": ("hole", "WITH_HOLE", "WITHOUT_HOLE"),
}


# Named BSP sets (animal → list of categories). Each set is intended to be
# evaluated as a unit; the union (337) is a *menu*, not a usable set, because
# gorilla and hawk are alternative bases for the same threat concepts (see
# BSP-schema-summary.md: "Gorilla ↔ Hawk Correspondence"). Always evaluate
# gorilla and hawk separately and report per-category.
BSP_SETS: dict[str, list[str]] = {
    "gorilla": [
        "cell_occupancy",
        "cell_attribute",
        "threat_line",
        "threat_square_2x2",
        "offered_piece",
        "game_phase",
        "global",
    ],
    "hawk": [
        "reframed_count",
        "reframed_completable",
        "reframed_any_threat",
        "reframed_sq_count",
        "reframed_sq_completable",
        "reframed_sq_any_threat",
        "reframed_global",
    ],
    "tiger": [
        # Agent-relative threats — see
        # docs/diary/2026-05-22_reframings-audit-tiger.md
        "tiger_decision_global",
        "tiger_offered_completing_attr",
        "tiger_line_winnable",
        "tiger_square_winnable",
        "tiger_pool_winning_count",
        "tiger_pool_safe_count",
    ],
}


def get_all_bsp_definitions() -> list[dict]:
    """Return metadata for all available BSPs (337 total — a *menu*, not a set).

    The 337 returned BSPs are the union of two **alternative** sets that should
    be evaluated separately (see ``BSP_SETS`` above and BSP-schema-summary.md).
    Use ``compute_bsp_labels.py --name gorilla`` (or ``--name hawk``), which
    resolves to the appropriate categories automatically.


    All BSPs are binary (1 or 0).

    Gorilla (164 BSPs):
    - cell_occupancy: 16 BSPs
    - cell_attribute: 64 BSPs
    - threat_line: 40 BSPs (rows, cols, diagonals)
    - threat_square_2x2: 36 BSPs (2x2 square patterns)
    - offered_piece: 4 BSPs
    - game_phase: 3 BSPs
    - global: 1 BSP

    Hawk — reframed (173 BSPs):
    - reframed_count: 40 BSPs (line count >=3)
    - reframed_completable: 40 BSPs (line threat + offered matches)
    - reframed_any_threat: 10 BSPs (any attribute threat per line)
    - reframed_sq_count: 36 BSPs (2x2 square count >=3)
    - reframed_sq_completable: 36 BSPs (2x2 square threat + offered matches)
    - reframed_sq_any_threat: 9 BSPs (any attribute threat per square)
    - reframed_global: 2 BSPs (board-level threat/completable)
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
            for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
                bsps.append(
                    {
                        "id": f"cell_{r}_{c}_{suffix}",
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
    lines.append(("diag", "main", [(i, i) for i in range(4)]))
    lines.append(("diag", "anti", [(i, 3 - i) for i in range(4)]))

    for line_type, line_idx, cells in lines:
        for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
            bsps.append(
                {
                    "id": f"{line_type}_{line_idx}_threat_{suffix}",
                    "description": f"{line_type.capitalize()} {line_idx}: 3 of 4 are {pos_val}",
                    "type": "binary",
                    "category": "threat_line",
                }
            )

    # 2x2 square threats (36)
    for top_r in range(3):
        for left_c in range(3):
            for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
                bsps.append(
                    {
                        "id": f"square_{top_r}_{left_c}_threat_{suffix}",
                        "description": f"2x2 at ({top_r},{left_c}): 3 of 4 are {pos_val}",
                        "type": "binary",
                        "category": "threat_square_2x2",
                    }
                )

    # Offered piece (4)
    for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
        bsps.append(
            {
                "id": f"offered_{suffix}",
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

    # ── Reframed BSPs (hawk set — Nanda-inspired) ──────────────────────────
    # These reframe threat concepts under alternative bases that may match
    # the model's internal representation more closely.

    # Reframed count: ≥3 pieces in line share attribute (40)
    for line_type, line_idx, cells_coords in lines:
        for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
            bsps.append(
                {
                    "id": f"{line_type}_{line_idx}_count_ge3_{suffix}",
                    "description": f"{line_type.capitalize()} {line_idx}: >=3 occupied cells are {pos_val}",
                    "type": "binary",
                    "category": "reframed_count",
                }
            )

    # Reframed completable: threat AND offered piece has that attribute (40)
    for line_type, line_idx, cells_coords in lines:
        for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
            bsps.append(
                {
                    "id": f"{line_type}_{line_idx}_completable_{suffix}",
                    "description": f"{line_type.capitalize()} {line_idx}: threat in {pos_val} AND offered piece is {pos_val}",
                    "type": "binary",
                    "category": "reframed_completable",
                }
            )

    # Reframed any-threat: any attribute creates a threat in this line (10)
    for line_type, line_idx, cells_coords in lines:
        bsps.append(
            {
                "id": f"{line_type}_{line_idx}_any_threat",
                "description": f"{line_type.capitalize()} {line_idx}: at least one attribute has a threat pattern",
                "type": "binary",
                "category": "reframed_any_threat",
            }
        )

    # ── Reframed 2×2 square BSPs (hawk set) ──────────────────────────────
    squares = []
    for top_r in range(3):
        for left_c in range(3):
            squares.append((top_r, left_c))

    # Reframed square count: ≥3 pieces in 2×2 square share attribute (36)
    for top_r, left_c in squares:
        for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
            bsps.append(
                {
                    "id": f"square_{top_r}_{left_c}_count_ge3_{suffix}",
                    "description": f"2x2 at ({top_r},{left_c}): >=3 occupied cells are {pos_val}",
                    "type": "binary",
                    "category": "reframed_sq_count",
                }
            )

    # Reframed square completable: threat AND offered piece has attribute (36)
    for top_r, left_c in squares:
        for suffix, (meta_key, pos_val, neg_val) in BINARY_ATTRS.items():
            bsps.append(
                {
                    "id": f"square_{top_r}_{left_c}_completable_{suffix}",
                    "description": f"2x2 at ({top_r},{left_c}): threat in {pos_val} AND offered piece is {pos_val}",
                    "type": "binary",
                    "category": "reframed_sq_completable",
                }
            )

    # Reframed square any-threat: any attribute creates a threat in this square (9)
    for top_r, left_c in squares:
        bsps.append(
            {
                "id": f"square_{top_r}_{left_c}_any_threat",
                "description": f"2x2 at ({top_r},{left_c}): at least one attribute has a threat pattern",
                "type": "binary",
                "category": "reframed_sq_any_threat",
            }
        )

    # Reframed global (2)
    bsps.append(
        {
            "id": "board_threat_exists",
            "description": "At least one threat exists on any line",
            "type": "binary",
            "category": "reframed_global",
        }
    )
    bsps.append(
        {
            "id": "board_completable_exists",
            "description": "At least one threat is completable with the offered piece",
            "type": "binary",
            "category": "reframed_global",
        }
    )

    # ── Tiger BSPs (agent-relative; see 2026-05-22_reframings-audit-tiger.md) ──

    # tiger_decision_global (5)
    bsps.append(
        {
            "id": "tiger_win_now_exists",
            "description": "Current player can win this turn by placing the offered piece somewhere",
            "type": "binary",
            "category": "tiger_decision_global",
        }
    )
    bsps.append(
        {
            "id": "tiger_opp_winning_offer_exists",
            "description": "At least one pool piece, if offered, lets the opponent win on their next turn",
            "type": "binary",
            "category": "tiger_decision_global",
        }
    )
    bsps.append(
        {
            "id": "tiger_lose_next_forced",
            "description": "Every pool piece, if offered, lets the opponent win — i.e. forced to gift a win",
            "type": "binary",
            "category": "tiger_decision_global",
        }
    )
    bsps.append(
        {
            "id": "tiger_safe_offer_exists",
            "description": "At least one pool piece can be offered without giving opponent an immediate win",
            "type": "binary",
            "category": "tiger_decision_global",
        }
    )
    bsps.append(
        {
            "id": "tiger_every_offer_safe",
            "description": "Every pool piece is a safe offer (no pool piece gives opponent an immediate win)",
            "type": "binary",
            "category": "tiger_decision_global",
        }
    )

    # tiger_offered_completing_attr (4) — offered piece is a winning completer for attribute X
    for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
        bsps.append(
            {
                "id": f"tiger_offered_completes_{suffix}",
                "description": (
                    f"Offered piece is {pos_val} AND at least one line/2x2 on the board has "
                    f"3 cells already matching {pos_val} (offered piece is a winning completer "
                    f"via the {meta_key} attribute)"
                ),
                "type": "binary",
                "category": "tiger_offered_completing_attr",
            }
        )

    # tiger_line_winnable (10) — per-line: placing offered on the empty cell wins this line
    for line_type, line_idx, _coords in _ALL_LINE_COORDS:
        bsps.append(
            {
                "id": f"tiger_line_{line_type}_{line_idx}_winnable",
                "description": (
                    f"{line_type.capitalize()} {line_idx}: line has exactly 1 empty cell AND "
                    f"placing the offered piece there completes the line (≥1 shared attribute)"
                ),
                "type": "binary",
                "category": "tiger_line_winnable",
            }
        )

    # tiger_square_winnable (9) — per 2x2 square
    for top_r, left_c, _coords in _ALL_SQUARE_COORDS:
        bsps.append(
            {
                "id": f"tiger_square_{top_r}_{left_c}_winnable",
                "description": (
                    f"2x2 at ({top_r},{left_c}): exactly 1 empty cell AND placing the "
                    f"offered piece there completes the square (≥1 shared attribute)"
                ),
                "type": "binary",
                "category": "tiger_square_winnable",
            }
        )

    # tiger_pool_winning_count (4) — bucketed count of "poison" pool pieces
    for bucket in ("ge1", "ge2", "ge4", "all"):
        bsps.append(
            {
                "id": f"tiger_pool_winning_count_{bucket}",
                "description": (
                    f"Count of pool pieces that, if offered, let opponent win immediately. "
                    f"Bucket: {bucket} (≥1 / ≥2 / ≥4 / = pool_size)"
                ),
                "type": "binary",
                "category": "tiger_pool_winning_count",
            }
        )

    # tiger_pool_safe_count (4) — bucketed count of safe-to-offer pool pieces
    for bucket in ("ge1", "ge2", "ge4", "eq0"):
        bsps.append(
            {
                "id": f"tiger_pool_safe_count_{bucket}",
                "description": (
                    f"Count of pool pieces that can be safely offered (opponent cannot win). "
                    f"Bucket: {bucket} (≥1 / ≥2 / ≥4 / = 0, the last being the forced-loss signal)"
                ),
                "type": "binary",
                "category": "tiger_pool_safe_count",
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

    # Cell attributes — e.g. "cell_0_0_tall", "cell_2_3_with_hole"
    if bsp_id.startswith("cell_") and not bsp_id.endswith("_occupied"):
        parts = bsp_id.split("_")  # ['cell', r, c, suffix...]
        r, c = parts[1], parts[2]
        suffix = "_".join(parts[3:])  # 'tall', 'black', 'with_hole', etc.

        if suffix not in BINARY_ATTRS:
            return 0.0

        if not cells.get(f"{r}_{c}_occupied", False):
            return 0.0

        meta_key, pos_val, _ = BINARY_ATTRS[suffix]
        return 1.0 if cells.get(f"{r}_{c}_{meta_key}", "") == pos_val else 0.0

    # Line threats
    if (
        bsp_id.startswith(("row_", "col_", "diag_"))
        and "threat" in bsp_id
        and "any_threat" not in bsp_id
    ):
        return _compute_line_threat(bsp_id, cells)

    # Square threats (gorilla)
    if bsp_id.startswith("square_") and "_threat_" in bsp_id:
        return _compute_square_threat(bsp_id, cells)

    # ── Reframed BSPs (hawk) — lines ─────────────────────────────────────

    # Count ≥3 in line — e.g. "row_0_count_ge3_tall"
    if "count_ge3" in bsp_id and bsp_id.startswith(("row_", "col_", "diag_")):
        return _compute_line_count_ge3(bsp_id, cells)

    # Completable threat — e.g. "row_0_completable_tall"
    if "completable_" in bsp_id and bsp_id.startswith(("row_", "col_", "diag_")):
        return _compute_line_completable(bsp_id, cells, offered)

    # Any threat in line — e.g. "row_0_any_threat"
    if bsp_id.endswith("_any_threat") and bsp_id.startswith(("row_", "col_", "diag_")):
        return _compute_line_any_threat(bsp_id, cells)

    # ── Reframed BSPs (hawk) — 2×2 squares ──────────────────────────────

    # Count ≥3 in square — e.g. "square_0_0_count_ge3_tall"
    if bsp_id.startswith("square_") and "count_ge3" in bsp_id:
        return _compute_square_count_ge3(bsp_id, cells)

    # Completable threat in square — e.g. "square_0_0_completable_tall"
    if bsp_id.startswith("square_") and "completable_" in bsp_id:
        return _compute_square_completable(bsp_id, cells, offered)

    # Any threat in square — e.g. "square_0_0_any_threat"
    if bsp_id.startswith("square_") and "any_threat" in bsp_id:
        return _compute_square_any_threat(bsp_id, cells)

    # Board-level reframed globals
    if bsp_id == "board_threat_exists":
        return _compute_board_threat_exists(cells)

    if bsp_id == "board_completable_exists":
        return _compute_board_completable_exists(cells, offered)

    # Offered piece — e.g. "offered_tall", "offered_with_hole"
    if bsp_id.startswith("offered_"):
        suffix = bsp_id.replace("offered_", "")  # 'tall', 'black', etc.
        if suffix in BINARY_ATTRS:
            meta_key, pos_val, _ = BINARY_ATTRS[suffix]
            return 1.0 if offered.get(meta_key, "") == pos_val else 0.0
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

    # ── Tiger BSPs (agent-relative) ───────────────────────────────────────
    if bsp_id.startswith("tiger_"):
        return _compute_tiger_bsp(bsp_id, cells, offered)

    return 0.0


# ── Tiger BSP helpers (agent-relative — pool reasoning) ──────────────────

# All 16 Quarto pieces as (size, coloration, shape, hole) attribute dicts.
_ALL_PIECES: tuple[dict[str, str], ...] = tuple(
    {
        "size": BINARY_ATTRS["tall"][1 if a else 2],
        "coloration": BINARY_ATTRS["black"][1 if b else 2],
        "shape": BINARY_ATTRS["square"][1 if c else 2],
        "hole": BINARY_ATTRS["with_hole"][1 if d else 2],
    }
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
    for d in (0, 1)
)


def _piece_key(attrs: dict) -> tuple[str, str, str, str]:
    return (
        attrs.get("size", ""),
        attrs.get("coloration", ""),
        attrs.get("shape", ""),
        attrs.get("hole", ""),
    )


def _enumerate_pool(cells: dict, offered: dict) -> list[dict]:
    """Return the unplaced, unoffered pieces — the pool the current player draws from."""
    placed_keys: set[tuple[str, str, str, str]] = set()
    for r in range(4):
        for c in range(4):
            if cells.get(f"{r}_{c}_occupied", False):
                placed_keys.add(
                    (
                        cells.get(f"{r}_{c}_size", ""),
                        cells.get(f"{r}_{c}_coloration", ""),
                        cells.get(f"{r}_{c}_shape", ""),
                        cells.get(f"{r}_{c}_hole", ""),
                    )
                )
    offered_key = _piece_key(offered) if offered else None
    pool = []
    for p in _ALL_PIECES:
        k = _piece_key(p)
        if k in placed_keys:
            continue
        if offered_key is not None and k == offered_key:
            continue
        pool.append(p)
    return pool


def _tiger_pool_stats(cells: dict, offered: dict) -> dict:
    """Compute and cache pool-relative stats on ``cells`` for reuse across tiger BSPs.

    Stashes under ``cells['_tiger_pool_stats']`` so subsequent tiger BSPs on the
    same position pay zero pool-enumeration cost. ``cells`` is freshly built per
    position in ``generate_positions``, so mutation is safe.
    """
    cached = cells.get("_tiger_pool_stats")
    if cached is not None:
        return cached
    pool = _enumerate_pool(cells, offered)
    pool_wins = [
        bool(_compute_winning_move_exists(cells, p)) for p in pool
    ]  # True ↔ "offering this piece lets opponent win"
    stats = {
        "pool_size": len(pool),
        "n_winning": sum(pool_wins),
        "n_safe": sum(1 for w in pool_wins if not w),
    }
    cells["_tiger_pool_stats"] = stats
    return stats


def _tiger_line_winnable(line_type: str, line_idx: str, cells: dict, offered: dict) -> float:
    """Line has exactly 1 empty cell AND placing the offered piece there completes it."""
    if not offered:
        return 0.0
    coords = _parse_line_coords([line_type, line_idx])
    if coords is None:
        return 0.0
    empty_cells = [(r, c) for r, c in coords if not cells.get(f"{r}_{c}_occupied", False)]
    if len(empty_cells) != 1:
        return 0.0
    r, c = empty_cells[0]
    offered_attrs = {
        "size": offered.get("size", ""),
        "coloration": offered.get("coloration", ""),
        "shape": offered.get("shape", ""),
        "hole": offered.get("hole", ""),
    }
    return 1.0 if _line_would_be_complete(coords, r, c, offered_attrs, cells) else 0.0


def _tiger_square_winnable(top_r: int, left_c: int, cells: dict, offered: dict) -> float:
    """2x2 square has exactly 1 empty cell AND placing the offered piece there completes it."""
    if not offered:
        return 0.0
    coords = [
        (top_r, left_c),
        (top_r, left_c + 1),
        (top_r + 1, left_c),
        (top_r + 1, left_c + 1),
    ]
    empty_cells = [(r, c) for r, c in coords if not cells.get(f"{r}_{c}_occupied", False)]
    if len(empty_cells) != 1:
        return 0.0
    r, c = empty_cells[0]
    offered_attrs = {
        "size": offered.get("size", ""),
        "coloration": offered.get("coloration", ""),
        "shape": offered.get("shape", ""),
        "hole": offered.get("hole", ""),
    }
    return 1.0 if _square_would_be_complete(coords, r, c, offered_attrs, cells) else 0.0


def _tiger_offered_completes(suffix: str, cells: dict, offered: dict) -> float:
    """Offered piece has attribute X AND at least one line/2x2 has 3 cells matching X."""
    if not offered or suffix not in BINARY_ATTRS:
        return 0.0
    meta_key, pos_val, _ = BINARY_ATTRS[suffix]
    if offered.get(meta_key, "") != pos_val:
        return 0.0
    # Look for any line/square with exactly 3 matching + 1 empty
    for _, _, coords in _ALL_LINE_COORDS:
        matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
        if matching == 3 and empty == 1:
            return 1.0
    for _, _, coords in _ALL_SQUARE_COORDS:
        matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
        if matching == 3 and empty == 1:
            return 1.0
    return 0.0


def _compute_tiger_bsp(bsp_id: str, cells: dict, offered: dict) -> float:
    """Dispatch for tiger_* BSPs."""

    # tiger_decision_global
    if bsp_id == "tiger_win_now_exists":
        return _compute_winning_move_exists(cells, offered)

    if bsp_id in (
        "tiger_opp_winning_offer_exists",
        "tiger_lose_next_forced",
        "tiger_safe_offer_exists",
        "tiger_every_offer_safe",
    ):
        stats = _tiger_pool_stats(cells, offered)
        if stats["pool_size"] == 0:
            # No pool → no offering decision to be made (terminal position).
            return 0.0
        n_win = stats["n_winning"]
        n_safe = stats["n_safe"]
        if bsp_id == "tiger_opp_winning_offer_exists":
            return 1.0 if n_win >= 1 else 0.0
        if bsp_id == "tiger_lose_next_forced":
            return 1.0 if n_win == stats["pool_size"] else 0.0
        if bsp_id == "tiger_safe_offer_exists":
            return 1.0 if n_safe >= 1 else 0.0
        if bsp_id == "tiger_every_offer_safe":
            return 1.0 if n_safe == stats["pool_size"] else 0.0

    # tiger_pool_winning_count_{ge1,ge2,ge4,all}
    if bsp_id.startswith("tiger_pool_winning_count_"):
        bucket = bsp_id[len("tiger_pool_winning_count_") :]
        stats = _tiger_pool_stats(cells, offered)
        if stats["pool_size"] == 0:
            return 0.0
        n = stats["n_winning"]
        if bucket == "ge1":
            return 1.0 if n >= 1 else 0.0
        if bucket == "ge2":
            return 1.0 if n >= 2 else 0.0
        if bucket == "ge4":
            return 1.0 if n >= 4 else 0.0
        if bucket == "all":
            return 1.0 if n == stats["pool_size"] else 0.0
        return 0.0

    # tiger_pool_safe_count_{ge1,ge2,ge4,eq0}
    if bsp_id.startswith("tiger_pool_safe_count_"):
        bucket = bsp_id[len("tiger_pool_safe_count_") :]
        stats = _tiger_pool_stats(cells, offered)
        if stats["pool_size"] == 0:
            return 0.0
        n = stats["n_safe"]
        if bucket == "ge1":
            return 1.0 if n >= 1 else 0.0
        if bucket == "ge2":
            return 1.0 if n >= 2 else 0.0
        if bucket == "ge4":
            return 1.0 if n >= 4 else 0.0
        if bucket == "eq0":
            return 1.0 if n == 0 else 0.0
        return 0.0

    # tiger_offered_completes_{tall,black,square,with_hole}
    if bsp_id.startswith("tiger_offered_completes_"):
        suffix = bsp_id[len("tiger_offered_completes_") :]
        return _tiger_offered_completes(suffix, cells, offered)

    # tiger_line_{row,col,diag}_{idx}_winnable
    if bsp_id.startswith("tiger_line_") and bsp_id.endswith("_winnable"):
        # "tiger_line_row_2_winnable" → parts: ['tiger','line','row','2','winnable']
        parts = bsp_id.split("_")
        line_type = parts[2]
        line_idx = parts[3]
        return _tiger_line_winnable(line_type, line_idx, cells, offered)

    # tiger_square_{top_r}_{left_c}_winnable
    if bsp_id.startswith("tiger_square_") and bsp_id.endswith("_winnable"):
        # "tiger_square_0_1_winnable" → parts: ['tiger','square','0','1','winnable']
        parts = bsp_id.split("_")
        try:
            top_r = int(parts[2])
            left_c = int(parts[3])
        except (ValueError, IndexError):
            return 0.0
        return _tiger_square_winnable(top_r, left_c, cells, offered)

    return 0.0


def _compute_line_threat(bsp_id: str, cells: dict) -> float:
    """Check if a line has 3 of 4 pieces sharing an attribute."""
    # Parse: "row_2_threat_tall" → line_type='row', idx='2', suffix='tall'
    #        "diag_main_threat_with_hole" → line_type='diag', idx='main', suffix='with_hole'
    parts = bsp_id.split("_")
    line_type = parts[0]  # "row", "col", "diag"
    line_idx = parts[1]  # numeric string or "main"/"anti"
    suffix = "_".join(parts[3:])  # 'tall', 'black', 'with_hole', etc.

    # Get cell coordinates
    if line_type == "row":
        coords = [(int(line_idx), c) for c in range(4)]
    elif line_type == "col":
        coords = [(r, int(line_idx)) for r in range(4)]
    elif line_type == "diag" and line_idx == "main":
        coords = [(i, i) for i in range(4)]
    elif line_type == "diag" and line_idx == "anti":
        coords = [(i, 3 - i) for i in range(4)]
    else:
        return 0.0

    meta_key, pos_val, _ = BINARY_ATTRS[suffix]

    matching = 0
    empty = 0

    for r, c in coords:
        if not cells.get(f"{r}_{c}_occupied", False):
            empty += 1
        else:
            if cells.get(f"{r}_{c}_{meta_key}", "") == pos_val:
                matching += 1

    # Threat: exactly 3 matching, 1 empty
    return 1.0 if (matching == 3 and empty == 1) else 0.0


def _compute_square_threat(bsp_id: str, cells: dict) -> float:
    """Check if a 2x2 square has 3 of 4 pieces sharing an attribute."""
    # Parse: "square_1_2_threat_square" → top_r=1, left_c=2, suffix='square'
    #        "square_0_0_threat_with_hole" → top_r=0, left_c=0, suffix='with_hole'
    parts = bsp_id.split("_")
    top_r = int(parts[1])
    left_c = int(parts[2])
    suffix = "_".join(parts[4:])  # 'tall', 'black', 'with_hole', etc.

    coords = [
        (top_r, left_c),
        (top_r, left_c + 1),
        (top_r + 1, left_c),
        (top_r + 1, left_c + 1),
    ]

    meta_key, pos_val, _ = BINARY_ATTRS[suffix]

    matching = 0
    empty = 0

    for r, c in coords:
        if not cells.get(f"{r}_{c}_occupied", False):
            empty += 1
        else:
            if cells.get(f"{r}_{c}_{meta_key}", "") == pos_val:
                matching += 1

    # Threat: exactly 3 matching, 1 empty
    return 1.0 if (matching == 3 and empty == 1) else 0.0


# ── Reframed BSP helpers (hawk set) ──────────────────────────────────────


def _parse_line_coords(parts: list[str]) -> list[tuple[int, int]] | None:
    """Parse line type and index from split BSP ID parts, return cell coordinates."""
    line_type = parts[0]
    line_idx = parts[1]
    if line_type == "row":
        return [(int(line_idx), c) for c in range(4)]
    elif line_type == "col":
        return [(r, int(line_idx)) for r in range(4)]
    elif line_type == "diag" and line_idx == "main":
        return [(i, i) for i in range(4)]
    elif line_type == "diag" and line_idx == "anti":
        return [(i, 3 - i) for i in range(4)]
    return None


def _parse_square_coords(parts: list[str]) -> list[tuple[int, int]] | None:
    """Parse square top-left from split BSP ID parts, return 4 cell coordinates."""
    # "square_0_0_count_ge3_tall" -> parts[0]='square', [1]='0', [2]='0', ...
    if parts[0] != "square":
        return None
    top_r = int(parts[1])
    left_c = int(parts[2])
    return [
        (top_r, left_c),
        (top_r, left_c + 1),
        (top_r + 1, left_c),
        (top_r + 1, left_c + 1),
    ]


def _count_matching_in_line(
    coords: list[tuple[int, int]], meta_key: str, pos_val: str, cells: dict
) -> tuple[int, int]:
    """Count pieces matching attribute and empty cells in a line.

    Returns (matching_count, empty_count).
    """
    matching = 0
    empty = 0
    for r, c in coords:
        if not cells.get(f"{r}_{c}_occupied", False):
            empty += 1
        elif cells.get(f"{r}_{c}_{meta_key}", "") == pos_val:
            matching += 1
    return matching, empty


def _compute_line_count_ge3(bsp_id: str, cells: dict) -> float:
    """>=3 occupied pieces in line share attribute (ignores empty cells).

    E.g. "row_0_count_ge3_tall": among occupied cells in row 0, are >=3 TALL?
    """
    parts = bsp_id.split("_")
    # "row_0_count_ge3_tall" -> parts[0]='row', [1]='0', [2]='count', [3]='ge3', [4:]='tall'
    # "diag_main_count_ge3_with_hole" -> [0]='diag', [1]='main', [2]='count', [3]='ge3', [4:]='with_hole'
    suffix = "_".join(parts[4:])
    coords = _parse_line_coords(parts)
    if coords is None or suffix not in BINARY_ATTRS:
        return 0.0

    meta_key, pos_val, _ = BINARY_ATTRS[suffix]
    matching, _ = _count_matching_in_line(coords, meta_key, pos_val, cells)
    return 1.0 if matching >= 3 else 0.0


def _compute_line_completable(bsp_id: str, cells: dict, offered: dict) -> float:
    """Threat exists AND offered piece has the same attribute.

    E.g. "row_0_completable_tall": row 0 has threat in TALL AND offered piece is TALL.
    """
    parts = bsp_id.split("_")
    # "row_0_completable_tall" -> [0]='row', [1]='0', [2]='completable', [3:]='tall'
    suffix = "_".join(parts[3:])
    coords = _parse_line_coords(parts)
    if coords is None or suffix not in BINARY_ATTRS:
        return 0.0

    meta_key, pos_val, _ = BINARY_ATTRS[suffix]

    # Check threat first
    matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
    if not (matching == 3 and empty == 1):
        return 0.0

    # Check offered piece has the attribute
    return 1.0 if offered.get(meta_key, "") == pos_val else 0.0


def _compute_line_any_threat(bsp_id: str, cells: dict) -> float:
    """Any attribute creates a threat in this line.

    E.g. "row_0_any_threat": row 0 has a threat for at least one of tall/black/square/with_hole.
    """
    parts = bsp_id.split("_")
    # "row_0_any_threat" -> [0]='row', [1]='0', [2]='any', [3]='threat'
    coords = _parse_line_coords(parts)
    if coords is None:
        return 0.0

    for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
        matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
        if matching == 3 and empty == 1:
            return 1.0
    return 0.0


# ── Reframed 2×2 square BSP helpers ─────────────────────────────────────


def _compute_square_count_ge3(bsp_id: str, cells: dict) -> float:
    """>=3 occupied pieces in 2x2 square share attribute.

    E.g. "square_0_0_count_ge3_tall": among occupied cells in 2x2 at (0,0), are >=3 TALL?
    """
    parts = bsp_id.split("_")
    # "square_0_0_count_ge3_tall" -> [0]='square', [1]='0', [2]='0', [3]='count', [4]='ge3', [5:]='tall'
    suffix = "_".join(parts[5:])
    coords = _parse_square_coords(parts)
    if coords is None or suffix not in BINARY_ATTRS:
        return 0.0

    meta_key, pos_val, _ = BINARY_ATTRS[suffix]
    matching, _ = _count_matching_in_line(coords, meta_key, pos_val, cells)
    return 1.0 if matching >= 3 else 0.0


def _compute_square_completable(bsp_id: str, cells: dict, offered: dict) -> float:
    """Threat in 2x2 square AND offered piece has matching attribute.

    E.g. "square_0_0_completable_tall": 2x2 at (0,0) has threat in TALL AND offered is TALL.
    """
    parts = bsp_id.split("_")
    # "square_0_0_completable_tall" -> [0]='square', [1]='0', [2]='0', [3]='completable', [4:]='tall'
    suffix = "_".join(parts[4:])
    coords = _parse_square_coords(parts)
    if coords is None or suffix not in BINARY_ATTRS:
        return 0.0

    meta_key, pos_val, _ = BINARY_ATTRS[suffix]

    # Check threat first
    matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
    if not (matching == 3 and empty == 1):
        return 0.0

    # Check offered piece has the attribute
    return 1.0 if offered.get(meta_key, "") == pos_val else 0.0


def _compute_square_any_threat(bsp_id: str, cells: dict) -> float:
    """Any attribute creates a threat in this 2x2 square.

    E.g. "square_0_0_any_threat": 2x2 at (0,0) has a threat for any attribute.
    """
    parts = bsp_id.split("_")
    # "square_0_0_any_threat" -> [0]='square', [1]='0', [2]='0', [3]='any', [4]='threat'
    coords = _parse_square_coords(parts)
    if coords is None:
        return 0.0

    for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
        matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
        if matching == 3 and empty == 1:
            return 1.0
    return 0.0


_ALL_LINE_COORDS = (
    [("row", i, [(i, c) for c in range(4)]) for i in range(4)]
    + [("col", i, [(r, i) for r in range(4)]) for i in range(4)]
    + [("diag", "main", [(i, i) for i in range(4)])]
    + [("diag", "anti", [(i, 3 - i) for i in range(4)])]
)

_ALL_SQUARE_COORDS = [
    (
        top_r,
        left_c,
        [
            (top_r, left_c),
            (top_r, left_c + 1),
            (top_r + 1, left_c),
            (top_r + 1, left_c + 1),
        ],
    )
    for top_r in range(3)
    for left_c in range(3)
]


def _compute_board_threat_exists(cells: dict) -> float:
    """At least one line or 2x2 square has a threat for any attribute."""
    for _, _, coords in _ALL_LINE_COORDS:
        for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
            matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
            if matching == 3 and empty == 1:
                return 1.0
    for _, _, coords in _ALL_SQUARE_COORDS:
        for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
            matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
            if matching == 3 and empty == 1:
                return 1.0
    return 0.0


def _compute_board_completable_exists(cells: dict, offered: dict) -> float:
    """At least one line or 2x2 square has a threat completable with the offered piece."""
    if not offered:
        return 0.0
    for _, _, coords in _ALL_LINE_COORDS:
        for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
            matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
            if matching == 3 and empty == 1:
                if offered.get(meta_key, "") == pos_val:
                    return 1.0
    for _, _, coords in _ALL_SQUARE_COORDS:
        for suffix, (meta_key, pos_val, _) in BINARY_ATTRS.items():
            matching, empty = _count_matching_in_line(coords, meta_key, pos_val, cells)
            if matching == 3 and empty == 1:
                if offered.get(meta_key, "") == pos_val:
                    return 1.0
    return 0.0


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
