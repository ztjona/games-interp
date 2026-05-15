"""Quarto game module for the unified-aux autoregressive champion (S4).

This is a sibling of :mod:`scripts.games.quarto` that supports models of class
:class:`models.quarto.CNN_autoreg_sa.QuartoCNNAutoregUnifiedS4`. That family
differs from ``QuartoCNN`` (uncoupled) in three important ways:

1. Forward signature is ``forward(x_board, x_aux, phase="place")`` where
   ``x_aux`` is the **32-d** vector ``[offered_one_hot ⊕ available_mask]``,
   not the 16-d offered piece.
2. Inference is phase-aware via ``predict_phase(..., phase="place"|"select")``.
3. The fc1 trunk is 512-wide (not 128).

To slot into the existing pipeline (``generate_positions.py``,
``collect_activations.py``, ``compute_bsp_labels.py``), this module exposes:

* ``load_model(path, device)`` — returns an :class:`S4Wrapper` whose
  ``forward(board, piece16)`` builds the 32-d aux internally. The wrapper
  stores the underlying S4 model as the child attribute ``s4`` so hookable
  layers are named ``s4.conv1``, ``s4.conv2``, ``s4.fc1`` etc.
* ``generate_positions(...)`` — same signature as ``quarto.generate_positions``;
  uses :class:`S4ModelBot` (backed by ``predict_phase``) for the model side
  in any opponent mode that involves the trained model.
* ``OPPONENT_MODES`` — identical four-mode set.

The on-disk position format (``boards``, ``pieces`` = 16-d offered, ``metadata``)
is unchanged, so the resulting positions files, BSP-label files, and BSP
schemas are interoperable with the existing ``compute_bsp_labels.py`` and
``sae_eval.py`` flows.
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


def _build_aux32(x_board: torch.Tensor, x_piece: torch.Tensor) -> torch.Tensor:
    """Construct the phase-stable 32-d aux from (board, offered_piece).

    ``aux = concat(offered_one_hot[16], available_pieces_mask[16])``.

    The available-mask is derived from the board: a piece type k is
    *placed* iff ``x_board[:, k, :, :].sum(dim=(1,2)) > 0``. Available =
    not placed and not offered.

    Args:
        x_board: (B, 16, 4, 4) one-hot board encoding.
        x_piece: (B, 16) one-hot offered piece (zeros if no piece is
            currently offered, e.g. at game start).

    Returns:
        (B, 32) tensor matching the contract of
        :class:`_QuartoCNNAutoregUnifiedBase`.
    """
    occupied = x_board.sum(dim=(2, 3))  # (B, 16) – piece-type presence on board
    occupied = (occupied > 0).to(x_board.dtype)
    available = (1.0 - occupied - x_piece).clamp(0.0, 1.0)
    return torch.cat([x_piece, available], dim=1)


class S4Wrapper(nn.Module):
    """Adapter exposing a ``(board, piece16)`` forward to the pipeline.

    Internally builds the 32-d aux and calls the underlying S4 model with
    ``phase="place"`` (the trunk is phase-agnostic for the unified variant,
    so the choice of phase does not affect any hookable trunk activation).
    """

    def __init__(self, s4_model: nn.Module):
        super().__init__()
        self.s4 = s4_model

    @property
    def name(self) -> str:
        inner = getattr(self.s4, "name", self.s4.__class__.__name__)
        return f"S4Wrapper({inner})"

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x_board: torch.Tensor | np.ndarray,
        x_piece: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x_board, np.ndarray):
            x_board = torch.from_numpy(x_board).float()
        if isinstance(x_piece, np.ndarray):
            x_piece = torch.from_numpy(x_piece).float()
        dev = next(self.s4.parameters()).device
        x_board = x_board.to(dev)
        x_piece = x_piece.to(dev)
        x_aux = _build_aux32(x_board, x_piece)
        return self.s4(x_board, x_aux, phase="place")


def load_model(model_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load the S4 champion and wrap it for the pipeline."""
    sys.path.insert(0, str(PROJECT_DIR))
    from models.quarto.CNN_autoreg_sa import QuartoCNNAutoregUnifiedS4

    inner = QuartoCNNAutoregUnifiedS4.from_file(str(model_path))
    inner.eval()
    inner.to(device)
    wrapper = S4Wrapper(inner)
    wrapper.eval()
    wrapper.to(device)
    return wrapper


# ──────────────────────────────────────────────────────────────────────
# Bots (BotAI subclasses, mirroring quarto.ModelBot/RandomBot)
# ──────────────────────────────────────────────────────────────────────

from quartopy import BotAI, Piece


class RandomBot(BotAI):
    """Random bot — identical to ``quarto.RandomBot``, redefined for locality."""

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


class S4ModelBot(BotAI):
    """Bot that plays via ``QuartoCNNAutoregUnifiedS4.predict_phase``.

    Mirrors :class:`scripts.games.quarto.ModelBot`'s caching contract, but uses
    the phase-aware unified-aux API. The bot receives an :class:`S4Wrapper`
    (so we can reach the underlying S4 via ``.s4``) for parity with the rest
    of the pipeline.
    """

    @property
    def name(self):
        return f"S4ModelBot({self._label})"

    def __init__(
        self,
        wrapper: S4Wrapper,
        deterministic: bool = False,
        temperature: float = 0.1,
        label: str = "S4",
        **kw,
    ):
        super().__init__()
        assert isinstance(
            wrapper, S4Wrapper
        ), "S4ModelBot expects an S4Wrapper instance produced by load_model()."
        self.wrapper = wrapper
        self.model = wrapper.s4  # the underlying autoreg-unified model
        self.DETERMINISTIC = deterministic
        self.TEMPERATURE = temperature
        self._label = label
        # Caches for ranking re-use across ith_option retries.
        self._place_ranking: torch.Tensor | None = None
        self._select_ranking: torch.Tensor | None = None
        self._place_recompute = True
        self._select_recompute = True

    # ── helpers ──────────────────────────────────────────────────────

    def _encode_board(self, game) -> torch.Tensor:
        # game_board.encode() returns shape (1, 16, 4, 4) numpy float.
        return torch.from_numpy(game.game_board.encode()).float().to(self.model.device)

    def _piece_one_hot(self, piece_index: int) -> np.ndarray:
        oh = np.zeros(16, dtype=np.float32)
        if 0 <= piece_index < 16:
            oh[piece_index] = 1.0
        return oh

    def _available_pieces_mask(self, game) -> np.ndarray:
        mask = np.zeros(16, dtype=np.float32)
        for piece in game.storage_board.get_valid_pieces():
            mask += piece.vectorize_onehot().reshape(-1)
        return mask

    def _selected_piece_index(self, game) -> int:
        if isinstance(game.selected_piece, Piece):
            return int(np.argmax(game.selected_piece.vectorize_onehot()))
        return -1

    def _build_aux(self, game, *, offered_index: int) -> torch.Tensor:
        offered = self._piece_one_hot(offered_index)
        available = self._available_pieces_mask(game)
        aux = np.concatenate([offered, available]).astype(np.float32).reshape(1, -1)
        return torch.from_numpy(aux).to(self.model.device)

    # ── placement ────────────────────────────────────────────────────

    def place_piece(self, game, piece, ith_option=0, *a, **kw):
        if ith_option == 0:
            self._place_recompute = True
        if self._place_recompute:
            board_t = self._encode_board(game)
            # The piece passed in is the one this player is about to place.
            offered_idx = int(np.argmax(piece.vectorize_onehot()))
            aux_t = self._build_aux(game, offered_index=offered_idx)
            self._place_ranking = self.model.predict_phase(
                board_t,
                aux_t,
                phase="place",
                TEMPERATURE=self.TEMPERATURE,
                DETERMINISTIC=self.DETERMINISTIC,
            )
            self._place_recompute = False
            # After a placement the cached select ranking is stale.
            self._select_recompute = True

        valid_moves = set(game.game_board.get_valid_moves())
        for idx in self._place_ranking[0].detach().cpu().tolist():
            pos = game.game_board.get_position_index(int(idx))
            if pos in valid_moves:
                if ith_option == 0:
                    return pos
                ith_option -= 1
        raise RuntimeError("S4ModelBot: no valid placement found in ranking.")

    # ── selection ────────────────────────────────────────────────────

    def select(self, game, ith_option=0, *a, **kw):
        if ith_option == 0:
            self._select_recompute = True
        if self._select_recompute:
            board_t = self._encode_board(game)
            # In the select phase, the "offered" slot is the piece this player
            # most recently placed; that piece is no longer present in
            # ``game.storage_board``. We approximate it as "no offered" (zeros).
            # The unified trunk only consumes the 32-d aux through the
            # phase-agnostic encoder, so this matches the contract used at
            # training time as long as the available_mask is correct.
            aux_t = self._build_aux(game, offered_index=-1)
            self._select_ranking = self.model.predict_phase(
                board_t,
                aux_t,
                phase="select",
                TEMPERATURE=self.TEMPERATURE,
                DETERMINISTIC=self.DETERMINISTIC,
            )
            self._select_recompute = False
            self._place_recompute = True

        valid_indices = {
            int(np.argmax(p.vectorize_onehot()))
            for p in game.storage_board.get_valid_pieces()
        }
        for idx in self._select_ranking[0].detach().cpu().tolist():
            i = int(idx)
            if i in valid_indices:
                if ith_option == 0:
                    return Piece.from_index(i)
                ith_option -= 1
        raise RuntimeError("S4ModelBot: no valid piece found in ranking.")


# ──────────────────────────────────────────────────────────────────────
# Position generation (mirrors quarto.generate_positions exactly,
# substituting S4ModelBot for ModelBot)
# ──────────────────────────────────────────────────────────────────────


def _load_shared_model(
    opponents: str,
    model_path: str | Path | None,
    model2_path: str | Path | None,
    device: str,
):
    if opponents == "random_v_random":
        return None, None
    if model_path is None:
        raise ValueError(
            f"opponents='{opponents}' requires a model, but no model_path was given"
        )
    model1 = load_model(model_path, device=device)
    if opponents == "model_v_random":
        return model1, None
    if opponents == "random_v_model":
        return None, model1
    if opponents == "model_v_model":
        if model2_path:
            model2 = load_model(model2_path, device=device)
            return model1, model2
        return model1, model1
    return model1, None


def _make_bots(opponents, model1, model2):
    if opponents == "random_v_random":
        return RandomBot(), RandomBot()
    if opponents == "model_v_random":
        return (
            S4ModelBot(model1, deterministic=False, temperature=0.1),
            RandomBot(),
        )
    if opponents == "random_v_model":
        return (
            RandomBot(),
            S4ModelBot(model2, deterministic=False, temperature=0.1),
        )
    if opponents == "model_v_model":
        return (
            S4ModelBot(model1, deterministic=False, temperature=0.1),
            S4ModelBot(model2, deterministic=False, temperature=0.1),
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
    """Self-play positions using the S4 champion (or random) as either side.

    Output layout is identical to :func:`scripts.games.quarto.generate_positions`
    so downstream pipeline files (BSP labels, activations) remain compatible.
    """
    from random import seed as rseed

    from quartopy import QuartoGame
    from tqdm.auto import tqdm

    rseed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    shared_model = _load_shared_model(opponents, model_path, None, device)

    boards: list[np.ndarray] = []
    pieces: list[np.ndarray] = []
    metadata: list[dict] = []

    desc = f"Quarto-S4 [{opponents}]"
    for game_idx in tqdm(range(num_games), desc=desc):
        bot1, bot2 = _make_bots(opponents, *shared_model)
        game = QuartoGame(player1=bot1, player2=bot2, mode_2x2=True)

        turn_count = 0
        while not game.player_won and not game.game_board.is_full():
            game.play_turn()
            if not game.pick:
                pass
            else:
                board_enc = game.game_board.encode()[0]
                offered = game.selected_piece
                if isinstance(offered, Piece):
                    piece_vec = offered.vectorize_onehot()
                else:
                    piece_vec = np.zeros(16, dtype=np.float32)

                n_pieces = sum(
                    1
                    for r in range(4)
                    for c in range(4)
                    if not game.game_board.is_empty(r, c)
                )

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

    return boards, pieces, metadata


# ──────────────────────────────────────────────────────────────────────
# BSP delegation — the schema is position-only, identical to the
# uncoupled-model game. We import lazily so this module stays optional.
# ──────────────────────────────────────────────────────────────────────


def get_all_bsp_definitions(*args, **kwargs):
    from .quarto import get_all_bsp_definitions as _impl

    return _impl(*args, **kwargs)


def compute_bsp_vector(*args, **kwargs):
    from .quarto import compute_bsp_vector as _impl

    return _impl(*args, **kwargs)


# Mirror BSP_SETS if the sibling module exposes it.
try:
    from .quarto import BSP_SETS  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pass
