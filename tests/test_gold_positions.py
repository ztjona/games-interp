"""The gold<k> position generator (3B-causal Wave 1b pre-registration, 2026-09-14, S4.1).

A gold game is the champion against itself: every decision random up to and
including the k-th placement and the selection after it, then the legal argmax
for both sides; only placement decisions from the (k+1)-th on are recorded.
Checked on an untrained S4Hot network (CPU, seconds): what the protocol says is
a property of the generator, not of a trained champion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.games import quarto_s4  # noqa: E402


@pytest.fixture(scope="module")
def wrapper():
    from models.quarto.CNN_autoreg_sa import QuartoCNNAutoregUnifiedS4Hot

    torch.manual_seed(0)
    inner = QuartoCNNAutoregUnifiedS4Hot()
    inner.device = torch.device("cpu")
    inner.eval()
    return quarto_s4.S4Wrapper(inner).eval()


def play(monkeypatch, wrapper, mode, n_games=12, seed=7):
    monkeypatch.setattr(quarto_s4, "load_model", lambda path, device="cpu": wrapper)
    boards, pieces, meta = quarto_s4.generate_positions(
        n_games, seed=seed, opponents=mode, model_path="untrained")
    return torch.from_numpy(np.stack(boards)), torch.from_numpy(np.stack(pieces)), meta


def legal_argmax(q, legal):
    return int(torch.where(legal, q, torch.full_like(q, -torch.inf)).argmax())


def greedy_next(wrapper, board, piece):
    """The legal-argmax placement of `piece` on `board`, then the legal-argmax selection."""
    b, p = board[None], piece[None]
    q = wrapper.s4.q_values_phase(b, quarto_s4._build_aux32(b, p), phase="place")[0]
    cell = legal_argmax(q, board.sum(0).flatten() == 0)
    after = board.clone()
    after[:, cell // 4, cell % 4] = piece
    q = wrapper.s4.q_values_phase(after[None], quarto_s4._build_aux32(after[None], torch.zeros(1, 16)),
                                  phase="select")[0]
    return after, legal_argmax(q, after.sum((1, 2)) == 0)


@pytest.mark.parametrize("mode,k", [("gold3", 3), ("gold12", 12), ("gold0", 0)])
def test_gold_prefix_parses(mode, k):
    assert quarto_s4.gold_prefix(mode) == k


@pytest.mark.parametrize("mode", ["gold", "model_v_model", "random_v_random", "xgold3", "gold3x"])
def test_other_modes_are_not_gold(mode):
    assert quarto_s4.gold_prefix(mode) is None


@pytest.mark.parametrize("k", [1, 3, 5])
def test_records_only_placements_from_k_plus_one(monkeypatch, wrapper, k):
    boards, pieces, meta = play(monkeypatch, wrapper, f"gold{k}")
    n_on_board = boards.sum((1, 2, 3))
    assert int(n_on_board.min()) >= k
    assert all(m["turn"] == m["n_pieces"] == int(n) for m, n in zip(meta, n_on_board))
    assert torch.all(pieces.sum(1) == 1)                       # a piece is always in hand
    assert torch.all((boards.sum((2, 3)) * pieces).sum(1) == 0)  # ... and it is not on the board
    if k <= 3:  # no Quarto before four pieces: every game reaches its (k+1)-th placement
        firsts = [m for m in meta if m["turn"] == k]
        assert sorted(m["game_idx"] for m in firsts) == list(range(12))


@pytest.mark.parametrize("k", [3, 5])
def test_continuation_is_the_legal_argmax(monkeypatch, wrapper, k):
    boards, pieces, meta = play(monkeypatch, wrapper, f"gold{k}")
    checked = 0
    for i in range(len(meta) - 1):
        if meta[i + 1]["game_idx"] != meta[i]["game_idx"]:
            continue
        after, sel = greedy_next(wrapper, boards[i], pieces[i])
        assert torch.equal(after, boards[i + 1]), f"row {i}: placement is not the legal argmax"
        assert sel == int(pieces[i + 1].argmax()), f"row {i}: selection is not the legal argmax"
        checked += 1
    assert checked > 20


def test_prefix_is_random_and_seeded(monkeypatch, wrapper):
    b1, p1, m1 = play(monkeypatch, wrapper, "gold3", seed=11)
    b2, p2, _ = play(monkeypatch, wrapper, "gold3", seed=11)
    assert torch.equal(b1, b2) and torch.equal(p1, p2)
    first = {(b1[i].numpy().tobytes(), p1[i].numpy().tobytes()) for i, m in enumerate(m1) if m["turn"] == 3}
    assert len(first) > 6  # 12 random 3-placement prefixes are almost surely distinct
    b3, _, _ = play(monkeypatch, wrapper, "gold3", seed=12)
    assert not (b3.shape == b1.shape and torch.equal(b3, b1))


def test_gold0_only_the_first_piece_is_random(monkeypatch, wrapper):
    """k = 0: the random player hands over the first piece, then leaves. Everything
    after is deterministic, so there are at most 16 distinct games."""
    boards, pieces, meta = play(monkeypatch, wrapper, "gold0", n_games=40)
    games = {}
    for i, m in enumerate(meta):
        games.setdefault(m["game_idx"], []).append((boards[i].numpy().tobytes(), pieces[i].numpy().tobytes()))
    distinct = {tuple(g) for g in games.values()}
    first_pieces = {g[0][1] for g in games.values()}
    assert 1 < len(distinct) == len(first_pieces) <= 16  # 1 would mean no hand-over at all
