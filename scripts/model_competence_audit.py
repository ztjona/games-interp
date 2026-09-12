"""Model competence audit for Quarto DQN.

Five behavioral tests that probe whether the model's *outputs* reflect
strategic understanding — independent of any SAE. Use this to gate
interpretability claims: if the model itself does not represent a concept
(e.g. completable threats), no SAE will recover it.

Tests
-----
A. Winning placement       — When a winning placement exists for the offered
                              piece, does the model's argmax legal placement
                              land on a winning cell?
B. Losing-piece avoidance  — When at least one safe AND one losing piece are
                              available, does the model's argmax legal piece
                              selection avoid the losing pieces?  Forced-loss
                              positions (every available piece loses) are
                              excluded.
C. Offered-piece sensitivity
                            — For each base position, swap the offered piece
                              for every other available piece and re-run
                              placement.  Reports the mean fraction of
                              distinct chosen cells (1.0 = every piece picks
                              a different cell, ~1/n_empty = invariant).
D. Q-occupancy gap         — Mean Q(empty cells) − mean Q(occupied cells)
                              from the unmasked board head.  A model that
                              has learned legality should give occupied
                              cells much lower Q.
E. Phase-stratified Q entropy
                            — Entropy of softmax(legal Q-values) bucketed by
                              piece count.  A competent model should be more
                              decisive (lower entropy) late game.
F. Raw illegal first choice
                            — How often the UNMASKED top-1 is illegal: an
                              occupied cell (place head, full dataset) or a
                              piece not in storage (select head, on the
                              post-placement states Test B builds).  Also the
                              number of engine retries (ith_option) before a
                              legal move.  The Quartopy engine retries invalid
                              moves instead of penalising them, so a model can
                              play well with a high F -- which is exactly why
                              Q on illegal actions must never be read as a
                              preference.  Bucketed by pieces on board.

Usage:
    model_competence_audit.py [--model-config=<yaml>]
                              [--model=<path>] [--random-model=<path>]
                              [--game=<name>]
                              [--positions=<path>] [--num-positions=<N>]
                              [--device=<dev>] [--output=<path>]
                              [--seed=<int>]
    model_competence_audit.py (-h | --help)

Options:
    --model-config=<yaml>   Champion model YAML (configs/models/<champ>.yaml).
                            Provides `path`, `random_path`, and `game`.
                            Overrides --model / --random-model / --game when given.
    --model=<path>          Trained model checkpoint
                            [default: models/quarto/20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt]
    --random-model=<path>   Random-init checkpoint for control comparison.
                            Pass "none" to skip.
                            [default: models/quarto/20260226_1420-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_0000.pt]
    --game=<name>           Game module name (e.g. quarto, quarto_s4).
                            [default: quarto]
    --positions=<path>      Position dataset
                            [default: data/quarto/positions-amalgam_unique.pt]
    --num-positions=<N>     Random subsample for tests A/B/C (D/E use full
                            set since they're vectorizable). [default: 2000]
    --device=<dev>          torch device [default: cuda]
    --output=<path>         JSON output path
                            [default: data/quarto/model_competence_audit.json]
    --seed=<int>            RNG seed for subsampling. [default: 42]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from docopt import docopt
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import yaml  # noqa: E402
from quartopy import Board, Piece  # noqa: E402

from scripts.games import get_game_module  # noqa: E402


def resolve_model_config(args: dict) -> dict:
    """Resolve the (game, trained_path, random_path, label) tuple.

    If --model-config is given, the YAML is the source of truth and overrides
    --model / --random-model / --game. Otherwise the CLI defaults are used and
    the champion is unlabeled.
    """
    if args.get("--model-config"):
        cfg_path = Path(args["--model-config"])
        cfg = yaml.safe_load(cfg_path.read_text())
        return {
            "label": cfg.get("name", cfg_path.stem),
            "display_name": cfg.get("display_name", cfg.get("name", cfg_path.stem)),
            "game": cfg["game"],
            "trained_path": cfg["path"],
            "random_path": cfg.get("random_path"),
            "config_path": str(cfg_path),
        }
    rand = args["--random-model"]
    return {
        "label": "unnamed",
        "display_name": None,
        "game": args["--game"],
        "trained_path": args["--model"],
        "random_path": None if rand.lower() == "none" else rand,
        "config_path": None,
    }


# ──────────────────────────────────────────────────────────────────────
# State reconstruction
# ──────────────────────────────────────────────────────────────────────


def reconstruct_state(
    board_enc: np.ndarray, piece_enc: np.ndarray
) -> tuple[Board, Piece | None, list[Piece], list[tuple[int, int]]]:
    """Rebuild a Board, the offered Piece, and the list of pieces still in
    storage from the (16,4,4) board encoding and (16,) piece encoding.

    Returns
    -------
    game_board : Board
    offered    : Piece | None  (None if no piece was on offer at this turn)
    storage    : list[Piece]   (pieces neither on board nor currently offered)
    empties    : list[(r, c)]  (empty cell coordinates for convenience)
    """
    game_board = Board("audit", storage=False, rows=4, cols=4)
    on_board: set[int] = set()
    empties: list[tuple[int, int]] = []
    for r in range(4):
        for c in range(4):
            channel = np.flatnonzero(board_enc[:, r, c])
            if len(channel):
                idx = int(channel[0])
                game_board.put_piece(Piece.from_index(idx), r, c)
                on_board.add(idx)
            else:
                empties.append((r, c))

    offered: Piece | None = None
    if piece_enc.any():
        offered = Piece.from_index(int(np.argmax(piece_enc)))

    used = set(on_board)
    if offered is not None:
        used.add(offered.index())
    storage = [Piece.from_index(i) for i in range(16) if i not in used]

    return game_board, offered, storage, empties


def placing_wins(board: Board, piece: Piece, r: int, c: int) -> bool:
    """Try placing ``piece`` at (r,c) on a fresh copy of ``board`` and return
    whether the placement completes a 4-in-a-row (incl. 2x2)."""
    test = Board("test", storage=False, rows=4, cols=4)
    for rr in range(4):
        for cc in range(4):
            cell = board.board[rr][cc]
            if isinstance(cell, Piece):
                test.put_piece(cell, rr, cc)
    test.put_piece(piece, r, c)
    won, _ = test.check_win(mode_2x2=True)
    return won


def winning_cells_for(
    board: Board, piece: Piece, empties: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [(r, c) for (r, c) in empties if placing_wins(board, piece, r, c)]


def piece_is_losing(board: Board, piece: Piece, empties: list[tuple[int, int]]) -> bool:
    """A piece is *losing* if placing it on any empty cell wins → giving it to
    the opponent lets them win immediately."""
    return any(placing_wins(board, piece, r, c) for (r, c) in empties)


# ──────────────────────────────────────────────────────────────────────
# Model query helpers
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def forward_qvalues(
    model: torch.nn.Module,
    board: Board,
    offered: Piece | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (qav_board, qav_piece) for a single state, both shape (16,)."""
    b = torch.from_numpy(board.encode().astype(np.float32)).to(device)  # (1,16,4,4)
    if offered is not None:
        p_vec = offered.vectorize_onehot().reshape(1, -1).astype(np.float32)
    else:
        p_vec = np.zeros((1, 16), dtype=np.float32)
    p = torch.from_numpy(p_vec).to(device)
    qb, qp = model(b, p)
    return qb[0].cpu().numpy(), qp[0].cpu().numpy()


def argmax_legal_cell(
    qav_board: np.ndarray, empties: list[tuple[int, int]]
) -> tuple[int, int]:
    empty_idx = [r * 4 + c for (r, c) in empties]
    best = max(empty_idx, key=lambda i: qav_board[i])
    return best // 4, best % 4


def argmax_legal_piece(qav_piece: np.ndarray, available: list[Piece]) -> Piece:
    idx_to_piece = {p.index(): p for p in available}
    best_idx = max(idx_to_piece, key=lambda i: qav_piece[i])
    return idx_to_piece[best_idx]


# ──────────────────────────────────────────────────────────────────────
# Tests A / B / C  (per-position; rely on game simulation)
# ──────────────────────────────────────────────────────────────────────


def run_tests_abc(
    model: torch.nn.Module,
    boards: np.ndarray,
    pieces: np.ndarray,
    indices: np.ndarray,
    device: str,
    label: str,
) -> dict[str, Any]:
    """Run tests A, B, C on the subsample ``indices``."""
    a_total = a_correct = 0
    b_total = b_correct = b_forced_loss = 0
    c_distinct_fractions: list[float] = []
    # Test F, select head: raw top-1 over all 16 pieces, on the post-placement
    # state; illegal = not in storage.
    f_sel_total = f_sel_illegal = 0
    f_sel_retries: list[int] = []

    for i in tqdm(indices, desc=f"A/B/C [{label}]"):
        b_enc = boards[i]
        p_enc = pieces[i]
        board, offered, storage, empties = reconstruct_state(b_enc, p_enc)
        if offered is None or not empties:
            continue

        # ── Test A ──────────────────────────────────────────────────
        win_cells = winning_cells_for(board, offered, empties)
        if win_cells:
            a_total += 1
            qb, _ = forward_qvalues(model, board, offered, device)
            chosen = argmax_legal_cell(qb, empties)
            if chosen in win_cells:
                a_correct += 1

        # ── Test B ──────────────────────────────────────────────────
        # We need at least one safe AND one losing piece in storage to even
        # ask the question.  Test what the model *would give* the opponent
        # next, conditional on already having placed.  We approximate
        # post-placement state by applying argmax-legal placement of the
        # offered piece (Test A's chosen cell), then querying selection.
        if storage:
            # Re-use qb from test A if available; otherwise compute now.
            if not win_cells:
                qb, _ = forward_qvalues(model, board, offered, device)
            placed_r, placed_c = argmax_legal_cell(qb, empties)
            after = Board("after", storage=False, rows=4, cols=4)
            for rr in range(4):
                for cc in range(4):
                    cell = board.board[rr][cc]
                    if isinstance(cell, Piece):
                        after.put_piece(cell, rr, cc)
            after.put_piece(offered, placed_r, placed_c)
            # Did the placement itself already end the game?  If so, opponent
            # never gets a piece — skip test B for this position.
            if not after.check_win(mode_2x2=True)[0]:
                # ── Test F (select head) ─────────────────────────────
                _, qp_raw = forward_qvalues(model, after, None, device)
                avail = {p.index() for p in storage}
                ranking = np.argsort(-qp_raw)
                f_sel_total += 1
                f_sel_illegal += int(int(ranking[0]) not in avail)
                f_sel_retries.append(
                    next(k for k, i in enumerate(ranking) if int(i) in avail))
                empties_after = [
                    (r, c) for (r, c) in empties if (r, c) != (placed_r, placed_c)
                ]
                losing = [
                    p for p in storage if piece_is_losing(after, p, empties_after)
                ]
                safe = [p for p in storage if p not in losing]
                if losing and safe:
                    b_total += 1
                    _, qp = forward_qvalues(model, after, None, device)
                    chosen_piece = argmax_legal_piece(qp, storage)
                    if chosen_piece not in losing:
                        b_correct += 1
                elif losing and not safe:
                    b_forced_loss += 1

        # ── Test C ──────────────────────────────────────────────────
        # Sweep offered piece across (offered ∪ storage).  Skip if no
        # alternative pieces are available.
        candidates = [offered] + storage
        if len(candidates) >= 2:
            chosen_cells = set()
            for alt in candidates:
                qb_alt, _ = forward_qvalues(model, board, alt, device)
                chosen_cells.add(argmax_legal_cell(qb_alt, empties))
            c_distinct_fractions.append(len(chosen_cells) / len(candidates))

    return {
        "test_A_winning_placement": {
            "n_winnable": a_total,
            "n_correct": a_correct,
            "accuracy": (a_correct / a_total) if a_total else None,
            "chance_baseline": "fraction of empty cells that win (varies); "
            "see per-position breakdown if needed",
        },
        "test_B_losing_piece_avoidance": {
            "n_auditable": b_total,
            "n_avoided": b_correct,
            "accuracy": (b_correct / b_total) if b_total else None,
            "n_forced_loss_skipped": b_forced_loss,
        },
        "test_F_raw_illegal_select": {
            "n_positions": f_sel_total,
            "raw_top1_illegal_rate": (
                f_sel_illegal / f_sel_total if f_sel_total else None),
            "mean_retries": (
                float(np.mean(f_sel_retries)) if f_sel_retries else None),
            "p90_retries": (
                float(np.percentile(f_sel_retries, 90)) if f_sel_retries else None),
            "interpretation": (
                "Unmasked select top-1 is a piece NOT in storage, on the "
                "post-placement state (model's own legal placement). The bot "
                "filters these; training masks them."),
        },
        "test_C_offered_piece_sensitivity": {
            "n_positions": len(c_distinct_fractions),
            "mean_distinct_fraction": (
                float(np.mean(c_distinct_fractions)) if c_distinct_fractions else None
            ),
            "median_distinct_fraction": (
                float(np.median(c_distinct_fractions)) if c_distinct_fractions else None
            ),
            "interpretation": (
                "1.0 = every offered piece picks a different cell; "
                "low = model ignores the offered piece"
            ),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Tests D / E  (vectorizable across the full dataset)
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_tests_de(
    model: torch.nn.Module,
    boards: torch.Tensor,
    pieces: torch.Tensor,
    metadata: list[dict],
    device: str,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Tests D and E only need a forward pass per position; batch them."""
    n = boards.shape[0]
    occupancy = boards.any(dim=1)  # (N, 4, 4) — True where any channel is hot
    occ_flat = occupancy.view(n, 16)  # (N, 16)
    n_pieces = occ_flat.sum(dim=1)  # (N,) — pieces on board

    qb_all = torch.empty((n, 16), dtype=torch.float32)
    qp_all = torch.empty((n, 16), dtype=torch.float32)

    for start in tqdm(range(0, n, batch_size), desc="D/E forward"):
        end = min(start + batch_size, n)
        b = boards[start:end].to(device)
        p = pieces[start:end].to(device)
        qb, qp = model(b, p)
        qb_all[start:end] = qb.cpu()
        qp_all[start:end] = qp.cpu()

    # ── Test D ──────────────────────────────────────────────────────
    occ_mask = occ_flat  # bool (N,16)
    empty_mask = ~occ_mask
    # Avoid div-by-zero with .clamp(min=1)
    q_empty_mean = (qb_all * empty_mask).sum(dim=1) / empty_mask.sum(dim=1).clamp(min=1)
    q_occ_mean = (qb_all * occ_mask).sum(dim=1) / occ_mask.sum(dim=1).clamp(min=1)
    has_both = (occ_mask.any(dim=1)) & (empty_mask.any(dim=1))
    gap = (q_empty_mean - q_occ_mean)[has_both]

    test_d = {
        "n_positions": int(has_both.sum().item()),
        "mean_gap": float(gap.mean().item()),
        "median_gap": float(gap.median().item()),
        "std_gap": float(gap.std().item()),
        "fraction_positive_gap": float((gap > 0).float().mean().item()),
        "interpretation": (
            "Q(empty) − Q(occupied) on the unmasked board head. "
            ">0 means the model has learned legality."
        ),
    }

    # ── Test E ──────────────────────────────────────────────────────
    # Mask occupied cells before softmax; entropy over legal moves only.
    masked_q = qb_all.clone()
    masked_q[occ_mask] = -1e9
    log_probs = F.log_softmax(masked_q, dim=1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)  # nats

    by_phase: dict[str, dict[str, float | int]] = {}
    n_pieces_np = n_pieces.cpu().numpy()
    entropy_np = entropy.cpu().numpy()
    # Bucket by piece count: 0 (empty), 1-4 early, 5-10 mid, 11-15 late
    buckets = {
        "all": np.ones_like(n_pieces_np, dtype=bool),
        "early_0_4": n_pieces_np <= 4,
        "mid_5_10": (n_pieces_np >= 5) & (n_pieces_np <= 10),
        "late_11_15": n_pieces_np >= 11,
    }
    for name, mask in buckets.items():
        mask = mask & (empty_mask.any(dim=1).cpu().numpy())  # need ≥1 legal cell
        if mask.sum() == 0:
            by_phase[name] = {"n": 0}
            continue
        e = entropy_np[mask]
        by_phase[name] = {
            "n": int(mask.sum()),
            "mean_entropy_nats": float(np.mean(e)),
            "median_entropy_nats": float(np.median(e)),
            "max_possible_at_uniform": float(
                np.log(empty_mask[mask].sum(dim=1).float().mean().item())
            ),
        }

    test_e = {
        "by_phase": by_phase,
        "interpretation": (
            "Lower entropy late-game → more decisive. Random model "
            "approaches log(n_legal_cells) uniformly."
        ),
    }

    # ── Test F (place head) ─────────────────────────────────────────
    # Raw top-1 over all 16 cells; illegal = occupied. Engine retries = rank of
    # the first empty cell in the full ranking (the ith_option the engine
    # would reach). Counted only where >=1 cell is occupied AND >=1 is empty.
    has_occ = occ_mask.any(dim=1) & empty_mask.any(dim=1)
    raw_top1 = qb_all.argmax(dim=1)
    illegal = occ_mask[torch.arange(n), raw_top1]
    ranks = qb_all.argsort(dim=1, descending=True)
    retries = (~occ_mask.gather(1, ranks)).float().argmax(dim=1)
    chance = n_pieces.float() / 16.0
    f_buckets = {
        "all": torch.ones(n, dtype=torch.bool),
        "pieces_1_4": (n_pieces >= 1) & (n_pieces <= 4),
        "pieces_5_8": (n_pieces >= 5) & (n_pieces <= 8),
        "pieces_9_12": (n_pieces >= 9) & (n_pieces <= 12),
        "pieces_13_15": n_pieces >= 13,
    }
    f_by_phase: dict[str, dict[str, float | int]] = {}
    for name, mask in f_buckets.items():
        m = mask & has_occ
        if not bool(m.any()):
            f_by_phase[name] = {"n": 0}
            continue
        f_by_phase[name] = {
            "n": int(m.sum()),
            "raw_top1_illegal_rate": float(illegal[m].float().mean()),
            "uniform_chance": float(chance[m].mean()),
        }
    test_f = {
        "n_positions": int(has_occ.sum()),
        "raw_top1_illegal_rate": float(illegal[has_occ].float().mean()),
        "uniform_chance": float(chance[has_occ].mean()),
        "mean_retries": float(retries[has_occ].float().mean()),
        "p90_retries": float(retries[has_occ].float().quantile(0.9)),
        "max_retries": int(retries[has_occ].max()),
        "by_phase": f_by_phase,
        "interpretation": (
            "Unmasked place top-1 is an OCCUPIED cell. The engine retries "
            "invalid moves (ith_option), so play stays legal; Q on illegal "
            "actions is untrained and must not be read as a preference."),
    }

    return {"test_D_q_occupancy_gap": test_d, "test_E_phase_entropy": test_e,
            "test_F_raw_illegal_place": test_f}


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def audit_model(
    model_path: str,
    label: str,
    game: str,
    boards_np: np.ndarray,
    pieces_np: np.ndarray,
    boards_t: torch.Tensor,
    pieces_t: torch.Tensor,
    metadata: list[dict],
    indices_abc: np.ndarray,
    device: str,
) -> dict[str, Any]:
    print(f"\n=== Auditing model: {label} ===")
    print(f"    Path: {model_path}")
    print(f"    Game: {game}")
    t0 = time.time()
    model = get_game_module(game).load_model(model_path, device=device)
    abc = run_tests_abc(model, boards_np, pieces_np, indices_abc, device, label)
    de = run_tests_de(model, boards_t, pieces_t, metadata, device)
    elapsed = time.time() - t0
    return {
        "model_path": str(model_path),
        "label": label,
        "game": game,
        "elapsed_seconds": round(elapsed, 1),
        "n_positions_abc": int(len(indices_abc)),
        "n_positions_de": int(boards_t.shape[0]),
        **abc,
        **de,
    }


def main() -> None:
    args = docopt(__doc__)
    device = args["--device"]
    seed = int(args["--seed"])
    n_abc = int(args["--num-positions"])
    pos_path = Path(args["--positions"])
    out_path = Path(args["--output"])

    mcfg = resolve_model_config(args)
    print(f"Champion: {mcfg['label']} ({mcfg.get('display_name') or '—'})")
    print(f"  game={mcfg['game']}  trained={mcfg['trained_path']}")
    print(f"  random={mcfg['random_path'] or '(skip)'}")

    print(f"Loading positions from {pos_path} ...")
    data = torch.load(pos_path, weights_only=False, map_location="cpu")
    boards_t: torch.Tensor = data["boards"]
    pieces_t: torch.Tensor = data["pieces"]
    metadata: list[dict] = data["metadata"]
    boards_np = boards_t.numpy()
    pieces_np = pieces_t.numpy()
    n_total = boards_t.shape[0]
    print(f"  -> {n_total} positions, {boards_t.dtype}")

    rng = np.random.default_rng(seed)
    n_abc = min(n_abc, n_total)
    indices_abc = rng.choice(n_total, size=n_abc, replace=False)
    indices_abc.sort()
    print(f"Subsampling {n_abc} positions for tests A/B/C (seed={seed})")

    results: dict[str, Any] = {
        "schema_version": 2,
        "champion": mcfg["label"],
        "champion_display_name": mcfg.get("display_name"),
        "champion_config": mcfg.get("config_path"),
        "game": mcfg["game"],
        "positions_path": str(pos_path),
        "n_total_positions": int(n_total),
        "n_abc": n_abc,
        "seed": seed,
        "device": device,
        "models": {},
    }

    results["models"]["trained"] = audit_model(
        mcfg["trained_path"],
        "trained",
        mcfg["game"],
        boards_np,
        pieces_np,
        boards_t,
        pieces_t,
        metadata,
        indices_abc,
        device,
    )

    if mcfg["random_path"]:
        results["models"]["random"] = audit_model(
            mcfg["random_path"],
            "random",
            mcfg["game"],
            boards_np,
            pieces_np,
            boards_t,
            pieces_t,
            metadata,
            indices_abc,
            device,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Compact summary table for stdout consumption
    print("\nSummary (trained vs random):")
    print(f"  {'metric':40s} {'trained':>10s} {'random':>10s}")
    t = results["models"].get("trained", {})
    r = results["models"].get("random", {})

    def cell(d: dict, *keys, fmt: str = "{:.3f}") -> str:
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return "  -  "
            cur = cur[k]
        if cur is None:
            return "  -  "
        try:
            return fmt.format(cur)
        except (ValueError, TypeError):
            return str(cur)

    for label, ks in [
        ("A: winning placement acc", ("test_A_winning_placement", "accuracy")),
        (
            "B: losing-piece avoidance acc",
            ("test_B_losing_piece_avoidance", "accuracy"),
        ),
        (
            "C: piece-sensitivity (mean distinct frac)",
            ("test_C_offered_piece_sensitivity", "mean_distinct_fraction"),
        ),
        ("D: Q(empty)-Q(occupied) mean gap", ("test_D_q_occupancy_gap", "mean_gap")),
        (
            "D: fraction positions with positive gap",
            ("test_D_q_occupancy_gap", "fraction_positive_gap"),
        ),
        (
            "E: late-game (11-15) mean entropy",
            ("test_E_phase_entropy", "by_phase", "late_11_15", "mean_entropy_nats"),
        ),
        (
            "E: early-game (0-4) mean entropy",
            ("test_E_phase_entropy", "by_phase", "early_0_4", "mean_entropy_nats"),
        ),
    ]:
        print(f"  {label:40s} {cell(t, *ks):>10s} {cell(r, *ks):>10s}")


if __name__ == "__main__":
    main()
