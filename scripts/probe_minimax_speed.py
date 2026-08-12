"""Timing probe: how expensive is MinimaxBot as a position-generation opponent?

Depth-2 alpha-beta over Quarto's branching factor is far slower than a network
forward pass, so before committing to 10k-game generation runs we need a
measured per-game cost rather than a guess. Reports seconds/game and the
extrapolated wall time for a 10,000-game mode, for each requested pairing.

Usage:
    probe_minimax_speed.py [options]
    probe_minimax_speed.py (-h | --help)

Options:
    -h --help          Show this help message.
    --games <int>      Games per pairing [default: 100]
    --depth <int>      MinimaxBot search depth [default: 2]
    --model <path>     Champion checkpoint for the minimax_v_model pairing.
                       Omit to probe only the random pairings.
    --game <name>      Game module [default: quarto_s4]
    --device <dev>     Device for the model bot [default: cpu]
    --seed <int>       Random seed [default: 42]
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt


def play_games(bot1_factory, bot2_factory, n_games: int, label: str) -> dict:
    """Play n_games and time them. Mirrors the loop in the game modules'
    ``generate_positions`` (mode_2x2=True), minus the position recording, so
    the measured cost is the bots' decisions plus the engine."""
    from quartopy import QuartoGame

    t0 = time.perf_counter()
    plies = 0
    for _ in range(n_games):
        game = QuartoGame(player1=bot1_factory(), player2=bot2_factory(),
                          mode_2x2=True)
        while not game.player_won and not game.game_board.is_full():
            # Guard the terminal state where every piece has been selected but
            # the board is not yet full: the select phase would then draw from
            # an empty storage board. The production loop in the game modules
            # does not carry this guard; it is here so a timing probe cannot die
            # on an end-of-game edge case.
            if game.pick and not game.storage_board.get_valid_moves():
                break
            game.play_turn()
            if game.pick:
                plies += 1
    dt = time.perf_counter() - t0
    return {
        "label": label,
        "games": n_games,
        "seconds": dt,
        "sec_per_game": dt / max(1, n_games),
        "positions": plies,
        "pos_per_game": plies / max(1, n_games),
    }


def main():
    args = docopt(__doc__)
    n = int(args["--games"])
    depth = int(args["--depth"])

    import random
    import numpy as np
    random.seed(int(args["--seed"]))
    np.random.seed(int(args["--seed"]))

    gm = importlib.import_module(f"scripts.games.{args['--game']}")
    from quartopy.bot.minimax_bot import MinimaxBot

    mk_random = lambda: gm.RandomBot()
    mk_minimax = lambda: MinimaxBot(depth=depth)

    pairings = [
        (mk_random, mk_random, "random_v_random (reference)"),
        (mk_minimax, mk_random, f"minimax(d={depth})_v_random"),
    ]

    if args["--model"]:
        wrapper = gm.load_model(args["--model"], device=args["--device"])
        mk_model = lambda: gm.S4ModelBot(wrapper, label="probe")
        pairings.append((mk_model, mk_random, "model_v_random (reference)"))
        pairings.append((mk_minimax, mk_model, f"minimax(d={depth})_v_model"))

    print(f"Probing {n} games per pairing, minimax depth={depth}\n")
    rows = []
    for b1, b2, label in pairings:
        r = play_games(b1, b2, n, label)
        rows.append(r)
        print(f"  {label:<32} {r['sec_per_game']:8.3f} s/game  "
              f"{r['pos_per_game']:5.1f} pos/game  ({r['seconds']:.1f}s total)")

    print(f"\nExtrapolated to a 10,000-game generation run:")
    base = rows[0]["sec_per_game"]
    for r in rows:
        hours = r["sec_per_game"] * 10_000 / 3600
        slow = r["sec_per_game"] / base if base > 0 else float("nan")
        print(f"  {r['label']:<32} {hours:7.2f} h   ({slow:6.1f}x random_v_random)")
    print("\nNote: single process, no batching. Generation modes are file-disjoint,")
    print("so distinct modes can run concurrently.")


if __name__ == "__main__":
    main()
