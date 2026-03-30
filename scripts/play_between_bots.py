"""Sanity check: replicate hierarchical-SAE win rate evaluation.

Uses quartopy's play_games() directly with proper BotAI subclasses.
Mirrors the exact pattern from hierarchical-SAE/play_between_bots.py.

Usage:
    python scripts/play_between_bots.py [--matches N] [--model PATH]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quartopy import play_games
from scripts.games.quarto import ModelBot, RandomBot, load_model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Play matches between bots")
    parser.add_argument("--matches", type=int, default=500)
    parser.add_argument(
        "--model",
        default="models/quarto/20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt",
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Matches per direction: {args.matches}")
    print()

    model = load_model(args.model, device="cpu")
    bot_model = ModelBot(model, deterministic=False, temperature=0.1, label="Aa_replay")
    bot_random = RandomBot()

    print("=== Model (P1) vs Random (P2) ===")
    _, wr1 = play_games(
        matches=args.matches,
        player1=bot_model,
        player2=bot_random,
        verbose=False,
        save_match=False,
        mode_2x2=True,
    )
    print(dict(wr1))

    print()
    print("=== Random (P1) vs Model (P2) ===")
    _, wr2 = play_games(
        matches=args.matches,
        player1=bot_random,
        player2=bot_model,
        verbose=False,
        save_match=False,
        mode_2x2=True,
    )
    print(dict(wr2))


if __name__ == "__main__":
    main()
