"""Game-specific position generators and model loaders.

Each game module must expose:
    generate_positions(num_games, seed) -> (boards, pieces, metadata)
    load_model(model_path, device) -> nn.Module
"""

from __future__ import annotations

AVAILABLE_GAMES = ["quarto", "quarto_s4"]  # extend as games are added


def get_game_module(game: str):
    """Lazily import and return the game module."""
    if game == "quarto":
        from . import quarto

        return quarto
    elif game == "quarto_s4":
        from . import quarto_s4

        return quarto_s4
    # elif game == "othello":
    #     from . import othello
    #     return othello
    # elif game == "tictactoe":
    #     from . import tictactoe
    #     return tictactoe
    else:
        raise ValueError(f"Unknown game '{game}'. Available: {AVAILABLE_GAMES}")
