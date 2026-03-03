"""Test configuration and shared fixtures for games-interp project."""

import pytest
from pathlib import Path
import torch
import numpy as np


# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def games():
    """Available games in the project."""
    return ["quarto"]  # extend as more games are added


@pytest.fixture(scope="session")
def test_fixtures_dir():
    """Directory containing test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def quarto_fixture_dir(test_fixtures_dir):
    """Quarto-specific test fixtures."""
    return test_fixtures_dir / "quarto"


@pytest.fixture
def sample_quarto_position():
    """Generate a minimal valid Quarto position for testing.

    Returns:
        dict with keys: board (16,4,4), piece (16,), metadata
    """
    # Empty board with one TALL DARK SQUARE HOLLOW piece at (0,0)
    board = np.zeros((16, 4, 4), dtype=np.float32)
    board[15, 0, 0] = 1.0  # Piece index 15 = 1111 binary = all positive attributes

    # Offered piece: TALL DARK SQUARE HOLLOW
    piece = np.zeros(16, dtype=np.float32)
    piece[15] = 1.0

    metadata = {
        "game_idx": 0,
        "turn": 1,
        "n_pieces": 1,
        "player_won": False,
        "cells": {
            "0_0_occupied": True,
            "0_0_size": "TALL",
            "0_0_coloration": "DARK",
            "0_0_shape": "SQUARE",
            "0_0_hole": "HOLLOW",
        },
        "offered_piece": {
            "size": "TALL",
            "coloration": "DARK",
            "shape": "SQUARE",
            "hole": "HOLLOW",
        },
    }

    return {"board": board, "piece": piece, "metadata": metadata}


@pytest.fixture
def sample_position_file(tmp_path, sample_quarto_position):
    """Create a temporary position file for testing.

    Returns:
        Path to the temporary .pt file
    """
    board = sample_quarto_position["board"]
    piece = sample_quarto_position["piece"]
    metadata = sample_quarto_position["metadata"]

    file_path = tmp_path / "test_positions_raw.pt"

    torch.save(
        {
            "boards": torch.tensor([board], dtype=torch.float32),
            "pieces": torch.tensor([piece], dtype=torch.float32),
            "metadata": [metadata],
            "provenance": {
                "game": "quarto",
                "opponents": "test",
                "n_positions": 1,
                "deduplicated": False,
            },
        },
        file_path,
    )

    return file_path
