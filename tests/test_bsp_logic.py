"""Test BSP (Board State Property) computation logic.

Critical tests for verifying that BSP computations are correct.
These tests use hand-crafted game states with known BSP values.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from games.quarto import compute_bsp_vector, get_all_bsp_definitions


class TestQuartoBSPs:
    """Test Quarto BSP computation correctness."""

    def test_empty_board(self):
        """Empty board should have all cells unoccupied, early game phase."""
        metadata = {
            "n_pieces": 0,
            "cells": {},
            "offered_piece": {},
        }

        bsp_ids = ["cell_0_0_occupied", "cell_1_1_occupied", "game_phase_early"]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 0.0, "Cell (0,0) should be empty"
        assert vec[1] == 0.0, "Cell (1,1) should be empty"
        assert vec[2] == 1.0, "Should be early game (0 pieces)"

    def test_cell_occupancy(self):
        """Test cell occupancy detection."""
        metadata = {
            "n_pieces": 1,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
            },
            "offered_piece": {},
        }

        bsp_ids = ["cell_0_0_occupied", "cell_0_1_occupied"]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 1.0, "Cell (0,0) should be occupied"
        assert vec[1] == 0.0, "Cell (0,1) should be empty"

    def test_binary_attributes(self):
        """Test binary attribute encoding (TALL=1, SHORT=0)."""
        metadata = {
            "n_pieces": 2,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
                "0_0_coloration": "DARK",
                "0_0_shape": "SQUARE",
                "0_0_hole": "HOLLOW",
                "1_1_occupied": True,
                "1_1_size": "SHORT",
                "1_1_coloration": "LIGHT",
                "1_1_shape": "ROUND",
                "1_1_hole": "SOLID",
            },
            "offered_piece": {},
        }

        bsp_ids = [
            "cell_0_0_size_tall",
            "cell_0_0_coloration_dark",
            "cell_0_0_shape_square",
            "cell_0_0_hole_hollow",
            "cell_1_1_size_tall",
            "cell_1_1_coloration_dark",
            "cell_1_1_shape_square",
            "cell_1_1_hole_hollow",
        ]
        vec = compute_bsp_vector(metadata, bsp_ids)

        # Cell (0,0): all positive attributes
        assert vec[0] == 1.0, "TALL should be 1"
        assert vec[1] == 1.0, "DARK should be 1"
        assert vec[2] == 1.0, "SQUARE should be 1"
        assert vec[3] == 1.0, "HOLLOW should be 1"

        # Cell (1,1): all negative attributes
        assert vec[4] == 0.0, "SHORT should be 0"
        assert vec[5] == 0.0, "LIGHT should be 0"
        assert vec[6] == 0.0, "ROUND should be 0"
        assert vec[7] == 0.0, "SOLID should be 0"

    def test_line_threat(self):
        """Test line threat detection (3 of 4 same attribute, 1 empty)."""
        # Row 0: positions (0,0), (0,1), (0,2) have TALL, (0,3) empty → threat!
        metadata = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
                "0_1_occupied": True,
                "0_1_size": "TALL",
                "0_2_occupied": True,
                "0_2_size": "TALL",
                "0_3_occupied": False,
            },
            "offered_piece": {},
        }

        bsp_ids = ["row_0_threat_size_tall", "row_0_threat_coloration_dark"]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 1.0, "Row 0 should have size threat (3 TALL + 1 empty)"
        assert vec[1] == 0.0, "Row 0 should NOT have coloration threat"

    def test_square_threat(self):
        """Test 2x2 square threat detection (3 of 4 same attribute, 1 empty)."""
        # Square at (0,0): (0,0), (0,1), (1,0) have DARK, (1,1) empty → threat!
        metadata = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "DARK",
                "0_1_occupied": True,
                "0_1_coloration": "DARK",
                "1_0_occupied": True,
                "1_0_coloration": "DARK",
                "1_1_occupied": False,
            },
            "offered_piece": {},
        }

        bsp_ids = ["square_0_0_threat_coloration_dark", "square_0_0_threat_size_tall"]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 1.0, "Square (0,0) should have coloration threat"
        assert vec[1] == 0.0, "Square (0,0) should NOT have size threat"

    def test_offered_piece(self):
        """Test offered piece attribute detection."""
        metadata = {
            "n_pieces": 1,
            "cells": {},
            "offered_piece": {
                "size": "TALL",
                "coloration": "LIGHT",
                "shape": "SQUARE",
                "hole": "SOLID",
            },
        }

        bsp_ids = [
            "offered_size_tall",
            "offered_coloration_dark",
            "offered_shape_square",
            "offered_hole_hollow",
        ]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 1.0, "Offered piece is TALL"
        assert vec[1] == 0.0, "Offered piece is LIGHT (not DARK)"
        assert vec[2] == 1.0, "Offered piece is SQUARE"
        assert vec[3] == 0.0, "Offered piece is SOLID (not HOLLOW)"

    def test_game_phases(self):
        """Test game phase detection (early/mid/late)."""
        test_cases = [
            (0, [1, 0, 0]),  # early: 0-5 pieces
            (3, [1, 0, 0]),  # early
            (5, [1, 0, 0]),  # early
            (6, [0, 1, 0]),  # mid: 6-11 pieces
            (11, [0, 1, 0]),  # mid
            (12, [0, 0, 1]),  # late: 12-16 pieces
            (16, [0, 0, 1]),  # late
        ]

        bsp_ids = ["game_phase_early", "game_phase_mid", "game_phase_late"]

        for n_pieces, expected in test_cases:
            metadata = {"n_pieces": n_pieces, "cells": {}, "offered_piece": {}}
            vec = compute_bsp_vector(metadata, bsp_ids)
            assert list(vec) == expected, f"Failed for {n_pieces} pieces"

    def test_all_bsp_definitions(self):
        """Test that get_all_bsp_definitions returns expected count and structure."""
        bsps = get_all_bsp_definitions()

        assert len(bsps) == 164, "Should have 164 total BSPs"

        # Check all BSPs are binary
        for bsp in bsps:
            assert bsp["type"] == "binary", f"BSP {bsp['id']} should be binary"
            assert "id" in bsp
            assert "description" in bsp
            assert "category" in bsp

        # Check category counts
        categories = {}
        for bsp in bsps:
            cat = bsp["category"]
            categories[cat] = categories.get(cat, 0) + 1

        expected_counts = {
            "cell_occupancy": 16,
            "cell_attribute": 64,
            "threat_line": 40,
            "threat_square_2x2": 36,
            "offered_piece": 4,
            "game_phase": 3,
            "global": 1,
        }

        assert categories == expected_counts, f"Category counts mismatch: {categories}"

    def test_empty_cell_attributes(self):
        """Empty cells should return 0 for all attribute BSPs."""
        metadata = {
            "n_pieces": 0,
            "cells": {
                "0_0_occupied": False,  # explicitly empty
            },
            "offered_piece": {},
        }

        bsp_ids = [
            "cell_0_0_size_tall",
            "cell_0_0_coloration_dark",
            "cell_0_0_shape_square",
            "cell_0_0_hole_hollow",
        ]
        vec = compute_bsp_vector(metadata, bsp_ids)

        # All attributes should be 0 for empty cell
        assert all(v == 0.0 for v in vec), "Empty cells should have all attributes = 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
