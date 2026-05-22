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
        """Test binary attribute encoding (TALL=1, LITTLE=0, etc.)."""
        metadata = {
            "n_pieces": 2,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
                "0_0_coloration": "BLACK",
                "0_0_shape": "SQUARE",
                "0_0_hole": "WITH_HOLE",
                "1_1_occupied": True,
                "1_1_size": "LITTLE",
                "1_1_coloration": "WHITE",
                "1_1_shape": "CIRCLE",
                "1_1_hole": "WITHOUT_HOLE",
            },
            "offered_piece": {},
        }

        bsp_ids = [
            "cell_0_0_tall",
            "cell_0_0_black",
            "cell_0_0_square",
            "cell_0_0_with_hole",
            "cell_1_1_tall",
            "cell_1_1_black",
            "cell_1_1_square",
            "cell_1_1_with_hole",
        ]
        vec = compute_bsp_vector(metadata, bsp_ids)

        # Cell (0,0): all positive attributes
        assert vec[0] == 1.0, "TALL should be 1"
        assert vec[1] == 1.0, "BLACK should be 1"
        assert vec[2] == 1.0, "SQUARE should be 1"
        assert vec[3] == 1.0, "WITH_HOLE should be 1"

        # Cell (1,1): all negative attributes
        assert vec[4] == 0.0, "LITTLE should be 0"
        assert vec[5] == 0.0, "WHITE should be 0"
        assert vec[6] == 0.0, "CIRCLE should be 0"
        assert vec[7] == 0.0, "WITHOUT_HOLE should be 0"

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

        bsp_ids = ["row_0_threat_tall", "row_0_threat_black"]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 1.0, "Row 0 should have size threat (3 TALL + 1 empty)"
        assert vec[1] == 0.0, "Row 0 should NOT have coloration threat"

    def test_square_threat(self):
        """Test 2x2 square threat detection (3 of 4 same attribute, 1 empty)."""
        # Square at (0,0): (0,0), (0,1), (1,0) have BLACK, (1,1) empty → threat!
        metadata = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "BLACK",
                "0_1_occupied": True,
                "0_1_coloration": "BLACK",
                "1_0_occupied": True,
                "1_0_coloration": "BLACK",
                "1_1_occupied": False,
            },
            "offered_piece": {},
        }

        bsp_ids = ["square_0_0_threat_black", "square_0_0_threat_tall"]
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
                "coloration": "WHITE",
                "shape": "SQUARE",
                "hole": "WITHOUT_HOLE",
            },
        }

        bsp_ids = [
            "offered_tall",
            "offered_black",
            "offered_square",
            "offered_with_hole",
        ]
        vec = compute_bsp_vector(metadata, bsp_ids)

        assert vec[0] == 1.0, "Offered piece is TALL"
        assert vec[1] == 0.0, "Offered piece is WHITE (not BLACK)"
        assert vec[2] == 1.0, "Offered piece is SQUARE"
        assert vec[3] == 0.0, "Offered piece is WITHOUT_HOLE"

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

        assert len(bsps) == 373, (
            "Should have 373 total BSPs (164 gorilla + 173 hawk + 36 tiger)"
        )

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
            "reframed_count": 40,
            "reframed_completable": 40,
            "reframed_any_threat": 10,
            "reframed_sq_count": 36,
            "reframed_sq_completable": 36,
            "reframed_sq_any_threat": 9,
            "reframed_global": 2,
            "tiger_decision_global": 5,
            "tiger_offered_completing_attr": 4,
            "tiger_line_winnable": 10,
            "tiger_square_winnable": 9,
            "tiger_pool_winning_count": 4,
            "tiger_pool_safe_count": 4,
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
            "cell_0_0_tall",
            "cell_0_0_black",
            "cell_0_0_square",
            "cell_0_0_with_hole",
        ]
        vec = compute_bsp_vector(metadata, bsp_ids)

        # All attributes should be 0 for empty cell
        assert all(v == 0.0 for v in vec), "Empty cells should have all attributes = 0"


class TestHawkBSPs:
    """Test reframed (hawk) BSP computation correctness."""

    def _threat_board(self):
        """Row 0: 3 TALL pieces + 1 empty = threat in TALL.

        All 3 are also BLACK, so row 0 also has threat in BLACK.
        """
        return {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
                "0_0_coloration": "BLACK",
                "0_0_shape": "SQUARE",
                "0_0_hole": "WITH_HOLE",
                "0_1_occupied": True,
                "0_1_size": "TALL",
                "0_1_coloration": "BLACK",
                "0_1_shape": "CIRCLE",
                "0_1_hole": "WITHOUT_HOLE",
                "0_2_occupied": True,
                "0_2_size": "TALL",
                "0_2_coloration": "BLACK",
                "0_2_shape": "SQUARE",
                "0_2_hole": "WITHOUT_HOLE",
                "0_3_occupied": False,
            },
            "offered_piece": {
                "size": "TALL",
                "coloration": "WHITE",
                "shape": "CIRCLE",
                "hole": "WITH_HOLE",
            },
        }

    def test_count_ge3(self):
        """count_ge3: >=3 occupied cells in line share attribute."""
        meta = self._threat_board()
        bsp_ids = [
            "row_0_count_ge3_tall",  # 3 TALL -> yes
            "row_0_count_ge3_black",  # 3 BLACK -> yes
            "row_0_count_ge3_square",  # 2 SQUARE -> no
            "row_0_count_ge3_with_hole",  # 1 WITH_HOLE -> no
        ]
        vec = compute_bsp_vector(meta, bsp_ids)
        assert vec[0] == 1.0, "3 TALL >= 3"
        assert vec[1] == 1.0, "3 BLACK >= 3"
        assert vec[2] == 0.0, "Only 2 SQUARE < 3"
        assert vec[3] == 0.0, "Only 1 WITH_HOLE < 3"

    def test_count_ge3_full_line(self):
        """count_ge3 fires even when all 4 cells are occupied (no empty cell)."""
        meta = {
            "n_pieces": 4,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
                "0_1_occupied": True,
                "0_1_size": "TALL",
                "0_2_occupied": True,
                "0_2_size": "TALL",
                "0_3_occupied": True,
                "0_3_size": "LITTLE",
            },
            "offered_piece": {},
        }
        vec = compute_bsp_vector(meta, ["row_0_count_ge3_tall"])
        assert vec[0] == 1.0, "3 TALL out of 4 occupied >= 3"

    def test_completable_yes(self):
        """completable: threat exists AND offered piece has attribute."""
        meta = self._threat_board()  # row 0 threat in TALL, offered is TALL
        vec = compute_bsp_vector(meta, ["row_0_completable_tall"])
        assert vec[0] == 1.0, "Threat in TALL + offered is TALL = completable"

    def test_completable_no_wrong_offered(self):
        """completable: threat exists BUT offered piece does NOT have attribute."""
        meta = self._threat_board()  # row 0 threat in BLACK, offered is WHITE
        vec = compute_bsp_vector(meta, ["row_0_completable_black"])
        assert vec[0] == 0.0, "Threat in BLACK but offered is WHITE = not completable"

    def test_completable_no_threat(self):
        """completable: no threat even though offered has attribute."""
        meta = self._threat_board()  # no threat in SQUARE (only 2 match)
        vec = compute_bsp_vector(meta, ["row_0_completable_square"])
        assert vec[0] == 0.0, "No threat in SQUARE = not completable"

    def test_any_threat(self):
        """any_threat: OR across all attribute threats for a line."""
        meta = self._threat_board()  # row 0 has threats in TALL and BLACK
        bsp_ids = [
            "row_0_any_threat",  # yes (TALL threat exists)
            "col_0_any_threat",  # no (only 1 piece in col 0)
        ]
        vec = compute_bsp_vector(meta, bsp_ids)
        assert vec[0] == 1.0, "Row 0 has at least one threat"
        assert vec[1] == 0.0, "Col 0 has only 1 piece, no threat"

    def test_board_threat_exists(self):
        """board_threat_exists: any threat anywhere on the board."""
        meta = self._threat_board()
        vec = compute_bsp_vector(meta, ["board_threat_exists"])
        assert vec[0] == 1.0, "Board has threats"

    def test_board_threat_exists_empty(self):
        """board_threat_exists: no threats on empty board."""
        meta = {"n_pieces": 0, "cells": {}, "offered_piece": {}}
        vec = compute_bsp_vector(meta, ["board_threat_exists"])
        assert vec[0] == 0.0, "Empty board has no threats"

    def test_board_completable_exists(self):
        """board_completable_exists: at least one completable threat."""
        meta = self._threat_board()  # TALL threat + offered is TALL
        vec = compute_bsp_vector(meta, ["board_completable_exists"])
        assert vec[0] == 1.0, "TALL threat completable with TALL offered"

    def test_board_completable_not_exists(self):
        """board_completable_exists: threats exist but none completable."""
        meta = self._threat_board()
        # Override offered piece to have NONE of the threat attributes
        meta["offered_piece"] = {
            "size": "LITTLE",  # not TALL
            "coloration": "WHITE",  # not BLACK
            "shape": "CIRCLE",
            "hole": "WITHOUT_HOLE",
        }
        vec = compute_bsp_vector(meta, ["board_completable_exists"])
        assert vec[0] == 0.0, "Threats exist but offered completes none"

    def test_hawk_definitions_count(self):
        """Hawk set should have exactly 173 BSPs across 7 reframed categories."""
        bsps = get_all_bsp_definitions()
        hawk = [b for b in bsps if b["category"].startswith("reframed_")]

        cats = {}
        for b in hawk:
            cats[b["category"]] = cats.get(b["category"], 0) + 1

        assert cats == {
            "reframed_count": 40,
            "reframed_completable": 40,
            "reframed_any_threat": 10,
            "reframed_sq_count": 36,
            "reframed_sq_completable": 36,
            "reframed_sq_any_threat": 9,
            "reframed_global": 2,
        }
        assert len(hawk) == 173

    def test_diag_reframed(self):
        """Diagonal reframed BSPs parse correctly."""
        # Main diagonal: (0,0), (1,1), (2,2), (3,3)
        # 3 TALL on diagonal + 1 empty
        meta = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_size": "TALL",
                "1_1_occupied": True,
                "1_1_size": "TALL",
                "2_2_occupied": True,
                "2_2_size": "TALL",
                "3_3_occupied": False,
            },
            "offered_piece": {
                "size": "TALL",
                "coloration": "BLACK",
                "shape": "SQUARE",
                "hole": "WITH_HOLE",
            },
        }
        bsp_ids = [
            "diag_main_count_ge3_tall",
            "diag_main_completable_tall",
            "diag_main_any_threat",
            "diag_anti_any_threat",
        ]
        vec = compute_bsp_vector(meta, bsp_ids)
        assert vec[0] == 1.0, "Diag main has 3 TALL"
        assert vec[1] == 1.0, "Diag main threat + offered TALL = completable"
        assert vec[2] == 1.0, "Diag main has a threat"
        assert vec[3] == 0.0, "Diag anti has no pieces = no threat"

    def test_square_count_ge3(self):
        """Square count_ge3: >=3 occupied cells in 2x2 square share attribute."""
        # 2x2 at (0,0): cells (0,0), (0,1), (1,0) are BLACK. (1,1) empty.
        meta = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "BLACK",
                "0_1_occupied": True,
                "0_1_coloration": "BLACK",
                "1_0_occupied": True,
                "1_0_coloration": "BLACK",
                "1_1_occupied": False,
            },
            "offered_piece": {},
        }
        bsp_ids = [
            "square_0_0_count_ge3_black",  # 3 BLACK -> yes
            "square_0_0_count_ge3_tall",  # no size info -> no
        ]
        vec = compute_bsp_vector(meta, bsp_ids)
        assert vec[0] == 1.0, "3 BLACK in 2x2 >= 3"
        assert vec[1] == 0.0, "No TALL data in 2x2"

    def test_square_completable(self):
        """Square completable: threat in 2x2 AND offered piece has attribute."""
        meta = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "BLACK",
                "0_1_occupied": True,
                "0_1_coloration": "BLACK",
                "1_0_occupied": True,
                "1_0_coloration": "BLACK",
                "1_1_occupied": False,
            },
            "offered_piece": {
                "size": "TALL",
                "coloration": "BLACK",
                "shape": "CIRCLE",
                "hole": "WITHOUT_HOLE",
            },
        }
        bsp_ids = [
            "square_0_0_completable_black",  # threat + offered BLACK -> yes
            "square_0_0_completable_tall",  # no threat in TALL -> no
        ]
        vec = compute_bsp_vector(meta, bsp_ids)
        assert vec[0] == 1.0, "BLACK threat + offered BLACK = completable"
        assert vec[1] == 0.0, "No TALL threat = not completable"

    def test_square_any_threat(self):
        """Square any_threat: any attribute creates a threat in 2x2 square."""
        meta = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "BLACK",
                "0_1_occupied": True,
                "0_1_coloration": "BLACK",
                "1_0_occupied": True,
                "1_0_coloration": "BLACK",
                "1_1_occupied": False,
            },
            "offered_piece": {},
        }
        bsp_ids = [
            "square_0_0_any_threat",  # BLACK threat -> yes
            "square_1_1_any_threat",  # no pieces in that square -> no
        ]
        vec = compute_bsp_vector(meta, bsp_ids)
        assert vec[0] == 1.0, "2x2 at (0,0) has BLACK threat"
        assert vec[1] == 0.0, "2x2 at (1,1) has no pieces"

    def test_board_threat_exists_via_square(self):
        """board_threat_exists detects threats in 2x2 squares, not just lines."""
        # Only have a 2x2 square threat, no line threats
        meta = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "BLACK",
                "0_1_occupied": True,
                "0_1_coloration": "BLACK",
                "1_0_occupied": True,
                "1_0_coloration": "BLACK",
                "1_1_occupied": False,
            },
            "offered_piece": {},
        }
        vec = compute_bsp_vector(meta, ["board_threat_exists"])
        assert vec[0] == 1.0, "Board has a 2x2 square threat"

    def test_gorilla_square_threat_still_works(self):
        """Gorilla square_threat dispatch not broken by hawk additions."""
        meta = {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True,
                "0_0_coloration": "BLACK",
                "0_1_occupied": True,
                "0_1_coloration": "BLACK",
                "1_0_occupied": True,
                "1_0_coloration": "BLACK",
                "1_1_occupied": False,
            },
            "offered_piece": {},
        }
        vec = compute_bsp_vector(meta, ["square_0_0_threat_black"])
        assert vec[0] == 1.0, "Gorilla square threat still detects correctly"


class TestTigerBSPs:
    """Test agent-relative (tiger) BSP computation."""

    def _winnable_board(self):
        """Row 0 cols 0,1,2 are all TALL (varying other attrs); col 3 empty.

        Offered piece is TALL → placing on (0,3) completes row 0 via TALL.
        """
        return {
            "n_pieces": 3,
            "cells": {
                "0_0_occupied": True, "0_0_size": "TALL", "0_0_coloration": "BLACK",
                "0_0_shape": "SQUARE", "0_0_hole": "WITH_HOLE",
                "0_1_occupied": True, "0_1_size": "TALL", "0_1_coloration": "WHITE",
                "0_1_shape": "SQUARE", "0_1_hole": "WITHOUT_HOLE",
                "0_2_occupied": True, "0_2_size": "TALL", "0_2_coloration": "BLACK",
                "0_2_shape": "CIRCLE", "0_2_hole": "WITH_HOLE",
                "0_3_occupied": False,
            },
            "offered_piece": {
                "size": "TALL", "coloration": "WHITE",
                "shape": "CIRCLE", "hole": "WITHOUT_HOLE",
            },
        }

    def test_win_now_and_line_winnable(self):
        meta = self._winnable_board()
        vec = compute_bsp_vector(
            meta, ["tiger_win_now_exists", "tiger_line_row_0_winnable", "tiger_offered_completes_tall"]
        )
        assert vec[0] == 1.0, "win_now_exists fires when offered completes a line"
        assert vec[1] == 1.0, "row_0 is winnable with the offered piece"
        assert vec[2] == 1.0, "offered piece is the TALL completer"

    def test_line_winnable_only_with_one_empty(self):
        """Lines with 2 empty cells should not be winnable in one move."""
        meta = self._winnable_board()
        # Wipe (0,2) so row 0 has 2 empties → cannot complete in one placement
        meta["cells"]["0_2_occupied"] = False
        vec = compute_bsp_vector(meta, ["tiger_line_row_0_winnable", "tiger_win_now_exists"])
        assert vec[0] == 0.0, "Row with 2 empties not winnable in one move"
        # win_now may still be true via square completion, but on this board nothing
        # else is near complete:
        assert vec[1] == 0.0, "No other line/square near complete"

    def test_pool_safe_on_empty_board(self):
        meta = {
            "n_pieces": 0,
            "cells": {f"{r}_{c}_occupied": False for r in range(4) for c in range(4)},
            "offered_piece": {
                "size": "TALL", "coloration": "BLACK",
                "shape": "SQUARE", "hole": "WITH_HOLE",
            },
        }
        vec = compute_bsp_vector(
            meta,
            [
                "tiger_every_offer_safe",
                "tiger_pool_winning_count_ge1",
                "tiger_pool_safe_count_eq0",
                "tiger_lose_next_forced",
            ],
        )
        assert vec[0] == 1.0, "Empty board: every offer is safe"
        assert vec[1] == 0.0, "Empty board: no pool piece wins"
        assert vec[2] == 0.0, "Empty board: not the forced-loss state"
        assert vec[3] == 0.0, "Empty board: not forced to lose"

    def test_pool_winning_count_buckets(self):
        """On the row-0 TALL near-completion board, several pool pieces would let opponent win.

        Specifically all 8 pool TALL pieces complete row 0 via TALL (except the offered itself,
        which is excluded from the pool). 7 TALL pool pieces ≥ 4, so ge1/ge2/ge4 all fire.
        """
        meta = self._winnable_board()
        vec = compute_bsp_vector(
            meta,
            [
                "tiger_pool_winning_count_ge1",
                "tiger_pool_winning_count_ge2",
                "tiger_pool_winning_count_ge4",
                "tiger_pool_winning_count_all",
                "tiger_opp_winning_offer_exists",
            ],
        )
        assert vec[0] == 1.0
        assert vec[1] == 1.0
        assert vec[2] == 1.0
        assert vec[3] == 0.0, "Not every pool piece is winning (LITTLE pieces are safe here)"
        assert vec[4] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
