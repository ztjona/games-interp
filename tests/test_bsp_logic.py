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

        assert len(bsps) == 546, (
            "Should have 546 total BSPs "
            "(164 gorilla + 173 hawk + 36 tiger + 173 hen)"
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
            # hen -- hawk's shape on the NEGATIVE attribute poles
            "neg_count": 40,
            "neg_completable": 40,
            "neg_any_threat": 10,
            "neg_sq_count": 36,
            "neg_sq_completable": 36,
            "neg_sq_any_threat": 9,
            "neg_global": 2,
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


class TestConceptFamilies:
    """The cross-basis concept-family map (gorilla/hawk/tiger -> one game fact).

    A basis is a packaging convention; a family is the concept. These tests
    guard the map itself -- the schema stamp and the analysis rollups are only
    as trustworthy as this mapping.
    """

    def test_every_category_in_every_basis_has_a_family(self):
        """An unclassified category silently vanishes from family rollups."""
        from games.quarto import BSP_SETS, CONCEPT_FAMILIES

        missing = {
            category
            for categories in BSP_SETS.values()
            for category in categories
            if category not in CONCEPT_FAMILIES
        }
        assert not missing, (
            f"categories with no concept_family: {sorted(missing)} -- add them "
            f"to CONCEPT_FAMILIES in scripts/games/quarto.py"
        )

    def test_family_roles_are_known(self):
        from games.quarto import CONCEPT_FAMILIES, FAMILY_ROLE_ORDER

        bad = {c: r for c, (_, r) in CONCEPT_FAMILIES.items()
               if r not in FAMILY_ROLE_ORDER}
        assert not bad, f"unknown family_role(s): {bad}"

    def test_triads_pick_one_category_per_basis(self):
        """A triad must be a 1:1 correspondence, else it is not a comparison."""
        from games.quarto import BSP_SETS, concept_triads

        triads = concept_triads()
        assert triads, "no cross-basis triads derived"
        for family, per_basis in triads.items():
            assert len(per_basis) > 1, f"{family} spans only one basis"
            for basis, category in per_basis.items():
                assert category in BSP_SETS[basis], (
                    f"{family}: {category} is not a {basis} category")

    def test_threat_triads_match_the_reframing_audit(self):
        """Regression: the line/square triads the 2026-05-22 audit compared.

        These were a hardcoded dict in scripts/basis_comparison.py before the
        schema carried the mapping; if the derivation ever stops reproducing
        them, the published basis comparison silently changes meaning.
        """
        from games.quarto import concept_triads

        triads = concept_triads()
        assert triads["line_threat"] == {
            "gorilla": "threat_line",
            "hawk": "reframed_completable",
            "tiger": "tiger_line_winnable",
        }
        assert triads["square_threat"] == {
            "gorilla": "threat_square_2x2",
            "hawk": "reframed_sq_completable",
            "tiger": "tiger_square_winnable",
        }

    def test_offered_piece_readout_is_not_a_threat_family(self):
        """gorilla ``offered_piece`` is a 4-bit readout carrying the F1~=0.667
        artefact; folding it in with a real threat concept would corrupt the
        family mean."""
        from games.quarto import CONCEPT_FAMILIES

        readout = CONCEPT_FAMILIES["offered_piece"][0]
        completion = CONCEPT_FAMILIES["tiger_offered_completing_attr"][0]
        assert readout != completion
        assert readout not in ("line_threat", "square_threat")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRunnerLabelNaming:
    """Runners must compute PER-DISTRIBUTION labels under a suffixed animal.

    Regression guard for the 2026-08-12 incident: champTa-rebuild.ps1 called
    ``compute_bsp_labels.py --name gorilla`` (the bare basis) instead of
    ``--name gorillaTa``. Because the baseline champion champAa is the un-tagged
    one, that wrote champTa's 290,147-row labels straight over champAa's
    ``bsp_labels-gorilla_164.pt`` and ``-hawk_173.pt``, destroying them, while
    champTa's own ``*Ta`` files were left stale at 88,524 rows. Nothing errored.

    A bare ``--name <basis>`` is only ever correct for champAa itself, which is
    driven by hand and not by a champion runner.
    """

    BASES = ("gorilla", "hawk", "tiger", "fox")

    def _runner_files(self):
        runners = PROJECT_ROOT / "runners"
        if not runners.is_dir():
            pytest.skip("no runners/ directory on this box")
        return sorted(runners.glob("*.ps1"))

    def test_no_runner_passes_a_bare_basis_as_name(self):
        import re

        offenders = []
        for path in self._runner_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if "compute_bsp_labels.py" not in line:
                    continue
                # Only flag a literal bare basis; a variable such as
                # --name $animal is resolved at runtime and cannot be judged here.
                m = re.search(r"--name\s+(['\"]?)([a-z]+)\1(?:\s|$)", line)
                if m and m.group(2) in self.BASES:
                    offenders.append(f"{path.name}:{line_no}: --name {m.group(2)}")
        assert not offenders, (
            "runner(s) pass a bare BASIS to compute_bsp_labels --name; labels are "
            "per-distribution and must carry the champion suffix (e.g. gorillaTa), "
            "or champAa's un-suffixed label files are silently overwritten:\n  "
            + "\n  ".join(offenders)
        )

    def test_rebuild_runners_verify_label_row_counts(self):
        """A runner that REPLACES an existing distribution must assert the labels
        it writes have one row per position.

        Scoped to rebuild runners (those that move the old positions file aside)
        because that is where the failure is silent and destructive: the stale
        label file from the previous distribution is still on disk, still loads,
        and still matches the stale activations, so every number stays
        self-consistent and wrong. A first-time build has no stale file to
        shadow it.
        """
        for path in self._runner_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            if "compute_bsp_labels.py" not in text:
                continue
            if "legacy_wrong_distribution" not in text:
                continue  # not a rebuild of an existing distribution
            assert "N_POSITIONS" in text, (
                f"{path.name} rebuilds a distribution and recomputes BSP labels "
                f"but never checks the label row count against the position "
                f"count")


class TestRunnerVariableHygiene:
    """PowerShell variable names are CASE-INSENSITIVE.

    Regression guard for the 2026-08-12 stage-4 failure: the runner held the
    positions-file path in ``$OUT`` and then assigned the per-hook activation
    path to ``$out`` inside the loop. Those are ONE variable, so ``$OUT`` was
    silently overwritten and ``collect_activations.py`` was handed its own
    output file as ``--positions-file`` -- surfacing only as
    ``IndexError: too many indices for tensor of dimension 2``, several frames
    away from the cause.

    Two spellings of one name is never intentional here, so flag it outright.
    """

    # Automatic / built-in names whose casing we do not control.
    AUTOMATIC = {
        "args", "input", "error", "host", "true", "false", "null", "matches",
        "_", "psitem", "pscmdlet", "psscriptroot", "lastexitcode", "pwd",
        "home", "pid", "profile",
    }

    def test_no_case_only_variable_collisions(self):
        import collections
        import re

        runners = PROJECT_ROOT / "runners"
        if not runners.is_dir():
            pytest.skip("no runners/ directory on this box")

        offenders = {}
        for path in sorted(runners.glob("*.ps1")):
            text = path.read_text(encoding="utf-8", errors="replace")
            spellings = collections.defaultdict(set)
            for m in re.finditer(r"\$([A-Za-z_]\w*)\s*(?:=|\+=)", text):
                name = m.group(1)
                if name.lower() in self.AUTOMATIC:
                    continue
                spellings[name.lower()].add(name)
            clashes = {k: sorted(v) for k, v in spellings.items() if len(v) > 1}
            if clashes:
                offenders[path.name] = clashes

        assert not offenders, (
            "PowerShell variables differing only in case are the SAME variable; "
            "one assignment silently clobbers the other:\n  "
            + "\n  ".join(f"{f}: {c}" for f, c in offenders.items())
        )


class TestHenNegativePoles:
    """`hen` -- the four attribute poles gorilla/hawk never probed.

    Quarto is won by four pieces sharing *a value* of an attribute, and LITTLE
    wins exactly as TALL does. Every gorilla/hawk threat BSP is keyed on the
    POSITIVE pole only, so hawk can state only half the threat menu: measured on
    champYb (296,045 positions), `hawk => tiger` with zero counterexamples and
    47.6% of tiger's line positives are wins hawk cannot express. `hen` mirrors
    hawk on the negative poles, which makes `tiger == OR(hawk UNION hen)` a
    checkable identity -- the guard below.

    Record: docs/diary/2026-08-21_3A-residuals-and-handoff.md §6.
    """

    LINES = ([f"row_{i}" for i in range(4)]
             + [f"col_{i}" for i in range(4)]
             + ["diag_main", "diag_anti"])
    SQUARES = [f"square_{r}_{c}" for r in range(3) for c in range(3)]
    POS = ("tall", "black", "square", "with_hole")
    NEG = ("little", "white", "circle", "without_hole")

    @staticmethod
    def _board(pieces, offered):
        """`pieces` maps (r, c) -> (size, coloration, shape, hole)."""
        cells = {}
        for (r, c), (size, col, shape, hole) in pieces.items():
            cells[f"{r}_{c}_occupied"] = True
            cells[f"{r}_{c}_size"] = size
            cells[f"{r}_{c}_coloration"] = col
            cells[f"{r}_{c}_shape"] = shape
            cells[f"{r}_{c}_hole"] = hole
        return {"n_pieces": len(pieces), "cells": cells, "offered_piece": offered}

    def test_hen_mirrors_hawk_category_for_category(self):
        """Matched-pair design: identical shape, only the polarity differs."""
        from games.quarto import BSP_SETS

        bsps = get_all_bsp_definitions()
        counts = {}
        for b in bsps:
            counts[b["category"]] = counts.get(b["category"], 0) + 1

        hawk_shape = sorted(counts[c] for c in BSP_SETS["hawk"])
        hen_shape = sorted(counts[c] for c in BSP_SETS["hen"])
        assert hen_shape == hawk_shape == [2, 9, 10, 36, 36, 40, 40], (
            f"hen {hen_shape} must mirror hawk {hawk_shape}; a shape mismatch "
            f"means hawk-vs-hen is no longer a matched pair"
        )
        assert sum(counts[c] for c in BSP_SETS["hen"]) == 173

    def test_hen_ids_do_not_collide_with_any_other_basis(self):
        """Duplicate ids in the menu would misalign every id -> index map."""
        bsps = get_all_bsp_definitions()
        ids = [b["id"] for b in bsps]
        assert len(ids) == len(set(ids)), "duplicate BSP ids in the 546-BSP menu"

    def test_negative_pole_threat_is_invisible_to_hawk(self):
        """An all-LITTLE line is a real win that only hen can state."""
        pieces = {
            (0, 0): ("LITTLE", "BLACK", "SQUARE", "WITH_HOLE"),
            (0, 1): ("LITTLE", "WHITE", "CIRCLE", "WITHOUT_HOLE"),
            (0, 2): ("LITTLE", "BLACK", "CIRCLE", "WITH_HOLE"),
        }
        offered = {"size": "LITTLE", "coloration": "WHITE",
                   "shape": "SQUARE", "hole": "WITH_HOLE"}
        md = self._board(pieces, offered)

        hawk_ids = [f"row_0_completable_{a}" for a in self.POS]
        hen_ids = [f"row_0_completable_{a}" for a in self.NEG]
        hawk = compute_bsp_vector(md, hawk_ids)
        hen = compute_bsp_vector(md, hen_ids)
        tiger = compute_bsp_vector(md, ["tiger_line_row_0_winnable"])

        assert max(hawk) == 0.0, "no positive pole is threatened here"
        assert hen[0] == 1.0, "row_0_completable_little must fire"
        assert tiger[0] == 1.0, "tiger sees the LITTLE win"

    def test_tiger_equals_or_of_hawk_and_hen(self):
        """The identity, on randomised boards -- lines AND 2x2 squares."""
        import random

        rng = random.Random(20260821)
        sizes = ("TALL", "LITTLE")
        cols = ("BLACK", "WHITE")
        shapes = ("SQUARE", "CIRCLE")
        holes = ("WITH_HOLE", "WITHOUT_HOLE")

        checked = fired = 0
        for _ in range(400):
            coords = [(r, c) for r in range(4) for c in range(4)]
            rng.shuffle(coords)
            # Keep boards dense so "exactly one empty cell" happens often.
            n = rng.randint(10, 15)
            pieces = {
                xy: (rng.choice(sizes), rng.choice(cols),
                     rng.choice(shapes), rng.choice(holes))
                for xy in coords[:n]
            }
            offered = {"size": rng.choice(sizes), "coloration": rng.choice(cols),
                       "shape": rng.choice(shapes), "hole": rng.choice(holes)}
            md = self._board(pieces, offered)

            for unit, tiger_id in (
                [(L, f"tiger_line_{L}_winnable") for L in self.LINES]
                + [(S, f"tiger_{S}_winnable") for S in self.SQUARES]
            ):
                hawk = compute_bsp_vector(
                    md, [f"{unit}_completable_{a}" for a in self.POS])
                hen = compute_bsp_vector(
                    md, [f"{unit}_completable_{a}" for a in self.NEG])
                tiger = compute_bsp_vector(md, [tiger_id])[0]
                union = 1.0 if (max(hawk) or max(hen)) else 0.0
                assert union == tiger, (
                    f"{unit}: OR(hawk={list(hawk)}, hen={list(hen)}) = {union} "
                    f"!= tiger {tiger}"
                )
                checked += 1
                fired += int(tiger)

        assert fired > 50, (
            f"only {fired} tiger positives over {checked} checks -- the identity "
            f"is being verified almost entirely on negatives, so it proves little"
        )

    def test_hen_aggregates_read_the_negative_poles_only(self):
        """`neg_any_threat` must not silently answer the hawk question.

        Both ids end in `_any_threat`, so a dispatch ordering slip routes hen's
        id to the positive-pole helper and the bug is invisible in the counts.
        """
        pieces = {
            (1, 0): ("LITTLE", "BLACK", "SQUARE", "WITH_HOLE"),
            (1, 1): ("LITTLE", "WHITE", "CIRCLE", "WITHOUT_HOLE"),
            (1, 2): ("LITTLE", "BLACK", "CIRCLE", "WITH_HOLE"),
        }
        md = self._board(pieces, {"size": "TALL", "coloration": "BLACK",
                                  "shape": "SQUARE", "hole": "WITH_HOLE"})

        vals = compute_bsp_vector(
            md, ["row_1_any_threat", "row_1_neg_any_threat",
                 "board_threat_exists", "board_neg_threat_exists"])
        assert vals[0] == 0.0, "no POSITIVE-pole threat on this board"
        assert vals[1] == 1.0, "three LITTLE pieces + one empty IS a neg threat"
        assert vals[2] == 0.0, "board_threat_exists is positive-pole only"
        assert vals[3] == 1.0, "board_neg_threat_exists must see it"

    def test_hen_categories_share_hawk_s_concept_families(self):
        """hen must roll up into the same families, else no comparison exists."""
        from games.quarto import CONCEPT_FAMILIES

        for hen_cat, hawk_cat in (
            ("neg_count", "reframed_count"),
            ("neg_completable", "reframed_completable"),
            ("neg_any_threat", "reframed_any_threat"),
            ("neg_sq_count", "reframed_sq_count"),
            ("neg_sq_completable", "reframed_sq_completable"),
            ("neg_sq_any_threat", "reframed_sq_any_threat"),
            ("neg_global", "reframed_global"),
        ):
            assert CONCEPT_FAMILIES[hen_cat] == CONCEPT_FAMILIES[hawk_cat], (
                f"{hen_cat} and {hawk_cat} are the same concept at opposite "
                f"polarity and must share (family, role)"
            )
