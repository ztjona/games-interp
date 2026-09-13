"""Tests for scripts/games/quarto_counterfactuals.py -- the Wave-1 pair generator.

The runtime guard (vectorised concepts == stored BSP labels on all 296k natural
positions) checks the concepts on real data; these tests pin the PAIR logic on
hand-built boards where the right pairs are known exactly.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("quartopy")
from scripts.games import quarto_counterfactuals as qc  # noqa: E402

P = qc.piece_pole_table()
POLE = {s: k for k, s in enumerate(qc.POLE_SUFFIXES)}


def piece(*poles):
    """Index of the unique piece carrying exactly these four poles."""
    want = torch.zeros(8, dtype=torch.bool)
    for p in poles:
        want[POLE[p]] = True
    hits = [i for i in range(16) if torch.equal(P[i], want)]
    assert len(hits) == 1, poles
    return hits[0]


# Three TALL pieces that share NOTHING else: colour B/W/B, shape S/C/C, hole W/W/WO.
ROW0 = [piece("tall", "black", "square", "with_hole"),
        piece("tall", "white", "circle", "with_hole"),
        piece("tall", "black", "circle", "without_hole")]


def board_with_row0_tall_threat(offered: int):
    """Row 0 holds the three TALL pieces in cells (0,0)-(0,2); (0,3) is empty;
    nothing else is on the board. The only threat anywhere is row 0 in TALL."""
    b = torch.zeros(1, 16, 4, 4)
    for c, pid in enumerate(ROW0):
        b[0, pid, 0, c] = 1.0
    p = torch.zeros(1, 16)
    p[0, offered] = 1.0
    return b, p


LITTLE_OFFER = piece("little", "white", "square", "without_hole")
TALL_OFFER = piece("tall", "white", "square", "without_hole")


class TestTables:
    def test_poles_and_groups(self):
        assert P.shape == (16, 8) and bool((P.sum(1) == 4).all())
        assert len(qc.groups()) == 19
        assert qc.group_index([(0, 0), (0, 1), (0, 2), (0, 3)]) == 0

    def test_threat_and_win_cells(self):
        t = qc.board_tables(*board_with_row0_tall_threat(LITTLE_OFFER))
        g = qc.group_index([(0, 0), (0, 1), (0, 2), (0, 3)])
        assert bool(t.threat[0, g, POLE["tall"]])
        assert int(t.threat.sum()) == 1                    # the only threat anywhere
        assert int(t.empty_cell[0, g]) == 3
        tall = qc.win_cells(t, P[TALL_OFFER].unsqueeze(0))
        assert tall[0].nonzero().flatten().tolist() == [3]
        assert not bool(qc.win_cells(t, P[LITTLE_OFFER].unsqueeze(0)).any())

    def test_availability_excludes_board_and_offer(self):
        t = qc.board_tables(*board_with_row0_tall_threat(LITTLE_OFFER))
        assert not any(bool(t.avail[0, i]) for i in ROW0 + [LITTLE_OFFER])
        assert int(t.avail.sum()) == 16 - 4


class TestParse:
    @pytest.mark.parametrize("bsp_id,basis,kind", [
        ("row_0_completable_tall", "hawk", "pinned"),
        ("square_1_2_completable_without_hole", "hen", "pinned"),
        ("diag_anti_completable_black", "hawk", "pinned"),
        ("tiger_line_col_3_winnable", "tiger", "winnable"),
        ("tiger_square_2_0_winnable", "tiger", "winnable"),
        ("tiger_offered_completes_circle", "tiger", "offered_attr"),
        ("tiger_win_now_exists", "tiger", "win_now"),
    ])
    def test_kinds(self, bsp_id, basis, kind):
        assert qc.parse_concept(bsp_id, basis).kind == kind

    def test_rejects_non_wave1(self):
        with pytest.raises(ValueError):
            qc.parse_concept("row_0_count_ge3_tall", "hawk")


class TestPairs:
    def _pairs(self, bsp_id, basis, offered):
        t = qc.board_tables(*board_with_row0_tall_threat(offered))
        return qc.build_pairs(t, qc.parse_concept(bsp_id, basis), qc.all_source_wins(t))

    def test_switch_on_sources_are_exactly_the_available_tall_pieces(self):
        ps = self._pairs("row_0_completable_tall", "hawk", LITTLE_OFFER)["switch_on"]
        tall_avail = {i for i in range(16) if bool(P[i, POLE["tall"]])} - set(ROW0) - {LITTLE_OFFER}
        assert set(ps.src_piece.tolist()) == tall_avail
        assert bool((ps.target[:, 3]).all()) and int(ps.target.sum()) == ps.n   # target {c}

    def test_legal_set_is_the_empty_cells_and_identical_for_every_pair(self):
        ps = self._pairs("row_0_completable_tall", "hawk", LITTLE_OFFER)["switch_on"]
        expect = torch.ones(16, dtype=torch.bool)
        expect[[0, 1, 2]] = False
        assert bool((ps.legal == expect).all())

    def test_the_or_concept_has_the_same_switch_on_pairs_here(self):
        """With a single-pole threat, tiger's OR and hawk's disjunct coincide."""
        a = self._pairs("row_0_completable_tall", "hawk", LITTLE_OFFER)["switch_on"]
        b = self._pairs("tiger_line_row_0_winnable", "tiger", LITTLE_OFFER)["switch_on"]
        assert sorted(a.src_piece.tolist()) == sorted(b.src_piece.tolist())
        assert bool(b.src_poles[:, POLE["tall"]].all()) and int(b.src_poles.sum()) == b.n

    def test_no_pairs_for_a_pole_with_no_threat(self):
        ps = self._pairs("row_0_completable_little", "hen", LITTLE_OFFER)
        assert ps["switch_on"].n == 0 and ps["switch_off"].n == 0

    def test_specificity_sources_win_nowhere(self):
        ps = self._pairs("row_0_completable_tall", "hawk", LITTLE_OFFER)["specificity"]
        assert ps.n > 0
        assert not any(bool(P[q, POLE["tall"]]) for q in ps.src_piece.tolist())
        assert LITTLE_OFFER not in ps.src_piece.tolist()

    def test_switch_off_needs_the_concept_true_in_the_base(self):
        off_little = self._pairs("row_0_completable_tall", "hawk", LITTLE_OFFER)["switch_off"]
        assert off_little.n == 0
        off = self._pairs("row_0_completable_tall", "hawk", TALL_OFFER)["switch_off"]
        assert off.n > 0
        assert not any(bool(P[q, POLE["tall"]]) for q in off.src_piece.tolist())
        assert not bool(off.target[:, 3].any())             # target: legal cells except c
        assert bool(off.expect_base[:, 3].all())            # base is expected to play c

    def test_cap_is_deterministic_and_concept_seeded(self):
        ps = self._pairs("row_0_completable_tall", "hawk", LITTLE_OFFER)["specificity"]
        a = qc.cap_pairs(ps, 3, "row_0_completable_tall", 0)
        b = qc.cap_pairs(ps, 3, "row_0_completable_tall", 0)
        assert a.n == 3 and torch.equal(a.src_piece, b.src_piece)
