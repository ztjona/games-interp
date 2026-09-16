"""Board-only orbit keys and the Wave 1b freshness filter (pre-registration
2026-09-14, S4.3): a gold position is dropped when its board, up to the 8 board
symmetries and whatever piece is in hand, is the board of any pilot pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compute_orbit_ids import board_keys, board_symmetry_images, orbit_ids  # noqa: E402
from scripts.freshness_filter import fresh_mask, pilot_board_keys  # noqa: E402


def random_positions(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    boards = torch.zeros(n, 16, 4, 4)
    pieces = torch.zeros(n, 16)
    for i in range(n):
        perm = torch.randperm(16, generator=g)
        cells = torch.randperm(16, generator=g)[: int(torch.randint(4, 12, (1,), generator=g))]
        for p, c in zip(perm, cells):
            boards[i, p, c // 4, c % 4] = 1
        pieces[i, perm[len(cells)]] = 1
    return boards, pieces


def test_board_key_is_symmetry_invariant_and_ignores_the_piece():
    boards, pieces = random_positions(40)
    k0 = board_keys(boards)
    for img in board_symmetry_images(boards):
        assert np.array_equal(board_keys(img), k0)
    assert np.unique(k0).size == 40                       # distinct boards, distinct keys
    other_piece = torch.roll(pieces, 1, dims=1)
    assert not np.array_equal(orbit_ids(boards, pieces), orbit_ids(boards, other_piece))


def test_orbit_ids_still_include_the_piece_and_the_symmetries():
    boards, pieces = random_positions(20, seed=3)
    ids = orbit_ids(boards, pieces)
    for img in board_symmetry_images(boards):
        assert np.array_equal(orbit_ids(img, pieces), ids)


def test_fresh_mask_drops_symmetric_images_of_pilot_boards():
    boards, _ = random_positions(30, seed=1)
    pilot = np.unique(board_keys(boards[:10]))
    rotated = torch.flip(torch.rot90(boards, 1, dims=(-2, -1)), dims=(-1,))
    keep = fresh_mask(rotated, pilot)
    assert not keep[:10].any() and keep[10:].all()


@pytest.fixture
def pilot(tmp_path):
    boards, pieces = random_positions(50, seed=2)
    pos = tmp_path / "positions.pt"
    torch.save({"boards": boards, "pieces": pieces, "metadata": [{}] * 50}, pos)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"positions": str(pos)}), encoding="utf-8")
    power = {"c1": {"switch_on": {"n": 3}, "switch_off": {"n": 0}, "specificity": {"n": 2}},
             "c2": {"switch_on": {"n": 2}, "switch_off": {"n": 1}, "specificity": {"n": 0}}}
    rec = {"c1|R7|switch_on": {"base": torch.tensor([0, 1, 1])},
           "c1|R7|specificity": {"base": torch.tensor([2, 3])},
           "c2|R7|switch_on": {"base": torch.tensor([4, 0])},
           "c2|R7|switch_off": {"base": torch.tensor([5])},
           "c1|R1|switch_on": {"base": torch.tensor([40, 41, 42])}}  # other reps are never read
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"config": str(cfg), "power": power}), encoding="utf-8")
    torch.save(rec, tmp_path / "run_pairs.pt")
    return run, boards, rec


def test_pilot_keys_are_the_r7_pair_boards(pilot):
    run, boards, _ = pilot
    keys, info = pilot_board_keys(run)
    assert np.array_equal(keys, np.unique(board_keys(boards[:6])))
    assert info["pilot_pair_rows"] == 6
    assert fresh_mask(boards, keys).tolist() == [False] * 6 + [True] * 44


def test_incomplete_pilot_records_abort(pilot):
    run, _, rec = pilot
    rec["c1|R7|specificity"] = {"base": torch.tensor([2])}
    torch.save(rec, str(run).replace(".json", "_pairs.pt"))
    with pytest.raises(SystemExit, match="incomplete"):
        pilot_board_keys(run)


def test_several_runs_are_excluded_together(pilot, tmp_path):
    """Wave 1c S4.2: the pilot and both Wave-1b runs are excluded at once."""
    run, boards, _ = pilot
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"config": json.loads(run.read_text(encoding="utf-8"))["config"],
                                 "power": {"c9": {"switch_on": {"n": 2}, "switch_off": {"n": 1},
                                                  "specificity": {"n": 0}}}}), encoding="utf-8")
    torch.save({"c9|R7|switch_on": {"base": torch.tensor([10, 11])},
                "c9|R7|switch_off": {"base": torch.tensor([12])}}, tmp_path / "other_pairs.pt")
    keys, info = pilot_board_keys([run, other])
    assert len(info["runs"]) == 2 and info["pilot_pair_rows"] == 9
    assert fresh_mask(boards, keys).tolist()[:14] == [False] * 6 + [True] * 4 + [False] * 3 + [True]
