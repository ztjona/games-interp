"""Compute symmetry-orbit IDs for a position dataset.

Positions are deduplicated by exact ``(board, piece)`` bytes, so the 8 board
symmetries (the dihedral group D4: 4 rotations x optional reflection) survive as
SEPARATE rows. That is correct for training -- the CNN is not rotation
equivariant, so a rotated board is a genuinely different input producing
genuinely different activations -- but it makes those rows statistically
DEPENDENT. For a rotation-invariant concept (global counts, decision-level BSPs)
a position and its rotation are a near-duplicate ``(x, y)`` pair, so letting them
land on opposite sides of a train/test split leaks and inflates held-out R2.

This script assigns every position an integer orbit ID: two positions share an ID
iff one is a board symmetry of the other. Downstream splitters keep a whole orbit
on one side of the split (see ``lib/sae/dilution.py``).

Only the board group is used, not the 384-element attribute-relabelling group.
Attribute relabelling permutes WHICH BSP is which (``_tall`` <-> ``_black``), so
those images are not near-duplicates of each other for a fixed BSP index, and
including them would merge rows that carry genuinely different labels.

Hashing is a random 64-bit polynomial over the board tensor -- vectorised over
all rows and all 8 transforms, so a 677k dataset takes seconds. Collision
probability is negligible (~N^2/2^64).

Usage:
    compute_orbit_ids.py <positions_file> [options]
    compute_orbit_ids.py (-h | --help)

Options:
    -h --help          Show this help message.
    --output <path>    Output .pt path [default: auto] (auto = sibling
                       ``orbit_ids-<stem>.pt`` next to the positions file).
    --seed <int>       Seed for the hash weights [default: 0].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt

# The 8 elements of D4, as (number of 90-degree rotations, whether to flip).
D4 = [(k, f) for f in (False, True) for k in range(4)]


def board_symmetry_images(boards: torch.Tensor) -> list[torch.Tensor]:
    """The 8 D4 images of a (N, C, H, W) board batch, acting on H and W only.

    Channels are piece identity, not space, so they are untouched.
    """
    out = []
    for k, flip in D4:
        b = torch.rot90(boards, k, dims=(-2, -1))
        if flip:
            b = torch.flip(b, dims=(-1,))
        out.append(b.contiguous())
    return out


def orbit_ids(boards: torch.Tensor, pieces: torch.Tensor, seed: int = 0) -> np.ndarray:
    """(N,) int64 orbit IDs: equal iff the positions are board-symmetric images.

    The canonical key is the MINIMUM hash over the orbit, which is invariant to
    which member we started from.
    """
    n = boards.shape[0]
    flat_dim = int(np.prod(boards.shape[1:]))
    rng = np.random.default_rng(seed)
    # Random odd 64-bit weights; int64 arithmetic wraps, which is the intended
    # mod-2^64 polynomial hash.
    w = torch.from_numpy(
        (rng.integers(1, 2**62, size=flat_dim, dtype=np.int64) | 1)
    )
    wp = torch.from_numpy(
        (rng.integers(1, 2**62, size=int(np.prod(pieces.shape[1:])) or 1,
                      dtype=np.int64) | 1)
    )

    piece_flat = pieces.reshape(n, -1).to(torch.int64)
    piece_term = (piece_flat * wp[: piece_flat.shape[1]]).sum(dim=1)

    best = None
    for img in board_symmetry_images(boards):
        h = (img.reshape(n, -1).to(torch.int64) * w).sum(dim=1) + piece_term
        best = h if best is None else torch.minimum(best, h)
    return best.numpy()


def main():
    args = docopt(__doc__)
    path = Path(args["<positions_file>"])
    if not path.exists():
        sys.exit(f"error: {path} not found")

    data = torch.load(path, map_location="cpu", weights_only=False)
    boards, pieces = data["boards"], data["pieces"]
    if not isinstance(boards, torch.Tensor):
        boards = torch.stack(list(boards))
    if not isinstance(pieces, torch.Tensor):
        pieces = torch.stack(list(pieces)) if len(pieces) else torch.zeros(len(boards), 1)
    if pieces.ndim == 1:
        pieces = pieces.unsqueeze(1)

    ids = orbit_ids(boards, pieces, seed=int(args["--seed"]))
    n_orbits = int(np.unique(ids).size)
    n = ids.shape[0]
    print(f"positions      : {n}")
    print(f"distinct orbits: {n_orbits}")
    print(f"mean orbit size: {n / max(1, n_orbits):.2f}  (1.00 = no symmetric "
          f"pairs present; 8.00 = every orbit fully represented)")

    out = args["--output"]
    if out == "auto":
        stem = path.stem.replace("positions-", "")
        out = path.parent / f"orbit_ids-{stem}.pt"
    else:
        out = Path(out)
    torch.save(torch.from_numpy(ids), out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
