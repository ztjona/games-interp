"""Deduplicate position files (combine multiple sources and remove duplicates).

Loads one or more position files, deduplicates based on (board, piece) hash,
and saves the unique positions to a new file.

Usage:
    deduplicate_positions.py <input_files>... --output <path> [options]
    deduplicate_positions.py (-h | --help)

Arguments:
    <input_files>...       One or more position .pt files to deduplicate

Options:
    -h --help              Show this help message
    --output <path>        Output .pt file for deduplicated positions
    --game <name>          Game name (auto-detected from first file if not specified)

Examples:
    # Deduplicate a single file
    python deduplicate_positions.py data/quarto/positions-random_v_random_raw.pt \
        --output data/quarto/positions-random_v_random_unique.pt

    # Combine and deduplicate multiple files (for SAE training)
    python deduplicate_positions.py \
        data/quarto/positions-random_v_random_raw.pt \
        data/quarto/positions-model_v_random-Aa_replay_raw.pt \
        data/quarto/positions-random_v_model-Aa_replay_raw.pt \
        --output data/quarto/positions-combined_unique.pt

    # All files matching a pattern (use shell expansion, not regex)
    python deduplicate_positions.py data/quarto/positions-*-Aa_replay_raw.pt \
        --output data/quarto/positions-Aa_replay-all_unique.pt
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_positions(
    file_paths: list[Path],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], list[dict]]:
    """Load positions from multiple files.

    Returns:
        boards: Combined list of board states
        pieces: Combined list of piece states
        metadata: Combined list of metadata dicts
        provenances: List of provenance dicts from each file
    """
    all_boards = []
    all_pieces = []
    all_metadata = []
    provenances = []

    for path in file_paths:
        print(f"Loading {path}...", file=sys.stderr)
        data = torch.load(path, map_location="cpu", weights_only=False)

        boards_tensor = data["boards"]
        pieces_tensor = data["pieces"]
        metadata_list = data["metadata"]
        provenance = data.get("provenance", {})

        # Convert tensors to numpy arrays (optimized: convert once, then split)
        boards_np = boards_tensor.numpy()
        pieces_np = pieces_tensor.numpy()
        boards = [boards_np[i] for i in range(boards_np.shape[0])]
        pieces = [pieces_np[i] for i in range(pieces_np.shape[0])]

        all_boards.extend(boards)
        all_pieces.extend(pieces)
        all_metadata.extend(metadata_list)
        provenances.append({"file": str(path), **provenance})

        print(f"  Loaded {len(boards)} positions", file=sys.stderr)

    print(
        f"\nTotal: {len(all_boards)} positions from {len(file_paths)} file(s)",
        file=sys.stderr,
    )
    return all_boards, all_pieces, all_metadata, provenances


def deduplicate_positions(
    boards: list[np.ndarray],
    pieces: list[np.ndarray],
    metadata: list[dict],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], dict]:
    """Remove duplicate positions based on (board, piece) hash.

    Returns:
        unique_boards, unique_pieces, unique_metadata, stats
    """
    print(f"\nDeduplicating {len(boards)} positions...", file=sys.stderr)
    t0 = time.time()

    seen = set()
    unique_boards = []
    unique_pieces = []
    unique_metadata = []

    for i, (b, p, m) in enumerate(zip(boards, pieces, metadata)):
        if i % 10000 == 0 and i > 0:
            print(
                f"  Progress: {i}/{len(boards)} ({len(unique_boards)} unique so far)",
                file=sys.stderr,
            )

        # Hash based on board and piece state
        key = (b.tobytes(), p.tobytes())
        if key not in seen:
            seen.add(key)
            unique_boards.append(b)
            unique_pieces.append(p)
            unique_metadata.append(m)

    n_unique = len(unique_boards)
    n_duplicates = len(boards) - n_unique
    pct_unique = 100.0 * n_unique / len(boards) if boards else 0
    dedup_time = time.time() - t0

    stats = {
        "n_raw": len(boards),
        "n_unique": n_unique,
        "n_duplicates_removed": n_duplicates,
        "pct_unique": f"{pct_unique:.1f}%",
        "dedup_time_seconds": f"{dedup_time:.1f}",
    }

    print(
        f"\nKept {n_unique} unique positions ({pct_unique:.1f}%), "
        f"removed {n_duplicates} duplicates ({dedup_time:.1f}s)",
        file=sys.stderr,
    )

    return unique_boards, unique_pieces, unique_metadata, stats


def main():
    args = docopt(__doc__)

    input_files = [Path(f) for f in args["<input_files>"]]
    output_path = Path(args["--output"])
    game = args.get("--game")

    # Validate input files
    for path in input_files:
        if not path.exists():
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all position files
    boards, pieces, metadata, provenances = load_positions(input_files)

    # Auto-detect game if not specified
    if not game and provenances:
        game = provenances[0].get("game", "unknown")

    # Deduplicate
    unique_boards, unique_pieces, unique_metadata, stats = deduplicate_positions(
        boards, pieces, metadata
    )

    # Save deduplicated positions
    board_tensor = torch.tensor(np.stack(unique_boards), dtype=torch.float32)
    piece_tensor = torch.tensor(np.stack(unique_pieces), dtype=torch.float32)

    provenance = {
        "game": game,
        "n_positions": len(unique_boards),
        "deduplicated": True,
        "source_files": [str(p) for p in input_files],
        "source_provenances": provenances,
        "deduplication_stats": stats,
        "deduplication_date": datetime.now().isoformat(),
    }

    torch.save(
        {
            "boards": board_tensor,
            "pieces": piece_tensor,
            "metadata": unique_metadata,
            "provenance": provenance,
        },
        output_path,
    )

    print(f"\nSaved to: {output_path}", file=sys.stderr)

    # Summary JSON to stdout
    summary = {
        "output": str(output_path),
        "game": game,
        "n_input_files": len(input_files),
        **stats,
        "board_shape": list(board_tensor.shape),
        "piece_shape": list(piece_tensor.shape),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
