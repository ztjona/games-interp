"""Generate game positions via self-play (without activation collection).

Saves board states and metadata for later activation collection with multiple models.
Supports deduplication for uniform state coverage.

Usage:
    generate_positions.py --game <name> --opponents <mode> [options]
    generate_positions.py (-h | --help)

Options:
    -h --help              Show this help message
    --game <name>          Game name: quarto, othello, tictactoe
    --opponents <mode>     Bot matchup [default: random_v_random]
                           random_v_random  – uniform game-tree coverage
                           model_v_random   – model (P1) vs random (P2)
                           random_v_model   – random (P1) vs model (P2)
                           model_v_model    – self-play (same model both players)
    --model <path>         Model path for P1 (required for model_v_* modes)
    --model2 <path>        Model path for P2 (optional, only for model_v_model with different models)
    --num-games <int>      Number of games to play [default: 10000]
    --output-dir <path>    Output directory [default: auto]
    --seed <int>           Random seed [default: 42]
    --device <str>         Device for model bots [default: cpu]

Examples:
    # Generate random_v_random positions (raw only, with natural duplicates)
    python generate_positions.py --game quarto --opponents random_v_random

    # Generate model_v_random positions (model name included in filename)
    python generate_positions.py --game quarto --opponents model_v_random \
        --model models/quarto/Aa_replay.pt

    # Generate model_v_model with SAME model (self-play)
    python generate_positions.py --game quarto --opponents model_v_model \
        --model models/quarto/Aa_replay.pt

    # Generate model_v_model with DIFFERENT models
    python generate_positions.py --game quarto --opponents model_v_model \
        --model models/quarto/Aa_replay.pt --model2 models/quarto/Bb_something.pt

Note: This script always saves raw positions with natural duplicates. Use
deduplicate_positions.py for deduplication after aggregation.
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


def deduplicate_positions(
    boards: list[np.ndarray],
    pieces: list[np.ndarray],
    metadata: list[dict],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Remove duplicate positions based on (board, piece) hash.

    Returns:
        Deduplicated boards, pieces, metadata (preserves first occurrence)
    """
    print(f"Deduplicating {len(boards)} positions...", file=sys.stderr)

    seen = set()
    unique_boards = []
    unique_pieces = []
    unique_metadata = []

    for b, p, m in zip(boards, pieces, metadata):
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

    print(
        f"Kept {n_unique} unique positions ({pct_unique:.1f}%), "
        f"removed {n_duplicates} duplicates",
        file=sys.stderr,
    )

    return unique_boards, unique_pieces, unique_metadata


def extract_model_name(model_path: str | Path) -> str:
    """Extract clean model name from checkpoint path.

    Examples:
        "20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt" -> "Aa_replay"
        "Bb_something.pt" -> "Bb_something"
    """
    import re

    model_name = Path(model_path).stem
    # Remove date patterns like "20260227_1103-"
    model_name = re.sub(r"^\d{8}_\d{4}-", "", model_name)
    # Take first meaningful part before complex suffixes
    model_name = (
        model_name.split("(")[0].split("_NUM_")[0].split("_BUFFER")[0].split("_E_")[0]
    )
    return model_name


def main():
    args = docopt(__doc__)

    game = args["--game"]
    opponents = args["--opponents"]
    model_path = args.get("--model")
    model2_path = args.get("--model2")
    num_games = int(args["--num-games"])
    seed = int(args["--seed"])
    device = args["--device"]

    # Resolve game module
    from games import get_game_module

    try:
        game_mod = get_game_module(game)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate model requirement
    if opponents != "random_v_random" and model_path is None:
        print(
            f"ERROR: --opponents {opponents} requires --model <path>",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build opponents tag for filename
    if opponents == "random_v_random":
        opponents_tag = opponents
        model_names = None
    elif opponents == "model_v_model" and model2_path:
        # Different models playing each other
        model1_name = extract_model_name(model_path)
        model2_name = extract_model_name(model2_path)
        opponents_tag = f"{opponents}-{model1_name}_v_{model2_name}"
        model_names = f"{model1_name} vs {model2_name}"
    elif model_path:
        # Single model (model_v_random, random_v_model, or model_v_model self-play)
        model_name = extract_model_name(model_path)
        opponents_tag = f"{opponents}-{model_name}"
        model_names = model_name
    else:
        opponents_tag = opponents
        model_names = None

    # Determine output directory
    if args["--output-dir"] == "auto" or args["--output-dir"] is None:
        output_dir = PROJECT_DIR / "data" / game
    else:
        output_dir = Path(args["--output-dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Game:       {game}", file=sys.stderr)
    print(f"Opponents:  {opponents}", file=sys.stderr)
    print(f"Model:      {model_path or 'None (random only)'}", file=sys.stderr)
    if model_names:
        print(f"Model(s):   {model_names}", file=sys.stderr)
    print(f"Device:     {device}", file=sys.stderr)
    print(f"Games:      {num_games}", file=sys.stderr)
    print(
        f"Save mode:  raw only (use deduplicate_positions.py for dedup)",
        file=sys.stderr,
    )

    # Generate positions
    t0 = time.time()
    boards, pieces, metadata = game_mod.generate_positions(
        num_games,
        seed=seed,
        opponents=opponents,
        model_path=model_path,
        device=device,
    )
    n_raw = len(boards)
    gen_time = time.time() - t0
    print(
        f"\nGenerated {n_raw} positions from {num_games} games ({gen_time:.1f}s)",
        file=sys.stderr,
    )

    # Save raw version
    raw_fname = f"positions-{opponents_tag}_raw.pt"
    raw_path = output_dir / raw_fname

    board_tensor = torch.tensor(np.stack(boards), dtype=torch.float32)
    piece_tensor = torch.tensor(np.stack(pieces), dtype=torch.float32)

    provenance = {
        "game": game,
        "opponents": opponents,
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "model2_path": str(Path(model2_path).resolve()) if model2_path else None,
        "num_games": num_games,
        "n_positions": n_raw,
        "deduplicated": False,
        "seed": seed,
        "device": device,
        "generation_date": datetime.now().isoformat(),
    }

    torch.save(
        {
            "boards": board_tensor,
            "pieces": piece_tensor,
            "metadata": metadata,
            "provenance": provenance,
        },
        raw_path,
    )
    print(f"\nSaved to: {raw_path}", file=sys.stderr)

    # Summary JSON to stdout
    summary = {
        "output": str(raw_path),
        **provenance,
        "board_shape": list(board_tensor.shape),
        "piece_shape": list(piece_tensor.shape),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
