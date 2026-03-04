"""Collect activations from a board game model at a specified hook point.

Runs the model on game positions and saves activations + metadata.

**Two modes:**
1. Load pre-generated positions (--positions-file) — recommended for reusing positions
2. Generate on-the-fly (--opponents, --num-games) — legacy backward compatibility

Game-specific logic lives in ``scripts/games/<game>.py``.

Usage:
    collect_activations.py <model_path> --hook <layer> --game <game> --positions-file <path> [options]
    collect_activations.py <model_path> --hook <layer> --game <game> --opponents <mode> [options]
    collect_activations.py (-h | --help)

Arguments:
    <model_path>    Path to the game model .pt weights file

Options:
    -h --help              Show this help message
    --hook <layer>         Named layer to hook (e.g. fc1, conv2)
    --game <game>          Game name: quarto, othello, tictactoe
    --positions-file <p>   Load pre-generated positions from this file
    --opponents <mode>     Bot matchup (used if no --positions-file) [default: random_v_random]
                           random_v_random  – uniform game-tree coverage
                           model_v_random   – model (P1) vs random (P2)
                           random_v_model   – random (P1) vs model (P2)
                           model_v_model    – competent self-play
    --output <path>        Output .pt file path [default: auto]
    --num-games <int>      Number of games to play (if generating) [default: 10000]
    --device <str>         Device [default: cpu]
    --flatten              Flatten conv activations (B,C,H,W) -> (B*H*W, C)
    --seed <int>           Random seed [default: 42]
    --batch-size <int>     Forward-pass batch size [default: 256]

Examples:
    # Use pre-generated positions (recommended)
    python collect_activations.py models/quarto/model.pt --hook fc1 --game quarto \
        --positions-file data/quarto/positions-random_v_random_raw.pt

    # Generate positions on-the-fly (legacy)
    python collect_activations.py models/quarto/model.pt --hook fc1 --game quarto \
        --opponents random_v_random --num-games 10000
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# Activation hooking (game-agnostic)
# ──────────────────────────────────────────────────────────────────────


class ActivationCollector:
    """Hook into a model layer and collect activations."""

    def __init__(self, model: nn.Module, layer_name: str, flatten: bool = False):
        self.activations: list[torch.Tensor] = []
        self.flatten = flatten
        self._hook = None

        found = False
        for name, module in model.named_modules():
            if name == layer_name:
                self._hook = module.register_forward_hook(self._collect_hook)
                found = True
                break

        if not found:
            available = [n for n, _ in model.named_modules() if n]
            raise ValueError(f"Layer '{layer_name}' not found. Available: {available}")

    def _collect_hook(self, module, input, output):
        act = output.detach()
        if self.flatten and act.ndim == 4:
            B, C, H, W = act.shape
            act = act.permute(0, 2, 3, 1).reshape(B * H * W, C)
        self.activations.append(act.cpu())

    def collect(self) -> torch.Tensor:
        return torch.cat(self.activations, dim=0)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()

    def clear(self):
        self.activations = []


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    args = docopt(__doc__)

    model_path = Path(args["<model_path>"])
    hook = args["--hook"]
    game = args["--game"]
    positions_file = args.get("--positions-file")
    opponents = args["--opponents"]
    device = args["--device"]
    flatten = args["--flatten"]
    seed = int(args["--seed"])
    num_games = int(args["--num-games"])
    batch_size = int(args["--batch-size"])

    # Resolve game module
    from games import get_game_module

    try:
        game_mod = get_game_module(game)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Derive checkpoint tag for filenames ──────────────────────────
    checkpoint_stem = model_path.stem  # full checkpoint filename (no ext)

    # ── Step 1: Load or generate positions ───────────────────────────
    if positions_file:
        # Load pre-generated positions
        print(f"Loading positions from: {positions_file}", file=sys.stderr)
        positions_path = Path(positions_file)
        if not positions_path.exists():
            print(f"ERROR: Positions file not found: {positions_file}", file=sys.stderr)
            sys.exit(1)

        t0 = time.time()
        data = torch.load(positions_path, map_location="cpu", weights_only=False)
        boards = data["boards"]
        pieces = data["pieces"]
        metadata = data["metadata"]
        provenance_gen = data.get("provenance", {})

        # Extract opponents from provenance
        opponents_used = provenance_gen.get("opponents", "unknown")

        print(
            f"Loaded {boards.shape[0]} positions ({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )
        print(
            f"Provenance: opponents={opponents_used}, "
            f"deduplicated={provenance_gen.get('deduplicated', 'unknown')}",
            file=sys.stderr,
        )

        # Convert tensors to numpy for compatibility
        boards = [boards[i].numpy() for i in range(boards.shape[0])]
        pieces = [pieces[i].numpy() for i in range(pieces.shape[0])]

        # Determine output naming
        opponents_tag = opponents_used

    else:
        # Generate positions on-the-fly (legacy mode)
        print("Generating positions on-the-fly...", file=sys.stderr)
        t0 = time.time()
        boards, pieces, metadata = game_mod.generate_positions(
            num_games,
            seed=seed,
            opponents=opponents,
            model_path=str(model_path),
            device=device,
        )
        print(
            f"Generated {len(boards)} positions from {num_games} games "
            f"({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )
        opponents_tag = opponents
        provenance_gen = {}

    n_samples = len(boards)

    # Determine output paths
    if args["--output"] == "auto" or args["--output"] is None:
        suffix = "_flat" if flatten else ""
        fname = f"{hook}_{checkpoint_stem}_{opponents_tag}{suffix}_activations.pt"
        output_path = PROJECT_DIR / "data" / game / fname
    else:
        output_path = Path(args["--output"])

    meta_path = output_path.with_name(output_path.stem + "_meta.pt") if not positions_file else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nModel:      {model_path}", file=sys.stderr)
    print(f"Checkpoint: {checkpoint_stem}", file=sys.stderr)
    print(f"Hook:       {hook}", file=sys.stderr)
    print(f"Game:       {game}", file=sys.stderr)
    print(f"Device:     {device}", file=sys.stderr)
    print(f"Flatten:    {flatten}", file=sys.stderr)
    print(f"Output:     {output_path}", file=sys.stderr)
    if meta_path:
        print(f"Meta:       {meta_path}", file=sys.stderr)
    else:
        print(f"Meta:       (none — positions file is the source of record)", file=sys.stderr)
    print(f"Samples:    {n_samples}", file=sys.stderr)
    print(f"Batch size: {batch_size}", file=sys.stderr)

    # ── Step 2: Load model for activation collection ─────────────────
    torch.manual_seed(seed)
    model = game_mod.load_model(model_path, device=device)
    arch_name = getattr(model, "name", model.__class__.__name__)
    print(f"\nModel loaded: {arch_name} on {device}", file=sys.stderr)

    # ── Step 3: Collect activations ──────────────────────────────────
    collector = ActivationCollector(model, hook, flatten=flatten)

    board_tensor = torch.tensor(np.stack(boards), dtype=torch.float32)
    piece_tensor = torch.tensor(np.stack(pieces), dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b = board_tensor[i : i + batch_size].to(device)
            p = piece_tensor[i : i + batch_size].to(device)
            model(b, p)

    activations = collector.collect()
    collector.remove()
    print(
        f"Activations collected: {activations.shape} ({time.time() - t0:.1f}s)",
        file=sys.stderr,
    )

    # ── Step 4: Save with full provenance ────────────────────────────
    provenance = {
        "model_architecture": arch_name,
        "checkpoint_path": str(model_path.resolve()),
        "checkpoint_stem": checkpoint_stem,
        "hook": hook,
        "opponents": opponents_tag,
        "game": game,
        "positions_file": (
            str(Path(positions_file).resolve()) if positions_file else None
        ),
        "n_samples": n_samples,
        "flatten": flatten,
        "seed": seed,
        "device": device,
        "collection_date": datetime.now().isoformat(),
        "position_provenance": provenance_gen,
    }

    torch.save(activations, output_path)
    if meta_path:
        torch.save(
            {
                "boards": board_tensor,
                "pieces": piece_tensor,
                "metadata": metadata,
                "provenance": provenance,
            },
            meta_path,
        )

    # Summary JSON to stdout
    summary = {
        "activations_path": str(output_path),
        "meta_path": str(meta_path) if meta_path else None,
        **provenance,
        "activation_shape": list(activations.shape),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
