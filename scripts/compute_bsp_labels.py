"""Compute Board State Property (BSP) labels from position or activation metadata.

Accepts either:
  - A positions file (positions-*.pt from deduplicate_positions.py)
  - An activation metadata file (*_meta.pt from collect_activations.py --opponents mode)

Both file formats share the same structure: {boards, pieces, metadata, provenance}.
The standard workflow uses positions files directly, avoiding redundant meta files.

Usage:
    compute_bsp_labels.py <data_file> --game <name> [options]
    compute_bsp_labels.py (-h | --help)

Arguments:
    <data_file>    Path to positions-*.pt or *_meta.pt file

Options:
    -h --help                      Show this help message
    --game <name>                  Game name: quarto, othello, tictactoe
    --name <str>                   BSP set name (e.g., 'gorilla', 'fox'). Auto-detected from --output if omitted.
    --output <path>                Output .pt file for labels [default: auto]
    --schema-out <path>            Output JSON schema file [default: auto]
    --categories <cats>            Comma-separated categories to include (e.g., "cell_occupancy,threat_line")
                                   If not specified, includes all BSPs
    --exclude-categories <cats>    Comma-separated categories to exclude
    --list-categories              List available BSP categories and exit

Examples:
    # Compute all BSPs from a positions file (standard workflow)
    python compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt --game quarto --name gorilla

    # Only cell properties
    python compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt --game quarto --name fox \\
        --categories cell_occupancy,cell_attribute,offered_piece,game_phase

    # From legacy _meta.pt file
    python compute_bsp_labels.py data/quarto/fc1_random_v_random_meta.pt --game quarto --name gorilla
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))


def load_game_module(game: str):
    """Load the game-specific module."""
    from games import get_game_module

    try:
        return get_game_module(game)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def filter_bsp_definitions(
    all_bsps: list[dict],
    include_categories: list[str] | None,
    exclude_categories: list[str] | None,
) -> list[dict]:
    """Filter BSP definitions by category."""
    if include_categories:
        all_bsps = [b for b in all_bsps if b.get("category") in include_categories]

    if exclude_categories:
        all_bsps = [b for b in all_bsps if b.get("category") not in exclude_categories]

    return all_bsps


def main():
    args = docopt(__doc__)

    meta_path = Path(args["<data_file>"])
    game = args["--game"]

    # Load game module
    game_mod = load_game_module(game)

    # Get all BSP definitions
    all_bsps = game_mod.get_all_bsp_definitions()

    # List categories mode
    if args["--list-categories"]:
        categories = sorted(set(b.get("category", "unknown") for b in all_bsps))
        category_counts = {
            cat: sum(1 for b in all_bsps if b.get("category") == cat)
            for cat in categories
        }
        print(
            json.dumps(
                {
                    "game": game,
                    "total_bsps": len(all_bsps),
                    "categories": category_counts,
                },
                indent=2,
            )
        )
        return

    # Parse category filters
    include_cats = None
    if args["--categories"]:
        include_cats = [c.strip() for c in args["--categories"].split(",")]

    exclude_cats = None
    if args["--exclude-categories"]:
        exclude_cats = [c.strip() for c in args["--exclude-categories"].split(",")]

    # Filter BSPs
    selected_bsps = filter_bsp_definitions(all_bsps, include_cats, exclude_cats)

    if not selected_bsps:
        print("ERROR: No BSPs selected after filtering.", file=sys.stderr)
        sys.exit(1)

    selected_ids = [b["id"] for b in selected_bsps]

    print(
        f"Selected {len(selected_bsps)} BSPs from {len(all_bsps)} total",
        file=sys.stderr,
    )

    # Load metadata — same structure for positions-*.pt and *_meta.pt files
    print(f"Loading data from {meta_path}...", file=sys.stderr)
    meta_data = torch.load(meta_path, map_location="cpu", weights_only=False)
    metadata_list = meta_data.get("metadata", [])
    n_samples = len(metadata_list)

    if n_samples == 0:
        print("ERROR: No metadata found in file.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing BSP labels for {n_samples} samples...", file=sys.stderr)

    # Compute BSP vectors
    bsp_vectors = []
    for i, meta in enumerate(metadata_list):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{n_samples}", file=sys.stderr)
        vec = game_mod.compute_bsp_vector(meta, selected_ids)
        bsp_vectors.append(vec)

    bsp_labels = np.stack(bsp_vectors, axis=0)  # (N, num_bsps)

    print(
        f"BSP labels computed: {bsp_labels.shape} (dtype={bsp_labels.dtype})",
        file=sys.stderr,
    )

    # Ask user for BSP set name (animal + count)
    print(
        f"\nYou selected {len(selected_bsps)} BSPs from these categories:",
        file=sys.stderr,
    )
    category_summary = {
        cat: sum(1 for b in selected_bsps if b.get("category") == cat)
        for cat in set(b.get("category", "unknown") for b in selected_bsps)
    }
    for cat, count in sorted(category_summary.items()):
        print(f"  {cat}: {count}", file=sys.stderr)

    # Get BSP set name from CLI args or auto-detect from output path
    animal_name = args["--name"]

    if animal_name is None:
        # Try to extract from --output path (e.g., "bsp_labels-fox_87.pt" -> "fox")
        output_arg = args["--output"]
        if output_arg and output_arg != "auto":
            output_stem = Path(output_arg).stem  # "bsp_labels-fox_87"
            # Try to extract name between "bsp_labels-" and "_{count}"
            if output_stem.startswith("bsp_labels-"):
                name_part = output_stem[len("bsp_labels-") :]  # "fox_87"
                # Remove "_{count}" suffix if present
                animal_name = name_part.rsplit("_", 1)[0]  # "fox"

    if not animal_name:
        print(
            "ERROR: Must provide --name <str> or use --output with embedded name (e.g., bsp_labels-fox_87.pt)",
            file=sys.stderr,
        )
        sys.exit(1)

    animal_name = animal_name.strip().lower()
    bsp_set_name = f"{animal_name}_{len(selected_bsps)}"
    print(f"\nUsing BSP set name: {bsp_set_name}", file=sys.stderr)

    # Determine output paths — always bsp_labels-{bsp_set_name}.pt (no hook/opponents in name)
    data_dir = meta_path.parent

    if args["--output"] == "auto" or args["--output"] is None:
        output_path = data_dir / f"bsp_labels-{bsp_set_name}.pt"
    else:
        output_path = Path(args["--output"])

    if args["--schema-out"] == "auto" or args["--schema-out"] is None:
        schema_path = data_dir / f"bsp_schema-{bsp_set_name}.json"
    else:
        schema_path = Path(args["--schema-out"])

    # Save labels
    torch.save(torch.from_numpy(bsp_labels), output_path)
    print(f"Saved BSP labels to: {output_path}", file=sys.stderr)

    # Save schema
    schema_doc = {
        "bsp_set_name": bsp_set_name,
        "animal": animal_name,
        "game": game,
        "num_bsps": len(selected_bsps),
        "categories": category_summary,
        "filters": {
            "include_categories": include_cats,
            "exclude_categories": exclude_cats,
        },
        "bsps": selected_bsps,
    }

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_doc, f, indent=2)

    print(f"Saved BSP schema to: {schema_path}", file=sys.stderr)

    # Summary to stdout
    summary = {
        "bsp_set_name": bsp_set_name,
        "data_file": str(meta_path),
        "game": game,
        "n_samples": n_samples,
        "num_bsps": len(selected_bsps),
        "bsp_labels_shape": list(bsp_labels.shape),
        "output_labels": str(output_path),
        "output_schema": str(schema_path),
        "category_counts": category_summary,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
