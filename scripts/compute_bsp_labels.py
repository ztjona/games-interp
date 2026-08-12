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
    --name <str>                   BSP set name (e.g., 'gorilla', 'hawk'). If the name matches an
                                   entry in ``<game_module>.BSP_SETS`` and ``--only-categories`` is
                                   not given, the corresponding category list is applied
                                   automatically. Auto-detected from --output if omitted.
    --output <path>                Output .pt file for labels [default: auto]
    --schema-out <path>            Output JSON schema file [default: auto]
    --only-categories <cats>       Comma-separated categories to include. Overrides any
                                   ``--name``-implied filter. If omitted and ``--name`` does not
                                   resolve to a known set, ALL BSPs are included (not recommended).
    --exclude-categories <cats>    Comma-separated categories to exclude
    --list-categories              List available BSP categories and exit

Examples:
    # Compute all BSPs from a positions file (standard workflow)
    compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt --game quarto --name gorilla

    # Only cell properties
    compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt --game quarto --name fox --only-categories cell_occupancy,cell_attribute

    # From legacy _meta.pt file
    compute_bsp_labels.py data/quarto/fc1_random_v_random_meta.pt --game quarto --name gorilla
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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

    # Resolve animal_name early so it can drive auto-filtering when --name
    # matches a known set in game_mod.BSP_SETS.
    animal_name = args["--name"]
    if animal_name is None:
        output_arg = args["--output"]
        if output_arg and output_arg != "auto":
            output_stem = Path(output_arg).stem
            if output_stem.startswith("bsp_labels-"):
                name_part = output_stem[len("bsp_labels-") :]
                animal_name = name_part.rsplit("_", 1)[0]
    if animal_name:
        animal_name = animal_name.strip()

    bsp_sets = getattr(game_mod, "BSP_SETS", {})

    # Resolve basis: animal_name is either {basis} (e.g. 'tiger', 'gorilla') or
    # {basis}{ChampionSuffix} (e.g. 'tigerS4', 'gorillaTa'). Basis is the
    # longest BSP_SETS key that is a case-insensitive prefix of animal_name.
    # Used for the category-filter lookup AND for the schema filename (which is
    # basis-only by convention; see CLAUDE.md § Domain conventions).
    basis: str | None = None
    if animal_name and bsp_sets:
        lower = animal_name.lower()
        candidates = [b for b in bsp_sets if lower.startswith(b)]
        if candidates:
            basis = max(candidates, key=len)

    # Parse category filters. Precedence:
    #   1. --only-categories (explicit) wins.
    #   2. --name X resolves to BSP_SETS[basis(X)] when X starts with a known basis.
    #   3. Fallback to all BSPs (legacy), with a stderr warning — see
    #      CLAUDE.md "Things that have bitten" for why this matters.
    include_cats = None
    if args["--only-categories"]:
        include_cats = [c.strip() for c in args["--only-categories"].split(",")]
        if basis and set(include_cats) != set(bsp_sets[basis]):
            print(
                f"WARNING: --name '{animal_name}' (basis '{basis}') implies categories "
                f"{bsp_sets[basis]} but --only-categories {include_cats} was given. "
                f"Using --only-categories.",
                file=sys.stderr,
            )
    elif basis:
        include_cats = list(bsp_sets[basis])
        if basis == animal_name:
            print(
                f"--name '{animal_name}' resolved to {len(include_cats)} categories: {include_cats}",
                file=sys.stderr,
            )
        else:
            print(
                f"--name '{animal_name}' resolved via basis '{basis}' to "
                f"{len(include_cats)} categories: {include_cats}",
                file=sys.stderr,
            )
    else:
        known = sorted(bsp_sets.keys())
        print(
            f"WARNING: no --only-categories given and --name {'(unset)' if not animal_name else f'{animal_name!r}'} "
            f"does not match a known BSP set ({known}). Falling back to ALL BSPs "
            f"({len(all_bsps)} total). This is almost never what you want — see "
            f"BSP-schema-summary.md.",
            file=sys.stderr,
        )

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
    for meta in tqdm(metadata_list, desc="Computing BSPs", file=sys.stderr):
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

    # animal_name was resolved above (before filtering); now finalize the
    # combined bsp_set_name used for output filenames.
    if not animal_name:
        print(
            "ERROR: Must provide --name <str> or use --output with embedded name (e.g., bsp_labels-fox_87.pt)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Labels are per-distribution → file keyed by the (possibly suffixed)
    # animal_name. Schema is distribution-independent → file keyed by the
    # basis only (e.g. ``gorilla``, not ``gorillaS4``); see CLAUDE.md §
    # "Domain conventions". ``basis`` is resolved above; fall back to
    # animal_name when no known basis matches (legacy / one-off sets).
    schema_key = basis if basis else animal_name
    bsp_set_name = f"{animal_name}_{len(selected_bsps)}"
    schema_basis_name = f"{schema_key}_{len(selected_bsps)}"
    print(f"\nUsing BSP set name: {bsp_set_name}", file=sys.stderr)
    if schema_key != animal_name:
        print(
            f"  Schema keyed by basis '{schema_key}' (champion-independent).",
            file=sys.stderr,
        )

    # Determine output paths — always bsp_labels-{bsp_set_name}.pt (no hook/opponents in name)
    data_dir = meta_path.parent

    if args["--output"] == "auto" or args["--output"] is None:
        output_path = data_dir / f"bsp_labels-{bsp_set_name}.pt"
    else:
        output_path = Path(args["--output"])

    if args["--schema-out"] == "auto" or args["--schema-out"] is None:
        schema_path = data_dir / f"bsp_schema-{schema_basis_name}.json"
    else:
        schema_path = Path(args["--schema-out"])

    # Save labels
    torch.save(torch.from_numpy(bsp_labels), output_path)
    print(f"Saved BSP labels to: {output_path}", file=sys.stderr)

    # Stamp the cross-basis concept family onto every BSP. A basis is a
    # packaging convention; the family is the underlying game fact, so this is
    # what makes gorilla/hawk/tiger comparable. Keeping it in the schema (rather
    # than in whichever analysis script needs it) means every consumer reads one
    # mapping that cannot drift -- see scripts/games/quarto.py CONCEPT_FAMILIES.
    family_of = getattr(game_mod, "concept_family_of", None)
    family_summary: dict[str, int] = {}
    category_families: dict[str, dict[str, str]] = {}
    unclassified: set[str] = set()
    if family_of is not None:
        for b in selected_bsps:
            cat = b.get("category", "unknown")
            fam = family_of(cat)
            if fam is None:
                unclassified.add(cat)
                continue
            family, role = fam
            b["concept_family"] = family
            b["family_role"] = role
            family_summary[family] = family_summary.get(family, 0) + 1
            category_families[cat] = {"concept_family": family,
                                      "family_role": role}
        if unclassified:
            # Loud but non-fatal: an unclassified category still produces valid
            # labels, it just cannot take part in a cross-basis comparison.
            print(
                f"WARNING: {len(unclassified)} category/categories have no "
                f"concept_family and will be excluded from family rollups: "
                f"{sorted(unclassified)}. Add them to CONCEPT_FAMILIES in "
                f"scripts/games/quarto.py (the s4 modules re-export it).",
                file=sys.stderr,
            )

    # Save schema (basis-keyed; content is identical across champion
    # distributions, so overwriting an existing basis schema is a no-op).
    schema_doc = {
        "bsp_set_name": schema_basis_name,
        "animal": basis,
        "game": game,
        "num_bsps": len(selected_bsps),
        "categories": category_summary,
        "concept_families": family_summary,
        "category_families": category_families,
        "filters": {
            "only_categories": include_cats,
            "exclude_categories": exclude_cats,
        },
        "bsps": selected_bsps,
    }

    with open(schema_path, "w", encoding="utf-8") as f:
        # sort_keys for reproducibility: dict key iteration order is otherwise
        # non-deterministic across runs and produces spurious git churn (the
        # "categories" summary is a dict; the "bsps" list order is preserved).
        json.dump(schema_doc, f, indent=2, sort_keys=True)

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
