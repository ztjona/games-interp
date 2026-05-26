"""Unify position datasets across all champions for cross-champion evaluation.

Merges per-champion amalgam files into a single deduplicated position pool,
then (optionally) recomputes BSP labels and collects activations for each
champion model on the unified positions.

The unified BSP set suffix encodes the position count in thousands
(e.g. gorilla156k) so it is always clear which pool was used.

Usage:
    unify_positions.py [options]
    unify_positions.py (-h | --help)

Options:
    -h --help              Show this help message
    --game-dir <path>      Data directory [default: data/quarto]
    --config-dir <path>    Champion config directory [default: configs/models]
    --bsp-names <list>     Comma-separated BSP basis names [default: gorilla,hawk,tiger]
    --bsp-game <name>      Game module for BSP computation [default: quarto]
    --skip-labels          Skip BSP label computation
    --skip-activations     Skip activation collection
    --device <dev>         Device for activation collection [default: cuda]
    --dry-run              Show what would be done without writing anything
    --force                Overwrite existing unified files

Examples:
    # Dry run: see what would be merged
    python scripts/unify_positions.py --dry-run

    # Merge positions + compute labels only (no activations)
    python scripts/unify_positions.py --skip-activations

    # Full pipeline: merge + labels + activations
    python scripts/unify_positions.py --device cuda
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

HOOKS_BY_GAME = {
    "quarto": ["fc1", "conv2"],
    "quarto_s4": ["s4.fc1", "s4.conv2"],
}

CONV_HOOKS = {"conv2", "s4.conv2", "conv1", "s4.conv1"}


def discover_amalgams(game_dir: Path) -> list[Path]:
    """Find all per-champion amalgam files, excluding the unified one."""
    candidates = sorted(game_dir.glob("positions-amalgam*_unique.pt"))
    return [p for p in candidates if "_all_" not in p.name]


def load_champion_configs(config_dir: Path) -> list[dict]:
    """Load all champ*.yaml configs."""
    configs = []
    for path in sorted(config_dir.glob("champ*.yaml")):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["_config_path"] = str(path)
        configs.append(cfg)
    return configs


def position_count_suffix(n_positions: int) -> str:
    """Format position count as a compact suffix: 156234 -> '156k'."""
    return f"{n_positions // 1000}k"


def run_cmd(args: list[str], description: str, dry_run: bool = False) -> bool:
    """Run a subprocess, printing the command and streaming output."""
    cmd_str = " ".join(str(a) for a in args)
    if dry_run:
        print(f"  [DRY-RUN] {description}")
        print(f"    $ {cmd_str}")
        return True

    print(f"  [RUN] {description}")
    print(f"    $ {cmd_str}", file=sys.stderr)
    result = subprocess.run(args, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"  [FAIL] exit code {result.returncode}", file=sys.stderr)
        return False
    return True


def get_unified_position_count(unified_path: Path) -> int:
    """Load the unified file and return the position count."""
    import torch
    data = torch.load(unified_path, map_location="cpu", weights_only=False)
    n = data["boards"].shape[0]
    del data
    return n


def main():
    args = docopt(__doc__)

    game_dir = Path(args["--game-dir"])
    config_dir = Path(args["--config-dir"])
    bsp_names = [s.strip() for s in args["--bsp-names"].split(",")]
    bsp_game = args["--bsp-game"]
    skip_labels = args["--skip-labels"]
    skip_activations = args["--skip-activations"]
    device = args["--device"]
    dry_run = args["--dry-run"]
    force = args["--force"]

    # -- Step 1: Discover amalgams --
    print("=== Step 1: Discover per-champion amalgam files ===")
    amalgams = discover_amalgams(game_dir)
    if not amalgams:
        print(f"ERROR: No amalgam files found in {game_dir}", file=sys.stderr)
        sys.exit(1)

    for p in amalgams:
        print(f"  Found: {p.name}")
    print(f"  Total: {len(amalgams)} amalgam files")

    # -- Step 2: Merge + dedup --
    print("\n=== Step 2: Merge and deduplicate positions ===")
    unified_path = game_dir / "positions-amalgam_all_unique.pt"
    need_rebuild = force or not unified_path.exists()

    if unified_path.exists() and not force:
        import torch
        existing = torch.load(unified_path, map_location="cpu", weights_only=False)
        existing_sources = set(
            existing.get("provenance", {}).get("source_files", [])
        )
        current_sources = set(str(p) for p in amalgams)
        n_positions = existing["boards"].shape[0]
        del existing

        if current_sources == existing_sources:
            suffix = position_count_suffix(n_positions)
            print(f"  Already up-to-date: {n_positions} positions ({suffix})")
        else:
            new_sources = current_sources - existing_sources
            print(f"  Stale: {len(new_sources)} new source(s) found")
            for s in sorted(new_sources):
                print(f"    + {Path(s).name}")
            need_rebuild = True

    if need_rebuild:
        dedup_args = [
            PYTHON, str(PROJECT_DIR / "scripts" / "deduplicate_positions.py"),
            *[str(p) for p in amalgams],
            "--output", str(unified_path),
        ]
        if not run_cmd(dedup_args, "Merge + deduplicate", dry_run):
            sys.exit(1)
        if not dry_run:
            n_positions = get_unified_position_count(unified_path)
            suffix = position_count_suffix(n_positions)
            print(f"  Unified: {n_positions} positions ({suffix})")
        else:
            suffix = "???k"
            n_positions = None
    elif not unified_path.exists():
        print("ERROR: unified file missing and --dry-run prevents creation")
        sys.exit(1)

    # -- Step 3: BSP labels --
    if not skip_labels:
        print(f"\n=== Step 3: Compute BSP labels (suffix={suffix}) ===")
        for basis in bsp_names:
            animal = f"{basis}{suffix}"
            schema_matches = sorted(game_dir.glob(f"bsp_schema-{basis}_*.json"))
            if not schema_matches:
                print(f"  WARNING: No schema found for basis '{basis}', skipping")
                continue
            count_match = re.search(r"_(\d+)\.json$", schema_matches[0].name)
            if not count_match:
                print(f"  WARNING: Cannot parse count from {schema_matches[0].name}")
                continue
            count = count_match.group(1)

            label_path = game_dir / f"bsp_labels-{animal}_{count}.pt"
            schema_out = game_dir / f"bsp_schema-{animal}_{count}.json"

            if label_path.exists() and not force:
                print(f"  {animal}_{count}: already exists, skipping")
                continue

            print(f"  -> {animal}_{count}")
            label_args = [
                PYTHON, str(PROJECT_DIR / "scripts" / "compute_bsp_labels.py"),
                str(unified_path),
                "--game", bsp_game,
                "--name", basis,
                "--output", str(label_path),
                "--schema-out", str(schema_out),
            ]
            if not run_cmd(label_args, f"BSP labels: {animal}", dry_run):
                print(f"  FAILED: {animal}", file=sys.stderr)
    else:
        print("\n=== Step 3: BSP labels (skipped) ===")

    # -- Step 4: Collect activations --
    if not skip_activations:
        print("\n=== Step 4: Collect activations on unified positions ===")
        configs = load_champion_configs(config_dir)
        if not configs:
            print(f"  WARNING: No champion configs found in {config_dir}")

        for cfg in configs:
            name = cfg["name"]
            game = cfg["game"]
            model_path = cfg["path"]
            random_path = cfg.get("random_path")
            hooks = HOOKS_BY_GAME.get(game, [])

            if not hooks:
                print(f"  WARNING: No hooks for game '{game}' ({name}), skipping")
                continue

            tag = name.replace("champ", "").lower()

            for hook in hooks:
                needs_flatten = hook in CONV_HOOKS
                for is_random in [False, True]:
                    if is_random and not random_path:
                        continue
                    mp = random_path if is_random else model_path
                    rand_suffix = "_random" if is_random else ""
                    act_path = (
                        game_dir
                        / f"{hook}_amalgam_all_{tag}{rand_suffix}_activations.pt"
                    )
                    label = f"{name}/{hook}" + (" (random)" if is_random else "")

                    if act_path.exists() and not force:
                        print(f"  {act_path.name}: already exists, skipping")
                        continue

                    print(f"  -> {label}")
                    act_args = [
                        PYTHON,
                        str(PROJECT_DIR / "scripts" / "collect_activations.py"),
                        str(PROJECT_DIR / mp),
                        "--hook", hook,
                        "--game", game,
                        "--positions-file", str(unified_path),
                        "--output", str(act_path),
                        "--device", device,
                    ]
                    if needs_flatten:
                        act_args.append("--flatten-position")

                    if not run_cmd(act_args, f"Activations: {label}", dry_run):
                        print(f"  FAILED: {label}", file=sys.stderr)
    else:
        print("\n=== Step 4: Activations (skipped) ===")

    # -- Summary --
    print("\n=== Done ===")
    print(f"  Unified positions: {unified_path}")
    if n_positions is not None:
        print(f"  Position count:    {n_positions} ({suffix})")
    print(f"  BSP set suffix:    {suffix}")
    if not skip_labels:
        for basis in bsp_names:
            print(f"  Eval name:         {basis}{suffix}")
    print()
    print("  To evaluate an SAE against the unified dataset:")
    print(f"    python sae_eval.py evaluate saes/quarto/<ckpt>.pt \\")
    print(f"        --bsps=gorilla{suffix} \\")
    print(f"        --data=data/quarto/<hook>_amalgam_all_<tag>_activations.pt \\")
    print(f"        --force")


if __name__ == "__main__":
    main()
