"""SAE evaluation script: coverage, board reconstruction, comparison.

Usage:
    sae_eval.py evaluate <checkpoint> [options]
    sae_eval.py compare [<run_ids>...] [options]
    sae_eval.py history [--game=<name>] [--arch=<name>]
    sae_eval.py -h | --help

Commands:
    evaluate    Run Layer 1 evaluation (coverage + board reconstruction)
    compare     Side-by-side comparison of evaluated runs
    history     Browse evaluation registry

Arguments:
    <checkpoint>    Path to trained SAE .pt file
    <run_ids>       One or more eval registry keys (checkpoint stems)

Options:
    --bsps=<name>           BSP set animal name (e.g. gorilla, fox) [default: gorilla]
    --data=<path>           Activation data path [default: auto]
    --bsp-labels=<path>     BSP label tensor path [default: auto]
    --bsp-schema=<path>     BSP schema JSON path [default: auto]
    --precision-thresh=<f>  Board reconstruction precision threshold [default: 0.9]
    --batch-size=<n>        Evaluation batch size [default: 4096]
    --tag=<str>             Human-readable tag for this evaluation
    --device=<dev>          Device (cuda|cpu|auto) [default: auto]
    --force                 Re-evaluate even if results exist in the registry
    --config=<path>         YAML config listing run_ids for compare
    --game=<name>           Filter history by game
    --arch=<name>           Filter history by architecture
    --verbose               Enable debug-level logging
    -h --help               Show this help

Auto-resolution (from checkpoint metadata):
    data    → data/{game}/{hook}_amalgam_activations.pt
    bsps    → data/{game}/bsp_labels-{animal}_{count}.pt  (by glob)
    schema  → data/{game}/bsp_schema-{animal}_{count}.json

Examples:
    # Evaluate with full BSP set (gorilla)
    python sae_eval.py evaluate saes/quarto/arnold-topk-k32-exp8-fc1.pt

    # Re-evaluate (force recompute)
    python sae_eval.py evaluate saes/quarto/arnold-topk-k32-exp8-fc1.pt --force

    # Compare two runs by ID
    python sae_eval.py compare arnold-topk-k32-exp8-fc1 arnold-topk-k64-exp8-fc1

    # Compare runs from a config file
    python sae_eval.py compare --config configs/compare-arnold.yaml

    # Browse all evaluations
    python sae_eval.py history

    # Filter history
    python sae_eval.py history --arch topk
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from docopt import docopt

from lib.sae import load_checkpoint, load_activation_data, evaluate_sae

log = logging.getLogger("sae_eval")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_data_path(game: str, hook: str) -> Path:
    """Auto-resolve activation data from game and hook."""
    path = Path(f"data/{game}/{hook}_amalgam_activations.pt")
    if not path.exists():
        log.error("Activation data not found: %s", path)
        log.error("  Expected: data/%s/%s_amalgam_activations.pt", game, hook)
        sys.exit(1)
    return path


def _resolve_bsp_paths(game: str, animal: str) -> tuple[Path, Path]:
    """Resolve BSP label and schema paths from animal name via glob.

    Looks for: data/{game}/bsp_labels-{animal}_*.pt
    """
    data_dir = Path(f"data/{game}")
    label_matches = sorted(data_dir.glob(f"bsp_labels-{animal}_*.pt"))
    schema_matches = sorted(data_dir.glob(f"bsp_schema-{animal}_*.json"))

    if not label_matches:
        available = sorted(data_dir.glob("bsp_labels-*.pt"))
        available_names = [p.stem.split("-", 1)[1].rsplit("_", 1)[0] for p in available]
        log.error("No BSP labels found for animal '%s' in data/%s/", animal, game)
        if available_names:
            log.error("  Available BSP sets: %s", ", ".join(available_names))
        sys.exit(1)

    if not schema_matches:
        log.error("BSP schema not found for animal '%s' in data/%s/", animal, game)
        sys.exit(1)

    return label_matches[0], schema_matches[0]


def _extract_checkpoint_info(metadata: dict) -> dict[str, str]:
    """Extract game, hook, architecture from checkpoint metadata."""
    return {
        "game": metadata.get("game", "quarto"),
        "hook": metadata.get("hook", "fc1"),
        "architecture": metadata.get("architecture", "unknown"),
        "experiment": metadata.get("experiment", "unknown"),
    }


# ---------------------------------------------------------------------------
# Eval registry
# ---------------------------------------------------------------------------


def _registry_path(game: str) -> Path:
    return Path(f"saes/{game}/eval_registry.json")


def _load_registry(game: str) -> dict[str, Any]:
    path = _registry_path(game)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(game: str, registry: dict[str, Any]) -> None:
    path = _registry_path(game)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def _register_eval(
    run_id: str,
    game: str,
    checkpoint_path: str,
    info: dict[str, str],
    bsp_set: str,
    metrics: dict,
    tag: str | None,
) -> None:
    """Save evaluation results to the eval registry."""
    registry = _load_registry(game)
    registry[run_id] = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "architecture": info["architecture"],
        "experiment": info["experiment"],
        "game": game,
        "hook": info["hook"],
        "bsp_set": bsp_set,
        "tag": tag or "",
        "metrics": metrics,
    }
    _save_registry(game, registry)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_evaluate(args: dict) -> None:
    """Run Layer 1 evaluation on a checkpoint."""
    checkpoint_path = args["<checkpoint>"]
    animal = args["--bsps"]
    precision_thresh = float(args["--precision-thresh"])
    batch_size = int(args["--batch-size"])
    tag = args["--tag"]
    force = args["--force"]

    device_arg = args["--device"]
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    run_id = Path(checkpoint_path).stem

    # ---- Cache check ----
    # Peek at metadata to find game before deciding on registry
    log.info("Loading checkpoint: %s", checkpoint_path)
    sae, metadata = load_checkpoint(Path(checkpoint_path), device=device)
    info = _extract_checkpoint_info(metadata)
    game, hook = info["game"], info["hook"]

    registry = _load_registry(game)
    if run_id in registry and not force:
        cached = registry[run_id]
        log.info("Found cached evaluation for '%s' (use --force to recompute)", run_id)
        output = {
            "run_id": run_id,
            "checkpoint": cached.get("checkpoint", checkpoint_path),
            "architecture": cached.get("architecture", "?"),
            "experiment": cached.get("experiment", "?"),
            "game": game,
            "hook": cached.get("hook", hook),
            "bsp_set": cached.get("bsp_set", animal),
            "cached": True,
            "metrics": cached.get("metrics", {}),
        }
        print(json.dumps(output, indent=2))
        _print_summary(
            run_id,
            cached.get("bsp_set", animal),
            cached.get("metrics", {}),
            game,
            cached=True,
        )
        return

    log.info(
        "  Architecture: %s, Game: %s, Hook: %s, d_input=%d, d_dict=%d",
        info["architecture"],
        game,
        hook,
        sae.d_input,
        sae.d_dict,
    )

    # Resolve paths
    data_arg = args["--data"]
    if data_arg and data_arg != "auto":
        data_path = Path(data_arg)
    else:
        data_path = _resolve_data_path(game, hook)

    bsp_label_arg = args["--bsp-labels"]
    bsp_schema_arg = args["--bsp-schema"]
    if bsp_label_arg and bsp_label_arg != "auto":
        bsp_label_path = Path(bsp_label_arg)
        bsp_schema_path = (
            Path(bsp_schema_arg)
            if bsp_schema_arg and bsp_schema_arg != "auto"
            else None
        )
    else:
        bsp_label_path, bsp_schema_path = _resolve_bsp_paths(game, animal)

    log.info("  Activations: %s", data_path)
    log.info("  BSP labels:  %s", bsp_label_path)
    if bsp_schema_path:
        log.info("  BSP schema:  %s", bsp_schema_path)

    # Load data
    activations = load_activation_data(str(data_path), device)
    bsp_labels = torch.load(bsp_label_path, map_location="cpu", weights_only=True)

    # Load schema for BSP names in output
    bsp_schema = None
    if bsp_schema_path and bsp_schema_path.exists():
        with open(bsp_schema_path, "r") as f:
            bsp_schema = json.load(f)

    log.info("  Samples: %d, BSPs: %d", activations.shape[0], bsp_labels.shape[1])

    # Run evaluation
    sae.eval()
    metrics = evaluate_sae(
        sae,
        activations,
        bsp_labels,
        precision_threshold=precision_thresh,
        batch_size=batch_size,
    )

    # Remove per_bsp_accuracy from top-level output (too verbose for JSON summary)
    metrics.pop("per_bsp_accuracy", [])

    # Register
    _register_eval(run_id, game, checkpoint_path, info, animal, metrics, tag)

    # Build JSON output
    output = {
        "run_id": run_id,
        "checkpoint": checkpoint_path,
        "architecture": info["architecture"],
        "experiment": info["experiment"],
        "game": game,
        "hook": hook,
        "bsp_set": animal,
        "num_samples": activations.shape[0],
        "cached": False,
        "metrics": metrics,
    }

    # Print JSON to stdout (machine-readable)
    print(json.dumps(output, indent=2))

    # Human-readable summary to stderr
    _print_summary(run_id, animal, metrics, game, num_bsps_label=bsp_labels.shape[1])


def _print_summary(
    run_id: str,
    animal: str,
    metrics: dict,
    game: str,
    num_bsps_label: int | None = None,
    cached: bool = False,
) -> None:
    """Print a human-readable summary to stderr."""
    log.info("")
    log.info("=" * 60)
    if cached:
        log.info("  %s  (cached)", run_id)
    else:
        log.info("  %s", run_id)
    bsp_count = num_bsps_label or metrics.get("num_bsps", "?")
    log.info("  BSP set: %s (%s BSPs)", animal, bsp_count)
    log.info("-" * 60)
    log.info("  FVU:                  %.6f", metrics.get("fvu", 0))
    log.info("  L0:                   %.1f", metrics.get("l0", 0))
    log.info("  Dead features:        %.1f%%", metrics.get("dead_features_pct", 0))
    log.info("  Coverage:             %.4f", metrics.get("coverage", 0))
    log.info("  Coverage >50%%:        %.4f", metrics.get("coverage_above_50", 0))
    log.info("  Coverage >75%%:        %.4f", metrics.get("coverage_above_75", 0))
    log.info("  Board reconstruction: %.4f", metrics.get("board_reconstruction", 0))
    log.info(
        "  Reconstructable BSPs: %d / %s (%.1f%%)",
        metrics.get("num_reconstructable_bsps", 0),
        metrics.get("num_bsps", "?"),
        metrics.get("fraction_reconstructable", 0) * 100,
    )
    log.info("=" * 60)
    log.info("  Saved to eval registry: %s", _registry_path(game))


def cmd_compare(args: dict) -> None:
    """Compare multiple evaluated runs side by side."""
    run_ids = args["<run_ids>"] or []
    config_path = args["--config"]

    # Load run_ids from config if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            log.error("Config file not found: %s", config_path)
            sys.exit(1)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or "run_ids" not in config:
            log.error("Config must be a YAML with a 'run_ids' list.")
            sys.exit(1)
        config_runs = config["run_ids"]
        if not isinstance(config_runs, list):
            log.error("'run_ids' must be a list in the config file.")
            sys.exit(1)
        # Each entry can be a string or a dict {id: ..., label: ...}
        for item in config_runs:
            if isinstance(item, str):
                run_ids.append(item)
            elif isinstance(item, dict) and "id" in item:
                run_ids.append(item["id"])
            else:
                log.warning("Skipping invalid entry in config run_ids: %s", item)

    if not run_ids:
        log.error("No run IDs specified. Use positional args or --config.")
        sys.exit(1)

    # Try each game's registry to find the runs
    registries: dict[str, dict] = {}
    for game_dir in Path("saes").iterdir():
        if game_dir.is_dir():
            reg = _load_registry(game_dir.name)
            registries.update(reg)

    # Collect entries
    entries = []
    for rid in run_ids:
        if rid not in registries:
            log.warning("Run '%s' not found in any eval registry", rid)
            continue
        entries.append((rid, registries[rid]))

    if len(entries) < 2:
        log.error("Need at least 2 valid runs to compare.")
        sys.exit(1)

    # Define which metrics to compare and their display format
    metric_rows = [
        ("architecture", None),
        ("bsp_set", None),
        ("fvu", ".6f"),
        ("l0", ".1f"),
        ("dead_features_pct", ".1f"),
        ("coverage", ".4f"),
        ("coverage_above_50", ".4f"),
        ("coverage_above_75", ".4f"),
        ("board_reconstruction", ".4f"),
        ("num_reconstructable_bsps", "d"),
        ("fraction_reconstructable", ".1%"),
    ]

    # Print table
    col_width = max(len(rid) for rid, _ in entries)
    col_width = max(col_width, 24)
    label_width = 26

    header = f"{'Metric':<{label_width}}"
    for rid, _ in entries:
        header += f"  {rid:>{col_width}}"
    print(header)
    print("-" * len(header))

    for key, fmt in metric_rows:
        row = f"{key:<{label_width}}"
        for _, entry in entries:
            if key in ("architecture", "bsp_set"):
                val = entry.get(key, "?")
                row += f"  {val:>{col_width}}"
            else:
                metrics = entry.get("metrics", {})
                val = metrics.get(key)
                if val is None:
                    row += f"  {'—':>{col_width}}"
                elif fmt:
                    row += f"  {val:{col_width}{fmt}}"
                else:
                    row += f"  {str(val):>{col_width}}"
        print(row)

    # Delta row for key metrics if exactly 2 runs
    if len(entries) == 2:
        print("-" * len(header))
        delta_keys = [
            ("coverage", ".4f", True),  # higher is better
            ("board_reconstruction", ".4f", True),
            ("fvu", ".6f", False),  # lower is better
            ("dead_features_pct", ".1f", False),
        ]
        m0 = entries[0][1].get("metrics", {})
        m1 = entries[1][1].get("metrics", {})
        row = f"{'Δ (B - A)':<{label_width}}"
        for key, fmt, higher_better in delta_keys:
            v0, v1 = m0.get(key), m1.get(key)
            if v0 is not None and v1 is not None:
                delta = v1 - v0
                sign = "+" if delta > 0 else ""
                marker = "✓" if (delta > 0) == higher_better else "✗"
                row += f"  {sign}{delta:{fmt}} {marker}"
        print(row)


def cmd_history(args: dict) -> None:
    """List all evaluation runs."""
    filter_game = args["--game"]
    filter_arch = args["--arch"]

    all_entries = []
    for game_dir in sorted(Path("saes").iterdir()):
        if not game_dir.is_dir():
            continue
        game = game_dir.name
        if filter_game and game != filter_game:
            continue

        registry = _load_registry(game)
        for run_id, entry in registry.items():
            if filter_arch and entry.get("architecture", "") != filter_arch:
                continue
            all_entries.append((run_id, entry))

    if not all_entries:
        log.info("No evaluation runs found.")
        return

    # Sort by timestamp
    all_entries.sort(key=lambda x: x[1].get("timestamp", ""))

    # Table header
    fmt = "{:<45} {:>8} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}"
    print(
        fmt.format("Run ID", "Arch", "L0", "FVU", "Cover", "Cov>75", "BRecon", "BSPs")
    )
    print("-" * 110)

    for run_id, entry in all_entries:
        m = entry.get("metrics", {})
        arch = entry.get("architecture", "?")[:8]
        print(
            fmt.format(
                run_id[:45],
                arch,
                f"{m.get('l0', 0):.1f}",
                f"{m.get('fvu', 0):.4f}",
                f"{m.get('coverage', 0):.4f}",
                f"{m.get('coverage_above_75', 0):.4f}",
                f"{m.get('board_reconstruction', 0):.4f}",
                entry.get("bsp_set", "?"),
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = docopt(__doc__)

    # Configure logging
    level = logging.DEBUG if args["--verbose"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )

    if args["evaluate"]:
        cmd_evaluate(args)
    elif args["compare"]:
        cmd_compare(args)
    elif args["history"]:
        cmd_history(args)


if __name__ == "__main__":
    main()
