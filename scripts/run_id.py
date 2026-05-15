#!/usr/bin/env python
"""Derive the deterministic run_id / checkpoint path from a training YAML config.

The run_id format produced by sae_train.py is::

    {experiment}-{arch}-{arch_param}-exp{E}-{hook}

where ``arch_param`` is:

==============   ==============================
architecture     suffix
==============   ==============================
topk             k{k}
batchtopk        k{k}
vanilla, gated   l1_{l1_weight as digits}
jumprelu         t{l0_target as int}
panneal          (none — only ``arch-exp{E}``)
==============   ==============================

This script reimplements the *same* function as sae_train.build_filename_suffix
so other tooling (shell scripts, registry_query, sweep orchestration) doesn't
have to reconstruct the rule manually.

Usage:
    run_id.py <config.yaml> [--checkpoint] [--metrics] [--game-dir]

Options:
    --checkpoint    Print the full checkpoint path (saes/<game>/<run_id>.pt).
    --metrics       Print the full metrics path (saes/<game>/<run_id>_metrics.jsonl).
    --game-dir      Print the per-game saes directory only.
    (default)       Print the run_id stem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from docopt import docopt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from sae_train import build_filename_suffix  # noqa: E402


def derive_run_id(config_path: Path) -> tuple[str, str]:
    """Return (run_id, game) for a training YAML config."""
    cfg = yaml.safe_load(config_path.read_text())

    required = ("experiment", "architecture", "game", "hook", "expansion")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"{config_path}: missing required keys {missing}. "
            f"Got keys: {sorted(cfg.keys())}"
        )

    arch = cfg["architecture"]
    arch_kwargs: dict = {}
    if arch in ("topk", "batchtopk"):
        arch_kwargs["k"] = int(cfg["k"])
    elif arch in ("vanilla", "gated"):
        arch_kwargs["l1_weight"] = float(cfg["l1_weight"])
    elif arch == "jumprelu":
        arch_kwargs["l0_target"] = int(cfg["l0_target"])

    suffix = build_filename_suffix(arch, arch_kwargs, int(cfg["expansion"]))
    run_id = f"{cfg['experiment']}-{suffix}-{cfg['hook']}"
    return run_id, cfg["game"]


def main() -> None:
    args = docopt(__doc__)
    cfg_path = Path(args["<config.yaml>"])
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    run_id, game = derive_run_id(cfg_path)
    game_dir = PROJECT_DIR / "saes" / game

    if args["--checkpoint"]:
        print(game_dir / f"{run_id}.pt")
    elif args["--metrics"]:
        print(game_dir / f"{run_id}_metrics.jsonl")
    elif args["--game-dir"]:
        print(game_dir)
    else:
        print(run_id)


if __name__ == "__main__":
    main()
