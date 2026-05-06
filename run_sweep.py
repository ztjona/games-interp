"""Anakin sweep orchestrator: train + evaluate all configs across GPUs.

Runs SAE training for each YAML config in sequence, with optional
evaluation after each training. Designed for multi-GPU parallelism via
split mode: run one instance per GPU in separate terminals.

Usage:
    run_sweep.py --configs=<dir> [--gpu=<id>] [--split=<N/M>] [options]
    run_sweep.py -h | --help

Options:
    --configs=<dir>         Directory with YAML config files [default: configs/anakin]
    --gpu=<id>              GPU device ID (sets CUDA_VISIBLE_DEVICES) [default: 0]
    --split=<N/M>           Run partition N of M (1-indexed). E.g. --split=1/3
    --eval                  Run evaluation after each training
    --bsps=<name>           BSP set for evaluation [default: gorilla]
    --skip-existing         Skip configs whose output .pt already exists
    --dry-run               Print what would be run without executing
    --tier=<ids>            Only run configs from specific tiers (comma-separated, e.g. A,B,C)
    --timeout=<sec>         Max seconds per training run [default: 86400]
    -h --help               Show this help

Examples:
    # All configs on GPU 0
    python run_sweep.py --configs=configs/anakin --gpu=0 --eval

    # Split across 3 GPUs (run each in a separate terminal window)
    python run_sweep.py --configs=configs/anakin --gpu=0 --split=1/3 --eval --skip-existing
    python run_sweep.py --configs=configs/anakin --gpu=1 --split=2/3 --eval --skip-existing
    python run_sweep.py --configs=configs/anakin --gpu=2 --split=3/3 --eval --skip-existing

    # Only architecture comparison tier on one GPU
    python run_sweep.py --configs=configs/anakin --gpu=0 --tier=A --eval

    # Dry run to see the plan
    python run_sweep.py --configs=configs/anakin --split=1/3 --dry-run

Notes:
    - Configs are sorted alphabetically and assigned to splits round-robin.
    - GPU 0 (RTX 4000) gets conv2 configs first (heavier), P4000s get fc1 first.
    - Failures are logged and skipped — the sweep continues.
    - Progress is logged to logs/sweep_{gpu}_{timestamp}.log
    - --skip-existing checks for a .pt checkpoint file. JSONL-only partial runs
      (no .pt saved) will NOT be skipped and will re-run from scratch, which is
      correct. If you want to skip a run that produced no .pt, delete its YAML.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    """Minimal arg parsing (avoiding docopt dependency for the runner)."""
    args = {
        "configs": "configs/anakin",
        "gpu": "0",
        "split": None,
        "eval": False,
        "bsps": "gorilla",
        "skip_existing": False,
        "timeout": 86400,
        "dry_run": False,
        "tier": None,
    }

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif a.startswith("--configs="):
            args["configs"] = a.split("=", 1)[1]
        elif a.startswith("--gpu="):
            args["gpu"] = a.split("=", 1)[1]
        elif a.startswith("--split="):
            args["split"] = a.split("=", 1)[1]
        elif a == "--eval":
            args["eval"] = True
        elif a.startswith("--bsps="):
            args["bsps"] = a.split("=", 1)[1]
        elif a == "--skip-existing":
            args["skip_existing"] = True
        elif a == "--dry-run":
            args["dry_run"] = True
        elif a.startswith("--tier="):
            args["tier"] = a.split("=", 1)[1]
        elif a.startswith("--timeout="):
            args["timeout"] = int(a.split("=", 1)[1])
        i += 1

    return args


def load_configs(config_dir: str, tier_filter: str | None = None) -> list[Path]:
    """Load and sort YAML configs from directory, optionally filtering by tier."""
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"ERROR: Config directory not found: {config_path}")
        sys.exit(1)

    configs = sorted(config_path.glob("*.yaml"))

    if tier_filter:
        allowed_tiers = set(t.strip().upper() for t in tier_filter.split(","))
        filtered = []
        for cfg_path in configs:
            with open(cfg_path, encoding="utf-8") as f:
                first_line = f.readline()
            tier = None
            # Method 1: "# Anakin Tier A: ..."
            if "Tier" in first_line:
                tier = first_line.split("Tier")[1].strip().split(":")[0].strip()
            # Method 2: "# Campaign C: ..."
            elif "Campaign" in first_line:
                parts = first_line.split("Campaign")[1].strip().split()
                if parts:
                    tier = parts[0].rstrip(":").upper()
            # Method 3: filename starts with tier letter (e.g., C01-..., D04-...)
            if tier is None and cfg_path.stem and cfg_path.stem[0].isalpha():
                tier = cfg_path.stem[0].upper()
            if tier and tier.upper() in allowed_tiers:
                filtered.append(cfg_path)
        configs = filtered

    return configs


def apply_split(configs: list[Path], split_str: str) -> list[Path]:
    """Select configs for this partition using round-robin."""
    n, m = split_str.split("/")
    n, m = int(n), int(m)
    if n < 1 or n > m:
        print(f"ERROR: Invalid split {split_str} (must be N/M where 1 <= N <= M)")
        sys.exit(1)
    return [c for i, c in enumerate(configs) if (i % m) == (n - 1)]


def get_output_path(cfg: dict) -> Path:
    """Predict the output checkpoint path for a config."""
    arch = cfg["architecture"]
    parts = [arch]
    if arch in ("topk", "batchtopk"):
        parts.append(f"k{cfg['k']}")
    elif arch in ("vanilla", "gated"):
        l1 = cfg.get("l1_weight", cfg.get("gated_l1", 0))
        l1_str = str(l1).replace("0.", "").replace(".", "")
        parts.append(f"l1_{l1_str}")
    elif arch == "jumprelu":
        parts.append(f"t{int(cfg['l0_target'])}")
    parts.append(f"exp{cfg['expansion']}")
    suffix = "-".join(parts)

    game = cfg.get("game", "quarto")
    hook = cfg.get("hook", "fc1")
    experiment = cfg["experiment"]
    return Path(f"saes/{game}/{experiment}-{suffix}-{hook}.pt")


def run_training(
    cfg_path: Path, device: str, timeout: int = 86400
) -> tuple[bool, str, float]:
    """Run sae_train.py for one config. Returns (success, output_path, wall_time)."""
    cmd = [
        sys.executable,
        "sae_train.py",
        f"--config={cfg_path}",
    ]
    # Set CUDA_VISIBLE_DEVICES so sae_train.py's --device=auto picks the right GPU
    env = os.environ.copy()
    gpu_id = device.split(":")[-1] if ":" in device else "0"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        wall = time.time() - start
        if result.returncode != 0:
            return (
                False,
                result.stderr[-500:] if result.stderr else "Unknown error",
                wall,
            )
        # Extract checkpoint path from output
        for line in result.stdout.split("\n"):
            if "Checkpoint:" in line:
                return True, line.split("Checkpoint:")[-1].strip(), wall
        return True, "completed", wall
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT ({timeout}s)", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def run_evaluation(
    checkpoint_path: str, bsps: str, device: str
) -> tuple[bool, str, float]:
    """Run sae_eval.py evaluate on a checkpoint."""
    cmd = [
        sys.executable,
        "sae_eval.py",
        "evaluate",
        checkpoint_path,
        f"--bsps={bsps}",
    ]
    env = os.environ.copy()
    gpu_id = device.split(":")[-1] if ":" in device else "0"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max per eval
            env=env,
        )
        wall = time.time() - start
        if result.returncode != 0:
            return (
                False,
                result.stderr[-500:] if result.stderr else "Unknown error",
                wall,
            )
        return True, "evaluated", wall
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (30 min)", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def main():
    args = parse_args()

    # Set GPU before any imports that might init CUDA
    device = f"cuda:{args['gpu']}" if args["gpu"] != "cpu" else "cpu"

    configs = load_configs(args["configs"], args["tier"])

    if args["split"]:
        configs = apply_split(configs, args["split"])

    if not configs:
        print("No configs to run.")
        return

    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"sweep_gpu{args['gpu']}_{timestamp}.log"

    def log(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Print plan
    total = len(configs)
    log(f"Anakin Sweep — GPU {args['gpu']}, {total} configs")
    if args["split"]:
        log(f"  Split: {args['split']}")
    if args["eval"]:
        log(f"  Eval: {args['bsps']} BSPs after each training")
    log(f"  Log: {log_path}")
    log("")

    for i, cfg_path in enumerate(configs, 1):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        name = cfg_path.stem
        out_path = get_output_path(cfg)

        # Skip existing
        if args["skip_existing"] and out_path.exists():
            log(f"[{i}/{total}] SKIP {name} — {out_path} exists")
            continue

        log(f"[{i}/{total}] TRAIN {name}")

        if args["dry_run"]:
            log(f"  -> would produce {out_path}")
            if args["eval"]:
                log(f"  → would evaluate with {args['bsps']}")
            continue

        # Train
        ok, info, wall = run_training(cfg_path, device, timeout=args["timeout"])
        if ok:
            log(f"  ✓ trained in {wall:.0f}s → {out_path}")
        else:
            log(f"  ✗ FAILED in {wall:.0f}s: {info}")
            continue  # Skip eval if training failed

        # Evaluate
        if args["eval"] and out_path.exists():
            ok_eval, info_eval, wall_eval = run_evaluation(
                str(out_path), args["bsps"], device
            )
            if ok_eval:
                log(f"  ✓ evaluated in {wall_eval:.0f}s")
            else:
                log(f"  ✗ eval FAILED in {wall_eval:.0f}s: {info_eval}")

    log("")
    log(f"Sweep complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()
