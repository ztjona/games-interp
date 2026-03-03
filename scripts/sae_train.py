"""SAE Trainer — CLI entry point.

Usage:
    sae_train.py train <architecture> --hook <layer> [options]
    sae_train.py resume <checkpoint> --data <path> [options]
    sae_train.py info <checkpoint>
    sae_train.py list-hooks <model_path> [--model-class <cls>] [--project-root <path>]
    sae_train.py (-h | --help)

Commands:
    train       Train a new SAE on pre-collected activation data
    resume      Resume training from an existing checkpoint
    info        Print checkpoint metadata as JSON
    list-hooks  List all named modules in a model (for choosing --hook)

Arguments:
    <architecture>    One of: vanilla, topk, batchtopk, gated, jumprelu, p-annealing
    <checkpoint>      Path to an existing .pt checkpoint file
    <model_path>      Path to a game model .pt file

Options:
    -h --help               Show this help
    --model <path>          Game model .pt file (used by list-hooks)
    --hook <layer>          Named layer to hook (e.g. fc1)
    --data <path>           Path to pre-collected activation .pt file
    --expansion <int>       Expansion factor: d_dict = expansion * d_input [default: 8]
    --k <int>               Top-K value for topk / batchtopk [default: 64]
    --lr <float>            Learning rate [default: 3e-4]
    --batch-size <int>      Batch size [default: 2048]
    --num-batches <int>     Number of training steps [default: 25000]
    --l1-weight <float>     L1 sparsity weight for vanilla / gated [default: 1e-3]
    --device <str>          Torch device [default: cuda]
    --tag <str>             Run tag for checkpoint naming [default: default]
    --game <str>            Game name for output directory [default: unknown]
    --seed <int>            Random seed [default: 42]
    --flatten               Flatten conv activations (B,C,H,W) -> (B*H*W,C)
    --model-class <cls>     Fully qualified model class for state_dict loading
    --project-root <path>   Project root added to sys.path for model imports
    --config <path>         Config YAML (auto-resolved if omitted)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from docopt import docopt

# ---------------------------------------------------------------------------
# Make lib/ importable regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.sae import (  # noqa: E402
    ARCHITECTURES,
    BatchTopKSAE,
    load_activation_data,
    load_checkpoint,
    load_game_model,
    list_hooks,
    save_checkpoint,
    train_sae,
)


def _stderr(msg: str):
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SKILL_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / ".agents"
    / "skills"
    / "sae-implementation"
)
_DEFAULT_CONFIG = _SKILL_DIR / "config.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def expand(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        return obj

    return expand(config)  # type: ignore[return-value]


def resolve_config(args: dict) -> dict[str, Any]:
    path = args.get("--config") or _DEFAULT_CONFIG
    try:
        return load_config(Path(path))
    except FileNotFoundError:
        _stderr(f"Config not found at {path}, using empty defaults.")
        return {}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_train(args: dict, config: dict):
    arch_name = args["<architecture>"]
    if arch_name not in ARCHITECTURES:
        _stderr(f"Unknown architecture: {arch_name}. Available: {list(ARCHITECTURES)}")
        sys.exit(1)

    hook_name = args["--hook"]
    device = args["--device"] or "cuda"
    tag = args["--tag"] or "default"
    game = args["--game"] or "unknown"
    seed = int(args["--seed"] or 42)
    flatten = bool(args.get("--flatten", False))

    defaults = config.get("defaults", {})
    batch_size = int(args["--batch-size"] or defaults.get("batch_size", 2048))
    lr = float(args["--lr"] or defaults.get("learning_rate", 3e-4))
    num_batches = int(args["--num-batches"] or defaults.get("num_batches", 25000))
    expansion = int(args["--expansion"] or defaults.get("expansion_factor", 8))
    log_every = int(defaults.get("log_every", 500))

    arch_defaults = config.get("architectures", {}).get(arch_name, {})
    k = int(args["--k"] or arch_defaults.get("k", 64))
    l1_weight = float(args["--l1-weight"] or arch_defaults.get("l1_weight", 1e-3))

    data_path = args["--data"]
    if not data_path:
        _stderr("ERROR: --data <path> is required.")
        sys.exit(1)

    data = load_activation_data(data_path, device)

    if flatten and data.dim() == 4:
        B, C, H, W = data.shape
        _stderr(f"Flattening: ({B},{C},{H},{W}) -> ({B * H * W},{C})")
        data = data.permute(0, 2, 3, 1).reshape(-1, C)

    d_input = data.shape[-1]
    d_dict = d_input * expansion
    _stderr(f"d_input={d_input}  d_dict={d_dict} ({expansion}×)")

    # Build architecture-specific constructor kwargs
    constructor_kwargs: dict[str, Any] = {}
    if arch_name == "vanilla":
        constructor_kwargs = {"l1_weight": l1_weight}
    elif arch_name == "topk":
        constructor_kwargs = {
            "k": k,
            "aux_loss_weight": float(arch_defaults.get("aux_loss_weight", 1e-2)),
        }
    elif arch_name == "batchtopk":
        constructor_kwargs = {"k": k}
    elif arch_name == "gated":
        constructor_kwargs = {"l1_weight": l1_weight}
    elif arch_name == "jumprelu":
        constructor_kwargs = {
            "theta_init": float(arch_defaults.get("theta_init", 0.001)),
            "bandwidth": float(arch_defaults.get("bandwidth", 0.001)),
            "l0_target": float(arch_defaults.get("l0_target", 50.0)),
            "l0_weight": float(arch_defaults.get("l0_weight", 1e-2)),
        }
    elif arch_name == "p-annealing":
        constructor_kwargs = {
            "p_start": float(arch_defaults.get("p_start", 1.0)),
            "p_end": float(arch_defaults.get("p_end", 0.2)),
            "p_anneal_steps": int(arch_defaults.get("p_anneal_steps", 15000)),
            "lp_weight": float(arch_defaults.get("lp_weight", 1e-3)),
        }

    sae = ARCHITECTURES[arch_name](
        d_input=d_input, d_dict=d_dict, device=device, **constructor_kwargs
    )

    results = train_sae(
        sae,
        data,
        num_batches=num_batches,
        batch_size=batch_size,
        lr=lr,
        log_every=log_every,
        seed=seed,
    )

    # Save checkpoint
    project_dir = config.get("project_dir", str(_PROJECT_ROOT))
    saes_dir = Path(project_dir) / config.get("saes_dir", "saes")
    ckpt_name = f"{game}-{arch_name}-{hook_name}-{expansion}x-{tag}.pt"
    ckpt_path = saes_dir / game / ckpt_name

    metadata: dict[str, Any] = {
        "architecture": arch_name,
        "game": game,
        "hook": hook_name,
        "d_input": d_input,
        "d_dict": d_dict,
        "expansion_factor": expansion,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_batches": num_batches,
        "seed": seed,
        "tag": tag,
        "flatten": flatten,
        "data_path": str(data_path),
        "constructor_kwargs": constructor_kwargs,
    }
    save_checkpoint(sae, ckpt_path, metadata)

    output = {
        "architecture": arch_name,
        "hook": hook_name,
        "activation_dim": d_input,
        "dictionary_size": d_dict,
        "checkpoint": str(ckpt_path),
        "training_metrics": {
            "final_loss": (
                results["metrics_log"][-1]["loss"] if results["metrics_log"] else None
            ),
            **results["final_metrics"],
            "num_steps": results["num_steps"],
            "num_epochs": results["num_epochs"],
            "wall_time_seconds": results["wall_time_seconds"],
        },
        "config": {
            "batch_size": batch_size,
            "learning_rate": lr,
            "expansion_factor": expansion,
            "tag": tag,
            "seed": seed,
            **constructor_kwargs,
        },
    }
    print(json.dumps(output, indent=2))


def cmd_resume(args: dict, config: dict):
    ckpt_path = Path(args["<checkpoint>"])
    device = args["--device"] or "cuda"
    data_path = args["--data"]
    if not data_path:
        _stderr("ERROR: --data <path> is required for resume.")
        sys.exit(1)

    sae, metadata = load_checkpoint(ckpt_path, device)
    data = load_activation_data(data_path, device)

    if args.get("--flatten", False) and data.dim() == 4:
        B, C, H, W = data.shape
        data = data.permute(0, 2, 3, 1).reshape(-1, C)

    defaults = config.get("defaults", {})
    batch_size = int(args["--batch-size"] or metadata.get("batch_size", 2048))
    lr = float(args["--lr"] or metadata.get("learning_rate", 3e-4))
    num_batches = int(args["--num-batches"] or defaults.get("num_batches", 25000))
    seed = int(args["--seed"] or metadata.get("seed", 42))

    _stderr(f"Resuming {metadata.get('architecture', '?')} from {ckpt_path}")

    results = train_sae(
        sae,
        data,
        num_batches=num_batches,
        batch_size=batch_size,
        lr=lr,
        log_every=int(defaults.get("log_every", 500)),
        seed=seed,
    )

    save_checkpoint(sae, ckpt_path, {**metadata, "resumed": True})

    output = {
        "resumed_from": str(ckpt_path),
        "training_metrics": {
            "final_loss": (
                results["metrics_log"][-1]["loss"] if results["metrics_log"] else None
            ),
            **results["final_metrics"],
            "num_steps": results["num_steps"],
            "wall_time_seconds": results["wall_time_seconds"],
        },
    }
    print(json.dumps(output, indent=2))


def cmd_info(args: dict):
    ckpt_path = Path(args["<checkpoint>"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    info = {
        "architecture": checkpoint.get("architecture"),
        "d_input": checkpoint.get("d_input"),
        "d_dict": checkpoint.get("d_dict"),
        "checkpoint": str(ckpt_path),
        **metadata,
    }
    print(json.dumps(info, indent=2, default=str))


def cmd_list_hooks(args: dict):
    model_path = args["<model_path>"]
    model_class = args.get("--model-class")
    project_root = args.get("--project-root")

    _stderr(f"Loading model from {model_path}...")
    try:
        model = load_game_model(
            model_path,
            model_class=model_class,
            project_root=project_root,
            device="cpu",
        )
    except Exception as e:
        _stderr(f"Error loading model: {e}")
        sys.exit(1)

    output = {"model_path": model_path, "modules": list_hooks(model)}
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = docopt(__doc__)
    config = resolve_config(args)

    if args["train"]:
        cmd_train(args, config)
    elif args["resume"]:
        cmd_resume(args, config)
    elif args["info"]:
        cmd_info(args)
    elif args["list-hooks"]:
        cmd_list_hooks(args)


if __name__ == "__main__":
    main()
