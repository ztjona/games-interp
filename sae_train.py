"""Unified SAE training script with YAML config support.

Usage:
    sae_train.py <experiment> <architecture> [options]
    sae_train.py --config=<yaml>
    sae_train.py -h | --help

Arguments:
    <experiment>        Experiment name (e.g., "pilot", "sweep-lr")
    <architecture>      SAE architecture: vanilla|topk|batchtopk|gated|jumprelu|p-annealing

Options:
    --game=<name>           Game name [default: quarto]
    --hook=<layer>          Hook layer [default: fc1]
    --data=<path>           Activation data path [default: auto]
    --expansion=<n>         Dictionary expansion factor [default: 8]
    --batch-size=<n>        Batch size [default: 4096]
    --num-batches=<n>       Number of training steps [default: 25000]
    --lr=<float>            Learning rate [default: 3e-4]
    --seed=<int>            Random seed [default: 42]
    --log-every=<n>         Log metrics every N steps [default: 500]
    --patience=<int>        Early-stop after N eval windows with no FVU improvement. 0=off [default: 0]
    --min-improvement=<f>   Min relative FVU improvement to reset patience [default: 0.01]
    --device=<dev>          Device (cuda|cpu|auto) [default: auto]

    # Architecture-specific hyperparameters
    --l1-weight=<float>     Vanilla: L1 sparsity weight [default: 1e-3]
    --k=<int>               TopK/BatchTopK: number of active features [default: 16]
    --aux-loss-weight=<f>   TopK/BatchTopK: auxiliary loss weight [default: 1e-2]
    --gated-l1=<float>      Gated: L1 weight on gate [default: 1e-3]
    --jump-threshold=<f>    JumpReLU: threshold parameter [default: 0.001]
    --l0-target=<float>     JumpReLU: target L0 sparsity [default: 50]
    --p-start=<float>       P-annealing: initial p [default: 1.0]
    --p-end=<float>         P-annealing: final p [default: 0.2]

    --config=<yaml>         Load all hyperparameters from YAML file (overrides CLI args)
    -h --help               Show this help

Examples:
    # Command line
    python sae_train.py pilot vanilla --expansion=8 --lr=3e-4
    python sae_train.py pilot topk --k=16 --expansion=8

    # From config file
    python sae_train.py --config=configs/pilot-vanilla.yaml

    # Parallel experiments (in separate terminals)
    python sae_train.py sweep-lr-1 vanilla --lr=1e-4 &
    python sae_train.py sweep-lr-2 vanilla --lr=3e-4 &
    python sae_train.py sweep-lr-3 vanilla --lr=1e-3 &

    # Live monitoring in another terminal
    python scripts/plot_training.py saes/quarto/pilot-*_metrics.jsonl --live
    # Then open http://localhost:8050 in your browser
    - Config files override all CLI arguments
    - Outputs: saes/{game}/{experiment}-{arch}-{hook}-*.pt and *_metrics.jsonl
    - Registry auto-updates at saes/{game}/training_registry.json
    - Use external terminal (not VS Code integrated) to avoid process limit issues
"""

import sys
from pathlib import Path

import yaml
import torch
from docopt import docopt

from lib.sae import (
    ARCHITECTURES,
    load_activation_data,
    train_sae,
    save_checkpoint,
    register_training_run,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    """Parse command line arguments and config file."""
    args = docopt(__doc__, argv=argv)

    # If config file provided, load it and override args
    if args["--config"]:
        config = load_config(args["--config"])
        # Map YAML keys to docopt arg format
        for key, value in config.items():
            if key == "experiment":
                args["<experiment>"] = value
            elif key == "architecture":
                args["<architecture>"] = value
            else:
                # Convert to docopt format: num_batches -> --num-batches
                arg_key = f"--{key.replace('_', '-')}"
                args[arg_key] = str(value)
        return args, config

    return args, None


def get_arch_kwargs(arch: str, args: dict) -> dict:
    """Extract architecture-specific constructor kwargs from args."""
    kwargs = {}

    if arch == "vanilla":
        kwargs["l1_weight"] = float(args["--l1-weight"])

    elif arch == "topk":
        kwargs["k"] = int(args["--k"])
        kwargs["aux_loss_weight"] = float(args["--aux-loss-weight"])

    elif arch == "batchtopk":
        kwargs["k"] = int(args["--k"])
        # BatchTopKSAE has no aux_loss_weight parameter

    elif arch == "gated":
        kwargs["l1_weight"] = float(args["--gated-l1"])

    elif arch == "jumprelu":
        kwargs["theta_init"] = float(args["--jump-threshold"])
        kwargs["l0_target"] = float(args["--l0-target"])
        # bandwidth, l0_weight use constructor defaults for now

    elif arch == "p-annealing":
        kwargs["p_start"] = float(args["--p-start"])
        kwargs["p_end"] = float(args["--p-end"])

    return kwargs


def build_filename_suffix(arch: str, arch_kwargs: dict, expansion: int) -> str:
    """Build descriptive filename suffix based on architecture and hyperparams."""
    parts = [arch]

    if arch in ("topk", "batchtopk"):
        parts.append(f"k{arch_kwargs['k']}")

    parts.append(f"exp{expansion}")

    return "-".join(parts)


def main():
    args, config = parse_args()

    # Extract core parameters
    experiment = args["<experiment>"]
    arch = args["<architecture>"]
    game = args["--game"]
    hook = args["--hook"]
    expansion = int(args["--expansion"])
    batch_size = int(args["--batch-size"])
    num_batches = int(args["--num-batches"])
    lr = float(args["--lr"])
    seed = int(args["--seed"])
    log_every = int(args["--log-every"])
    patience = int(args["--patience"])
    min_improvement = float(args["--min-improvement"])

    # Device
    device_arg = args["--device"]
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    # Data path
    data_path = args["--data"]
    if data_path == "auto":
        data_path = f"data/{game}/{hook}_amalgam_activations.pt"

    # Validate architecture
    if arch not in ARCHITECTURES:
        print(f"Error: Unknown architecture '{arch}'")
        print(f"Available: {', '.join(ARCHITECTURES.keys())}")
        sys.exit(1)

    print(f"Experiment: {experiment}")
    print(f"Architecture: {arch}")
    print(f"Game: {game}, Hook: {hook}")
    print(f"Device: {device}")
    es_info = f", patience={patience} (min_imp={min_improvement:.1%})" if patience > 0 else ""
    print(
        f"Hyperparameters: expansion={expansion}x, batch_size={batch_size}, num_batches={num_batches}, lr={lr}{es_info}"
    )

    # Load data
    data = load_activation_data(data_path, device)
    d_input = data.shape[-1]

    # Get architecture-specific kwargs
    arch_kwargs = get_arch_kwargs(arch, args)
    print(f"Architecture kwargs: {arch_kwargs}")

    # Create SAE
    sae = ARCHITECTURES[arch](
        d_input=d_input, d_dict=d_input * expansion, device=device, **arch_kwargs
    )

    # Build output paths
    output_dir = Path(f"saes/{game}")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = build_filename_suffix(arch, arch_kwargs, expansion)
    checkpoint_path = str(output_dir / f"{experiment}-{suffix}-{hook}.pt")
    metrics_file = str(output_dir / f"{experiment}-{suffix}-{hook}_metrics.jsonl")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_file}")
    print()

    # Train
    results = train_sae(
        sae,
        data,
        num_batches,
        batch_size,
        lr,
        log_every=log_every,
        seed=seed,
        metrics_file=metrics_file,
        patience=patience,
        min_improvement=min_improvement,
    )

    # Prepare metadata
    all_hyperparams = {
        "expansion": expansion,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "lr": lr,
        "seed": seed,
        "log_every": log_every,
        "patience": patience,
        "min_improvement": min_improvement,
        **arch_kwargs,
    }

    metadata = {
        "experiment": experiment,
        "architecture": arch,
        "hook": hook,
        "game": game,
        "data_path": data_path,
        "device": device,
        "hyperparameters": all_hyperparams,
        "constructor_kwargs": arch_kwargs,
        "training_results": {
            "final_step": results["final_step"],
            "num_epochs": results["num_epochs"],
            "wall_time_seconds": results["wall_time_seconds"],
            "early_stopped": results["early_stopped"],
        },
        "final_metrics": results["final_metrics"],
    }

    # Save checkpoint
    save_checkpoint(sae, checkpoint_path, metadata)

    # Register in experiment tracking
    register_training_run(
        experiment=experiment,
        architecture=arch,
        game=game,
        hook=hook,
        checkpoint_path=checkpoint_path,
        hyperparams=all_hyperparams,
        final_metrics=results["final_metrics"],
    )

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_file}")
    print(f"Final metrics: {results['final_metrics']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
