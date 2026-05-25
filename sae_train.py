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
    --l0-patience=<int>     Early-stop if L0 plateau lasts N eval windows. 0=off [default: 0]
    --l0-min-change=<f>     Min absolute L0 change per window to reset L0 counter [default: 5.0]
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
import json
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

    elif arch in ("anchored-jumprelu", "anchored-batchtopk"):
        # Base-architecture hyperparameters
        if arch == "anchored-jumprelu":
            kwargs["theta_init"] = float(args["--jump-threshold"])
            kwargs["l0_target"] = float(args["--l0-target"])
        else:  # anchored-batchtopk
            kwargs["k"] = int(args["--k"])
        # Anchor mapping is built later in main() once labels + schema are loaded;
        # placeholders here are filled in by main() before SAE instantiation.

    return kwargs


def _build_anchor_kwargs(config: dict | None, num_anchor_cols: int) -> tuple[dict, dict]:
    """Resolve anchored-SAE mapping from a YAML config + label tensor + schema.

    Anchored variants are YAML-only (no CLI flags) to keep the docopt
    surface clean.  The raw ``config`` dict is read directly so list values
    (``anchor_high_categories``) survive without being stringified.

    Returns:
        (anchor_kwargs, anchor_meta)
        anchor_kwargs feeds the SAE constructor (lists, not tensors, so the
        same dict can be serialized into the JSON training registry).
        anchor_meta carries the file paths and λ tier breakdown for
        bookkeeping / filename suffix.
    """
    if config is None:
        raise ValueError(
            "Anchored architectures are YAML-only; pass --config=<file>."
        )

    schema_path = config.get("anchor_schema")
    if not schema_path:
        raise ValueError("Anchored YAML config requires 'anchor_schema'")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    bsps = schema.get("bsps", schema) if isinstance(schema, dict) else schema
    if num_anchor_cols != len(bsps):
        raise ValueError(
            f"Anchor label tensor has {num_anchor_cols} columns but schema "
            f"lists {len(bsps)} BSPs — schema/label mismatch."
        )

    mode = config.get("anchor_index_mode", "prefix")
    if mode != "prefix":
        raise NotImplementedError(
            f"anchor_index_mode='{mode}' not implemented; only 'prefix' is supported."
        )
    anchor_feature_idx = list(range(num_anchor_cols))
    anchor_bsp_idx = list(range(num_anchor_cols))

    if "anchor_lambda_high" not in config or "anchor_lambda_medium" not in config:
        raise ValueError(
            "Anchored YAML config requires 'anchor_lambda_high' and "
            "'anchor_lambda_medium'."
        )
    lam_high = float(config["anchor_lambda_high"])
    lam_med = float(config["anchor_lambda_medium"])
    high_cats = list(config.get("anchor_high_categories") or [])
    high_cats_set = set(high_cats)

    anchor_lambda_per_feature: list[float] = []
    tier_counts = {"high": 0, "medium": 0}
    for bsp in bsps:
        cat = bsp.get("category", "")
        if cat in high_cats_set:
            anchor_lambda_per_feature.append(lam_high)
            tier_counts["high"] += 1
        else:
            anchor_lambda_per_feature.append(lam_med)
            tier_counts["medium"] += 1

    anchor_kwargs = {
        "anchor_feature_idx": anchor_feature_idx,
        "anchor_bsp_idx": anchor_bsp_idx,
        "anchor_lambda_per_feature": anchor_lambda_per_feature,
    }
    anchor_meta = {
        "anchor_bsps": config.get("anchor_bsps"),
        "anchor_schema": schema_path,
        "anchor_index_mode": mode,
        "anchor_lambda_high": lam_high,
        "anchor_lambda_medium": lam_med,
        "anchor_high_categories": high_cats,
        "anchor_tier_counts": tier_counts,
        "anchor_loss": config.get("anchor_loss", "bce"),
    }
    return anchor_kwargs, anchor_meta


def build_filename_suffix(arch: str, arch_kwargs: dict, expansion: int) -> str:
    """Build descriptive filename suffix based on architecture and hyperparams."""
    parts = [arch]

    if arch in ("topk", "batchtopk"):
        parts.append(f"k{arch_kwargs['k']}")
    elif arch in ("vanilla", "gated"):
        l1 = arch_kwargs.get("l1_weight", 0)
        # e.g. 0.005 -> "l1_005", 0.01 -> "l1_01"
        l1_str = str(l1).replace("0.", "").replace(".", "")
        parts.append(f"l1_{l1_str}")
    elif arch == "jumprelu":
        parts.append(f"t{int(arch_kwargs['l0_target'])}")
    elif arch == "anchored-jumprelu":
        parts.append(f"t{int(arch_kwargs['l0_target'])}")
    elif arch == "anchored-batchtopk":
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
    l0_patience = int(args["--l0-patience"])
    l0_min_change = float(args["--l0-min-change"])

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
    es_info = (
        f", patience={patience} (min_imp={min_improvement:.1%})"
        + (
            f", l0_patience={l0_patience} (min_chg={l0_min_change})"
            if l0_patience > 0
            else ""
        )
        if patience > 0 or l0_patience > 0
        else ""
    )
    print(
        f"Hyperparameters: expansion={expansion}x, batch_size={batch_size}, num_batches={num_batches}, lr={lr}{es_info}"
    )

    # Load data
    data = load_activation_data(data_path, device)
    d_input = data.shape[-1]

    # Get architecture-specific kwargs
    arch_kwargs = get_arch_kwargs(arch, args)

    # Anchored variants: load labels + schema and build per-feature λ vector
    anchor_labels: torch.Tensor | None = None
    anchor_meta: dict = {}
    if arch in ("anchored-jumprelu", "anchored-batchtopk"):
        anchor_labels_path = (config or {}).get("anchor_bsps")
        if not anchor_labels_path:
            print("Error: anchored YAML config requires 'anchor_bsps' (label tensor path).")
            sys.exit(1)
        print(f"Anchor labels: {anchor_labels_path}")
        anchor_labels = torch.load(
            anchor_labels_path, map_location=device, weights_only=True
        )
        if anchor_labels.ndim != 2:
            print(
                f"Error: anchor_bsps must be 2D (N, num_bsps); got {tuple(anchor_labels.shape)}"
            )
            sys.exit(1)
        if anchor_labels.shape[0] != data.shape[0]:
            print(
                f"Error: anchor label row count ({anchor_labels.shape[0]}) "
                f"does not match activation count ({data.shape[0]})."
            )
            sys.exit(1)
        anchor_kwargs, anchor_meta = _build_anchor_kwargs(
            config, num_anchor_cols=anchor_labels.shape[1]
        )
        arch_kwargs = {**arch_kwargs, **anchor_kwargs}
        print(
            f"Anchor map: {anchor_meta['anchor_tier_counts']['high']} high-tier x "
            f"lambda={anchor_meta['anchor_lambda_high']} + "
            f"{anchor_meta['anchor_tier_counts']['medium']} medium-tier x "
            f"lambda={anchor_meta['anchor_lambda_medium']} "
            f"(loss={anchor_meta['anchor_loss']}, mode={anchor_meta['anchor_index_mode']})"
        )

    print(f"Architecture kwargs: { {k: (f'<list len={len(v)}>' if isinstance(v, list) and len(v) > 10 else v) for k, v in arch_kwargs.items()} }")

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
    try:
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
            l0_patience=l0_patience,
            l0_min_change=l0_min_change,
            anchor_labels=anchor_labels,
        )
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Partial metrics written to JSONL. No checkpoint saved."
        )
        return

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
    if anchor_meta:
        metadata["anchor"] = anchor_meta

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
