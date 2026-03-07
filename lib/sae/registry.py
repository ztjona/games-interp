"""Training experiment registry for SAE runs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def register_training_run(
    experiment: str,
    architecture: str,
    game: str,
    hook: str,
    checkpoint_path: str,
    hyperparams: dict[str, Any],
    final_metrics: dict[str, Any],
    registry_path: str | None = None,
) -> None:
    """Register a completed training run in the experiment registry.

    Args:
        experiment: Experiment name (e.g., "pilot", "sweep1")
        architecture: SAE architecture (e.g., "vanilla", "topk")
        game: Game name (e.g., "quarto")
        hook: Hook point name (e.g., "fc1")
        checkpoint_path: Path to saved checkpoint
        hyperparams: Dict of hyperparameters (expansion, lr, seed, etc.)
        final_metrics: Dict of final metrics (loss, fvu, l0, dead_features_pct)
        registry_path: Optional custom registry path (default: saes/{game}/training_registry.json)
    """
    if registry_path is None:
        registry_path = f"saes/{game}/training_registry.json"

    registry_file = Path(registry_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing registry
    if registry_file.exists():
        with open(registry_file, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Create run ID
    run_id = Path(checkpoint_path).stem

    # Add new entry
    registry[run_id] = {
        "experiment": experiment,
        "architecture": architecture,
        "game": game,
        "hook": hook,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        **hyperparams,
        "final_metrics": final_metrics,
    }

    # Save registry
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Registered training run: {run_id}")
