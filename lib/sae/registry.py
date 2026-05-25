"""Training experiment registry for SAE runs."""

import json
import msvcrt
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

    Uses file locking for parallel-safe read-modify-write.
    """
    if registry_path is None:
        registry_path = f"saes/{game}/training_registry.json"

    registry_file = Path(registry_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    run_id = Path(checkpoint_path).stem
    entry = {
        "experiment": experiment,
        "architecture": architecture,
        "game": game,
        "hook": hook,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        **hyperparams,
        "final_metrics": final_metrics,
    }

    lock_path = registry_file.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        try:
            registry = {}
            if registry_file.exists():
                with open(registry_file, "r") as f:
                    registry = json.load(f)
            registry[run_id] = entry
            tmp = registry_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(registry, f, indent=2)
            tmp.replace(registry_file)
        finally:
            msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)

    print(f"Registered training run: {run_id}")
