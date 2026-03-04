"""SAE training loop, metrics, data utilities, and checkpoint I/O."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .architectures import BaseSAE, BatchTopKSAE, ARCHITECTURES


def _stderr(msg: str):
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(sae: BaseSAE, x: torch.Tensor) -> dict[str, float]:
    """Compute standard SAE evaluation metrics on a batch.

    Returns:
        l0:                 average active features per sample
        fvu:                fraction of variance unexplained
        mse:                mean squared reconstruction error
        dead_features_pct:  % of features inactive across the whole batch
    """
    sae.eval()
    with torch.no_grad():
        result = sae(x)
        x_hat, h = result["x_hat"], result["h"]

        l0 = (h > 0).float().sum(dim=-1).mean().item()

        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        var_x = x.var(dim=0).sum().item()
        fvu = mse / max(var_x, 1e-8)

        alive = (h > 0).any(dim=0).float()
        dead_pct = (1.0 - alive.mean().item()) * 100.0

    sae.train()
    return {
        "l0": round(l0, 2),
        "fvu": round(fvu, 6),
        "mse": round(mse, 6),
        "dead_features_pct": round(dead_pct, 2),
    }


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_activation_data(data_path: str, device: str) -> torch.Tensor:
    """Load pre-collected activation tensor from a .pt file.

    Expected shape: (N, d_activation) or (N, C, H, W) for conv layers.
    """
    _stderr(f"Loading activation data from {data_path}...")
    data = torch.load(data_path, map_location=device, weights_only=True)
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(data)}")
    _stderr(f"Loaded {data.shape[0]} samples, shape: {data.shape}")
    return data


def iter_batches(data: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Yield batches from a tensor dataset with optional shuffling."""
    N = data.shape[0]
    if shuffle:
        perm = torch.randperm(N, device=data.device)
        data = data[perm]
    for i in range(0, N, batch_size):
        yield data[i : i + batch_size]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_sae(
    sae: BaseSAE,
    data: torch.Tensor,
    num_batches: int,
    batch_size: int,
    lr: float,
    log_every: int = 500,
    seed: int = 42,
    metrics_file: Path | str | None = None,
) -> dict[str, Any]:
    """Train an SAE on pre-collected activation data.

    For BatchTopKSAE, automatically calibrates inference thresholds after
    the training loop using the same data.

    Args:
        metrics_file: Optional path to write metrics as JSONL for live monitoring

    Returns:
        dict with keys: num_steps, num_epochs, wall_time_seconds,
                        final_metrics, metrics_log
    """
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()

    metrics_log: list[dict] = []
    step = 0
    epoch = 0

    # Open metrics file for live logging
    metrics_fh = None
    if metrics_file:
        metrics_file = Path(metrics_file)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = open(metrics_file, "w")
    t_start = time.time()

    _stderr(
        f"Training {sae.__class__.__name__} | "
        f"d_input={sae.d_input}, d_dict={sae.d_dict} | "
        f"{num_batches} steps, batch_size={batch_size}, lr={lr}"
    )

    # Progress bar
    pbar = tqdm(total=num_batches, desc="Training", unit="step")

    while step < num_batches:
        epoch += 1
        for batch in iter_batches(data, batch_size, shuffle=True):
            if step >= num_batches:
                break

            result = sae(batch)
            losses = sae.compute_loss(result)
            loss = losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            pbar.update(1)
            sae.normalize_decoder()

            step += 1

            if step % log_every == 0 or step == num_batches:
                eval_metrics = compute_metrics(sae, batch)
                log_entry: dict[str, Any] = {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    **{
                        k: round(v.item(), 6) if isinstance(v, torch.Tensor) else v
                        for k, v in losses.items()
                        if k != "loss"
                    },
                    **eval_metrics,
                }

                metrics_log.append(log_entry)

                # Write to JSONL file for live monitoring
                if metrics_fh:
                    metrics_fh.write(json.dumps(log_entry) + "\n")
                    metrics_fh.flush()  # Ensure immediate write for live plotting

                # Update progress bar with latest metrics
                pbar.set_postfix(
                    {
                        "loss": f"{log_entry['loss']:.4f}",
                        "L0": f"{eval_metrics['l0']:.1f}",
                        "FVU": f"{eval_metrics['fvu']:.4f}",
                        "dead": f"{eval_metrics['dead_features_pct']:.1f}%",
                    }
                )

    pbar.close()

    # Close metrics file
    if metrics_fh:
        metrics_fh.close()

    wall_time = time.time() - t_start
    _stderr(f"Training complete. {step} steps in {wall_time:.1f}s ({epoch} epochs)")

    # BatchTopK: calibrate per-feature JumpReLU thresholds for eval/inference
    if isinstance(sae, BatchTopKSAE):
        _stderr("Calibrating BatchTopK inference thresholds...")
        sae.calibrate_thresholds(data, batch_size=batch_size)
        _stderr("Threshold calibration complete.")

    # Final evaluation on a larger held-out sample
    with torch.no_grad():
        eval_sample = data[: min(len(data), batch_size * 4)]
        sae.eval()
        final_metrics = compute_metrics(sae, eval_sample)
        sae.train()

    return {
        "num_steps": step,
        "num_epochs": epoch,
        "wall_time_seconds": round(wall_time, 2),
        "final_metrics": final_metrics,
        "metrics_log": metrics_log,
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(sae: BaseSAE, path: Path | str, metadata: dict[str, Any]):
    """Save SAE weights and metadata to a .pt checkpoint file."""
    path = Path(path)  # Convert to Path if string
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": sae.state_dict(),
        "architecture": sae.__class__.__name__,
        "d_input": sae.d_input,
        "d_dict": sae.d_dict,
        "metadata": metadata,
    }
    torch.save(checkpoint, path)
    _stderr(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path, device: str = "cuda") -> tuple[BaseSAE, dict]:
    """Load an SAE from a checkpoint file.

    Returns:
        (sae, metadata)  — sae is ready for inference (eval mode).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    arch_name = checkpoint["architecture"]
    d_input = checkpoint["d_input"]
    d_dict = checkpoint["d_dict"]
    metadata = checkpoint.get("metadata", {})

    arch_cls = next(
        (cls for cls in ARCHITECTURES.values() if cls.__name__ == arch_name), None
    )
    if arch_cls is None:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Available: {[c.__name__ for c in ARCHITECTURES.values()]}"
        )

    constructor_kwargs = metadata.get("constructor_kwargs", {})
    sae = arch_cls(d_input=d_input, d_dict=d_dict, device=device, **constructor_kwargs)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()

    return sae, metadata
