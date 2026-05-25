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

    Does NOT change the model's training/eval mode — caller is responsible.

    Returns:
        l0:                 average active features per sample
        l0_std:             std of active features per sample (plot as gray band)
        fvu:                fraction of variance unexplained
        mse:                mean squared reconstruction error
        dead_features_pct:  % of features inactive across the whole batch
        median_feat_freq:   median per-feature firing rate across the batch
                            (complements dead%: tracks typical feature utilisation)
    """
    with torch.no_grad():
        result = sae(x)
        x_hat, h = result["x_hat"], result["h"]

        active_per_sample = (h > 0).float().sum(dim=-1)
        l0 = active_per_sample.mean().item()
        l0_std = active_per_sample.std().item()

        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        var_x = x.var(dim=0).sum().item()
        fvu = mse / max(var_x, 1e-8)

        firing_rates = (h > 0).float().mean(dim=0)  # (d_dict,)
        alive = (firing_rates > 0).float()
        dead_pct = (1.0 - alive.mean().item()) * 100.0
        median_feat_freq = firing_rates.median().item()

    return {
        "l0": round(l0, 2),
        "l0_std": round(l0_std, 2),
        "fvu": round(fvu, 6),
        "mse": round(mse, 6),
        "dead_features_pct": round(dead_pct, 2),
        "median_feat_freq": round(median_feat_freq, 6),
    }


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_activation_data(data_path: str, device: str) -> torch.Tensor:
    """Load pre-collected activation tensor from a .pt file.

    Expected shape: (N, d_activation).  Conv layers must be pre-flattened
    at collection time (use --flatten-position in collect_activations.py).
    """
    _stderr(f"Loading activation data from {data_path}...")
    data = torch.load(data_path, map_location=device, weights_only=True)
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(data)}")
    if data.ndim != 2:
        raise ValueError(
            f"Activation tensor must be 2D (N, d_act), got shape {tuple(data.shape)}. "
            "Re-collect with --flatten-position for conv layers."
        )
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


def _shuffled_batches(
    data: torch.Tensor,
    batch_size: int,
    anchor_labels: torch.Tensor | None,
):
    """Yield (batch_x, batch_anchor_labels_or_None) using a fresh random perm.

    Used in place of :func:`iter_batches` when anchored SAEs require per-batch
    label slices aligned with the shuffled activations.
    """
    N = data.shape[0]
    perm = torch.randperm(N, device=data.device)
    for i in range(0, N, batch_size):
        idx = perm[i : i + batch_size]
        batch = data[idx]
        if anchor_labels is None:
            yield batch, None
        else:
            yield batch, anchor_labels[idx.to(anchor_labels.device)]


def train_sae(
    sae: BaseSAE,
    data: torch.Tensor,
    num_batches: int,
    batch_size: int,
    lr: float,
    log_every: int = 500,
    seed: int = 42,
    metrics_file: Path | str | None = None,
    patience: int = 0,
    min_improvement: float = 0.01,
    l0_patience: int = 0,
    l0_min_change: float = 5.0,
    anchor_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Train an SAE on pre-collected activation data.

    For BatchTopKSAE, automatically calibrates inference thresholds after
    the training loop using the same data.

    Args:
        metrics_file: Optional path to write metrics as JSONL for live monitoring
        patience: Early-stop after this many eval windows with no FVU improvement.
                  0 (default) disables early stopping.
        min_improvement: Minimum relative FVU improvement to reset patience counter.
                         E.g. 0.01 means FVU must drop by at least 1% relative.
        l0_patience: Early-stop if L0 changes by less than l0_min_change for this
                     many consecutive eval windows.  0 (default) disables.  Do NOT
                     enable for TopK/BatchTopK — their L0 is always exactly k.
        l0_min_change: Minimum absolute L0 change per window to reset the L0
                       plateau counter.  Default 5.0 units.
        anchor_labels: Optional (N, num_anchor_cols) tensor aligned with ``data``
                       row order.  When provided, the trainer slices the matching
                       rows for each batch and pushes them onto ``sae`` via
                       ``sae.set_batch_labels()`` before the forward pass.  Used
                       by ``AnchoredJumpReLUSAE`` / ``AnchoredBatchTopKSAE``.

    Returns:
        dict with keys: final_step, num_epochs, wall_time_seconds,
                        final_metrics, metrics_log, early_stopped
    """
    if anchor_labels is not None:
        if anchor_labels.shape[0] != data.shape[0]:
            raise ValueError(
                f"anchor_labels row count ({anchor_labels.shape[0]}) must match "
                f"data row count ({data.shape[0]})"
            )
        if not hasattr(sae, "set_batch_labels"):
            raise ValueError(
                f"anchor_labels provided but {type(sae).__name__} has no "
                "set_batch_labels() method; use an Anchored* architecture."
            )
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()

    metrics_log: list[dict] = []
    step = 0
    epoch = 0

    # Early stopping state — FVU-based
    best_fvu = float("inf")
    best_step = 0
    patience_counter = 0
    early_stopped = False
    # Early stopping state — L0 plateau
    prev_l0: float | None = None
    l0_patience_counter = 0

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
    pbar = tqdm(total=num_batches, desc="Training", unit="step", disable=None)

    stop_reason: str | None = None
    try:
        while step < num_batches:
            epoch += 1
            for batch, batch_labels in _shuffled_batches(
                data, batch_size, anchor_labels
            ):
                if step >= num_batches:
                    break

                if batch_labels is not None:
                    sae.set_batch_labels(batch_labels)  # type: ignore[attr-defined]

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
                    postfix = {
                        "loss": f"{log_entry['loss']:.4f}",
                        "L0": f"{eval_metrics['l0']:.1f}",
                        "FVU": f"{eval_metrics['fvu']:.4f}",
                        "dead": f"{eval_metrics['dead_features_pct']:.1f}%",
                    }

                    # Early stopping — FVU plateau
                    current_fvu = eval_metrics["fvu"]
                    if current_fvu < best_fvu * (1 - min_improvement):
                        best_fvu = current_fvu
                        best_step = step
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience > 0:
                        postfix["pat"] = f"{patience_counter}/{patience}"
                        if patience_counter >= patience:
                            early_stopped = True
                            _stderr(
                                f"Early stopping at step {step}: "
                                f"FVU {current_fvu:.6f} has not improved "
                                f"by >{min_improvement:.1%} over best "
                                f"{best_fvu:.6f} (step {best_step}) "
                                f"for {patience} eval windows."
                            )
                            pbar.set_postfix(postfix)
                            break

                    # Early stopping — L0 plateau
                    current_l0 = eval_metrics["l0"]
                    if prev_l0 is not None:
                        if abs(current_l0 - prev_l0) < l0_min_change:
                            l0_patience_counter += 1
                        else:
                            l0_patience_counter = 0
                        if l0_patience > 0:
                            postfix["l0pat"] = f"{l0_patience_counter}/{l0_patience}"
                            if l0_patience_counter >= l0_patience:
                                early_stopped = True
                                _stderr(
                                    f"Early stopping at step {step}: "
                                    f"L0 plateau — change {abs(current_l0 - prev_l0):.1f} "
                                    f"< {l0_min_change} for {l0_patience} eval windows "
                                    f"(current L0={current_l0:.1f})."
                                )
                                pbar.set_postfix(postfix)
                                break
                    prev_l0 = current_l0

                    pbar.set_postfix(postfix)

            if early_stopped:
                break

        stop_reason = "early_stopped" if early_stopped else "completed"
    finally:
        pbar.close()
        if metrics_fh:
            if stop_reason is not None:  # None means interrupted — write nothing
                metrics_fh.write(json.dumps({"stop_reason": stop_reason}) + "\n")
                metrics_fh.flush()
            metrics_fh.close()

    wall_time = time.time() - t_start
    _stderr(f"Training complete. {step} steps in {wall_time:.1f}s ({epoch} epochs)")

    # BatchTopK: calibrate per-feature JumpReLU thresholds for eval/inference
    if isinstance(sae, BatchTopKSAE):
        _stderr("Calibrating BatchTopK inference thresholds...")
        sae.calibrate_thresholds(data, batch_size=batch_size)
        _stderr("Threshold calibration complete.")

    # Final evaluation on a larger held-out sample
    # BatchTopK: evaluate in training mode (batch-level sparsity) since per-feature
    # inference thresholds produce incomparable metrics (different sparsity mechanism).
    with torch.no_grad():
        eval_sample = data[: min(len(data), batch_size * 4)]
        if isinstance(sae, BatchTopKSAE):
            sae.train()
        else:
            sae.eval()
        final_metrics = compute_metrics(sae, eval_sample)
        sae.train()

    return {
        "final_step": step,
        "num_epochs": epoch,
        "wall_time_seconds": round(wall_time, 2),
        "final_metrics": final_metrics,
        "metrics_log": metrics_log,
        "early_stopped": early_stopped,
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
