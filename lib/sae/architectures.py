"""SAE architecture definitions.

All six supported variants share BaseSAE and are registered in ARCHITECTURES.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseSAE(nn.Module):
    """Base class for all SAE variants.

    Shared parameters:
        W_enc  : (d_input, d_dict)  encoder weight
        b_enc  : (d_dict,)          encoder bias
        W_dec  : (d_dict, d_input)  decoder weight
        b_dec  : (d_input,)         pre-encoder centering bias / decoder bias
    """

    def __init__(self, d_input: int, d_dict: int, device: str = "cuda"):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.device_str = device

        # Shared decoder bias (pre-encoder centering)
        self.b_dec = nn.Parameter(torch.zeros(d_input, device=device))

        # Encoder: d_input -> d_dict
        self.W_enc = nn.Parameter(torch.empty(d_input, d_dict, device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_dict, device=device))

        # Decoder: d_dict -> d_input
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_input, device=device))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        # Decoder columns start at unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """x_hat = h @ W_dec + b_dec"""
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        x_hat = self.decode(h)
        return {"x_hat": x_hat, "h": h, "x": x}

    def compute_loss(self, result: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm (call after every optimizer step)."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


# ---------------------------------------------------------------------------
# 1. Vanilla SAE
# ---------------------------------------------------------------------------


class VanillaSAE(BaseSAE):
    """Standard SAE with ReLU activation and L1 sparsity penalty.

    Paper: Cunningham et al. 2023 (sae-interpretable-features)

    Equations:
        h     = ReLU(W_enc (x - b_dec) + b_enc)
        x_hat = W_dec h + b_dec
        L     = ||x - x_hat||^2 + alpha * ||h||_1
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        l1_weight: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__(d_input, d_dict, device)
        self.l1_weight = l1_weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat, h = result["x"], result["x_hat"], result["h"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()
        l_sparsity = h.abs().sum(dim=-1).mean()
        loss = l_reconstruct + self.l1_weight * l_sparsity
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l_sparsity": l_sparsity.detach(),
        }


# ---------------------------------------------------------------------------
# 2. TopK SAE
# ---------------------------------------------------------------------------


class TopKSAE(BaseSAE):
    """TopK SAE — keep only the K largest activations per input.

    Paper: Gao et al. 2024 (scaling-evaluating-sae)

    Equations:
        z = W_enc (x - b_dec) + b_enc
        h = TopK(ReLU(z), k)          # zeros all but K largest post-relu values
        L = ||x - x_hat||^2 + lambda_aux * L_aux
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        k: int = 64,
        aux_loss_weight: float = 1e-2,
        device: str = "cuda",
    ):
        super().__init__(d_input, d_dict, device)
        self.k = k
        self.aux_loss_weight = aux_loss_weight
        self._pre_act: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Apply ReLU first (matches TopK(ReLU(z), k) in the paper)
        pre_act = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        self._pre_act = pre_act  # stored for aux loss on dead features
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        h = torch.zeros_like(pre_act)
        h.scatter_(-1, topk_idx, topk_vals)
        return h

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat, h = result["x"], result["x_hat"], result["h"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Aux loss: reconstruct residual using dead features (Gao et al. 2024).
        # Previous version counted dead features — a step function with zero
        # gradient everywhere, so dead features could never revive.
        alive_mask = (h > 0).any(dim=0)  # (d_dict,) True if feature fired
        num_dead = int((~alive_mask).sum().item())

        if num_dead > 0 and self._pre_act is not None:
            # Mask pre-activations to dead features only
            dead_pre_act = self._pre_act * (~alive_mask).float().unsqueeze(0)

            # TopK among dead features to select which ones get gradient
            k_aux = min(self.k, num_dead)
            topk_vals, topk_idx = torch.topk(dead_pre_act, k_aux, dim=-1)
            h_dead = torch.zeros_like(dead_pre_act)
            h_dead.scatter_(-1, topk_idx, topk_vals)

            # Dead features reconstruct the residual (no b_dec — already in x_hat)
            residual = x - x_hat.detach()
            x_hat_dead = h_dead @ self.W_dec
            l_aux = (residual - x_hat_dead).pow(2).sum(dim=-1).mean()
        else:
            l_aux = torch.tensor(0.0, device=x.device)

        self._pre_act = None  # free memory

        loss = l_reconstruct + self.aux_loss_weight * l_aux
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l_aux": l_aux.detach(),
        }


# ---------------------------------------------------------------------------
# 3. BatchTopK SAE
# ---------------------------------------------------------------------------


class BatchTopKSAE(BaseSAE):
    """BatchTopK SAE — batch-level sparsity constraint.

    Paper: Bussmann et al. 2024 (batch-topk)

    Training:  total active latents = B * k  (distributed across whole batch)
    Inference: per-feature JumpReLU thresholds calibrated after training via
               calibrate_thresholds().

    Why board games: positions vary dramatically in complexity (opening vs.
    endgame), so per-sample sparsity varies naturally — BatchTopK captures this.
    """

    def __init__(self, d_input: int, d_dict: int, k: int = 64, device: str = "cuda"):
        super().__init__(d_input, d_dict, device)
        self.k = k
        # Per-feature JumpReLU thresholds; calibrated post-training
        self.register_buffer("_threshold_estimate", torch.zeros(d_dict, device=device))
        self._thresholds_calibrated = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = (x - self.b_dec) @ self.W_enc + self.b_enc
        B = pre_act.shape[0]

        if self.training:
            # Batch-level TopK: keep top B*k activations across the entire batch
            flat = pre_act.reshape(-1)
            total_k = min(B * self.k, flat.numel())
            topk_vals, topk_idx = torch.topk(flat, total_k)
            mask = torch.zeros_like(flat)
            mask.scatter_(0, topk_idx, 1.0)
            h = F.relu(pre_act) * mask.reshape_as(pre_act)
        else:
            # Inference: per-feature JumpReLU with calibrated thresholds
            threshold: torch.Tensor = self._threshold_estimate  # type: ignore[assignment]
            h = F.relu(pre_act) * (pre_act > threshold).float()

        return h

    def calibrate_thresholds(self, data: torch.Tensor, batch_size: int = 2048):
        """Estimate per-feature JumpReLU inference thresholds from training data.

        Runs a full forward pass in training mode to collect pre-activations,
        then for each feature sets the threshold at the quantile corresponding
        to the training sparsity target (k / d_dict active fraction per feature).

        Call this once immediately after the training loop finishes.
        """
        was_training = self.training
        self.train()

        all_pre_acts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                pre_act = (batch - self.b_dec) @ self.W_enc + self.b_enc
                all_pre_acts.append(F.relu(pre_act))

        pre_acts = torch.cat(all_pre_acts, dim=0)  # (N, d_dict)

        # Target active fraction per feature: k / d_dict
        # Threshold = (1 - target_frac) quantile of the relu'd activations
        target_active_frac = self.k / self.d_dict
        quantile_level = float(max(0.0, min(1.0 - target_active_frac, 1.0)))
        thresholds = torch.quantile(pre_acts, quantile_level, dim=0)  # (d_dict,)

        self._threshold_estimate.copy_(thresholds)
        self._thresholds_calibrated = True

        if not was_training:
            self.eval()

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat = result["x"], result["x_hat"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()
        return {
            "loss": l_reconstruct,
            "l_reconstruct": l_reconstruct.detach(),
        }


# ---------------------------------------------------------------------------
# 4. Gated SAE
# ---------------------------------------------------------------------------


class GatedSAE(BaseSAE):
    """Gated SAE — separate gate and magnitude paths.

    Paper: Rajamanoharan et al. 2024a (gated-sae)

    Equations:
        gate_pre = W_gate (x - b_dec) + b_gate
        g        = 1[gate_pre > 0]           (which features fire)
        m        = ReLU(W_enc (x - b_dec) + b_enc)  (how much)
        h        = g ⊙ m
        L        = ||x - x_hat||^2 + alpha * ||ReLU(gate_pre)||_1

    L1 on gate pre-activations eliminates shrinkage on h.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        l1_weight: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__(d_input, d_dict, device)
        self.l1_weight = l1_weight
        self.W_gate = nn.Parameter(torch.empty(d_input, d_dict, device=device))
        self.b_gate = nn.Parameter(torch.zeros(d_dict, device=device))
        nn.init.kaiming_uniform_(self.W_gate)
        self._gate_pre: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.b_dec

        # Gate path (which features fire)
        gate_pre = centered @ self.W_gate + self.b_gate
        self._gate_pre = gate_pre
        gate = (gate_pre > 0).float()

        # Magnitude path (how much each fires)
        mag = F.relu(centered @ self.W_enc + self.b_enc)

        return gate * mag

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat = result["x"], result["x_hat"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()

        # L1 on gate pre-activations (not on h — avoids shrinkage)
        assert self._gate_pre is not None
        l_sparsity = F.relu(self._gate_pre).sum(dim=-1).mean()

        loss = l_reconstruct + self.l1_weight * l_sparsity
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l_sparsity": l_sparsity.detach(),
        }


# ---------------------------------------------------------------------------
# 5. JumpReLU SAE
# ---------------------------------------------------------------------------


class JumpReLUSAE(BaseSAE):
    """JumpReLU SAE — per-feature learnable thresholds.

    Paper: Rajamanoharan et al. 2024b (jumprelu-sae)

    Equations:
        z    = W_enc (x - b_dec) + b_enc
        h_i  = z_i * 1[z_i > theta_i]       (JumpReLU, theta_i learnable)
        L    = ||x - x_hat||^2 + lambda * (L0 - L0_target)^2

    Gradients through the indicator use a sigmoid STE with bandwidth epsilon.
    Thresholds are parameterised in log-space to enforce theta_i > 0.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        theta_init: float = 0.001,
        bandwidth: float = 0.001,
        l0_target: float = 50.0,
        l0_weight: float = 1e-2,
        device: str = "cuda",
    ):
        super().__init__(d_input, d_dict, device)
        self.bandwidth = bandwidth
        self.l0_target = l0_target
        self.l0_weight = l0_weight

        # Log-space parameterisation keeps theta_i > 0 at all times
        self.log_theta = nn.Parameter(
            torch.full((d_dict,), float(np.log(theta_init)), device=device)
        )

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        theta = self.theta

        if self.training:
            # Straight-Through Estimator: forward = hard threshold, backward = sigmoid
            mask_hard = (z > theta).float()
            mask_soft = torch.sigmoid((z - theta) / self.bandwidth)
            mask = mask_hard + (mask_soft - mask_soft.detach())
        else:
            mask = (z > theta).float()

        return z * mask

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat, h = result["x"], result["x_hat"], result["h"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()
        l0 = (h > 0).float().sum(dim=-1).mean()
        l_sparsity = (l0 - self.l0_target).pow(2)
        loss = l_reconstruct + self.l0_weight * l_sparsity
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l0": l0.detach(),
        }


# ---------------------------------------------------------------------------
# 6. p-Annealing SAE
# ---------------------------------------------------------------------------


class PAnnealingSAE(BaseSAE):
    """p-Annealing SAE — Lp penalty with p annealed from 1 toward 0.

    Equations:
        h = ReLU(W_enc (x - b_dec) + b_enc)
        L = ||x - x_hat||^2 + w * sum_i (|h_i| + eps)^p

    p is linearly annealed from p_start (≈L1) down to p_end (≈L0) over
    p_anneal_steps training steps.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        p_start: float = 1.0,
        p_end: float = 0.2,
        p_anneal_steps: int = 15000,
        lp_weight: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__(d_input, d_dict, device)
        self.p_start = p_start
        self.p_end = p_end
        self.p_anneal_steps = p_anneal_steps
        self.lp_weight = lp_weight
        self._step = 0

    @property
    def current_p(self) -> float:
        t = min(self._step / max(self.p_anneal_steps, 1), 1.0)
        return self.p_start + (self.p_end - self.p_start) * t

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat, h = result["x"], result["x_hat"], result["h"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()
        p = self.current_p
        l_sparsity = (h.abs() + 1e-8).pow(p).sum(dim=-1).mean()
        self._step += 1
        loss = l_reconstruct + self.lp_weight * l_sparsity
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l_sparsity": l_sparsity.detach(),
            "current_p": torch.tensor(p),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURES: dict[str, type[BaseSAE]] = {
    "vanilla": VanillaSAE,
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
    "gated": GatedSAE,
    "jumprelu": JumpReLUSAE,
    "p-annealing": PAnnealingSAE,
}
