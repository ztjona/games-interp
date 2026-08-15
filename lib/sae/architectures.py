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

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        device: str = "cuda",
        aux_loss_weight: float = 0.0,
        aux_k: int | None = None,
        dead_window: int = 0,
        init_mode: str = "kaiming",
    ):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.device_str = device
        self.init_mode = init_mode

        # Dead-feature revival (Gao et al. 2024). Shared by every variant that
        # opts in, so a fix lands in all of them at once -- the previous
        # per-class implementation existed only on TopKSAE, which silently made
        # every architecture comparison partly a comparison of training
        # machinery. See docs/diary/2026-08-15_dead-feature-revival.md.
        self.aux_loss_weight = aux_loss_weight
        self.aux_k = aux_k
        self.dead_window = dead_window
        self._pre_act: torch.Tensor | None = None
        # Non-persistent: a training-time statistic, deliberately kept out of
        # state_dict so it cannot break `load_state_dict` on any existing
        # checkpoint (which is strict by default).
        self.register_buffer(
            "_steps_since_fired",
            torch.zeros(d_dict, dtype=torch.long, device=device),
            persistent=False,
        )

        # Shared decoder bias (pre-encoder centering)
        self.b_dec = nn.Parameter(torch.zeros(d_input, device=device))

        # Encoder: d_input -> d_dict
        self.W_enc = nn.Parameter(torch.empty(d_input, d_dict, device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_dict, device=device))

        # Decoder: d_dict -> d_input
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_input, device=device))

        # Initialisation. Gao et al. 2024 list TWO dead-latent mitigations and
        # this is the FIRST of them: "initialize W_enc = W_dec^T". It was absent
        # from this file entirely until 2026-08-15, for every architecture
        # including TopK -- which is the most likely reason our dictionaries sit
        # at 61-99% dead where Gao report 7% at 16M latents.
        #
        #   "kaiming"            -- W_enc and W_dec drawn independently. The
        #                           pre-2026-08-15 behaviour; every config
        #                           written before that date pins it explicitly
        #                           so those runs still reproduce.
        #   "decoder_transpose"  -- Gao et al.: unit-norm the decoder, then set
        #                           the encoder to its transpose, so every atom
        #                           starts as its own detector and no column
        #                           begins in the zero-gradient trap.
        if init_mode not in ("kaiming", "decoder_transpose"):
            raise ValueError(
                f"init_mode must be 'kaiming' or 'decoder_transpose', got {init_mode!r}"
            )
        # NOTE ON RNG ORDER: both draws consume the global generator, so the
        # ORDER of the two kaiming calls changes the values. The legacy path
        # must keep W_enc first, exactly as before, or seeded reruns of banked
        # configs would not reproduce.
        if init_mode == "kaiming":
            nn.init.kaiming_uniform_(self.W_enc)
            nn.init.kaiming_uniform_(self.W_dec)
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
        else:  # decoder_transpose
            nn.init.kaiming_uniform_(self.W_dec)
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
                self.W_enc.data = self.W_dec.data.T.contiguous().clone()

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

    # -- dead-feature revival ------------------------------------------------

    def _update_dead_mask(self, h: torch.Tensor) -> torch.Tensor:
        """(d_dict,) bool mask of features counted as dead, and tick the counter.

        ``dead_window = 0`` reproduces per-batch aliveness exactly (a feature is
        dead iff it did not fire in THIS batch) -- the semantics TopKSAE shipped
        with, kept as the default so existing runs stay reproducible. A positive
        window is the Gao et al. definition: dead iff it has not fired in the
        last ``dead_window`` steps. Per-batch aliveness over-counts, because at
        batch 4096 with k = 32 only B*k of B*d_dict slots can fire at all, so a
        genuinely healthy but rare feature reads as dead in most batches.
        """
        fired = (h > 0).any(dim=0)
        if self.training:
            self._steps_since_fired += 1
            self._steps_since_fired[fired] = 0
        return self._steps_since_fired > self.dead_window

    def _aux_dead_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, h: torch.Tensor, k_aux: int
    ) -> torch.Tensor:
        """Gao et al. 2024 auxiliary loss: dead features reconstruct the residual.

        The top ``k_aux`` dead features by pre-activation are decoded and asked
        to explain what the live reconstruction missed. ``x_hat`` is detached so
        this term only ever trains the dead columns -- it cannot degrade the
        main reconstruction path.

        Returns a scalar zero when the term is switched off, when nothing is
        dead, or when the subclass did not stash ``_pre_act``.
        """
        if self.aux_loss_weight == 0.0 or self._pre_act is None:
            self._pre_act = None
            return torch.zeros((), device=x.device)

        dead_mask = self._update_dead_mask(h)
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            self._pre_act = None
            return torch.zeros((), device=x.device)

        dead_pre_act = self._pre_act * dead_mask.float().unsqueeze(0)
        self._pre_act = None  # free memory

        k = min(int(k_aux), num_dead)
        topk_vals, topk_idx = torch.topk(dead_pre_act, k, dim=-1)
        h_dead = torch.zeros_like(dead_pre_act)
        h_dead.scatter_(-1, topk_idx, topk_vals)

        # No b_dec: it is already accounted for in x_hat.
        residual = x - x_hat.detach()
        x_hat_dead = h_dead @ self.W_dec
        return (residual - x_hat_dead).pow(2).sum(dim=-1).mean()


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
        aux_k: int | None = None,
        dead_window: int = 0,
        init_mode: str = "decoder_transpose",
        device: str = "cuda",
    ):
        # Canonical Gao et al. 2024 defaults: aux loss ON, W_enc = W_dec^T.
        super().__init__(
            d_input, d_dict, device,
            aux_loss_weight=aux_loss_weight, aux_k=aux_k, dead_window=dead_window,
            init_mode=init_mode,
        )
        self.k = k

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

        # Dead-feature revival, now shared with BatchTopK and JumpReLU. The
        # default k_aux = k and dead_window = 0 reproduce this class's original
        # behaviour exactly, so banked TopK results stay reproducible.
        l_aux = self._aux_dead_loss(x, x_hat, h, self.aux_k or self.k)

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

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        k: int = 64,
        aux_loss_weight: float = 1e-2,
        aux_k: int | None = None,
        dead_window: int = 0,
        init_mode: str = "decoder_transpose",
        device: str = "cuda",
    ):
        # Canonical Bussmann et al. 2024 defaults. The paper states the
        # auxiliary loss is "retained" from TopK, so aux ON is the published
        # form -- it was absent here entirely until 2026-08-15.
        super().__init__(
            d_input, d_dict, device,
            aux_loss_weight=aux_loss_weight, aux_k=aux_k, dead_window=dead_window,
            init_mode=init_mode,
        )
        self.k = k
        # Per-feature JumpReLU thresholds; calibrated post-training
        self.register_buffer("_threshold_estimate", torch.zeros(d_dict, device=device))
        self._thresholds_calibrated = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = (x - self.b_dec) @ self.W_enc + self.b_enc
        B = pre_act.shape[0]

        if self.training:
            # Batch-level TopK: keep top B*k activations across the entire batch
            self._pre_act = F.relu(pre_act)  # for the dead-feature aux loss
            flat = pre_act.reshape(-1)
            total_k = min(B * self.k, flat.numel())
            topk_vals, topk_idx = torch.topk(flat, total_k)
            mask = torch.zeros_like(flat)
            mask.scatter_(0, topk_idx, 1.0)
            h = F.relu(pre_act) * mask.reshape_as(pre_act)
        else:
            self._pre_act = None
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

        # Concatenate on CPU to avoid CUDA OOM on large datasets (N * d_dict can be ~1 GiB)
        pre_acts = torch.cat([t.cpu() for t in all_pre_acts], dim=0)  # (N, d_dict)

        # Target active fraction per feature: k / d_dict
        # Threshold = (1 - target_frac) quantile of the relu'd activations
        target_active_frac = self.k / self.d_dict
        quantile_level = float(max(0.0, min(1.0 - target_active_frac, 1.0)))
        thresholds = torch.quantile(pre_acts, quantile_level, dim=0).to(
            self._threshold_estimate.device
        )  # (d_dict,)

        self._threshold_estimate.copy_(thresholds)
        self._thresholds_calibrated = True

        if not was_training:
            self.eval()

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat, h = result["x"], result["x_hat"], result["h"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead-feature revival. OFF by default (aux_loss_weight = 0.0) so every
        # banked BatchTopK run reproduces bit-for-bit; enable per config.
        l_aux = self._aux_dead_loss(x, x_hat, h, self.aux_k or self.k)

        loss = l_reconstruct + self.aux_loss_weight * l_aux
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l_aux": l_aux.detach(),
        }


# ---------------------------------------------------------------------------
# 4. Gated SAE
# ---------------------------------------------------------------------------


class GatedSAE(BaseSAE):
    """Gated SAE — separate gate and magnitude paths.

    Paper: Rajamanoharan et al. 2024a (gated-sae)

    Equations:
        gate_pre  = W_gate (x - b_dec) + b_gate
        g         = 1[gate_pre > 0]                  (hard gate, which features fire)
        m         = ReLU(W_enc (x - b_dec) + b_enc)  (magnitude, how much)
        h         = g ⊙ m
        x_hat_via = ReLU(gate_pre) @ W_dec + b_dec   (auxiliary, soft-gate reconstruction)
        L         = ||x - x_hat||^2
                  + alpha * ||ReLU(gate_pre)||_1      (sparsity on gate)
                  + ||x - x_hat_via||^2              (via-gate: routes recon grad to W_gate)

    The hard gate g = 1[gate_pre > 0] is a step function with zero gradient,
    so W_gate would otherwise receive NO signal from the reconstruction loss —
    only the L1 penalty, which pushes all gate values toward zero (total
    collapse). The via-gate auxiliary loss fixes this by decoding ReLU(gate_pre)
    through a frozen W_dec, routing reconstruction gradients back to W_gate.
    W_dec.detach() ensures the auxiliary loss trains only gate parameters.
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

        # L1 on gate pre-activations (anti-shrinkage)
        assert self._gate_pre is not None
        gate_pre = self._gate_pre
        self._gate_pre = None  # free memory
        l_sparsity = F.relu(gate_pre).sum(dim=-1).mean()

        # Via-gate auxiliary loss: the only path for reconstruction gradients
        # to reach W_gate. Uses ReLU(gate_pre) as a soft, differentiable gate
        # decoded through frozen W_dec (detached so only W_gate is updated here).
        x_hat_via_gate = F.relu(gate_pre) @ self.W_dec.detach() + self.b_dec.detach()
        l_aux = (x - x_hat_via_gate).pow(2).sum(dim=-1).mean()

        loss = l_reconstruct + self.l1_weight * l_sparsity + l_aux
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l_sparsity": l_sparsity.detach(),
            "l_aux": l_aux.detach(),
        }


# ---------------------------------------------------------------------------
# 5. JumpReLU SAE
# ---------------------------------------------------------------------------


class JumpReLUSAE(BaseSAE):
    """JumpReLU SAE — per-feature learnable thresholds.

    Paper: Rajamanoharan et al. 2024b (jumprelu-sae)

    Equations:
        z         = W_enc (x - b_dec) + b_enc
        h_i       = z_i * 1[z_i > theta_i]            (JumpReLU, theta_i learnable)
        L0_approx = sum_i sigmoid((z_i - theta_i) / epsilon)   (differentiable L0)
        L         = ||x - x_hat||^2 + lambda * (L0_approx - L0_target)^2

    Two STE paths:
    1. Reconstruction loss → h via mask_hard + (mask_soft - mask_soft.detach())
       → trains both W_enc and theta toward better reconstruction.
    2. L0 penalty → L0_approx via sigmoid((z - theta) / epsilon)
       → trains theta to hit the sparsity target.

    Bug that was present: using (h > 0).float() for the penalty has zero
    gradient w.r.t. theta, so the L0 penalty was a no-op and theta drifted
    to near-zero (all features always active, L0 ≈ d_dict regardless of target).
    The fix stores z from encode() and uses sigmoid STE in compute_loss().

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
        aux_loss_weight: float = 0.0,
        aux_k: int | None = None,
        dead_window: int = 0,
        init_mode: str = "decoder_transpose",
        device: str = "cuda",
    ):
        # Rajamanoharan et al. 2024b: "No auxiliary losses, no resampling" --
        # aux OFF is the PUBLISHED form, not an omission, and the 2026-08-15 A/B
        # measured adding one as null (+/-0.013). It stays off.
        #
        # The init is different: the paper does not specify one, so this axis is
        # UNDERDETERMINED rather than settled, and our previous independent
        # kaiming draw was an arbitrary choice rather than the published form.
        # K05 measured the alternative at +6-8% coverage on all three BSP sets
        # (gorillaYb 0.424 -> 0.459) with 52% more live latents, so the tied
        # init is now the default -- a documented CHOICE on an axis the source
        # leaves open, not a deviation from it.
        super().__init__(
            d_input, d_dict, device,
            aux_loss_weight=aux_loss_weight, aux_k=aux_k, dead_window=dead_window,
            init_mode=init_mode,
        )
        self.bandwidth = bandwidth
        self.l0_target = l0_target
        self.l0_weight = l0_weight

        # Log-space parameterisation keeps theta_i > 0 at all times
        self.log_theta = nn.Parameter(
            torch.full((d_dict,), float(np.log(theta_init)), device=device)
        )
        self._z: torch.Tensor | None = (
            None  # stored during training for differentiable L0 penalty
        )

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        theta = self.theta

        if self.training:
            # STE path 1 (reconstruction): forward = hard threshold, backward = sigmoid
            mask_hard = (z > theta).float()
            mask_soft = torch.sigmoid((z - theta) / self.bandwidth)
            mask = mask_hard + (mask_soft - mask_soft.detach())
            self._z = z  # store for differentiable L0 penalty in compute_loss
            # ReLU(z), not z: the aux path asks dead columns to reconstruct a
            # residual, which needs non-negative magnitudes (same convention as
            # TopKSAE). z itself is signed.
            self._pre_act = F.relu(z)
        else:
            self._z = None
            self._pre_act = None
            mask = (z > theta).float()

        return z * mask

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        x, x_hat, h = result["x"], result["x_hat"], result["h"]
        l_reconstruct = (x - x_hat).pow(2).sum(dim=-1).mean()

        # True L0 for logging (step function — no gradient)
        l0_true = (h > 0).float().sum(dim=-1).mean()

        # STE path 2 (sparsity): differentiable L0 approximation via sigmoid.
        # (h > 0).float() has zero gradient w.r.t. theta — that was the bug.
        # sigmoid((z - theta) / epsilon) is the kernel estimator from the paper.
        if self.training and self._z is not None:
            l0_soft = (
                torch.sigmoid((self._z - self.theta) / self.bandwidth)
                .sum(dim=-1)
                .mean()
            )
            l_sparsity = (l0_soft - self.l0_target).pow(2)
            self._z = None  # free memory
        else:
            l_sparsity = torch.tensor(0.0, device=x.device)

        # Dead-feature revival. OFF by default (aux_loss_weight = 0.0) so every
        # banked JumpReLU run reproduces bit-for-bit; enable per config.
        # k_aux defaults to the L0 target, this variant's analogue of TopK's k.
        l_aux = self._aux_dead_loss(x, x_hat, h, self.aux_k or int(self.l0_target))

        loss = l_reconstruct + self.l0_weight * l_sparsity + self.aux_loss_weight * l_aux
        return {
            "loss": loss,
            "l_reconstruct": l_reconstruct.detach(),
            "l0": l0_true.detach(),  # true L0 for monitoring
            "l_aux": l_aux.detach(),
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
# 7. Anchored SAEs (supervised BCE penalty on a subset of features)
# ---------------------------------------------------------------------------


class _AnchorMixin:
    """Mixin that adds a per-batch BCE-on-pre-activation anchor loss.

    A pre-specified subset of dictionary slots ("anchored features") is
    trained to fire in alignment with a corresponding subset of binary BSP
    labels.  Loss form: BCE_with_logits on the encoder pre-activation
    ``z = (x - b_dec) @ W_enc + b_enc`` for each anchored slot vs its bound
    BSP label.  This routes a smooth gradient through W_enc / b_enc so the
    anchored columns behave like a tiny logistic probe embedded in the SAE.

    The mapping is fixed at construction time:
        anchor_feature_idx[k]  →  dictionary slot anchored by anchor k
        anchor_bsp_idx[k]      →  column of the label tensor used as target
        anchor_lambda_per_feature[k] → per-anchor weight

    Per-batch labels are injected by the training loop via
    :meth:`set_batch_labels` immediately before ``sae(batch)``; the labels
    are consumed (cleared to ``None``) during ``compute_loss``.  When no
    labels are set (e.g. during eval), the anchor term is 0.
    """

    def _init_anchor(
        self,
        anchor_feature_idx,
        anchor_bsp_idx,
        anchor_lambda_per_feature,
        device: str,
    ) -> None:
        feat = torch.as_tensor(anchor_feature_idx, dtype=torch.long, device=device)
        bsp = torch.as_tensor(anchor_bsp_idx, dtype=torch.long, device=device)
        lam = torch.as_tensor(
            anchor_lambda_per_feature, dtype=torch.float32, device=device
        )
        if not (feat.shape == bsp.shape == lam.shape):
            raise ValueError(
                "anchor_feature_idx, anchor_bsp_idx, anchor_lambda_per_feature "
                f"must have matching shape; got {feat.shape}, {bsp.shape}, {lam.shape}"
            )
        # Buffers (not parameters): travel with state_dict / device moves.
        self.register_buffer("anchor_feature_idx", feat)
        self.register_buffer("anchor_bsp_idx", bsp)
        self.register_buffer("anchor_lambda_per_feature", lam)
        self._batch_labels: torch.Tensor | None = None

    def set_batch_labels(self, labels: torch.Tensor | None) -> None:
        """Stash the current batch's anchor labels (full label matrix slice).

        ``labels`` shape: (B, num_anchor_columns); the column for each anchor
        is selected internally via ``anchor_bsp_idx``.
        """
        self._batch_labels = labels

    def _anchor_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self._batch_labels is None or self.anchor_lambda_per_feature.numel() == 0:
            return torch.zeros((), device=x.device)
        # Recompute z to keep the autograd path clean and not piggyback on
        # any architecture-specific stash (which may be cleared, masked, or
        # absent depending on the variant).
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        z_anchor = z[:, self.anchor_feature_idx]  # (B, num_anchored)
        y = self._batch_labels.to(device=z_anchor.device, dtype=z_anchor.dtype)
        y_anchor = y[:, self.anchor_bsp_idx]  # (B, num_anchored)
        bce = F.binary_cross_entropy_with_logits(
            z_anchor, y_anchor, reduction="none"
        )  # (B, num_anchored)
        weighted = (bce * self.anchor_lambda_per_feature.unsqueeze(0)).sum(dim=-1).mean()
        # Consume labels so a missing set_batch_labels() in the next call
        # raises the explicit "no labels" path rather than silently reusing.
        self._batch_labels = None
        return weighted


class AnchoredJumpReLUSAE(_AnchorMixin, JumpReLUSAE):
    """JumpReLU SAE with a supervised BCE anchor loss on a subset of features."""

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        anchor_feature_idx,
        anchor_bsp_idx,
        anchor_lambda_per_feature,
        theta_init: float = 0.001,
        bandwidth: float = 0.001,
        l0_target: float = 50.0,
        l0_weight: float = 1e-2,
        aux_loss_weight: float = 0.0,
        aux_k: int | None = None,
        dead_window: int = 0,
        # Must track JumpReLUSAE's default, or the anchored variant silently
        # trains a differently-initialised dictionary than its own base class.
        init_mode: str = "decoder_transpose",
        device: str = "cuda",
    ):
        # Keyword form deliberately: this call was positional through `device`,
        # so adding a base-class parameter before it silently placed `device`
        # into another slot and every tensor landed on the default device.
        JumpReLUSAE.__init__(
            self, d_input=d_input, d_dict=d_dict, theta_init=theta_init,
            bandwidth=bandwidth, l0_target=l0_target, l0_weight=l0_weight,
            aux_loss_weight=aux_loss_weight, aux_k=aux_k, dead_window=dead_window,
            init_mode=init_mode, device=device,
        )
        self._init_anchor(
            anchor_feature_idx, anchor_bsp_idx, anchor_lambda_per_feature, device
        )

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        losses = super().compute_loss(result)
        l_anchor = self._anchor_loss(result["x"])
        losses["loss"] = losses["loss"] + l_anchor
        losses["l_anchor"] = l_anchor.detach()
        return losses


class AnchoredBatchTopKSAE(_AnchorMixin, BatchTopKSAE):
    """BatchTopK SAE with a supervised BCE anchor loss on a subset of features."""

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        anchor_feature_idx,
        anchor_bsp_idx,
        anchor_lambda_per_feature,
        k: int = 64,
        aux_loss_weight: float = 1e-2,
        aux_k: int | None = None,
        dead_window: int = 0,
        init_mode: str = "decoder_transpose",
        device: str = "cuda",
    ):
        # Keyword form: see the note in AnchoredJumpReLUSAE.
        BatchTopKSAE.__init__(
            self, d_input=d_input, d_dict=d_dict, k=k,
            aux_loss_weight=aux_loss_weight, aux_k=aux_k, dead_window=dead_window,
            init_mode=init_mode, device=device,
        )
        self._init_anchor(
            anchor_feature_idx, anchor_bsp_idx, anchor_lambda_per_feature, device
        )

    def compute_loss(self, result: dict) -> dict[str, torch.Tensor]:
        losses = super().compute_loss(result)
        l_anchor = self._anchor_loss(result["x"])
        losses["loss"] = losses["loss"] + l_anchor
        losses["l_anchor"] = l_anchor.detach()
        return losses


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
    "anchored-jumprelu": AnchoredJumpReLUSAE,
    "anchored-batchtopk": AnchoredBatchTopKSAE,
}
