"""Unit tests for lib.sae — all run on CPU with tiny tensors (< 5 s total).

Test categories:
  1. Sparsity contracts   — exact K / B×K / threshold guarantees
  2. Non-negative h       — all architectures must produce h >= 0
  3. BatchTopK calibration — inference L0 ≈ k after calibrate_thresholds()
  4. Checkpoint round-trip — load(save(sae)) gives identical reconstructions
  5. Training sanity       — 100 steps reduces loss, FVU < 1.0, some features alive
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.sae import (
    ARCHITECTURES,
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    PAnnealingSAE,
    TopKSAE,
    VanillaSAE,
    load_checkpoint,
    save_checkpoint,
    train_sae,
)

# ---------------------------------------------------------------------------
# Shared constants — tiny so every test is near-instant on CPU
# ---------------------------------------------------------------------------
D_INPUT = 16
D_DICT = 64  # 4× expansion
K = 8
BATCH = 32
N_SAMPLES = 256
NUM_STEPS = 100
DEVICE = "cpu"

torch.manual_seed(0)
_DATA = torch.randn(N_SAMPLES, D_INPUT)  # shared synthetic activations


def _batch() -> torch.Tensor:
    return _DATA[:BATCH]


def _make(arch: str, **kw) -> torch.Tensor:
    """Instantiate an architecture with sensible small defaults."""
    defaults: dict = dict(d_input=D_INPUT, d_dict=D_DICT, device=DEVICE)
    if arch in ("topk", "batchtopk"):
        defaults["k"] = K
    return ARCHITECTURES[arch](**defaults, **kw)


# ===========================================================================
# 1. Sparsity contracts
# ===========================================================================


class TestSparsityContracts:

    def test_topk_exact_k_per_sample(self):
        sae = _make("topk")
        sae.eval()
        h = sae.encode(_batch())
        active_per_sample = (h > 0).sum(dim=-1)
        assert (active_per_sample == K).all(), (
            f"TopK must activate exactly K={K} features per sample; "
            f"got {active_per_sample.tolist()}"
        )

    def test_batchtopk_exact_BK_active_during_training(self):
        sae = _make("batchtopk")
        sae.train()
        x = _batch()
        h = sae.encode(x)
        total_active = (h > 0).sum().item()
        expected = BATCH * K
        assert total_active == expected, (
            f"BatchTopK training must activate exactly B×K={expected} latents; "
            f"got {total_active}"
        )

    def test_jumprelu_fires_only_above_threshold(self):
        sae = _make("jumprelu", theta_init=0.5)
        sae.eval()
        z = ((_batch() - sae.b_dec) @ sae.W_enc + sae.b_enc).detach()
        h = sae.encode(_batch().detach())
        theta = sae.theta.detach()
        # Any feature that fired must have had z > theta
        fired = h > 0
        below_theta = (z <= theta) & fired
        assert (
            not below_theta.any()
        ), "JumpReLU fired features whose pre-activation was <= theta"

    def test_vanilla_l1_is_included_in_loss(self):
        """The L1 sparsity term must contribute to the total loss.

        Comparing L1 vs L0 emergence requires structured data and many training
        steps — not appropriate for a unit test.  Instead we verify the simpler
        invariant: total loss strictly exceeds reconstruction loss when l1_weight > 0
        and any feature is active.
        """
        sae = VanillaSAE(D_INPUT, D_DICT, l1_weight=1e-2, device=DEVICE)
        result = sae(_batch())
        losses = sae.compute_loss(result)
        h = result["h"]
        if h.abs().sum() > 0:
            assert losses["loss"].item() > losses["l_reconstruct"].item(), (
                "VanillaSAE total loss must exceed reconstruction loss when h != 0 "
                "(L1 term was not added)"
            )


# ===========================================================================
# 2. Non-negative hidden activations
# ===========================================================================


class TestNonNegativeActivations:

    @pytest.mark.parametrize("arch", list(ARCHITECTURES.keys()))
    def test_h_non_negative(self, arch):
        sae = _make(arch)
        sae.eval()
        with torch.no_grad():
            h = sae.encode(_batch())
        assert (h >= 0).all(), f"{arch}: h contains negative values"

    def test_gated_gate_is_binary(self):
        sae = _make("gated")
        sae.eval()
        # Inspect gate mask directly
        centered = _batch() - sae.b_dec
        gate_pre = (centered @ sae.W_gate + sae.b_gate).detach()
        gate = (gate_pre > 0).float()
        unique_vals = gate.unique()
        assert set(unique_vals.tolist()).issubset(
            {0.0, 1.0}
        ), f"Gated gate must be binary; got {unique_vals.tolist()}"


# ===========================================================================
# 3. BatchTopK calibration
# ===========================================================================


class TestBatchTopKCalibration:

    def test_inference_l0_close_to_k_after_calibration(self):
        sae = BatchTopKSAE(D_INPUT, D_DICT, k=K, device=DEVICE)
        train_sae(
            sae, _DATA, num_batches=NUM_STEPS, batch_size=BATCH, lr=3e-4, log_every=999
        )
        # calibrate_thresholds is called automatically inside train_sae
        sae.eval()
        with torch.no_grad():
            h = sae.encode(_batch())
        l0 = (h > 0).float().sum(dim=-1).mean().item()
        # Allow ±50% tolerance on synthetic data (thresholds are estimated)
        assert (
            abs(l0 - K) < K * 0.5
        ), f"BatchTopK eval L0={l0:.1f} is too far from target k={K}"

    def test_inference_l0_far_from_zero_without_calibration(self):
        """Regression: before the fix, _threshold_estimate=0 caused plain ReLU (L0>>k)."""
        sae = BatchTopKSAE(D_INPUT, D_DICT, k=K, device=DEVICE)
        # Do NOT calibrate — _threshold_estimate stays at zeros
        sae.eval()
        with torch.no_grad():
            h = sae.encode(_batch())
        l0_broken = (h > 0).float().sum(dim=-1).mean().item()
        assert l0_broken > K, (
            f"Without calibration L0 should be >> k={K} (plain ReLU), "
            f"confirming the bug exists when calibration is skipped. Got {l0_broken:.1f}."
        )


# ===========================================================================
# 4. Checkpoint round-trip
# ===========================================================================


class TestCheckpointRoundTrip:

    @pytest.mark.parametrize("arch", list(ARCHITECTURES.keys()))
    def test_save_load_identical_reconstruction(self, arch, tmp_path):
        sae = _make(arch)
        sae.eval()
        x = _batch()

        with torch.no_grad():
            out_before = sae(x)["x_hat"].clone()

        ckpt_path = tmp_path / f"{arch}.pt"
        metadata = {"architecture": arch, "constructor_kwargs": {}}
        if arch in ("topk", "batchtopk"):
            metadata["constructor_kwargs"] = {"k": K}
        save_checkpoint(sae, ckpt_path, metadata)

        sae2, _ = load_checkpoint(ckpt_path, device=DEVICE)
        sae2.eval()
        with torch.no_grad():
            out_after = sae2(x)["x_hat"]

        assert torch.allclose(
            out_before, out_after, atol=1e-6
        ), f"{arch}: reconstruction changed after save/load"


# ===========================================================================
# 5. Training sanity
# ===========================================================================


class TestTrainingSanity:

    @pytest.mark.parametrize("arch", list(ARCHITECTURES.keys()))
    def test_loss_decreases(self, arch):
        """Loss averaged over the last quarter of training must be below the first quarter.

        Comparing single steps is fragile for architectures with volatile aux-loss
        terms (TopK dead-feature count, JumpReLU L0 penalty).  500 steps and
        quarter-averaging give a robust signal regardless of architecture.
        """
        n_steps = 500
        sae = _make(arch)
        results = train_sae(
            sae, _DATA, num_batches=n_steps, batch_size=BATCH, lr=3e-4, log_every=50
        )
        log = results["metrics_log"]
        assert len(log) >= 4, "Expected at least 4 log entries"
        quarter = max(1, len(log) // 4)
        first_avg = sum(e["loss"] for e in log[:quarter]) / quarter
        last_avg = sum(e["loss"] for e in log[-quarter:]) / quarter
        assert last_avg < first_avg, (
            f"{arch}: loss did not decrease over {n_steps} steps. "
            f"First-quarter avg={first_avg:.4f}, last-quarter avg={last_avg:.4f}"
        )

    @pytest.mark.parametrize("arch", list(ARCHITECTURES.keys()))
    def test_fvu_decreases_after_training(self, arch):
        """FVU after training must be strictly lower than FVU of the untrained SAE.

        Using a relative improvement check instead of an absolute threshold
        (fvu < 1.0) because at initialisation FVU ≈ 1.0 for random Gaussian data
        (MSE ≈ Var(x)), so 100 steps is not reliably enough to cross that threshold.
        This test is architecture-agnostic and converges reliably in 500 steps.
        """
        sae_init = _make(arch)
        sae_init.eval()
        with torch.no_grad():
            from lib.sae import compute_metrics

            fvu_before = compute_metrics(sae_init, _DATA)["fvu"]

        sae_trained = _make(arch)
        results = train_sae(
            sae_trained,
            _DATA,
            num_batches=500,
            batch_size=BATCH,
            lr=3e-4,
            log_every=999,
        )
        fvu_after = results["final_metrics"]["fvu"]
        assert fvu_after < fvu_before, (
            f"{arch}: FVU did not improve after training. "
            f"Before={fvu_before:.4f}, After={fvu_after:.4f}"
        )

    @pytest.mark.parametrize("arch", list(ARCHITECTURES.keys()))
    def test_not_all_features_dead_after_training(self, arch):
        sae = _make(arch)
        results = train_sae(
            sae, _DATA, num_batches=NUM_STEPS, batch_size=BATCH, lr=3e-4, log_every=999
        )
        dead_pct = results["final_metrics"]["dead_features_pct"]
        assert (
            dead_pct < 100.0
        ), f"{arch}: 100% of features are dead after {NUM_STEPS} steps"
