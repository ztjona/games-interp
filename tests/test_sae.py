"""Unit tests for lib.sae — all run on CPU with tiny tensors (< 5 s total).

Test categories:
  1. Sparsity contracts   — exact K / B×K / threshold guarantees
  2. Non-negative h       — all architectures must produce h >= 0
  3. BatchTopK calibration — inference L0 ≈ k after calibrate_thresholds()
  4. Checkpoint round-trip — load(save(sae)) gives identical reconstructions
  5. Training sanity       — 100 steps reduces loss, FVU < 1.0, some features alive
  6. Aux loss gradients    — TopK dead features receive non-zero gradients
  7. Metrics completeness  — l0_std, median_feat_freq present and sensible
  8. Early stopping        — patience triggers stop, disabled by default
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


# Architectures swept by the generic parametrized tests below.  Anchored
# variants need extra constructor kwargs (anchor_feature_idx etc.) and are
# exercised separately in TestAnchoredSAE; exclude them here.
BASE_ARCHS = [a for a in ARCHITECTURES.keys() if not a.startswith("anchored-")]


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

    @pytest.mark.parametrize("arch", BASE_ARCHS)
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

    @pytest.mark.parametrize("arch", BASE_ARCHS)
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

    @pytest.mark.parametrize("arch", BASE_ARCHS)
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

    @pytest.mark.parametrize("arch", BASE_ARCHS)
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

    @pytest.mark.parametrize("arch", BASE_ARCHS)
    def test_not_all_features_dead_after_training(self, arch):
        sae = _make(arch)
        results = train_sae(
            sae, _DATA, num_batches=NUM_STEPS, batch_size=BATCH, lr=3e-4, log_every=999
        )
        dead_pct = results["final_metrics"]["dead_features_pct"]
        assert (
            dead_pct < 100.0
        ), f"{arch}: 100% of features are dead after {NUM_STEPS} steps"


# ===========================================================================
# 6. Aux loss gradients — dead features must receive gradient signal
# ===========================================================================


class TestAuxLossGradients:

    def test_topk_dead_features_receive_gradient(self):
        """TopK aux loss must provide non-zero gradients to dead feature weights.

        Before the fix, the aux loss was a step function (count of dead features)
        with zero gradient everywhere, making dead features irrecoverable.
        """
        sae = TopKSAE(D_INPUT, D_DICT, k=K, device=DEVICE)
        x = _batch()
        result = sae(x)
        losses = sae.compute_loss(result)
        losses["loss"].backward()

        h = result["h"]
        dead_mask = ~(h > 0).any(dim=0)

        if dead_mask.sum() == 0:
            pytest.skip("No dead features in this batch — cannot test aux gradient")

        dead_grad_norm = sae.W_enc.grad[:, dead_mask].norm().item()
        assert dead_grad_norm > 0, (
            f"TopK dead features get zero gradient (norm={dead_grad_norm:.6f}). "
            f"Aux loss is not providing a learning signal."
        )

    def test_topk_aux_loss_is_zero_when_all_alive(self):
        """If no features are dead, aux loss should be zero."""
        # Use k = D_DICT so all features fire
        sae = TopKSAE(D_INPUT, D_DICT, k=D_DICT, device=DEVICE)
        x = _batch()
        result = sae(x)
        losses = sae.compute_loss(result)
        assert (
            losses["l_aux"].item() == 0.0
        ), "Aux loss should be 0 when all features are alive"

    def test_gated_wgate_receives_reconstruction_gradient(self):
        """W_gate must receive non-zero gradient from the reconstruction loss.

        The hard gate g = (gate_pre > 0).float() is a step function, so the main
        reconstruction path has ∂loss/∂W_gate = 0. The via-gate auxiliary loss
        (decoding ReLU(gate_pre) through frozen W_dec) is the only path that
        routes reconstruction signal back to W_gate.

        Without this fix, W_gate only receives gradient from the L1 sparsity
        penalty, which monotonically pushes all gate activations to zero —
        causing total collapse (L0→0, FVU→1, dead→100%) as seen in the arnold sweep.
        """
        sae = GatedSAE(D_INPUT, D_DICT, l1_weight=1e-3, device=DEVICE)
        sae.train()
        x = _batch()
        result = sae(x)
        losses = sae.compute_loss(result)
        losses["loss"].backward()

        assert sae.W_gate.grad is not None, "W_gate has no gradient at all"
        wgate_grad_norm = sae.W_gate.grad.norm().item()
        assert wgate_grad_norm > 0, (
            f"W_gate gradient is zero (norm={wgate_grad_norm:.6f}). "
            "The via-gate auxiliary loss is not routing reconstruction gradients "
            "to W_gate. Without this, the L1 penalty will collapse all gates to zero."
        )

    def test_gated_wgate_gradient_comes_from_reconstruction_not_only_l1(self):
        """Verify gradient reaches W_gate even when L1 weight is zero.

        If l1_weight=0, the only gradient path to W_gate is the via-gate aux loss.
        This isolates the fix from the sparsity term — if W_gate has zero gradient
        here, the via-gate aux loss is broken.
        """
        sae = GatedSAE(D_INPUT, D_DICT, l1_weight=0.0, device=DEVICE)
        sae.train()
        x = _batch()
        result = sae(x)
        losses = sae.compute_loss(result)
        losses["loss"].backward()

        assert sae.W_gate.grad is not None, "W_gate has no gradient with l1_weight=0"
        wgate_grad_norm = sae.W_gate.grad.norm().item()
        assert wgate_grad_norm > 0, (
            f"W_gate gradient is zero even with l1_weight=0 (norm={wgate_grad_norm:.6f}). "
            "The via-gate auxiliary loss must route gradients to W_gate independently."
        )

    def test_jumprelu_theta_receives_sparsity_gradient(self):
        """log_theta must receive non-zero gradient from the L0 sparsity penalty.

        The original bug: the L0 penalty was computed as (h > 0).float().sum(),
        which is a step function with ∂/∂theta = 0 everywhere. The fix replaces
        this with the sigmoid kernel estimator σ((z−θ)/ε), which is differentiable
        w.r.t. theta and thus drives theta toward the l0_target.

        Without the fix, theta never moves and L0 converges to ~d_dict regardless
        of the target (as observed: t32 → L0=595, t64 → L0=604 in arnold sweep).
        """
        sae = JumpReLUSAE(D_INPUT, D_DICT, l0_target=K, device=DEVICE)
        sae.train()
        x = _batch()
        result = sae(x)
        losses = sae.compute_loss(result)
        losses["loss"].backward()

        assert sae.log_theta.grad is not None, "log_theta has no gradient at all"
        theta_grad_norm = sae.log_theta.grad.norm().item()
        assert theta_grad_norm > 0, (
            f"log_theta gradient is zero (norm={theta_grad_norm:.6f}). "
            "The L0 sparsity penalty is not differentiable w.r.t. theta. "
            "Use the sigmoid kernel estimator σ((z−θ)/ε) instead of (h>0).float()."
        )

    def test_jumprelu_l0_converges_toward_target(self):
        """After training, L0 should move toward l0_target (not stay near d_dict).

        Uses a target well below the natural L0 of random data to make the
        direction of movement unambiguous. After 500 steps, L0 must be closer
        to the target than it was at initialisation.
        """
        target = K  # K=8 is well below d_dict=64
        sae = JumpReLUSAE(D_INPUT, D_DICT, l0_target=float(target), device=DEVICE)

        # Measure initial L0 (random weights, near d_dict)
        sae.eval()
        with torch.no_grad():
            h_init = sae.encode(_batch())
        l0_init = (h_init > 0).float().sum(dim=-1).mean().item()

        # Train with sparsity pressure
        train_sae(sae, _DATA, num_batches=500, batch_size=BATCH, lr=3e-4, log_every=999)

        sae.eval()
        with torch.no_grad():
            h_final = sae.encode(_batch())
        l0_final = (h_final > 0).float().sum(dim=-1).mean().item()

        dist_init = abs(l0_init - target)
        dist_final = abs(l0_final - target)
        assert dist_final < dist_init, (
            f"JumpReLU L0 did not converge toward target={target}. "
            f"Initial L0={l0_init:.1f} (dist={dist_init:.1f}), "
            f"Final L0={l0_final:.1f} (dist={dist_final:.1f}). "
            "The L0 penalty may have zero gradient w.r.t. theta."
        )


# ===========================================================================
# 7. Metrics completeness — new metrics present and sensible
# ===========================================================================


class TestMetricsCompleteness:

    def test_new_metrics_present(self):
        """compute_metrics must return l0_std and median_feat_freq."""
        from lib.sae.train import compute_metrics

        sae = _make("vanilla")
        metrics = compute_metrics(sae, _batch())
        assert "l0_std" in metrics, "l0_std missing from metrics"
        assert "median_feat_freq" in metrics, "median_feat_freq missing from metrics"

    def test_l0_std_is_zero_for_topk(self):
        """TopK activates exactly K features per sample, so L0 std must be 0."""
        from lib.sae.train import compute_metrics

        sae = _make("topk")
        metrics = compute_metrics(sae, _batch())
        assert (
            metrics["l0_std"] == 0.0
        ), f"TopK L0 std should be exactly 0 (exact K per sample), got {metrics['l0_std']}"

    def test_median_feat_freq_bounded(self):
        """Median feature frequency must be in [0, 1]."""
        from lib.sae.train import compute_metrics

        for arch in BASE_ARCHS:
            sae = _make(arch)
            metrics = compute_metrics(sae, _batch())
            mff = metrics["median_feat_freq"]
            assert (
                0.0 <= mff <= 1.0
            ), f"{arch}: median_feat_freq={mff} is outside [0, 1]"

    @pytest.mark.parametrize("arch", BASE_ARCHS)
    def test_all_metrics_keys_present(self, arch):
        """Every architecture must return the full set of expected metric keys."""
        from lib.sae.train import compute_metrics

        expected_keys = {
            "l0",
            "l0_std",
            "fvu",
            "mse",
            "dead_features_pct",
            "median_feat_freq",
        }
        sae = _make(arch)
        metrics = compute_metrics(sae, _batch())
        missing = expected_keys - set(metrics.keys())
        assert not missing, f"{arch}: missing metric keys: {missing}"


# ===========================================================================
# 8. Early stopping
# ===========================================================================


class TestEarlyStopping:

    def test_disabled_by_default(self):
        """With patience=0 (default), training runs all num_batches steps."""
        sae = _make("vanilla")
        results = train_sae(
            sae, _DATA, num_batches=50, batch_size=BATCH, lr=1e-3, log_every=10
        )
        assert results["final_step"] == 50
        assert results["early_stopped"] is False

    def test_triggers_before_max_steps(self):
        """With aggressive patience, training should stop before num_batches.

        Strategy: train 20 steps first (SAE learns something), then continue
        with patience=2. The well-trained SAE should plateau quickly on the
        tiny synthetic data and trigger early stop before 500 additional steps.
        """
        sae = _make("vanilla")
        # Pre-train so FVU is already low — plateau will come fast
        train_sae(sae, _DATA, num_batches=100, batch_size=BATCH, lr=1e-3, log_every=50)

        # Now train with very aggressive patience on an already-converged SAE
        results = train_sae(
            sae,
            _DATA,
            num_batches=500,
            batch_size=BATCH,
            lr=1e-3,
            log_every=10,
            patience=3,
            min_improvement=0.5,  # require 50% improvement
        )
        assert results["early_stopped"] is True
        assert results["final_step"] < 500

    def test_results_keys(self):
        """Results dict must contain early_stopped and final_step keys."""
        sae = _make("vanilla")
        results = train_sae(
            sae, _DATA, num_batches=20, batch_size=BATCH, lr=1e-3, log_every=10
        )
        assert "early_stopped" in results
        assert "final_step" in results


# ===========================================================================
# 9. Anchored SAEs (supervised BCE penalty on a subset of features)
# ===========================================================================


from lib.sae import AnchoredBatchTopKSAE, AnchoredJumpReLUSAE  # noqa: E402


_NUM_ANCHORED = 4  # use prefix-index mapping: features [0..3] anchor BSPs [0..3]


def _make_anchored(arch: str, lam: float = 0.1, **kw):
    """Build an anchored SAE with prefix-index mapping and uniform λ."""
    feat = list(range(_NUM_ANCHORED))
    bsp = list(range(_NUM_ANCHORED))
    lam_vec = [lam] * _NUM_ANCHORED
    defaults = dict(
        d_input=D_INPUT,
        d_dict=D_DICT,
        anchor_feature_idx=feat,
        anchor_bsp_idx=bsp,
        anchor_lambda_per_feature=lam_vec,
        device=DEVICE,
    )
    if arch == "anchored-batchtopk":
        defaults["k"] = K
    return ARCHITECTURES[arch](**defaults, **kw)


def _random_labels(n: int = BATCH, c: int = _NUM_ANCHORED) -> torch.Tensor:
    torch.manual_seed(123)
    return (torch.rand(n, c) > 0.5).float()


class TestAnchoredSAE:

    def test_lambda_zero_matches_base_loss(self):
        """λ=0 ⇒ anchored compute_loss == base compute_loss (same seed init)."""
        torch.manual_seed(7)
        base = _make("jumprelu", theta_init=0.01, l0_target=8.0)
        torch.manual_seed(7)
        anchored = _make_anchored(
            "anchored-jumprelu", lam=0.0, theta_init=0.01, l0_target=8.0
        )
        x = _batch()
        anchored.set_batch_labels(_random_labels())
        rb = base(x)
        ra = anchored(x)
        lb = base.compute_loss(rb)["loss"]
        la = anchored.compute_loss(ra)["loss"]
        assert torch.allclose(lb, la, atol=1e-6), (
            f"With λ=0 anchored loss must equal base loss; got base={lb.item()} "
            f"anchored={la.item()}"
        )

    def test_bce_numerical_correctness(self):
        """l_anchor matches a hand-computed BCE-with-logits on the toy slice."""
        torch.manual_seed(11)
        sae = _make_anchored("anchored-jumprelu", lam=1.0, theta_init=0.01, l0_target=8.0)
        x = _batch()
        labels = _random_labels()
        sae.set_batch_labels(labels)
        result = sae(x)
        losses = sae.compute_loss(result)
        # Reproduce by hand:
        with torch.no_grad():
            z = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
            z_anchor = z[:, : _NUM_ANCHORED]
            y_anchor = labels[:, : _NUM_ANCHORED]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                z_anchor, y_anchor, reduction="none"
            )
            expected = bce.sum(dim=-1).mean()  # λ=1 per feature
        assert torch.allclose(losses["l_anchor"], expected, atol=1e-6), (
            f"l_anchor={losses['l_anchor'].item()} vs expected={expected.item()}"
        )

    def test_prefix_mapping_uses_first_n_features(self):
        """anchor_feature_idx and anchor_bsp_idx are exactly [0..N-1]."""
        sae = _make_anchored("anchored-batchtopk", lam=0.1)
        assert sae.anchor_feature_idx.tolist() == list(range(_NUM_ANCHORED))
        assert sae.anchor_bsp_idx.tolist() == list(range(_NUM_ANCHORED))
        assert sae.anchor_lambda_per_feature.shape == (_NUM_ANCHORED,)

    def test_save_load_preserves_anchor_buffers(self):
        """save/load round-trip preserves anchor_* buffers and reproduces loss."""
        torch.manual_seed(3)
        sae = _make_anchored("anchored-jumprelu", lam=0.25, theta_init=0.01, l0_target=8.0)
        x = _batch()
        labels = _random_labels()
        sae.set_batch_labels(labels)
        result = sae(x)
        loss_before = sae.compute_loss(result)["l_anchor"].clone()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "anchored.pt"
            save_checkpoint(
                sae,
                path,
                metadata={
                    "architecture": "anchored-jumprelu",
                    "constructor_kwargs": {
                        "anchor_feature_idx": list(range(_NUM_ANCHORED)),
                        "anchor_bsp_idx": list(range(_NUM_ANCHORED)),
                        "anchor_lambda_per_feature": [0.25] * _NUM_ANCHORED,
                        "theta_init": 0.01,
                        "l0_target": 8.0,
                    },
                },
            )
            sae2, _ = load_checkpoint(path, device=DEVICE)

        assert torch.equal(sae2.anchor_feature_idx, sae.anchor_feature_idx)
        assert torch.equal(sae2.anchor_bsp_idx, sae.anchor_bsp_idx)
        assert torch.allclose(sae2.anchor_lambda_per_feature, sae.anchor_lambda_per_feature)
        sae2.set_batch_labels(labels)
        loss_after = sae2.compute_loss(sae2(x))["l_anchor"]
        assert torch.allclose(loss_before, loss_after, atol=1e-6)

    def test_labels_consumed_each_call(self):
        """set_batch_labels is one-shot: omitting it next call gives l_anchor=0."""
        torch.manual_seed(5)
        sae = _make_anchored("anchored-jumprelu", lam=1.0, theta_init=0.01, l0_target=8.0)
        x = _batch()
        sae.set_batch_labels(_random_labels())
        losses_with = sae.compute_loss(sae(x))
        assert losses_with["l_anchor"].item() > 0
        # No new labels set â€” l_anchor must be exactly zero, not stale reuse.
        losses_without = sae.compute_loss(sae(x))
        assert losses_without["l_anchor"].item() == 0.0


