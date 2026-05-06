"""Tests for SAE evaluation metrics: coverage and board reconstruction.

Test categories:
  1. Feature–BSP matching  — precision, recall, F1 computed correctly
  2. Coverage              — mean best-F1 and threshold fractions
  3. Board reconstruction  — accuracy from high-precision features
  4. Edge cases            — dead features, constant BSPs, no qualifying features
  5. Integrated evaluate   — full pipeline with a trained SAE
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.sae import (
    ARCHITECTURES,
    TopKSAE,
    VanillaSAE,
    train_sae,
)
from lib.sae.eval import (
    FeatureBSPMatching,
    compute_board_reconstruction,
    compute_coverage,
    compute_feature_sharing,
    evaluate_sae,
    match_features_to_bsps,
)

# ---------------------------------------------------------------------------
# Shared constants — tiny, CPU-only
# ---------------------------------------------------------------------------
D_INPUT = 16
D_DICT = 32
K = 4
BATCH = 64
N_SAMPLES = 128
DEVICE = "cpu"

torch.manual_seed(42)


def _make_perfect_feature_bsp_pair(n: int = 100):
    """Create h and bsp_labels where feature 0 perfectly predicts BSP 0.

    Feature 0 fires on exactly the samples where BSP 0 is true.
    Other features are random noise. Other BSPs are random.
    """
    # BSP 0: first half True, second half False
    bsp_0 = torch.zeros(n)
    bsp_0[: n // 2] = 1.0

    # BSP 1: random
    bsp_1 = (torch.rand(n) > 0.5).float()

    bsp_labels = torch.stack([bsp_0, bsp_1], dim=1)  # (N, 2)

    # Feature 0: fires exactly when BSP 0 is true → perfect match
    h = torch.rand(n, 4) * 0.1  # low noise on features 1-3
    h[:, 0] = 0.0
    h[: n // 2, 0] = 1.0  # fires on first half = BSP 0 = True

    return h, bsp_labels


def _make_imperfect_pair(n: int = 100):
    """Feature partially overlaps BSP — precision < 1, recall < 1."""
    bsp_labels = torch.zeros(n, 1)
    bsp_labels[:40] = 1.0  # 40 positives

    h = torch.zeros(n, 2)
    # Feature 0: fires on samples 0-49 (50 fires)
    # BSP positive on 0-39 → TP=40, FP=10, FN=0
    h[:50, 0] = 1.0

    # Feature 1: fires on samples 0-19 (20 fires)
    # BSP positive on 0-39 → TP=20, FP=0, FN=20
    h[:20, 1] = 1.0

    return h, bsp_labels


# ===========================================================================
# 1. Feature–BSP matching
# ===========================================================================


class TestFeatureBSPMatching:

    def test_perfect_match_gives_f1_one(self):
        """A feature that perfectly predicts a BSP should have F1 = 1.0."""
        h, bsp_labels = _make_perfect_feature_bsp_pair()
        matching = match_features_to_bsps(h, bsp_labels)

        # Feature 0 vs BSP 0: perfect
        assert matching.precision[0, 0].item() == pytest.approx(1.0, abs=1e-4)
        assert matching.recall[0, 0].item() == pytest.approx(1.0, abs=1e-4)
        assert matching.f1[0, 0].item() == pytest.approx(1.0, abs=1e-4)

    def test_imperfect_precision_recall(self):
        """Verify precision and recall with known overlap."""
        h, bsp_labels = _make_imperfect_pair()
        matching = match_features_to_bsps(h, bsp_labels)

        # Feature 0: TP=40, FP=10, FN=0
        # precision = 40/50 = 0.8, recall = 40/40 = 1.0
        assert matching.precision[0, 0].item() == pytest.approx(0.8, abs=1e-3)
        assert matching.recall[0, 0].item() == pytest.approx(1.0, abs=1e-3)

        # Feature 1: TP=20, FP=0, FN=20
        # precision = 20/20 = 1.0, recall = 20/40 = 0.5
        assert matching.precision[1, 0].item() == pytest.approx(1.0, abs=1e-3)
        assert matching.recall[1, 0].item() == pytest.approx(0.5, abs=1e-3)

    def test_best_feature_per_bsp(self):
        """best_feature_per_bsp should select the feature with highest F1."""
        h, bsp_labels = _make_imperfect_pair()
        matching = match_features_to_bsps(h, bsp_labels)

        # Feature 0: F1 = 2*(0.8*1.0)/(0.8+1.0) = 0.8889
        # Feature 1: F1 = 2*(1.0*0.5)/(1.0+0.5) = 0.6667
        assert matching.best_feature_per_bsp[0].item() == 0
        assert matching.best_f1_per_bsp[0].item() == pytest.approx(0.8889, abs=1e-3)

    def test_output_shapes(self):
        """Matching tensors should have correct shapes."""
        d_dict, num_bsps, n = 8, 3, 50
        h = torch.rand(n, d_dict)
        bsp_labels = (torch.rand(n, num_bsps) > 0.5).float()
        matching = match_features_to_bsps(h, bsp_labels)

        assert matching.precision.shape == (d_dict, num_bsps)
        assert matching.recall.shape == (d_dict, num_bsps)
        assert matching.f1.shape == (d_dict, num_bsps)
        assert matching.best_f1_per_bsp.shape == (num_bsps,)
        assert matching.best_feature_per_bsp.shape == (num_bsps,)

    def test_sample_count_mismatch_raises(self):
        """Should raise ValueError when h and bsp_labels have different N."""
        h = torch.rand(50, 4)
        bsp_labels = torch.rand(30, 2)
        with pytest.raises(ValueError, match="Sample count mismatch"):
            match_features_to_bsps(h, bsp_labels)

    def test_all_values_in_valid_range(self):
        """Precision, recall, F1 should all be in [0, 1]."""
        h = torch.rand(100, 16)
        bsp_labels = (torch.rand(100, 8) > 0.5).float()
        matching = match_features_to_bsps(h, bsp_labels)

        for name, tensor in [
            ("precision", matching.precision),
            ("recall", matching.recall),
            ("f1", matching.f1),
        ]:
            assert (tensor >= 0).all(), f"{name} has negative values"
            assert (tensor <= 1.0 + 1e-6).all(), f"{name} has values > 1"


# ===========================================================================
# 2. Coverage
# ===========================================================================


class TestCoverage:

    def test_perfect_coverage(self):
        """If every BSP has a perfect-F1 feature, coverage = 1.0."""
        h, bsp_labels = _make_perfect_feature_bsp_pair()
        matching = match_features_to_bsps(h, bsp_labels)

        # Manually set best_f1 to 1.0 for all BSPs to isolate coverage logic
        matching.best_f1_per_bsp = torch.ones(2)
        cov = compute_coverage(matching)

        assert cov["coverage"] == pytest.approx(1.0, abs=1e-4)
        assert cov["coverage_above_50"] == pytest.approx(1.0, abs=1e-4)
        assert cov["coverage_above_75"] == pytest.approx(1.0, abs=1e-4)

    def test_zero_coverage(self):
        """If best_f1 is 0 for all BSPs, coverage = 0."""
        matching = FeatureBSPMatching(
            precision=torch.zeros(4, 3),
            recall=torch.zeros(4, 3),
            f1=torch.zeros(4, 3),
            best_f1_per_bsp=torch.zeros(3),
            best_feature_per_bsp=torch.zeros(3, dtype=torch.long),
        )
        cov = compute_coverage(matching)
        assert cov["coverage"] == pytest.approx(0.0, abs=1e-4)
        assert cov["coverage_above_50"] == pytest.approx(0.0, abs=1e-4)

    def test_partial_coverage(self):
        """Coverage is the mean of best-F1 values."""
        matching = FeatureBSPMatching(
            precision=torch.zeros(1, 4),
            recall=torch.zeros(1, 4),
            f1=torch.zeros(1, 4),
            best_f1_per_bsp=torch.tensor([1.0, 0.8, 0.4, 0.0]),
            best_feature_per_bsp=torch.zeros(4, dtype=torch.long),
        )
        cov = compute_coverage(matching)

        assert cov["coverage"] == pytest.approx(0.55, abs=1e-4)
        assert cov["num_bsps"] == 4
        assert cov["min_f1"] == pytest.approx(0.0, abs=1e-4)
        assert cov["max_f1"] == pytest.approx(1.0, abs=1e-4)
        # Above 50: 1.0 and 0.8 → 2/4 = 0.5
        assert cov["coverage_above_50"] == pytest.approx(0.5, abs=1e-4)
        # Above 75: only 1.0 and 0.8 → 2/4 = 0.5
        assert cov["coverage_above_75"] == pytest.approx(0.5, abs=1e-4)

    def test_required_keys_present(self):
        """Coverage dict must contain all expected keys."""
        h = torch.rand(50, 4)
        bsp = (torch.rand(50, 2) > 0.5).float()
        matching = match_features_to_bsps(h, bsp)
        cov = compute_coverage(matching)

        expected_keys = {
            "coverage",
            "coverage_above_50",
            "coverage_above_75",
            "num_bsps",
            "min_f1",
            "max_f1",
            "median_f1",
        }
        assert expected_keys == set(cov.keys())


class TestFeatureSharing:

    def test_feature_sharing_detects_reuse(self):
        matching = FeatureBSPMatching(
            precision=torch.zeros(4, 4),
            recall=torch.zeros(4, 4),
            f1=torch.zeros(4, 4),
            best_f1_per_bsp=torch.tensor([1.0, 0.9, 0.8, 0.7]),
            best_feature_per_bsp=torch.tensor([0, 0, 2, 3]),
        )

        sharing = compute_feature_sharing(matching)

        assert sharing["num_features_used_by_best_matches"] == 3
        assert sharing["max_bsps_per_feature"] == 2
        assert sharing["num_shared_features"] == 1
        assert sharing["num_bsps_with_shared_best_feature"] == 2
        assert sharing["fraction_bsps_with_shared_best_feature"] == pytest.approx(
            0.5, abs=1e-4
        )

    def test_feature_sharing_all_unique(self):
        matching = FeatureBSPMatching(
            precision=torch.zeros(4, 4),
            recall=torch.zeros(4, 4),
            f1=torch.zeros(4, 4),
            best_f1_per_bsp=torch.ones(4),
            best_feature_per_bsp=torch.tensor([0, 1, 2, 3]),
        )

        sharing = compute_feature_sharing(matching)

        assert sharing["num_features_used_by_best_matches"] == 4
        assert sharing["num_shared_features"] == 0
        assert sharing["fraction_bsps_with_unique_best_feature"] == pytest.approx(
            1.0, abs=1e-4
        )


# ===========================================================================
# 3. Board reconstruction
# ===========================================================================


class TestBoardReconstruction:

    def test_perfect_reconstruction(self):
        """Feature with precision=1.0 and matching all samples → accuracy 1.0."""
        h, bsp_labels = _make_perfect_feature_bsp_pair()
        matching = match_features_to_bsps(h, bsp_labels)

        recon = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=0.9
        )

        # BSP 0 should be reconstructable (feature 0 has precision=1.0)
        assert recon["num_reconstructable_bsps"] >= 1
        assert recon["board_reconstruction"] == pytest.approx(1.0, abs=1e-3)

    def test_no_qualifying_features(self):
        """When no feature has precision above threshold, reconstruction = 0."""
        n = 100
        h = torch.rand(n, 4) * 0.5 + 0.5  # all features fire on all samples
        bsp_labels = (torch.rand(n, 2) > 0.5).float()  # random BSPs

        matching = match_features_to_bsps(h, bsp_labels)

        # Features fire on everything → precision ≈ base rate → likely < 0.9
        recon = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=0.99
        )

        assert recon["num_reconstructable_bsps"] == 0
        assert recon["board_reconstruction"] == 0.0
        assert recon["fraction_reconstructable"] == 0.0

    def test_partial_reconstruction(self):
        """Only BSPs with high-precision features are reconstructed."""
        n = 200
        # BSP 0: well-predicted, BSP 1: poorly predicted
        bsp_labels = torch.zeros(n, 2)
        bsp_labels[:100, 0] = 1.0  # BSP 0: first half positive
        bsp_labels[:50, 1] = 1.0  # BSP 1: first quarter positive

        h = torch.zeros(n, 4)
        # Feature 0: fires exactly on BSP 0=True → precision=1.0 for BSP 0
        h[:100, 0] = 1.0
        # Feature 1: fires on everything → precision=0.25 for BSP 1
        h[:, 1] = 1.0

        matching = match_features_to_bsps(h, bsp_labels)
        recon = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=0.9
        )

        # Only BSP 0 should qualify
        assert recon["num_reconstructable_bsps"] == 1
        assert recon["fraction_reconstructable"] == pytest.approx(0.5, abs=1e-4)
        # BSP 0 accuracy should be 1.0 (perfect feature)
        assert recon["board_reconstruction"] == pytest.approx(1.0, abs=1e-3)

    def test_reconstruction_accuracy_calculation(self):
        """Verify accuracy is correct for a feature with known error rate."""
        n = 200
        bsp_labels = torch.zeros(n, 1)
        bsp_labels[:100] = 1.0  # 100 positives

        h = torch.zeros(n, 2)
        # Feature 0: fires on samples 0..109 (100 TP + 10 FP)
        # precision = 100/110 = 0.909 (above 0.9)
        # accuracy: 100 correct positive + 90 correct negative = 190/200 = 0.95
        h[:110, 0] = 1.0

        matching = match_features_to_bsps(h, bsp_labels)
        recon = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=0.9
        )

        assert recon["num_reconstructable_bsps"] == 1
        assert recon["board_reconstruction"] == pytest.approx(0.95, abs=1e-3)

    def test_per_bsp_accuracy_length(self):
        """per_bsp_accuracy list has one entry per BSP (None for non-qualifying)."""
        n = 100
        h = torch.zeros(n, 2)
        h[:50, 0] = 1.0
        bsp_labels = torch.zeros(n, 3)
        bsp_labels[:50, 0] = 1.0  # Feature 0 matches BSP 0 perfectly
        bsp_labels[:25, 1] = 1.0  # Feature 0 won't match BSP 1 well

        matching = match_features_to_bsps(h, bsp_labels)
        recon = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=0.9
        )

        assert len(recon["per_bsp_accuracy"]) == 3

    def test_required_keys_present(self):
        """Board reconstruction dict must contain all expected keys."""
        h, bsp_labels = _make_perfect_feature_bsp_pair()
        matching = match_features_to_bsps(h, bsp_labels)
        recon = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=0.9
        )

        expected_keys = {
            "board_reconstruction",
            "num_reconstructable_bsps",
            "fraction_reconstructable",
            "mean_accuracy",
            "per_bsp_accuracy",
        }
        assert expected_keys == set(recon.keys())


# ===========================================================================
# 4. Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_dead_features_dont_crash(self):
        """Features that never fire (h=0) should produce precision=0, recall=0."""
        n = 50
        h = torch.zeros(n, 4)  # all features dead
        bsp_labels = (torch.rand(n, 2) > 0.5).float()

        matching = match_features_to_bsps(h, bsp_labels)
        assert (matching.f1 == 0).all()

        cov = compute_coverage(matching)
        assert cov["coverage"] == pytest.approx(0.0, abs=1e-4)

    def test_constant_bsp_all_true(self):
        """A BSP that is always true: recall=1 for any firing feature."""
        n = 50
        h = torch.zeros(n, 2)
        h[:25, 0] = 1.0  # feature 0 fires on half

        bsp_labels = torch.ones(n, 1)  # BSP always true

        matching = match_features_to_bsps(h, bsp_labels)

        # Feature 0: TP=25, FP=0, FN=25 → precision=1.0, recall=0.5
        assert matching.precision[0, 0].item() == pytest.approx(1.0, abs=1e-3)
        assert matching.recall[0, 0].item() == pytest.approx(0.5, abs=1e-3)

    def test_constant_bsp_all_false(self):
        """A BSP that is always false: any firing feature has recall=0."""
        n = 50
        h = torch.zeros(n, 2)
        h[:25, 0] = 1.0

        bsp_labels = torch.zeros(n, 1)  # BSP always false

        matching = match_features_to_bsps(h, bsp_labels)

        # Feature 0: TP=0, FP=25, FN=0 → precision=0, recall=0
        assert matching.precision[0, 0].item() == pytest.approx(0.0, abs=1e-3)
        assert matching.f1[0, 0].item() == pytest.approx(0.0, abs=1e-3)

    def test_single_sample(self):
        """Should work with a single sample."""
        h = torch.tensor([[1.0, 0.0]])  # 1 sample, 2 features
        bsp_labels = torch.tensor([[1.0]])  # 1 sample, 1 BSP

        matching = match_features_to_bsps(h, bsp_labels)
        assert matching.f1.shape == (2, 1)

        # Feature 0 fires and BSP is true → TP=1, FP=0, FN=0 → F1=1.0
        assert matching.f1[0, 0].item() == pytest.approx(1.0, abs=1e-3)

    def test_large_dict_small_bsp(self):
        """Many features, few BSPs — coverage shouldn't exceed 1.0."""
        n = 100
        h = torch.rand(n, 256)
        bsp_labels = (torch.rand(n, 3) > 0.5).float()

        matching = match_features_to_bsps(h, bsp_labels)
        cov = compute_coverage(matching)

        assert 0.0 <= cov["coverage"] <= 1.0


# ===========================================================================
# 5. Integrated evaluation with a trained SAE
# ===========================================================================


class TestEvaluateSAE:

    def test_evaluate_sae_returns_all_keys(self):
        """evaluate_sae must return both structural and BSP-based metrics."""
        sae = TopKSAE(D_INPUT, D_DICT, k=K, device=DEVICE)
        train_sae(
            sae,
            torch.randn(N_SAMPLES, D_INPUT),
            num_batches=50,
            batch_size=32,
            lr=3e-4,
            log_every=999,
        )
        sae.eval()

        activations = torch.randn(N_SAMPLES, D_INPUT)
        bsp_labels = (torch.rand(N_SAMPLES, 10) > 0.5).float()

        results = evaluate_sae(sae, activations, bsp_labels)

        # Structural metrics
        assert "fvu" in results
        assert "l0" in results
        assert "mse" in results
        assert "dead_features_pct" in results

        # Coverage metrics
        assert "coverage" in results
        assert "coverage_above_50" in results
        assert "num_bsps" in results
        assert results["num_bsps"] == 10

        # Board reconstruction metrics
        assert "board_reconstruction" in results
        assert "num_reconstructable_bsps" in results
        assert "feature_sharing" in results

    def test_trained_sae_has_nonzero_coverage(self):
        """A trained SAE on structured data should have some coverage > 0.

        We embed two BSPs directly into the activation data so a well-trained
        SAE has a chance to discover them.
        """
        torch.manual_seed(7)
        n = 256

        # Create activations with 2 latent BSPs embedded
        bsp_0 = (torch.rand(n) > 0.5).float()
        bsp_1 = (torch.rand(n) > 0.7).float()

        # Construct activations where BSPs are linearly encoded
        base = torch.randn(n, D_INPUT) * 0.1
        direction_0 = torch.randn(D_INPUT)
        direction_1 = torch.randn(D_INPUT)
        activations = (
            base + bsp_0.unsqueeze(1) * direction_0 + bsp_1.unsqueeze(1) * direction_1
        )

        bsp_labels = torch.stack([bsp_0, bsp_1], dim=1)

        sae = TopKSAE(D_INPUT, D_DICT, k=K, device=DEVICE)
        train_sae(
            sae, activations, num_batches=500, batch_size=32, lr=3e-4, log_every=999
        )
        sae.eval()

        results = evaluate_sae(sae, activations, bsp_labels)

        # With BSPs linearly embedded and a trained SAE, coverage should be > 0
        assert (
            results["coverage"] > 0.0
        ), f"Expected nonzero coverage on structured data, got {results['coverage']}"

    @pytest.mark.parametrize("arch", ["vanilla", "topk"])
    def test_works_across_architectures(self, arch):
        """evaluate_sae should work for different SAE architectures."""
        defaults = dict(d_input=D_INPUT, d_dict=D_DICT, device=DEVICE)
        if arch in ("topk", "batchtopk"):
            defaults["k"] = K

        sae = ARCHITECTURES[arch](**defaults)
        sae.eval()

        activations = torch.randn(64, D_INPUT)
        bsp_labels = (torch.rand(64, 5) > 0.5).float()

        results = evaluate_sae(sae, activations, bsp_labels, batch_size=32)

        assert "coverage" in results
        assert "board_reconstruction" in results
        assert "feature_sharing" in results
        assert 0.0 <= results["coverage"] <= 1.0
