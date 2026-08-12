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
        # F1 threshold-fraction keys were removed 2026-08-11; the MCC
        # equivalents (coverage_mcc_above_25/50) carry that role now.
        assert "coverage_above_50" not in cov
        assert "coverage_f1_lift" not in cov

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
        # F1 derivatives (min/max/median/above_N/lift) were removed 2026-08-11;
        # `coverage` is the only F1 number kept, for literature comparison.

    def test_required_keys_present(self):
        """Coverage dict must contain all expected keys."""
        h = torch.rand(50, 4)
        bsp = (torch.rand(50, 2) > 0.5).float()
        matching = match_features_to_bsps(h, bsp)
        cov = compute_coverage(matching)

        expected_keys = {
            "coverage",
            "num_bsps",
            # MCC-based (added when matching has MCC fields populated)
            "coverage_mcc",
            "coverage_mcc_above_25",
            "coverage_mcc_above_50",
            "median_mcc",
            "min_mcc",
            "max_mcc",
            # Youden's J = TPR - FPR: the prevalence-INVARIANT companion to MCC.
            # F1, F1-lift, MCC and precision all move with the base rate for a
            # detector of fixed quality; J does not. See scripts/prevalence_audit.py.
            "coverage_youden_j",
            "median_youden_j",
            "min_youden_j",
            "max_youden_j",
            # MCC standardised to a reference prevalence: the number that IS
            # comparable across populations with different base rates. p_ref is
            # stored so a standardised value is never ambiguous.
            "coverage_mcc_at_pref",
            "median_mcc_at_pref",
            "p_ref",
            # Prevalence belongs in the registry: without it a reader cannot
            # tell a weak SAE from a rare concept.
            "mean_base_rate",
            "median_base_rate",
            "min_base_rate",
            "max_base_rate",
            "frac_bsps_very_rare",
            "frac_bsps_trivial_f1",
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
        assert "coverage_mcc" in results
        assert "coverage_youden_j" in results
        assert "coverage_mcc_at_pref" in results
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


class TestPrevalenceStandardisation:
    """MCC@p_ref must be the noise-free equivalent of subsample matching:
    a detector of fixed quality scored on two populations that differ ONLY in
    base rate must land on the same standardised number."""

    def test_mcc_at_pref_removes_a_pure_prevalence_gap(self):
        import torch

        from lib.sae.eval import match_features_to_bsps

        torch.manual_seed(0)
        a, b = 0.55, 0.985  # fixed detector quality

        def population(n, p):
            y = (torch.rand(n) < p).float()
            fires = torch.where(
                y > 0, (torch.rand(n) < a).float(), (torch.rand(n) > b).float()
            )
            h = fires.unsqueeze(1)  # one feature
            return h, y.unsqueeze(1)

        hX, yX = population(300_000, 0.050)
        hY, yY = population(300_000, 0.013)
        mX = match_features_to_bsps(hX, yX)
        mY = match_features_to_bsps(hY, yY)

        raw_gap = abs(
            float(mY.best_mcc_per_bsp[0]) / float(mX.best_mcc_per_bsp[0]) - 1.0
        )
        std_gap = abs(
            float(mY.best_mcc_at_pref_per_bsp[0])
            / float(mX.best_mcc_at_pref_per_bsp[0])
            - 1.0
        )
        # Raw MCC shows a large spurious gap; standardised MCC must not.
        assert raw_gap > 0.15, f"expected a raw prevalence gap, got {raw_gap:.3f}"
        assert std_gap < 0.05, f"standardisation left a {std_gap:.3f} gap"

    def test_p_ref_is_frozen_and_recorded(self):
        from lib.sae.eval import P_REF_DEFAULT

        # Frozen by decision (docs/methods-reference.md S1.4). Changing it
        # changes every standardised number ever reported.
        assert P_REF_DEFAULT == 0.025

    def test_youden_j_is_prevalence_invariant_where_mcc_is_not(self):
        import torch

        from lib.sae.eval import match_features_to_bsps

        torch.manual_seed(1)
        a, b = 0.60, 0.98

        def population(n, p):
            y = (torch.rand(n) < p).float()
            fires = torch.where(
                y > 0, (torch.rand(n) < a).float(), (torch.rand(n) > b).float()
            )
            return fires.unsqueeze(1), y.unsqueeze(1)

        hX, yX = population(300_000, 0.100)
        hY, yY = population(300_000, 0.013)
        jX = float(match_features_to_bsps(hX, yX).best_j_per_bsp[0])
        jY = float(match_features_to_bsps(hY, yY).best_j_per_bsp[0])
        assert abs(jY - jX) < 0.02, f"J moved with prevalence: {jX:.3f} vs {jY:.3f}"


class TestConceptFamilyRollups:
    """Schema-driven family grouping (lib side).

    The library must never carry its own category->family mapping: it reads
    only what the schema declares. These tests pin that contract.
    """

    @staticmethod
    def _schema(*rows):
        """Build a minimal schema from ``(category, family, role, n)`` rows."""
        bsps = []
        for category, family, role, n in rows:
            for i in range(n):
                bsps.append({
                    "id": f"{category}_{i}",
                    "category": category,
                    "concept_family": family,
                    "family_role": role,
                })
        return {"bsps": bsps}

    def test_category_families_reads_per_bsp_stamps(self):
        from lib.sae.eval import category_families

        schema = self._schema(("threat_line", "line_threat", "state", 2))
        assert category_families(schema) == {
            "threat_line": {"concept_family": "line_threat",
                            "family_role": "state"}
        }

    def test_unstamped_schema_yields_empty_not_a_guess(self):
        """A pre-family schema must report 'nothing declared', so callers can
        tell the user to re-stamp rather than silently reporting no families."""
        from lib.sae.eval import category_families

        assert category_families({"bsps": [{"id": "x", "category": "threat_line"}]}) == {}

    def test_triads_prefer_the_most_agent_relative_category(self):
        from lib.sae.eval import derive_triads

        schemas = {
            "gorilla": self._schema(("threat_line", "line_threat", "state", 1)),
            "hawk": self._schema(
                ("reframed_count", "line_threat", "state", 1),
                ("reframed_completable", "line_threat", "agent_relative", 1),
            ),
        }
        assert derive_triads(schemas) == {
            "line_threat": {"gorilla": "threat_line",
                            "hawk": "reframed_completable"}
        }

    def test_triads_drop_single_basis_families(self):
        """A family only one basis has is not a cross-basis comparison."""
        from lib.sae.eval import derive_triads

        schemas = {
            "tiger": self._schema(
                ("tiger_pool_safe_count", "pool_reasoning", "agent_relative", 4)),
            "gorilla": self._schema(
                ("cell_occupancy", "board_occupancy", "state", 16)),
        }
        assert derive_triads(schemas) == {}

    def test_family_means_are_weighted_by_bsp_count(self):
        """Categories differ in size by an order of magnitude, so an unweighted
        mean over categories would be a different statistic than the mean over
        BSPs -- and would over-weight tiny categories like ``global`` (1 BSP)."""
        from lib.sae.eval import aggregate_per_category_by_family

        schema = self._schema(
            ("threat_line", "line_threat", "state", 40),
            ("reframed_any_threat", "line_threat", "state_any", 10),
        )
        per_category = {
            "threat_line": {"count": 40, "mean_mcc": 1.0},
            "reframed_any_threat": {"count": 10, "mean_mcc": 0.0},
        }
        out = aggregate_per_category_by_family(per_category, schema)
        assert out["line_threat"]["count"] == 50
        # Weighted: (40*1.0 + 10*0.0) / 50 = 0.8, NOT the unweighted 0.5.
        assert out["line_threat"]["mean_mcc"] == pytest.approx(0.8)
        assert out["line_threat"]["categories"] == [
            "reframed_any_threat", "threat_line"]

    def test_family_rollup_ignores_categories_the_schema_does_not_declare(self):
        from lib.sae.eval import aggregate_per_category_by_family

        schema = self._schema(("threat_line", "line_threat", "state", 4))
        out = aggregate_per_category_by_family(
            {"threat_line": {"count": 4, "mean_mcc": 0.5},
             "mystery_cat": {"count": 9, "mean_mcc": 0.9}},
            schema,
        )
        assert set(out) == {"line_threat"}
        assert out["line_threat"]["count"] == 4


class TestShippedSchemasCarryFamilies:
    """The real schemas on disk must be stamped, or every family rollup in the
    analysis scripts degrades to an error message."""

    def test_each_basis_schema_is_stamped(self):
        import json

        from lib.sae.eval import category_families

        data_dir = PROJECT_ROOT / "data" / "quarto"
        for basis, count in (("gorilla", 164), ("hawk", 173), ("tiger", 36)):
            path = data_dir / f"bsp_schema-{basis}_{count}.json"
            if not path.exists():
                pytest.skip(f"{path.name} not on this box")
            schema = json.loads(path.read_text(encoding="utf-8"))
            fams = category_families(schema)
            declared = set(schema["categories"])
            assert declared == set(fams), (
                f"{path.name}: unstamped categories "
                f"{sorted(declared - set(fams))} -- run "
                f"scripts/stamp_concept_families.py")


class TestSchemaNamingConvention:
    """One schema per basis, and every animal resolves to it.

    A BSP *schema* is distribution-independent, so the convention is
    ``bsp_schema-<basis>_<count>.json`` with NO champion or pool suffix
    (CLAUDE.md, Domain conventions). Suffixed copies used to be minted per
    champion and per unified-pool size; each was byte-identical, each was
    preferred over the basis schema by ``resolve_schema_path``, and each was a
    place for the two to silently drift apart.
    """

    @pytest.mark.parametrize(
        "animal,expected",
        [
            ("gorilla", "gorilla"),
            ("gorillaVe", "gorilla"),      # champion suffix (upper-case initial)
            ("tigerTa", "tiger"),
            ("gorilla677k", "gorilla"),    # unified-pool suffix (<digits>k)
            ("hawk156k", "hawk"),
            ("fox", "fox"),
        ],
    )
    def test_animal_to_basis(self, animal, expected):
        from lib.sae.eval import animal_to_basis

        assert animal_to_basis(animal) == expected

    def test_pool_suffixed_animal_resolves_to_the_basis_schema(self, tmp_path):
        """The reason the duplicates existed: before the numeric suffix was
        handled, `gorilla677k` resolved to itself and the basis fallback never
        fired, so unify_positions.py had to write a suffixed copy to be
        findable at all."""
        from lib.sae.eval import resolve_schema_path

        basis = tmp_path / "bsp_schema-gorilla_164.json"
        basis.write_text("{}", encoding="utf-8")
        assert resolve_schema_path(tmp_path, "gorilla677k") == basis
        assert resolve_schema_path(tmp_path, "gorillaYb") == basis

    def test_no_suffixed_schema_files_on_disk(self):
        """Exactly one schema per basis. A second file for the same basis makes
        `resolve_schema_path`'s sorted()[0] pick load-bearing and arbitrary."""
        import collections
        import re

        data_dir = PROJECT_ROOT / "data" / "quarto"
        if not data_dir.exists():
            pytest.skip("no quarto data dir on this box")

        by_basis = collections.defaultdict(list)
        for path in sorted(data_dir.glob("bsp_schema-*.json")):
            m = re.match(r"bsp_schema-(.+)_(\d+)\.json$", path.name)
            assert m, f"{path.name} does not match bsp_schema-<name>_<count>.json"
            from lib.sae.eval import animal_to_basis

            name = m.group(1)
            assert animal_to_basis(name) == name, (
                f"{path.name} is suffixed ('{name}'); schemas are keyed by "
                f"BASIS only -- labels carry the suffix, schemas do not")
            by_basis[name].append(path.name)

        dupes = {b: names for b, names in by_basis.items() if len(names) > 1}
        assert not dupes, f"multiple schemas for one basis: {dupes}"
