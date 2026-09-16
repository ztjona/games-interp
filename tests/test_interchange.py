"""Tests for lib/sae/interchange.py -- the game-agnostic core of 3B-causal.

The Tier-A controls (pre-registration S7, as amended pre-data by amendment 1
section A1: A1a, A1b, A2, A3) are unit tests here, on toy models where the right
answer is known exactly; the runner re-checks them on the real champion as its
dry-run gate.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.sae.interchange import (  # noqa: E402
    ArmResult,
    ForwardHookReadout,
    LinearReluReadout,
    benjamini_hochberg,
    bootstrap_iia_star,
    chance_corrected,
    check_encoder_is_per_sample,
    classify,
    empirical_p,
    frequency_matched_sets,
    legal_argmax,
    orbit_folds,
    patch_direction,
    patch_full,
    patch_latents,
    patch_subspace,
    random_unit_directions,
    score_pairs,
    train_das_direction,
)


class Toy(nn.Module):
    """hook module (fc1) -> functional ReLU -> linear head: the S4 fc1 shape."""

    def __init__(self, d_in=5, d=8, a=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(d_in, d)
        self.head = nn.Linear(d, a)

    def forward(self, x):
        return self.head(torch.relu(self.fc1(x)))


def _readouts(model):
    closed = LinearReluReadout(model.head.weight, model.head.bias)
    fwd = ForwardHookReadout(model, model.fc1, model.head,
                             run=lambda inputs: model(*inputs))
    return closed, fwd


# ---------------------------------------------------------------------------
# Tier A -- implementation controls
# ---------------------------------------------------------------------------


class TestTierAControls:
    def test_a1a_null_space_of_head_is_exactly_inert_on_h(self):
        torch.manual_seed(1)
        W, b = torch.randn(4, 8), torch.randn(4)
        ro = LinearReluReadout(W, b)
        _, _, Vh = torch.linalg.svd(W, full_matrices=True)
        null = Vh[4:]                                   # rank 4 -> 4 null dirs
        h = torch.rand(32, 8)
        for v in null:
            assert (ro.head(h + 3.0 * v) - ro.head(h)).abs().max() < 1e-5

    def test_a1b_dead_units_are_exactly_inert_on_z(self):
        """Changing only units that are negative, and stay negative, changes
        nothing -- the z-space exact control."""
        torch.manual_seed(2)
        ro = LinearReluReadout(torch.randn(4, 8), torch.randn(4))
        z = torch.randn(64, 8)
        dead = z < 0
        z2 = torch.where(dead, z - torch.rand_like(z), z)   # push dead further down
        assert (ro(z2) - ro(z)).abs().max() < 1e-6

    def test_a2_full_patch_reproduces_the_source_decision(self):
        torch.manual_seed(3)
        ro = LinearReluReadout(torch.randn(4, 8), torch.randn(4))
        z_b, z_s = torch.randn(50, 8), torch.randn(50, 8)
        legal = torch.ones(50, 4, dtype=torch.bool)
        assert torch.equal(legal_argmax(ro(patch_full(z_b, z_s)), legal),
                           legal_argmax(ro(z_s), legal))

    def test_a3_forward_hook_path_equals_closed_form(self):
        """Replace the hooked module's output, run the real model: must equal
        relu(z) @ W.T + b for ANY z, including one the trunk never produced."""
        model = Toy().eval()
        closed, fwd = _readouts(model)
        x = torch.randn(40, 5)
        z_arbitrary = torch.randn(40, 8) * 3.0
        with torch.no_grad():
            assert (closed(z_arbitrary) - fwd(z_arbitrary, (x,))).abs().max() < 1e-5
            # and with the real hook value, both equal the plain forward pass
            z_real = model.fc1(x)
            assert (fwd(z_real, (x,)) - model(x)).abs().max() < 1e-5

    def test_forward_hook_readout_removes_its_hooks(self):
        model = Toy().eval()
        _, fwd = _readouts(model)
        x = torch.randn(4, 5)
        fwd(torch.zeros(4, 8), (x,))
        assert not model.fc1._forward_hooks and not model.head._forward_hooks


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


class TestPatches:
    def test_direction_patch_replaces_only_the_component(self):
        torch.manual_seed(4)
        z_b, z_s, w = torch.randn(10, 6), torch.randn(10, 6), torch.randn(6)
        out = patch_direction(z_b, z_s, w)
        wn = w / w.norm()
        assert torch.allclose(out @ wn, z_s @ wn, atol=1e-5)
        perp = lambda z: z - (z @ wn).unsqueeze(1) * wn  # noqa: E731
        assert torch.allclose(perp(out), perp(z_b), atol=1e-5)

    def test_subspace_patch_is_basis_invariant(self):
        torch.manual_seed(5)
        z_b, z_s = torch.randn(10, 6), torch.randn(10, 6)
        U = torch.randn(6, 2)
        mix = torch.tensor([[2.0, 1.0], [0.5, -1.0]])      # another basis, same span
        assert torch.allclose(patch_subspace(z_b, z_s, U),
                              patch_subspace(z_b, z_s, U @ mix), atol=1e-5)

    def test_latent_patch_is_error_preserving(self):
        """J = all latents moves z by exactly recon(s) - recon(b): the decoder
        bias cancels and the base's SAE error term survives untouched."""
        torch.manual_seed(6)
        d, k = 6, 10
        W_enc, W_dec, b_dec = torch.randn(d, k), torch.randn(k, d), torch.randn(d)
        encode = lambda z: torch.relu(z @ W_enc)  # noqa: E731
        recon = lambda z: encode(z) @ W_dec + b_dec  # noqa: E731
        z_b, z_s = torch.randn(8, d), torch.randn(8, d)
        out = patch_latents(z_b, z_s, encode, W_dec, list(range(k)))
        assert torch.allclose(out - z_b, recon(z_s) - recon(z_b), atol=1e-4)
        assert torch.equal(patch_latents(z_b, z_s, encode, W_dec, []), z_b)

    def test_per_sample_encoder_guard(self):
        torch.manual_seed(7)
        W = torch.randn(6, 12)
        per_sample = lambda z: torch.relu(z @ W)  # noqa: E731

        def batch_topk(z, k=3):                    # keeps the batch-wide top k*B
            a = torch.relu(z @ W)
            thr = a.flatten().topk(k * len(z)).values[-1]
            return a * (a >= thr)

        z = torch.randn(20, 6)
        check_encoder_is_per_sample(per_sample, z)
        with pytest.raises(ValueError, match="batch-dependent"):
            check_encoder_is_per_sample(batch_topk, z)


# ---------------------------------------------------------------------------
# Decisions and scores
# ---------------------------------------------------------------------------


class TestScoring:
    def test_legal_argmax_never_picks_an_illegal_action(self):
        logits = torch.tensor([[9.0, 1.0, 2.0], [0.0, 5.0, 7.0]])
        legal = torch.tensor([[False, True, True], [True, True, False]])
        assert legal_argmax(logits, legal).tolist() == [2, 1]
        with pytest.raises(ValueError, match="at least one legal"):
            legal_argmax(logits, torch.zeros(2, 3, dtype=torch.bool))

    def test_score_pairs_counts_iia_r0_and_flips(self):
        base = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        pat = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        target = torch.tensor([[False, True]] * 4)
        legal = torch.ones(4, 2, dtype=torch.bool)
        s = score_pairs(base, pat, target, legal)
        assert (s.n, s.iia, s.r0) == (4, 0.75, 0.25)
        assert s.iia_star == pytest.approx((0.75 - 0.25) / 0.75)
        assert s.flip_rate == 0.5

    def test_chance_correction_edges(self):
        assert chance_corrected(1.0, 1.0) is None
        assert chance_corrected(0.25, 0.25) == 0.0
        assert chance_corrected(0.0, 0.5) == -1.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_empirical_p_and_parametric_tail(self):
        null = torch.linspace(0, 1, 101)
        p, flag = empirical_p(0.5, null)
        assert not flag and p == pytest.approx(52 / 102)
        p, flag = empirical_p(5.0, null)
        assert flag and 0 < p < 1 / 102

    def test_benjamini_hochberg_known_case(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        # Classic BH example at q = 0.05: the first two are rejected.
        assert benjamini_hochberg(p, 0.05) == [True, True] + [False] * 8
        assert benjamini_hochberg([float("nan"), 0.001]) == [False, True]

    def test_frequency_matched_sets(self):
        freq = torch.tensor([0.10, 0.11, 0.50, 0.52, 0.09, 0.30, 0.0])
        alive = freq > 0
        g = torch.Generator().manual_seed(0)
        draws = frequency_matched_sets([0, 2], freq, alive, 50, g, tol=0.2)
        for d in draws:
            assert len(d) == 2 and 0 not in d.tolist() and 2 not in d.tolist()
            assert int(d[0]) in (1, 4) and int(d[1]) == 3

    def test_orbit_folds_keep_groups_whole(self):
        groups = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4, 5])
        folds = orbit_folds(groups, 3, seed=0)
        for gid in torch.unique(groups):
            assert len(torch.unique(folds[groups == gid])) == 1

    def test_bootstrap_ci_brackets_the_estimate(self):
        torch.manual_seed(8)
        n = 400
        hb = torch.rand(n) < 0.2
        hp = torch.rand(n) < 0.7
        groups = torch.arange(n) // 2
        lo, hi = bootstrap_iia_star(hp, hb, groups, n_boot=300, seed=1)
        point = chance_corrected(float(hp.float().mean()), float(hb.float().mean()))
        assert lo <= point <= hi

    def test_random_unit_directions_are_unit(self):
        v = random_unit_directions(7, 5, torch.Generator().manual_seed(0))
        assert torch.allclose(v.norm(dim=1), torch.ones(7), atol=1e-6)


# ---------------------------------------------------------------------------
# DAS-1 -- the positive control must be able to find a planted variable
# ---------------------------------------------------------------------------


class TestDAS:
    def test_das_recovers_a_planted_causal_direction(self):
        """Action 0's logit reads unit 3 only. Bases have unit 3 off, sources
        have it on; every other unit is noise. DAS-1 must find e_3, and a
        random direction must not transfer the decision."""
        torch.manual_seed(9)
        d, a, n = 10, 4, 600
        W = torch.randn(a, d) * 0.1
        W[0] = 0.0
        W[0, 3] = 6.0
        ro = LinearReluReadout(W, torch.zeros(a))
        z_b = torch.randn(n, d)
        z_s = torch.randn(n, d)
        z_b[:, 3] = -1.0
        z_s[:, 3] = 2.0
        target = torch.zeros(n, a, dtype=torch.bool)
        target[:, 0] = True
        legal = torch.ones(n, a, dtype=torch.bool)
        train, test = slice(0, 400), slice(400, n)
        w = train_das_direction(z_b[train], z_s[train], ro, target[train],
                                legal[train], steps=250, lr=0.1, seed=0)
        assert abs(float(w[3])) > 0.9
        s = score_pairs(ro(z_b[test]), ro(patch_direction(z_b[test], z_s[test], w)),
                        target[test], legal[test])
        assert s.iia_star > 0.9
        rnd = random_unit_directions(1, d, torch.Generator().manual_seed(3))[0]
        s0 = score_pairs(ro(z_b[test]), ro(patch_direction(z_b[test], z_s[test], rnd)),
                         target[test], legal[test])
        assert s0.iia_star < 0.5


# ---------------------------------------------------------------------------
# Verdict rule 3B.C1
# ---------------------------------------------------------------------------


def _arm(n=200, s=0.5, sig=True, **kw):
    return ArmResult(n=n, iia_star=s, significant=sig, **kw)


class TestVerdictRule:
    @pytest.mark.parametrize("on,spec,off,verdict", [
        (_arm(), _arm(s=0.0, sig=False), _arm(n=80), "concept-consistent"),
        (_arm(), _arm(s=0.0, sig=False), _arm(n=10), "concept-consistent (on-only)"),
        (_arm(), _arm(s=0.0, sig=False), None, "concept-consistent (on-only)"),
        (_arm(), _arm(s=0.0, sig=False), _arm(n=80, s=0.05, sig=False), "install-only"),
        (_arm(s=0.05, sig=False), _arm(s=0.0, sig=False), _arm(n=80), "remove-only"),
        (_arm(), _arm(s=0.4, sig=True), _arm(n=80), "context-blind"),
        (_arm(s=0.10, sig=True), _arm(s=0.0, sig=False), None, "inert"),   # below floor
        (_arm(s=-0.3, sig=False, below_null_p5=True), _arm(sig=False), None,
         "anti-consistent"),
        (_arm(s=0.0, sig=False, flip_significant=True), _arm(sig=False), None,
         "off-target"),
        (_arm(n=99), _arm(), None, "underpowered"),
        (_arm(), _arm(n=50), None, "underpowered"),
    ])
    def test_truth_table(self, on, spec, off, verdict):
        assert classify(on, spec, off) == verdict

    def test_floor_is_inclusive(self):
        assert classify(_arm(s=0.20), _arm(s=0.0, sig=False), None) \
            == "concept-consistent (on-only)"
        assert math.isclose(0.20, 0.20)


class TestFeasibility:
    """Wave 1b S4.4: a set enters only if >= 50% of concepts are powered in every arm."""

    @staticmethod
    def _power(on, spec, off):
        return {"switch_on": {"n": on}, "specificity": {"n": spec}, "switch_off": {"n": off}}

    def test_every_arm_must_reach_its_own_n_min(self):
        from lib.sae.interchange import underpowered_arms
        pw = {"a": self._power(100, 100, 50), "b": self._power(100, 100, 49),
              "c": self._power(99, 1000, 1000)}
        assert underpowered_arms(pw) == {"a": [], "b": ["switch_off"], "c": ["switch_on"]}

    def test_half_is_enough_and_less_is_not(self):
        from lib.sae.interchange import feasibility
        ok, bad = self._power(500, 500, 500), self._power(500, 500, 0)
        assert feasibility({"a": ok, "b": bad})["passed"]
        f = feasibility({"a": ok, "b": bad, "c": bad})
        assert not f["passed"] and f["powered_in_every_arm"] == 1
        assert f["powered_per_arm"] == {"switch_on": 3, "specificity": 3, "switch_off": 1}


class TestRuleC2:
    """Wave 1b S9: context-blind = installs AND E significant AND rho >= 0.5."""

    @staticmethod
    def _spec(excess, sig=True):
        return _arm(s=None, sig=sig, excess=excess)

    def test_small_relative_leak_is_not_context_blind(self):
        on = _arm(s=0.8)
        assert classify(on, self._spec(0.3), None, rule="3B.C2") == "concept-consistent (on-only)"
        assert classify(on, self._spec(0.3), None, rule="3B.C1") == "context-blind"

    def test_rho_threshold_is_inclusive(self):
        assert classify(_arm(s=0.8), self._spec(0.4), None, rule="3B.C2") == "context-blind"
        assert classify(_arm(s=0.8), self._spec(0.39), _arm(n=80), rule="3B.C2") == "concept-consistent"

    def test_insignificant_or_undefined_excess_is_not_context_blind(self):
        assert classify(_arm(s=0.8), self._spec(0.9, sig=False), None, rule="3B.C2") \
            == "concept-consistent (on-only)"
        assert classify(_arm(s=0.8), self._spec(None), None, rule="3B.C2") \
            == "concept-consistent (on-only)"

    def test_relative_leak(self):
        from lib.sae.interchange import relative_leak
        assert relative_leak(self._spec(0.2), _arm(s=0.4)) == pytest.approx(0.5)
        assert relative_leak(self._spec(0.2), _arm(s=0.0)) is None
        assert relative_leak(self._spec(0.2), _arm(s=-0.1)) is None

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError):
            classify(_arm(), _arm(), None, rule="3B.C9")

    @pytest.mark.parametrize("rule", ["3B.C1", "3B.C2"])
    def test_off_target_needs_an_insignificant_switch_on(self, rule):
        """Amendment 1 A3: significant-but-below-floor with a significant flip rate
        is inert, not off-target (the code omitted this until 2026-09-14)."""
        spec = _arm(s=0.0, sig=False)
        assert classify(_arm(s=0.1, sig=True, flip_significant=True), spec, None, rule=rule) == "inert"
        assert classify(_arm(s=0.1, sig=False, flip_significant=True), spec, None, rule=rule) \
            == "off-target"


class TestCovarianceMatchedNull:
    def test_unit_norm_and_inside_the_span_of_real_differences(self):
        from lib.sae.interchange import covariance_matched_directions
        g = torch.Generator().manual_seed(0)
        basis = torch.linalg.qr(torch.randn(64, 3, generator=g))[0]          # (64, 3)
        delta = torch.randn(500, 3, generator=g) @ basis.T + 5.0            # offset: centred away
        w = covariance_matched_directions(delta, 200, torch.Generator().manual_seed(1))
        assert w.shape == (200, 64)
        assert torch.allclose(w.norm(dim=1), torch.ones(200), atol=1e-5)
        resid = w - (w @ basis) @ basis.T
        assert resid.abs().max() < 1e-4

    def test_follows_the_covariance_not_isotropy(self):
        from lib.sae.interchange import covariance_matched_directions
        g = torch.Generator().manual_seed(0)
        delta = torch.randn(2000, 16, generator=g) * torch.tensor([10.0] + [1.0] * 15)
        w = covariance_matched_directions(delta, 500, torch.Generator().manual_seed(2))
        assert w[:, 0].abs().mean() > 3 * w[:, 1:].abs().mean()


class TestDasMulti:
    @staticmethod
    def _setup(seed=0, n=40, d=12, a=6):
        g = torch.Generator().manual_seed(seed)
        W, b = torch.randn(a, d, generator=g), torch.randn(a, generator=g)
        ro = LinearReluReadout(W, b)
        zb, zs = torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)
        legal = torch.ones(n, a, dtype=torch.bool)
        tgt = torch.zeros(n, a, dtype=torch.bool)
        tgt[torch.arange(n), torch.randint(a, (n,), generator=g)] = True
        return ro, zb, zs, tgt, legal

    def test_one_batch_is_the_single_kind_trainer(self):
        from lib.sae.interchange import DasBatch, train_das_direction, train_das_direction_multi
        ro, zb, zs, tgt, legal = self._setup()
        w1 = train_das_direction(zb, zs, ro, tgt, legal, steps=50, seed=3)
        w2 = train_das_direction_multi([DasBatch(zb, zs, tgt, legal)], ro, steps=50, seed=3)
        assert torch.equal(w1, w2)

    def test_each_kind_weighs_the_same_whatever_its_size(self):
        """Repeating one kind's rows 10x leaves the objective, hence w, unchanged."""
        from lib.sae.interchange import DasBatch, train_das_direction_multi
        ro, zb, zs, tgt, legal = self._setup()
        a = DasBatch(zb[:10], zs[:10], tgt[:10], legal[:10])
        b = DasBatch(zb[10:], zs[10:], tgt[10:], legal[10:])
        rep = lambda x: x.repeat(10, 1)  # noqa: E731
        b10 = DasBatch(rep(zb[10:]), rep(zs[10:]), rep(tgt[10:]), rep(legal[10:]))
        empty = DasBatch(zb[:0], zs[:0], tgt[:0], legal[:0])
        w = train_das_direction_multi([a, b], ro, steps=50, seed=4)
        assert torch.allclose(w, train_das_direction_multi([a, b10], ro, steps=50, seed=4), atol=1e-5)
        assert torch.equal(w, train_das_direction_multi([a, empty, b], ro, steps=50, seed=4))


class TestRuleC3:
    """Wave 1c S9: the verdict is about installing; removal is a separate label."""

    @staticmethod
    def _spec(excess=None, sig=False):
        return _arm(s=None, sig=sig, excess=excess)

    def test_installing_is_the_verdict_whatever_removal_does(self):
        for off in (None, _arm(n=80, s=0.6), _arm(n=80, s=0.0, sig=False), _arm(n=10)):
            assert classify(_arm(s=0.8), self._spec(), off, rule="3B.C3") == "installs"

    def test_context_blind_as_c2_and_no_remove_only(self):
        assert classify(_arm(s=0.8), self._spec(0.5, sig=True), None, rule="3B.C3") == "context-blind"
        # removes but does not install: 3B.C2 says remove-only; 3B.C3 has no such verdict
        on, off = _arm(s=0.05, sig=False), _arm(n=80, s=0.6)
        assert classify(on, self._spec(), off, rule="3B.C2") == "remove-only"
        assert classify(on, self._spec(), off, rule="3B.C3") == "inert"

    def test_removal_label(self):
        from lib.sae.interchange import removal_label
        assert removal_label(None) == "underpowered"
        assert removal_label(_arm(n=49, s=0.9)) == "underpowered"
        assert removal_label(_arm(n=50, s=0.20)) == "removes"
        assert removal_label(_arm(n=50, s=0.19)) == "does not remove"
        assert removal_label(_arm(n=50, s=0.9, sig=False)) == "does not remove"

    def test_feasibility_on_chosen_arms(self):
        from lib.sae.interchange import feasibility
        pw = {"a": {"switch_on": {"n": 500}, "specificity": {"n": 500}, "switch_off": {"n": 0}}}
        assert not feasibility(pw)["passed"]
        assert feasibility(pw, ("switch_on", "specificity"))["passed"]


class TestDasSubspace:
    def test_orthonormal_seeded_and_better_than_its_start(self):
        from lib.sae.interchange import DasBatch, train_das_subspace_multi
        ro, zb, zs, tgt, legal = TestDasMulti._setup(n=60, d=12)
        b = [DasBatch(zb, zs, tgt, legal)]
        Q = train_das_subspace_multi(b, ro, 3, steps=150, seed=5)
        assert Q.shape == (12, 3)
        assert torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5)
        assert torch.equal(Q, train_das_subspace_multi(b, ro, 3, steps=150, seed=5))

        def loss(Qm):
            logits = ro(zb + ((zs - zb) @ Qm) @ Qm.T)
            return float((torch.logsumexp(logits, 1) - logits[tgt]).mean())
        Q0 = train_das_subspace_multi(b, ro, 3, steps=0, seed=5)
        assert loss(Q) < loss(Q0)

    def test_all_folds_at_once_equal_one_fold_at_a_time(self):
        """Stacking the folds' parameters under one Adam changes the overhead,
        not the optimisation: every fold's subspace matches a separate fit on
        its training rows (same seed, same per-kind averaging)."""
        from lib.sae.interchange import (DasBatch, train_das_subspace_folds,
                                         train_das_subspace_multi)
        ro, zb, zs, tgt, legal = TestDasMulti._setup(n=90, d=12)
        g = torch.Generator().manual_seed(9)
        batches = [DasBatch(zb[:60], zs[:60], tgt[:60], legal[:60]),
                   DasBatch(zb[60:], zs[60:], tgt[60:], legal[60:])]
        folds = [torch.randint(3, (60,), generator=g), torch.randint(3, (30,), generator=g)]
        folds[1][folds[1] == 2] = 0                       # kind 2 has no rows in fold 2
        seeds = [11, 12, 13]
        Qs = train_das_subspace_folds(batches, folds, 3, ro, 2, seeds, steps=60)
        for f in range(3):
            sub = [DasBatch(b.z_b[fo != f], b.z_s[fo != f], b.target[fo != f], b.legal[fo != f])
                   for b, fo in zip(batches, folds)]
            Q1 = train_das_subspace_multi(sub, ro, 2, steps=60, seed=seeds[f])
            assert torch.allclose(Qs[f] @ Qs[f].T, Q1 @ Q1.T, atol=1e-4)

    def test_random_subspaces_live_in_the_span_of_real_differences(self):
        from lib.sae.interchange import covariance_matched_subspaces
        g = torch.Generator().manual_seed(0)
        basis = torch.linalg.qr(torch.randn(32, 5, generator=g))[0]
        delta = torch.randn(400, 5, generator=g) @ basis.T
        Qs = covariance_matched_subspaces(delta, 50, 3, torch.Generator().manual_seed(1))
        assert Qs.shape == (50, 32, 3)
        eye = torch.eye(3).expand(50, 3, 3)
        assert torch.allclose(Qs.transpose(1, 2) @ Qs, eye, atol=1e-5)
        assert (Qs - basis @ (basis.T @ Qs)).abs().max() < 1e-4
