"""Tests for the Phase 3A dilution diagnostic (lib/sae/dilution.py).

Two layers:
  1. classify() truth-table -- the verdict logic is a pure function of the
     metrics dict, so the four branches are pinned directly and deterministically.
  2. diagnose_concept() end-to-end on synthetic planted concepts -- confirms the
     metrics that feed classify() come out with the right qualitative shape for
     captured / absent / spread structure.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.sae.dilution import (
    GLOSSARY,
    VERDICT_GLOSSARY,
    DilutionConfig,
    classify,
    classify_with_stability,
    diagnose_concept,
    participation_ratio,
    solo_frac_of,
)
from scripts.dilution_diagnostic import VERDICTS  # noqa: E402


CFG = DilutionConfig(max_rows=10_000)  # small; tests use N <= 6000


def _metrics(**over):
    """A 'recoverable, spread' baseline metrics dict; override per test."""
    base = dict(
        asymptote_r2=0.40, null_r2=0.00, knee_k=12, community_size=16,
        neg_coupling_frac=0.30, support_overlap=0.40, intrinsic_dim=3.0,
        solo_frac=0.25,
    )
    base.update(over)
    return base


# --- 1. classify() truth table ------------------------------------------------

def test_classify_absent_by_margin():
    assert classify(_metrics(asymptote_r2=0.05, null_r2=0.045), CFG) == "absent"


def test_classify_absent_by_floor():
    assert classify(_metrics(asymptote_r2=0.01, null_r2=-0.01), CFG) == "absent"


def test_classify_captured():
    m = _metrics(asymptote_r2=0.5, null_r2=0.0, knee_k=1, community_size=2)
    assert classify(m, CFG) == "captured"


def test_classify_tiled_shape_is_now_spread():
    """Rule 3A.3 collapsed diluted/tiled into `spread`; the shape still must not
    be read as captured or absent."""
    m = _metrics(support_overlap=0.05, neg_coupling_frac=0.7)
    assert classify(m, CFG) == "spread"


def test_classify_spread():
    # recoverable, many latents, overlapping supports, mixed-sign couplings
    m = _metrics(support_overlap=0.4, neg_coupling_frac=0.3, knee_k=12)
    assert classify(m, CFG) == "spread"


def test_classify_spread_regardless_of_overlap():
    # Pre-3A.3 this distinguished diluted from tiled; both are now `spread`.
    m = _metrics(support_overlap=0.5, neg_coupling_frac=0.8)
    assert classify(m, CFG) == "spread"


# --- 1b. rule 3A.2 calibration ------------------------------------------------
# These pin the regression found in the 2026-07-27 run: the supervised anchored
# positive control (I04) came out 'diluted' on 22/23 concepts under rule 3A.1
# despite community_size 2 and intrinsic_dim 1.01, because knee_k was inflated
# by noise-level creep in the tail of the R2 curve. See the classify() docstring
# and docs/diary/2026-07-27_3A-dilution-results.md.

def test_classify_captured_via_solo_frac_despite_large_knee():
    """The I04-champYb anchored shape: one latent carries ~79% of everything
    recoverable and the community is 1-dimensional -> captured, even though the
    90%-of-asymptote crossing sits at k=17."""
    m = _metrics(asymptote_r2=0.51, null_r2=0.0, knee_k=17, community_size=2,
                 intrinsic_dim=1.01, solo_frac=0.79)
    assert classify(m, CFG) == "captured"


def test_classify_captured_via_knee_despite_large_community():
    """The F04-champYb/gorillaYb shape: latent #1 reaches 99% of the asymptote,
    but 9 redundant latents co-fire. Redundancy is not dilution."""
    m = _metrics(asymptote_r2=0.53, null_r2=0.0, knee_k=1, community_size=9,
                 intrinsic_dim=2.30, solo_frac=0.99)
    assert classify(m, CFG) == "captured"


def test_classify_still_spread_when_signal_is_spread():
    """The unsupervised shape must NOT drift to captured: no single latent
    dominates and the community spans several effective dimensions."""
    m = _metrics(asymptote_r2=0.38, null_r2=0.0, knee_k=8, community_size=10,
                 intrinsic_dim=2.99, solo_frac=0.22)
    assert classify(m, CFG) == "spread"


def test_classify_high_solo_frac_but_high_dim_is_not_captured():
    """Both halves of the new condition are required."""
    m = _metrics(solo_frac=0.95, intrinsic_dim=6.0)
    assert classify(m, CFG) == "spread"


def test_classify_low_dim_but_low_solo_frac_is_not_captured():
    m = _metrics(solo_frac=0.30, intrinsic_dim=1.0)
    assert classify(m, CFG) == "spread"


def test_rule_3a1_captured_verdicts_are_preserved():
    """3A.2 only ADDS a sufficient condition -- nothing that was captured under
    the old rule may become non-captured."""
    m = _metrics(asymptote_r2=0.5, null_r2=0.0, knee_k=1, community_size=2,
                 solo_frac=0.0, intrinsic_dim=9.0)
    assert classify(m, CFG) == "captured"


def test_absent_still_wins_over_captured():
    """A concentrated but signal-free concept is absent, not captured."""
    m = _metrics(asymptote_r2=0.01, null_r2=0.0, solo_frac=1.0, intrinsic_dim=1.0)
    assert classify(m, CFG) == "absent"


def test_solo_frac_backfills_from_curve_for_old_reports():
    """Reports written before 3A.2 have no solo_frac; it must be recoverable
    from the stored curve so `reclassify` never needs the _h caches."""
    m = {"curve": [0.40, 0.45, 0.50], "asymptote_r2": 0.50}
    assert solo_frac_of(m) == 0.8
    assert solo_frac_of({"curve": [], "asymptote_r2": 0.0}) == 0.0
    # an explicit value always wins over the back-fill
    assert solo_frac_of({"solo_frac": 0.1, "curve": [0.4], "asymptote_r2": 0.5}) == 0.1


def test_glossary_covers_every_reported_metric_and_verdict():
    """The embedded glossary is the contract that makes a report readable on its
    own -- it must not silently fall behind the metrics dict."""
    rng = np.random.default_rng(7)
    N, d = 3000, 30
    y = (rng.random(N) < 0.3).astype(np.float32)
    h = _make_codes(N, d, rng)
    h[:, 0] = y * 1.0
    m = diagnose_concept(torch.tensor(h), torch.tensor(y), CFG)
    reported = set(m) - {"verdict", "base_rate", "n_candidates", "solo_r2",
                         "random_asymptote_r2"}
    assert reported <= set(GLOSSARY), f"missing from GLOSSARY: {reported - set(GLOSSARY)}"
    # `diluted`/`tiled` are retained as RETIRED entries so a pre-3A.3
    # report stays readable; the live set is three-way.
    assert {"absent", "captured", "spread"} <= set(VERDICT_GLOSSARY)
    assert set(VERDICTS) == {"absent", "captured", "spread"}


# --- 2. diagnose_concept() end-to-end -----------------------------------------

def _make_codes(N, d_dict, rng):
    """Background sparse codes: each latent fires ~5% of the time at value ~1."""
    fire = (rng.random((N, d_dict)) < 0.05).astype(np.float32)
    return fire * rng.uniform(0.5, 1.5, size=(N, d_dict)).astype(np.float32)


def test_diagnose_captured_single_latent():
    rng = np.random.default_rng(0)
    N, d = 6000, 40
    y = (rng.random(N) < 0.3).astype(np.float32)
    h = _make_codes(N, d, rng)
    # latent 0 is a clean monosemantic detector of y
    h[:, 0] = y * rng.uniform(0.8, 1.2, size=N).astype(np.float32)
    m = diagnose_concept(torch.tensor(h), torch.tensor(y), CFG)
    assert m["asymptote_r2"] - m["null_r2"] > 0.2   # clearly recoverable
    assert m["knee_k"] <= CFG.captured_k            # one latent suffices
    assert m["verdict"] == "captured"


def test_diagnose_absent_independent():
    rng = np.random.default_rng(1)
    N, d = 6000, 40
    y = (rng.random(N) < 0.3).astype(np.float32)
    h = _make_codes(N, d, rng)  # codes carry no info about y
    m = diagnose_concept(torch.tensor(h), torch.tensor(y), CFG)
    assert m["asymptote_r2"] - m["null_r2"] < CFG.absent_margin
    assert m["verdict"] == "absent"


def test_diagnose_disjoint_latents_is_geometric():
    rng = np.random.default_rng(2)
    N, d = 6000, 40
    y = (rng.random(N) < 0.5).astype(np.float32)
    h = _make_codes(N, d, rng)
    # y=1 rows are partitioned across 8 mutually-exclusive latents (disjoint
    # supports, competing -> negative couplings). None fires unless y=1.
    pos = np.where(y > 0)[0]
    which = rng.integers(0, 8, size=pos.size)
    for j in range(8):
        h[:, j] = 0.0
        sel = pos[which == j]
        h[sel, j] = rng.uniform(0.8, 1.2, size=sel.size).astype(np.float32)
    m = diagnose_concept(torch.tensor(h), torch.tensor(y), CFG)
    assert m["asymptote_r2"] - m["null_r2"] > 0.2   # recoverable in aggregate
    assert m["support_overlap"] < 0.2               # disjoint supports
    assert m["verdict"] == "spread"     # geometric, not captured/absent
    assert m["verdict"] != "captured"


def test_nested_ols_matches_sklearn_refits():
    """The incremental Gram solve replaced m independent LinearRegression fits
    for speed; it must be numerically the same estimator, not an approximation."""
    from lib.sae.dilution import _NestedOLS, _r2_heldout

    rng = np.random.default_rng(11)
    n, m = 4000, 12
    X = rng.normal(size=(n, m))
    y = X[:, 0] * 0.8 + X[:, 3] * 0.3 + rng.normal(scale=0.5, size=n)
    Xtr, Xte, ytr, yte = X[:3000], X[3000:], y[:3000], y[3000:]

    fast = _NestedOLS(Xtr, ytr, Xte, yte).curve(ytr)
    slow = [_r2_heldout(Xtr[:, :k], ytr, Xte[:, :k], yte) for k in range(1, m + 1)]
    assert np.allclose(fast, slow, atol=1e-6), f"{fast}\n{slow}"


def test_curve_is_averaged_over_splits_not_one_draw():
    """A single train/test split makes knee_k hostage to one draw. Averaging
    over n_splits must actually happen -- and must report its own spread."""
    rng = np.random.default_rng(12)
    N, d = 6000, 30
    y = (rng.random(N) < 0.3).astype(np.float32)
    h = _make_codes(N, d, rng)
    # A NOISY detector, not a perfect one: a perfect latent gives R2 == 1.0 in
    # every split and a spread of exactly 0, which would not exercise anything.
    flip = rng.random(N) < 0.25
    h[:, 0] = np.where(flip, 1.0 - y, y) * 1.0
    m1 = diagnose_concept(torch.tensor(h), torch.tensor(y),
                          DilutionConfig(max_rows=10_000, n_splits=1))
    m5 = diagnose_concept(torch.tensor(h), torch.tensor(y),
                          DilutionConfig(max_rows=10_000, n_splits=5))
    assert m1["asymptote_r2_std"] == 0.0        # undefined for a single split
    assert m5["asymptote_r2_std"] > 0.0         # real across-split spread
    assert abs(m5["asymptote_r2"] - m1["asymptote_r2"]) < 0.1


def test_rare_concept_gets_more_rows_via_min_positives():
    """A base-rate-0.02 concept must not be judged on a few hundred positives:
    the curve stage widens its row budget until min_positives is met."""
    rng = np.random.default_rng(13)
    N, d = 60_000, 20
    y = (rng.random(N) < 0.02).astype(np.float32)
    h = _make_codes(N, d, rng)
    h[:, 0] = y * 1.0
    cfg = DilutionConfig(max_rows=5_000, curve_rows=10_000, min_positives=800)
    m = diagnose_concept(torch.tensor(h), torch.tensor(y), cfg)
    # ~800 positives at this base rate needs ~min_positives/base_rate rows,
    # far above curve_rows -- the budget must have been widened to reach it.
    needed = cfg.min_positives / m["base_rate"]
    assert m["n_curve_rows"] >= 0.95 * needed > cfg.curve_rows
    assert m["n_curve_positives"] >= 0.85 * cfg.min_positives
    assert m["verdict"] == "captured"


def test_ranking_cache_gives_identical_results():
    """Hoisting stage A out of the per-concept loop is an optimisation only --
    it must not change a single number."""
    from lib.sae.dilution import RankingCache

    rng = np.random.default_rng(14)
    N, d = 5000, 25
    y = (rng.random(N) < 0.25).astype(np.float32)
    h = _make_codes(N, d, rng)
    h[:, 0] = y * 1.0
    cfg = DilutionConfig(max_rows=3000)
    ht, yt = torch.tensor(h), torch.tensor(y)
    assert diagnose_concept(ht, yt, cfg) == diagnose_concept(
        ht, yt, cfg, cache=RankingCache(ht, cfg))


def test_support_overlap_singleton_is_flagged():
    """support_overlap returns 1.0 both for a lone latent and for perfect
    duplication; the singleton case must be distinguishable."""
    rng = np.random.default_rng(15)
    N, d = 4000, 20
    y = (rng.random(N) < 0.3).astype(np.float32)
    h = _make_codes(N, d, rng)
    h[:, 0] = y * 1.0
    m = diagnose_concept(torch.tensor(h), torch.tensor(y),
                         DilutionConfig(max_rows=4000))
    if m["community_size"] <= 1:
        assert m["singleton_community"] is True
        assert m["support_overlap"] == 1.0
    else:
        assert m["singleton_community"] is False


def test_orbit_aware_split_keeps_orbits_together():
    """A position and its board symmetries must never straddle a split."""
    from lib.sae.dilution import split_train_test

    rng = np.random.default_rng(21)
    groups = np.repeat(np.arange(500), 8)          # 500 orbits of 8
    te, tr = split_train_test(groups.size, int(0.3 * groups.size), rng, groups)
    assert set(groups[te]).isdisjoint(set(groups[tr]))
    assert te.size + tr.size == groups.size
    assert sorted(np.concatenate([te, tr]).tolist()) == list(range(groups.size))
    assert 0.2 < te.size / groups.size < 0.4       # budget roughly respected


def test_orbit_aware_split_falls_back_to_rowwise():
    from lib.sae.dilution import split_train_test

    rng = np.random.default_rng(22)
    te, tr = split_train_test(1000, 300, rng, None)
    assert te.size == 300 and tr.size == 700
    assert sorted(np.concatenate([te, tr]).tolist()) == list(range(1000))


def test_orbit_aware_split_removes_duplicate_leakage():
    """With exact duplicates present, row-wise splitting inflates held-out R2
    and orbit-aware splitting does not. This is the whole point of the change."""
    rng = np.random.default_rng(23)
    N, d = 3000, 20
    y_base = (rng.random(N) < 0.3).astype(np.float32)
    h_base = _make_codes(N, d, rng)
    # A latent that predicts y only through memorisable noise: no generalisable
    # signal, so honest held-out R2 must be ~0.
    h_base[:, 0] = rng.normal(size=N).astype(np.float32)
    # Duplicate every row: each position appears twice, same orbit id.
    h = np.repeat(h_base, 2, axis=0)
    y = np.repeat(y_base, 2)
    groups = np.repeat(np.arange(N), 2)
    cfg = DilutionConfig(max_rows=6000, curve_rows=6000, min_positives=1,
                         n_splits=3, top_k=8)
    m_row = diagnose_concept(torch.tensor(h), torch.tensor(y), cfg)
    m_orb = diagnose_concept(torch.tensor(h), torch.tensor(y), cfg,
                             orbit_ids=groups)
    assert m_row["orbit_aware_split"] is False
    assert m_orb["orbit_aware_split"] is True
    # The orbit-aware estimate must not be the more optimistic of the two.
    assert m_orb["asymptote_r2"] <= m_row["asymptote_r2"] + 1e-9


def test_participation_ratio_bounds():
    rng = np.random.default_rng(3)
    # one dominant axis -> PR near 1
    x = rng.normal(size=(2000, 1)) @ rng.normal(size=(1, 6))
    assert participation_ratio(x) < 1.5
    # six isotropic axes -> PR near 6
    y = rng.normal(size=(2000, 6))
    assert participation_ratio(y) > 4.0


def test_diagnose_row_subsample_is_deterministic():
    rng = np.random.default_rng(4)
    N, d = 12000, 30
    y = (rng.random(N) < 0.3).astype(np.float32)
    h = _make_codes(N, d, rng)
    h[:, 0] = y * 1.0
    cfg = DilutionConfig(max_rows=5000)  # forces subsample
    m1 = diagnose_concept(torch.tensor(h), torch.tensor(y), cfg)
    m2 = diagnose_concept(torch.tensor(h), torch.tensor(y), cfg)
    assert m1 == m2


# ---------------------------------------------------------------------------
# Rule 3A.3: the learned-signal floor, and the diluted/tiled collapse
# ---------------------------------------------------------------------------


def _metrics(**over):
    """A concept that is clearly present and clearly NOT concentrated."""
    base = dict(
        asymptote_r2=0.30, null_r2=0.00, solo_frac=0.20, knee_k=20,
        community_size=12, intrinsic_dim=5.0, support_overlap=0.10,
        neg_coupling_frac=0.60,
    )
    base.update(over)
    return base


def test_random_control_floor_turns_unlearned_signal_into_absent():
    """A concept a RANDOM-model dictionary recovers equally well is not learned.

    The permutation null cannot catch this: it shuffles the labels but leaves
    the dictionary intact, so it tests "is the signal real", not "is it
    learned". Measured 2026-08-14, the random-model conv2 dictionary scores
    R2 0.025-0.034 on every real gorilla threat, so under rule 3A.2 an
    untrained network passed gate G-3A at 100%.
    """
    cfg = DilutionConfig()
    # trained 0.30 vs random 0.29 -> the model contributed 0.01, below the floor
    assert classify(_metrics(random_asymptote_r2=0.29), cfg) == "absent"
    # trained 0.30 vs random 0.05 -> a real learned margin
    assert classify(_metrics(random_asymptote_r2=0.05), cfg) == "spread"


def test_missing_random_control_does_not_silently_pass():
    """No control number => the floor cannot fire, and that must be visible.

    12 of 17 reports on 2026-08-14 named a random control in their metadata
    while storing no number, because the fc1 control SAE has zero alive
    latents. The verdict is then provisional, not tested.
    """
    cfg = DilutionConfig()
    m = _metrics()  # no random_asymptote_r2 key at all
    assert classify(m, cfg) == "spread"
    assert m.get("random_control_applied") is None


def test_verdicts_are_three_way_at_rule_3A3():
    """diluted/tiled are collapsed into `spread`.

    The tiling test fired 42/1100 times and EVERY firing had community_size==2
    with one negative edge -- a coin flip on the sign of a single partial
    correlation. Rescoping to the candidate list does not help: mean pairwise
    Jaccard is ~0.10 for captured concepts and ~0.10 for spread ones.
    """
    cfg = DilutionConfig()
    # 3A.4 only ADDS the stability band; the point verdict is unchanged, so the
    # three-way collapse below must still hold at every later rule version.
    assert cfg.rule_version == "3A.4"
    seen = {
        classify(_metrics(asymptote_r2=0.001, null_r2=0.0), cfg),      # absent
        classify(_metrics(solo_frac=0.95, intrinsic_dim=1.0), cfg),    # captured
        classify(_metrics(), cfg),                                     # spread
    }
    assert seen == {"absent", "captured", "spread"}
    # the old tiling signature must no longer produce a separate label
    tiled_shaped = _metrics(support_overlap=0.0, neg_coupling_frac=1.0,
                            community_size=2)
    assert classify(tiled_shaped, cfg) == "spread"


def test_captured_still_wins_over_the_random_floor_ordering():
    """A concept that clears the floor AND is concentrated is `captured`."""
    cfg = DilutionConfig()
    m = _metrics(solo_frac=0.95, intrinsic_dim=1.0, random_asymptote_r2=0.05)
    assert classify(m, cfg) == "captured"


def test_random_floor_precedes_captured():
    """Not-learned beats concentrated: a concept the random dictionary also
    recovers is `absent` even if one latent carries all of it."""
    cfg = DilutionConfig()
    m = _metrics(solo_frac=0.99, intrinsic_dim=1.0, random_asymptote_r2=0.295)
    assert classify(m, cfg) == "absent"


def test_gate_summary_handles_both_rule_versions(tmp_path):
    """The combined summary must not crash on either gate schema.

    On 2026-08-17 `runners/3A-dilution.ps1` finished all 17 cells and then died
    writing the summary, because its inline script still read the `n_diluted` /
    `n_tiled` keys that rule 3A.3 replaced with `n_spread`. The logic now lives
    in scripts/summarize_3a_gate.py so it can be tested; this is that test.
    """
    import json
    import subprocess
    import sys as _sys

    analysis = tmp_path / "saes" / "quarto" / "analysis"
    analysis.mkdir(parents=True)

    def _report(name, gate):
        base = {
            "summary": {"run_id": name, "bsp_set": "gorillaYb",
                        "config": {"rule_version": gate.pop("_rule")}},
            "gate_g3a": gate,
            "concepts": [{"bsp_id": "x", "asymptote_r2": 0.3, "solo_frac": 0.2,
                          "intrinsic_dim": 4.0, "top_phi": 0.3,
                          "verdict": "spread"}],
        }
        (analysis / f"{name}_dilution-gorillaYb.json").write_text(json.dumps(base))

    # pre-3A.3 schema
    _report("legacy-run", {"_rule": "3A.2", "n_threat_bsps": 10, "n_diluted": 6,
                           "n_tiled": 1, "n_captured": 3, "n_absent": 0,
                           "geometric_frac": 0.7, "verdict": "3C-proceeds"})
    # current schema
    _report("current-run", {"_rule": "3A.3", "n_threat_bsps": 10, "n_spread": 7,
                            "n_captured": 3, "n_absent": 0, "geometric_frac": 0.7,
                            "verdict": "3C-proceeds",
                            "random_control_coverage": 1.0,
                            "gate_is_provisional": False})

    script = PROJECT_ROOT / "scripts" / "summarize_3a_gate.py"
    r = subprocess.run([_sys.executable, str(script)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"summary crashed:\n{r.stdout}\n{r.stderr}"
    out = json.loads((analysis / "3A_gate_summary.json").read_text())
    got = {row["run_id"]: row["n_spread"] for row in out["runs"]}
    assert got == {"legacy-run": 7, "current-run": 7}, got
    # a stale rule must be reported, not silently accepted
    assert "3A.2" in r.stdout and "reclassify" in r.stdout


# --- rule 3A.4: the verdict stability band ------------------------------------
#
# Why these exist: `classify` maps a POINT estimate onto a threshold that cuts a
# CONTINUUM (the 2026-08-21 bimodality retraction), and direct seed replication
# measured 12-17% of per-concept verdicts flipping on seed alone. The band is
# what keeps a near-boundary verdict from being quoted as if it were determined.

def test_band_marks_a_near_boundary_captured_verdict_undecided():
    cfg = DilutionConfig()
    on_edge = _metrics(solo_frac=cfg.captured_solo_frac, intrinsic_dim=1.0,
                       asymptote_r2_std=0.001)
    verdict, st = classify_with_stability(on_edge, cfg)
    assert verdict == "captured"
    assert st["stability"] == "undecided"
    assert "solo_frac" in st["flips_on"]


def test_band_leaves_a_far_from_boundary_verdict_confident():
    cfg = DilutionConfig()
    clear = _metrics(solo_frac=0.99, intrinsic_dim=1.0, asymptote_r2_std=0.001)
    verdict, st = classify_with_stability(clear, cfg)
    assert verdict == "captured"
    assert st["stability"] == "confident"
    assert st["flips_on"] == []


def test_band_covers_the_absent_floor_not_only_solo_frac():
    """The flip mode a solo_frac-only band cannot see.

    Regression on the 2026-08-21 finding: of K04-champYb's four measured
    cross-seed verdict flips, ZERO went through solo_frac -- all four were
    `absent <-> spread`, i.e. the rule-3A.3 learned-signal floor. A band that
    watches only the `captured` threshold reports those four as confident.
    """
    cfg = DilutionConfig()
    # asymptote_r2 sits just above the random-model control by ~random_margin.
    near_floor = _metrics(asymptote_r2=0.30, null_r2=0.01,
                          random_asymptote_r2=0.28, asymptote_r2_std=0.004,
                          solo_frac=0.20)
    verdict, st = classify_with_stability(near_floor, cfg)
    assert st["stability"] == "undecided"
    assert st["flips_on"] == ["asymptote_r2"], (
        "the learned-signal floor must be what this verdict rests on")


def test_band_widens_with_band_sds():
    """More sd = more undecided. Guards the knob against being inert."""
    m = _metrics(solo_frac=0.75, intrinsic_dim=1.0, asymptote_r2_std=0.001)
    tight = classify_with_stability(m, DilutionConfig(band_sds=0.1))[1]
    wide = classify_with_stability(m, DilutionConfig(band_sds=6.0))[1]
    assert tight["stability"] == "confident"
    assert wide["stability"] == "undecided"


def test_band_is_a_pure_function_of_stored_metrics():
    """Same guarantee `classify` has: re-bandable from a stored report."""
    cfg = DilutionConfig()
    m = _metrics(solo_frac=0.71, intrinsic_dim=1.0, asymptote_r2_std=0.002)
    import copy
    before = copy.deepcopy(m)
    classify_with_stability(m, cfg)
    assert m == before, "classify_with_stability must not mutate its input"
