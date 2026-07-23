"""Tests for the Phase 3A dilution diagnostic (lib/sae/dilution.py).

Two layers:
  1. classify() truth-table -- the verdict logic is a pure function of the
     metrics dict, so the four branches are pinned directly and deterministically.
  2. diagnose_concept() end-to-end on synthetic planted concepts -- confirms the
     metrics that feed classify() come out with the right qualitative shape for
     captured / absent / tiled structure.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.sae.dilution import (
    DilutionConfig,
    classify,
    diagnose_concept,
    participation_ratio,
)


CFG = DilutionConfig(max_rows=10_000)  # small; tests use N <= 6000


def _metrics(**over):
    """A 'recoverable, spread' baseline metrics dict; override per test."""
    base = dict(
        asymptote_r2=0.40, null_r2=0.00, knee_k=12, community_size=16,
        neg_coupling_frac=0.30, support_overlap=0.40, intrinsic_dim=3.0,
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


def test_classify_tiled():
    m = _metrics(support_overlap=0.05, neg_coupling_frac=0.7)
    assert classify(m, CFG) == "tiled"


def test_classify_diluted():
    # recoverable, many latents, overlapping supports, mixed-sign couplings
    m = _metrics(support_overlap=0.4, neg_coupling_frac=0.3, knee_k=12)
    assert classify(m, CFG) == "diluted"


def test_classify_diluted_not_tiled_when_overlap_high():
    # negative couplings alone must NOT trigger tiled if supports overlap
    m = _metrics(support_overlap=0.5, neg_coupling_frac=0.8)
    assert classify(m, CFG) == "diluted"


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


def test_diagnose_tiled_disjoint_latents():
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
    assert m["verdict"] in ("tiled", "diluted")     # geometric, not captured/absent
    assert m["verdict"] != "captured"


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
