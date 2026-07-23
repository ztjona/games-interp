"""Phase 3A dilution diagnostic -- game-agnostic core.

Given a trained SAE's cached codes ``h`` of shape (N, d_dict) and a binary
BSP label vector ``y`` of shape (N,), decide how the concept ``y`` is carried
in the SAE dictionary:

    absent    -- the codes carry no more signal about y than a permutation null
                 (or a random-model SAE control); the concept is not present in
                 dictionary-accessible form.
    captured  -- the concept is recovered by a small, low-dimensional set of
                 latents (clean monosemantic-ish capture).
    diluted   -- the concept is recoverable but only by aggregating many
                 mutually-overlapping latents (feature splitting / dilution);
                 the co-firing community is large with mixed-sign couplings.
    tiled     -- the concept is recoverable but spread over near-disjoint,
                 competing latents (shattered / tiled manifold); the community
                 has low support overlap and predominantly negative couplings.

The design and thresholds are documented in
``docs/diary/2026-07-21_3A-dilution-diagnostic.md``. All functions here operate
on plain tensors and know nothing about Quarto -- game knowledge enters only
through ``y`` (a column of a BSP label tensor) and the schema metadata handled
by the CLI wrapper ``scripts/dilution_diagnostic.py``.

Method summary (per concept):
  1. rank every alive latent by its signed association (phi/MCC) with y;
  2. estimate signed pairwise couplings (partial correlations) among the top
     candidates and detect the co-firing community around the top latent;
  3. build a restricted-R2 support curve (held-out R2 of y regressed on the
     first k association-ranked latents) against a permutation null and, if
     provided, a random-model SAE control;
  4. estimate the intrinsic dimension of the concept-conditioned community
     code (PCA participation ratio);
  5. combine into a verdict via documented, config-exposed thresholds.

Dependencies: numpy, torch, scikit-learn, networkx (all already required by
the repo). No non-ASCII characters -- the host prints under cp1252.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DilutionConfig:
    """Thresholds for the 3A diagnostic. All exposed so verdicts are auditable.

    Defaults were chosen to be conservative and are recorded verbatim in every
    output JSON so a verdict can always be recomputed from the raw metrics.
    """

    top_k: int = 64             # candidate latents ranked by |assoc| with y
    max_rows: int = 40000       # deterministic row cap for the numeric work
    min_freq: float = 1e-4      # alive-latent firing-rate floor
    max_freq: float = 0.999     # alive-latent firing-rate ceiling
    ridge: float = 1e-2         # regularizer for the precision (coupling) matrix
    coupling_tau: float = 0.05  # |partial corr| threshold for a community edge
    test_frac: float = 0.30     # held-out fraction for honest restricted-R2
    knee_frac: float = 0.90     # fraction of asymptotic R2 that defines the knee
    n_perm: int = 3             # label permutations for the absent null

    # verdict thresholds
    absent_margin: float = 0.02  # real_R2 - null_R2 below this -> absent
    absent_floor: float = 0.02   # asymptotic R2 below this -> absent
    captured_k: int = 2          # knee at or below this many latents -> captured
    captured_size: int = 3       # community at or below this size -> captured
    tile_overlap: float = 0.15   # mean support Jaccard below this -> tiled-ish
    tile_neg_frac: float = 0.50  # negative-coupling fraction above this -> tiled

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def phi_association(firing: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed phi coefficient (== MCC for 2x2) of each column of ``firing``
    against binary ``y``. Shape (d_dict,). Constant columns give 0.

    firing: (N, d_dict) in {0,1}; y: (N,) in {0,1}.
    """
    N = y.shape[0]
    y = y.astype(np.float64)
    f = firing.astype(np.float64)
    py = y.mean()
    pf = f.mean(axis=0)
    # covariance and standard deviations
    cov = (f * y[:, None]).mean(axis=0) - pf * py
    var_f = pf * (1.0 - pf)
    var_y = py * (1.0 - py)
    denom = np.sqrt(var_f * var_y)
    phi = np.zeros_like(pf)
    ok = denom > 0
    phi[ok] = cov[ok] / denom[ok]
    return phi


def select_concept_features(
    firing: np.ndarray, y: np.ndarray, cfg: DilutionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, phi) of the top-k alive latents by |phi| with y.

    Indices are ordered by descending |phi| (the association ranking used by
    the restricted-R2 curve). ``phi`` is the signed association for those
    indices, same order.
    """
    freq = firing.mean(axis=0)
    alive = (freq >= cfg.min_freq) & (freq <= cfg.max_freq)
    phi = phi_association(firing, y)
    score = np.abs(phi)
    score[~alive] = -1.0
    k = min(cfg.top_k, int(alive.sum()))
    if k <= 0:
        return np.array([], dtype=int), np.array([])
    order = np.argsort(-score)[:k]
    return order, phi[order]


def signed_partial_correlations(
    firing_sub: np.ndarray, cfg: DilutionConfig
) -> np.ndarray:
    """Signed partial-correlation matrix among candidate latents.

    Uses the Gaussian-graphical-model estimator on the binarized firing:
    partial_corr_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj), where Theta is the
    (ridge-regularized) inverse correlation matrix. This is an approximation
    for binary variables but recovers the sign structure the verdict needs
    (mixed-sign redundancy vs. competing negative couplings). Diagonal set to 0.
    """
    m = firing_sub.shape[1]
    if m <= 1:
        return np.zeros((m, m))
    X = firing_sub.astype(np.float64)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / std
    R = np.corrcoef(Xs, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    R += cfg.ridge * np.eye(m)
    try:
        Theta = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Theta = np.linalg.pinv(R)
    d = np.sqrt(np.clip(np.diag(Theta), 1e-12, None))
    P = -Theta / np.outer(d, d)
    np.fill_diagonal(P, 0.0)
    return P


def detect_community(P: np.ndarray, seed_idx: int, cfg: DilutionConfig) -> list[int]:
    """Greedy-modularity community (over |P| > tau) containing ``seed_idx``.

    Deterministic. Falls back to the connected component, then to {seed_idx}.
    Returns local indices into the candidate array.
    """
    import networkx as nx

    m = P.shape[0]
    if m <= 1:
        return [seed_idx]
    G = nx.Graph()
    G.add_nodes_from(range(m))
    ii, jj = np.where(np.triu(np.abs(P) > cfg.coupling_tau, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        G.add_edge(a, b, weight=float(abs(P[a, b])))
    if G.number_of_edges() == 0:
        return [seed_idx]
    try:
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
    except Exception:
        comms = list(nx.connected_components(G))
    for c in comms:
        if seed_idx in c:
            return sorted(int(x) for x in c)
    return [seed_idx]


def support_overlap(firing_comm: np.ndarray) -> float:
    """Mean pairwise Jaccard overlap of the firing supports of community latents.

    High -> redundant/overlapping (dilution). Low -> disjoint (tiling).
    firing_comm: (N, c) in {0,1}.
    """
    c = firing_comm.shape[1]
    if c <= 1:
        return 1.0
    F = firing_comm.astype(bool)
    jac = []
    for a in range(c):
        for b in range(a + 1, c):
            inter = np.logical_and(F[:, a], F[:, b]).sum()
            union = np.logical_or(F[:, a], F[:, b]).sum()
            if union > 0:
                jac.append(inter / union)
    return float(np.mean(jac)) if jac else 0.0


def participation_ratio(codes: np.ndarray) -> float:
    """PCA participation ratio (effective dimensionality) of ``codes``.

    PR = (sum lambda)^2 / sum(lambda^2) over the covariance eigenvalues.
    Ranges in [1, n_features]; ~1 means a single dominant axis.
    """
    if codes.shape[0] < 2 or codes.shape[1] < 1:
        return 1.0
    Xc = codes - codes.mean(axis=0)
    cov = (Xc.T @ Xc) / max(1, codes.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    s = ev.sum()
    if s <= 0:
        return 1.0
    return float((s * s) / (ev ** 2).sum())


def _r2_heldout(Xtr, ytr, Xte, yte) -> float:
    """Held-out R2 of a plain linear regression. Guards degenerate variance."""
    from sklearn.linear_model import LinearRegression

    if Xtr.shape[1] == 0:
        return 0.0
    reg = LinearRegression()
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def restricted_r2_curve(
    codes: np.ndarray, y: np.ndarray, order: np.ndarray, cfg: DilutionConfig
) -> dict:
    """Held-out restricted-R2 as latents are added in ``order``.

    Returns curve (list over k=1..m), asymptote (== last point), knee (smallest
    k reaching knee_frac * asymptote), and a permutation-null asymptote (mean
    over cfg.n_perm label shuffles at full support).
    """
    rng = np.random.default_rng(cfg.seed)
    N = codes.shape[0]
    perm = rng.permutation(N)
    n_te = max(1, int(cfg.test_frac * N))
    te_idx, tr_idx = perm[:n_te], perm[n_te:]
    Xtr_all = codes[tr_idx][:, order]
    Xte_all = codes[te_idx][:, order]
    ytr, yte = y[tr_idx], y[te_idx]

    curve = []
    for k in range(1, len(order) + 1):
        curve.append(_r2_heldout(Xtr_all[:, :k], ytr, Xte_all[:, :k], yte))
    asymptote = curve[-1] if curve else 0.0

    knee = len(order)
    if asymptote > 0:
        target = cfg.knee_frac * asymptote
        for k, v in enumerate(curve, start=1):
            if v >= target:
                knee = k
                break

    # permutation null at full support
    null_vals = []
    for _ in range(cfg.n_perm):
        yp = ytr.copy()
        rng.shuffle(yp)
        null_vals.append(_r2_heldout(Xtr_all, yp, Xte_all, yte))
    null_r2 = float(np.mean(null_vals)) if null_vals else 0.0

    return {
        "curve": [round(float(v), 4) for v in curve],
        "asymptote_r2": round(float(asymptote), 4),
        "knee_k": int(knee),
        "null_r2": round(float(null_r2), 4),
    }


# ---------------------------------------------------------------------------
# Per-concept diagnosis
# ---------------------------------------------------------------------------


def classify(metrics: dict, cfg: DilutionConfig) -> str:
    """Map raw metrics to {absent, captured, diluted, tiled}. Pure function of
    the numbers in ``metrics`` and the thresholds in ``cfg`` -- auditable."""
    gap = metrics["asymptote_r2"] - metrics["null_r2"]
    if gap < cfg.absent_margin or metrics["asymptote_r2"] < cfg.absent_floor:
        return "absent"
    if metrics["knee_k"] <= cfg.captured_k and metrics["community_size"] <= cfg.captured_size:
        return "captured"
    if (metrics["support_overlap"] < cfg.tile_overlap
            and metrics["neg_coupling_frac"] > cfg.tile_neg_frac):
        return "tiled"
    return "diluted"


def diagnose_concept(
    h: torch.Tensor,
    y: torch.Tensor,
    cfg: DilutionConfig,
    h_random: torch.Tensor | None = None,
) -> dict:
    """Full 3A diagnosis of a single concept. Returns a JSON-ready dict.

    h:        (N, d_dict) SAE codes (post-activation feature values).
    y:        (N,) binary concept labels, row-aligned to h.
    h_random: optional (N, d_dict) codes from a random-model SAE, row-aligned;
              adds a random-control asymptote alongside the permutation null.
    """
    y_full = _to_numpy(y).astype(np.float64).ravel()
    base_rate = float(y_full.mean())

    # Deterministic row subsample for the numeric work. R2 / couplings / PCA
    # are stable well below the full N (hundreds of thousands); capping keeps
    # the per-concept cost to well under a second. The base rate above is taken
    # on the full data so the reported value is exact.
    N_full = h.shape[0]
    if N_full > cfg.max_rows:
        rng = np.random.default_rng(cfg.seed)
        rows = rng.choice(N_full, size=cfg.max_rows, replace=False)
        rows.sort()
        h_np = _to_numpy(h[rows]).astype(np.float32)
        y_np = y_full[rows]
    else:
        rows = None
        h_np = _to_numpy(h).astype(np.float32)
        y_np = y_full
    firing = (h_np > 0).astype(np.float64)
    order, phi = select_concept_features(firing, y_np, cfg)

    if order.size == 0:
        metrics = {
            "base_rate": round(base_rate, 4),
            "n_candidates": 0,
            "asymptote_r2": 0.0, "null_r2": 0.0, "knee_k": 0,
            "community_size": 0, "neg_coupling_frac": 0.0,
            "support_overlap": 0.0, "intrinsic_dim": 0.0,
            "top_phi": 0.0, "curve": [],
        }
        metrics["verdict"] = "absent"
        return metrics

    codes_sub = h_np[:, order]
    firing_sub = firing[:, order]

    # couplings + community around the top-associated latent (local idx 0)
    P = signed_partial_correlations(firing_sub, cfg)
    community = detect_community(P, 0, cfg)
    if community:
        sub = np.array(community)
        within = P[np.ix_(sub, sub)]
        offdiag = within[~np.eye(len(sub), dtype=bool)]
        nz = offdiag[np.abs(offdiag) > cfg.coupling_tau]
        neg_frac = float((nz < 0).mean()) if nz.size else 0.0
        overlap = support_overlap(firing_sub[:, sub])
        active = y_np > 0.5
        comm_codes = codes_sub[:, sub]
        idim = participation_ratio(comm_codes[active]) if active.sum() > 1 else 1.0
    else:
        neg_frac, overlap, idim = 0.0, 1.0, 1.0

    r2 = restricted_r2_curve(codes_sub, y_np, np.arange(len(order)), cfg)

    metrics = {
        "base_rate": round(base_rate, 4),
        "n_candidates": int(order.size),
        "top_phi": round(float(phi[0]), 4),
        "asymptote_r2": r2["asymptote_r2"],
        "null_r2": r2["null_r2"],
        "knee_k": r2["knee_k"],
        "community_size": int(len(community)),
        "neg_coupling_frac": round(neg_frac, 4),
        "support_overlap": round(overlap, 4),
        "intrinsic_dim": round(float(idim), 4),
        "knee_over_idim": round(r2["knee_k"] / max(1.0, idim), 4),
        "curve": r2["curve"],
    }

    if h_random is not None:
        hr = _to_numpy(h_random if rows is None else h_random[rows]).astype(np.float32)
        fr = (hr > 0).astype(np.float64)
        ro, _ = select_concept_features(fr, y_np, cfg)
        if ro.size:
            rr = restricted_r2_curve(hr[:, ro], y_np, np.arange(len(ro)), cfg)
            metrics["random_asymptote_r2"] = rr["asymptote_r2"]

    metrics["verdict"] = classify(metrics, cfg)
    return metrics
