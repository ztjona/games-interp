"""Phase 3A dilution diagnostic -- game-agnostic core.

Given a trained SAE's cached codes ``h`` of shape (N, d_dict) and a binary
BSP label vector ``y`` of shape (N,), decide how the concept ``y`` is carried
in the SAE dictionary:

    absent    -- the codes carry no more signal about y than a floor: the
                 permutation null (not real) or, where measured, a random-model
                 SAE control (real but NOT LEARNED).
    captured  -- the concept is recovered by a small, low-dimensional set of
                 latents (clean monosemantic-ish capture).
    spread    -- recoverable and above both floors, but no single latent carries
                 most of it. The GEOMETRIC verdict. Replaces the diluted/tiled
                 pair at rule 3A.3; see ``classify`` for why that split was not
                 measurable with the statistics available.

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

import itertools
import math
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

    # top_k is THE scale hyperparameter of this diagnostic: knee_k,
    # community_size and n_candidates are all bounded by it, and asymptote_r2
    # is measured at exactly this support. A metric "up to 64" means "up to
    # top_k", not anything intrinsic about the SAE. Changing it changes those
    # numbers, so it is recorded in every report and must be quoted with them.
    top_k: int = 64             # candidate latents ranked by |assoc| with y

    # Two row budgets, because the two stages have very different cost:
    #   ranking  -- needs ALL d_dict columns, so rows are expensive (memory);
    #   curve    -- needs only the top_k selected columns, so rows are cheap.
    # Splitting them lets the curve use far more rows (and therefore far more
    # positives for a rare concept) at negligible cost.
    max_rows: int = 40000       # rows for the phi ranking (all d_dict columns)
    curve_rows: int = 120000    # rows for the R2 curve / community (top_k cols)
    min_positives: int = 2000   # raise curve_rows until the concept has this
                                # many positive rows (capped at N). At base rate
                                # 0.02, 40k rows leave only ~240 positives in the
                                # held-out split -- too few to trust knee_k.

    min_freq: float = 1e-4      # alive-latent firing-rate floor
    max_freq: float = 0.999     # alive-latent firing-rate ceiling
    fire_threshold: float = 0.0  # a latent "fires" when h > this. NOT swept and
                                # NOT relative to h_max: TopK/BatchTopK/JumpReLU
                                # all emit exact structural zeros, so 0 is the
                                # architecture's own on/off boundary. Only change
                                # it for a dense (Vanilla/Gated) dictionary.
    ridge: float = 1e-2         # regularizer for the precision (coupling) matrix
    coupling_tau: float = 0.05  # |partial corr| threshold for a community edge
    test_frac: float = 0.30     # held-out fraction for honest restricted-R2
    knee_frac: float = 0.90     # fraction of asymptotic R2 that defines the knee
    n_splits: int = 5           # train/test resamples the R2 curve averages over
    n_perm: int = 5             # label permutations for the absent null

    # verdict thresholds
    absent_margin: float = 0.02  # real_R2 - null_R2 below this -> absent
    # A SECOND floor, against the random-model SAE rather than the permutation
    # null. The permutation null only destroys the concept's association; it
    # leaves the dictionary's own structure intact, so it answers "is this
    # signal real?" and NOT "is this signal learned?". Measured 2026-08-14: the
    # random-model conv2 dictionary scores R2 0.025-0.034 on every real gorilla
    # threat -- above absent_floor -- so under rule 3A.2 an UNTRAINED network
    # passed gate G-3A at 100%. This floor is what makes the gate a test.
    random_margin: float = 0.02  # real_R2 - random_model_R2 below this -> absent
    absent_floor: float = 0.02   # asymptotic R2 below this -> absent
    captured_k: int = 2          # knee at or below this many latents -> captured
    captured_size: int = 3       # community at or below this size -> captured
    captured_solo_frac: float = 0.70  # solo latent recovers >= this share -> captured
    captured_idim: float = 2.0        # ...AND community is this low-dimensional
    tile_overlap: float = 0.15   # mean support Jaccard below this -> tiled-ish
    tile_neg_frac: float = 0.50  # negative-coupling fraction above this -> tiled

    # --- verdict stability band (rule 3A.4) --------------------------------
    # Every threshold above discretises a CONTINUUM. The 2026-08-21 retraction
    # settled that: at category level 13 of 69 medians sit above 0.8 and 36
    # below 0.4, but 20 sit in between -- there is no valley for 0.70 to fall
    # into. A threshold on a continuum needs an uncertainty band or it reports
    # noise as a finding, so the band is REQUIRED, not decorative.
    # Calibrated 2026-08-24 against the seed grid's measured flip rates, which
    # is the first time this could be chosen by measurement rather than taste.
    # Sweep over 16 condition x basis cells (mean measured flip rate 11.4%):
    #   with the asymptote floor -- 0.5sd -> 8.2%, 1.0 -> 15.9%, 3.0 -> 25.3%
    #   without it               -- 3.0sd -> 7.1%, i.e. under-calls at EVERY
    #                               width, because a split sd cannot see seeds.
    # 1.0 sits at 1.40x the measured rate: deliberately a little conservative,
    # because a concept that did not flip across three draws can still be near
    # enough to flip on a fourth, and "undecided" should mean "could plausibly
    # land the other side", not "did land it in this sample". 3.0 was a
    # ~99.7% interval and flagged 91% of one cell -- too blunt to be useful.
    band_sds: float = 1.0
    # Per-concept cross-seed sd of solo_frac. Re-measured 2026-08-24 on the
    # seed grid: 4 conditions x 3 seeds x 4 bases, **1,764 concepts** (the
    # previous 0.0523 came from 99 concepts, 2 conditions, one basis).
    # The MEAN, deliberately, not the median: the distribution is heavily
    # right-skewed (median 0.0191, mean 0.0407, p90 0.1145), and a band is a
    # claim about the tail. Per-basis means span only 1.6x (gorilla 0.0300,
    # hawk 0.0366, tiger 0.0459, hen 0.0489), so one pooled constant is fair;
    # the within-basis skew is ~5x and dominates.
    # Re-derive with `python scripts/verdict_stability.py`, which prints the
    # GROUND-TRUTH flip rate and warns if this constant has drifted.
    solo_frac_seed_sd: float = 0.0407
    # Cross-seed sd of asymptote_r2, same grid, same 1,764 concepts.
    # `asymptote_r2_std` is a WITHIN-RUN split sd: it resamples rows but never
    # retrains the SAE, so it cannot see seed movement at all and was measured
    # **1.96x low (median) / 3.39x (mean)**. That is the single biggest reason
    # the band under-called the measured flip rate. `_band_sd` takes the LARGER
    # of the per-concept split sd and this floor, keeping concept-specific
    # noise where it exceeds typical seed movement without ever under-stating.
    asymptote_r2_seed_sd: float = 0.00648

    # Bumped whenever ``classify`` changes, so a stored verdict can always be
    # traced to the rule that produced it. 3A.1 = original (knee_k AND
    # community_size only); 3A.2 = adds the solo_frac/intrinsic_dim path;
    # 3A.3 = adds the random-model floor and collapses diluted/tiled -> spread;
    # 3A.4 = adds the stability band (``classify_with_stability``);
    # 3A.5 = band CALIBRATED against measured cross-seed flip rates (band_sds
    # 3.0 -> 1.0, asymptote_r2 gains a measured seed floor, solo_frac_seed_sd
    # 0.0523 -> 0.0407 on 1,764 concepts instead of 99).
    # The point verdict is UNCHANGED at both 3A.4 and 3A.5 -- only the
    # confidence annotation moves, so every 3A.3 verdict still reads the same.
    rule_version: str = "3A.5"

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Glossary -- embedded verbatim in every report so a JSON is self-describing
# ---------------------------------------------------------------------------

# Each entry: (range, ideal-for-"captured", what the number means).
GLOSSARY: dict[str, dict[str, str]] = {
    "base_rate": {
        "range": "0..1",
        "ideal": "n/a (a property of the data, not the SAE)",
        "means": "Fraction of positions where the concept is TRUE. Rare concepts "
                 "(~0.02) make F1 and R2 look small even when the signal is real, "
                 "so compare like base rates or use MCC/phi.",
    },
    "top_phi": {
        "range": "-1..1",
        "ideal": "|phi| near 1",
        "means": "Signed phi coefficient (identical to MCC for a 2x2 table) between "
                 "the single best-associated latent's firing and the concept. 0 = no "
                 "association. This is the strongest single-latent association that "
                 "exists in the dictionary for this concept.",
    },
    "asymptote_r2": {
        "range": "<=1 (can go slightly negative)",
        "ideal": "high",
        "means": "Held-out R2 of the concept regressed on ALL top_k candidate "
                 "latents together. Total information about the concept recoverable "
                 "from the dictionary, independent of how many latents it takes. "
                 "1.0 = perfectly predicted; 0 = no better than predicting the mean.",
    },
    "asymptote_r2_std": {
        "range": ">=0",
        "ideal": "small vs asymptote_r2",
        "means": "Standard deviation of asymptote_r2 across the n_splits train/test "
                 "resamples. A value comparable to asymptote_r2 itself means the "
                 "concept's numbers (especially knee_k) are split noise, not signal.",
    },
    "null_r2": {
        "range": "around 0",
        "ideal": "0",
        "means": "Same regression with the labels shuffled -- the overfitting floor. "
                 "Averaged over n_perm shuffles x n_splits splits. "
                 "asymptote_r2 minus null_r2 is the real signal.",
    },
    "solo_r2": {
        "range": "<=1",
        "ideal": "close to asymptote_r2",
        "means": "Held-out R2 using only the single best-associated latent (k=1).",
    },
    "solo_frac": {
        "range": "0..1",
        "ideal": ">= 0.70 (captured)",
        "means": "solo_r2 / asymptote_r2: the share of all recoverable signal that "
                 "ONE latent already carries. Low = the concept is smeared across "
                 "many latents (dilution). This is the primary concentration metric.",
    },
    "knee_k": {
        "range": "1..top_k",
        "ideal": "1-2",
        "means": "Smallest number of latents reaching 90% of asymptote_r2. Intuitive "
                 "but brittle: a slow ~2% upward drift in the tail of the curve pushes "
                 "the crossing far right even when latent #1 already did the work. "
                 "Read it alongside solo_frac, never alone.",
    },
    "community_size": {
        "range": "1..top_k",
        "ideal": "small (1-3)",
        "means": "Number of latents in the co-firing community around the top latent "
                 "(greedy-modularity community over |partial correlation| > tau). "
                 "Counts REDUNDANCY: near-duplicate latents inflate it without the "
                 "concept being any harder to read out.",
    },
    "intrinsic_dim": {
        "range": "1..community_size",
        "ideal": "~1",
        "means": "PCA participation ratio of the community's codes on positions where "
                 "the concept is TRUE. ~1 = one dominant direction (a genuine single "
                 "feature); 5 = the community spans ~5 effective dimensions. This is "
                 "the dimensionality that actually matters, vs community_size which "
                 "merely counts members.",
    },
    "support_overlap": {
        "range": "0..1",
        "ideal": "n/a -- it selects dilution vs tiling, not quality",
        "means": "Mean PAIRWISE JACCARD of the community latents' firing supports: "
                 "for each pair, |rows where both fire| / |rows where either fires|, "
                 "averaged over pairs. NOT 'how many latents fire per position'. "
                 "HIGH = same rows (redundant -> diluted); LOW = disjoint rows "
                 "(shattered -> tiled). Neither end is good. A single-latent "
                 "community has no pairs and returns 1.0 by convention -- check "
                 "singleton_community before reading a 1.0 as redundancy.",
    },
    "singleton_community": {
        "range": "true/false",
        "ideal": "n/a",
        "means": "True when the community is a single latent, i.e. support_overlap "
                 "and neg_coupling_frac are conventional defaults rather than "
                 "measurements.",
    },
    "n_curve_rows": {
        "range": "<= N",
        "ideal": "n/a",
        "means": "Rows used for the R2 curve / community stage. Raised above "
                 "curve_rows automatically until the concept has min_positives "
                 "positive rows, so rare concepts are not judged on a few hundred.",
    },
    "orbit_aware_split": {
        "range": "true/false",
        "ideal": "true",
        "means": "Whether train/test splits kept each position together with its 8 "
                 "board symmetries. Positions are deduplicated by exact bytes, so "
                 "symmetric images survive as separate rows; they are legitimately "
                 "distinct inputs but statistically dependent, and for a "
                 "rotation-invariant concept they form near-duplicate (x, y) pairs "
                 "that inflate held-out R2 if they straddle a split. False means the "
                 "orbit IDs were unavailable and the R2 may be slightly optimistic.",
    },
    "n_curve_positives": {
        "range": ">=0",
        "ideal": ">= min_positives",
        "means": "Positive rows available to the curve stage. If this is far below "
                 "min_positives the concept is too rare in this dataset to diagnose "
                 "and its verdict should be treated as provisional.",
    },
    "neg_coupling_frac": {
        "range": "0..1",
        "ideal": "n/a -- selects dilution vs tiling",
        "means": "Fraction of within-community couplings that are negative, i.e. "
                 "latents that suppress each other (competing for the same concept). "
                 "High + low support_overlap = tiled.",
    },
    "knee_over_idim": {
        "range": ">=1 typically",
        "ideal": "~1",
        "means": "knee_k / intrinsic_dim. >>1 means many more latents are needed than "
                 "the code's own dimensionality implies -- a splitting signature.",
    },
    "curve": {
        "range": "list of R2, length top_k",
        "ideal": "flat after k=1",
        "means": "The restricted-R2 support curve: entry k is the held-out R2 using "
                 "the k best-associated latents. A steep early rise then a plateau = "
                 "concentrated; a long slow climb = diluted.",
    },
    "random_asymptote_r2": {
        "range": "<=1",
        "ideal": "far BELOW asymptote_r2",
        "means": "The same restricted-R2 asymptote computed on an SAE trained on an "
                 "UNTRAINED network's activations, row-aligned to the same positions. "
                 "The learned-signal floor: asymptote_r2 minus this is what the "
                 "TRAINED model contributes over the architectural prior. Absent when "
                 "no usable control existed for that run -- check "
                 "random_control_applied before reading any verdict as tested.",
    },
    "random_control_applied": {
        "range": "true/false",
        "ideal": "true",
        "means": "Whether the random-model floor was actually applied to this "
                 "concept. FALSE means the verdict rests on the permutation null "
                 "alone and is PROVISIONAL: the permutation null cannot distinguish "
                 "learned structure from the architectural prior. A control named in "
                 "a report's metadata is not enough -- the fc1 control SAE has zero "
                 "alive latents and contributed no number.",
    },
    "geometric_frac": {
        "range": "0..1",
        "ideal": "n/a -- this is the gate quantity",
        "means": "(n_diluted + n_tiled) / n_threat_bsps for one run. The fraction of "
                 "threat concepts whose failure mode is GEOMETRIC (present in the code "
                 "but spread out) rather than absent or already clean. >= 0.50 -> "
                 "Gate G-3A says phase 3C proceeds.",
    },
    "verdict_stability": {
        "range": "confident | undecided",
        "ideal": "confident",
        "means": "Does the verdict survive pushing every quantity `classify` "
                 "thresholds on to +/- band_sds sd? `undecided` means the point "
                 "estimate sits close enough to a boundary that a re-run could "
                 "land the other side. Required because the thresholds cut a "
                 "CONTINUUM (solo_frac is not bimodal -- 2026-08-21 retraction), "
                 "and because seed replication measured 12-17% of per-concept "
                 "verdicts flipping. Rule 3A.4.",
    },
    "verdict_flips_on": {
        "range": "list of metric names (possibly empty)",
        "ideal": "empty",
        "means": "Which banded quantity, moved ALONE, changes this verdict -- "
                 "i.e. which threshold the verdict actually rests on. "
                 "`asymptote_r2` points at the absent / learned-signal floor; "
                 "`solo_frac` at the captured threshold.",
    },
    "n_confident_spread / _captured / _absent": {
        "range": "counts summing with n_undecided to n_threat_bsps",
        "ideal": "n/a -- this IS the spectrum",
        "means": "How many threat concepts hold each verdict ROBUSTLY, i.e. "
                 "survive the +/-band_sds perturbation, plus how many are in "
                 "play. Quote these, not an interval: turning the in-play "
                 "concepts into a range on geometric_frac requires assuming "
                 "how they would resolve, and every such assumption is "
                 "arbitrary. '2 spread / 0 captured / 0 absent / 21 in play "
                 "of 23' says plainly that the measurement is undetermined.",
    },
    "geometric_frac_worst_lo / _worst_hi": {
        "range": "0..1",
        "ideal": "n/a -- the band on the gate quantity",
        "means": "WORST CASE: geometric_frac if EVERY in-play concept resolved "
                 "the least / most geometric way AT ONCE. A union bound, not a "
                 "confidence interval -- measured 3-10x wider than the "
                 "cross-seed range on 2026-08-24, and vacuous at small n (at "
                 "tiger's 23 concepts one concept is 0.043 of the fraction). "
                 "Its one legitimate use is `gate_verdict_is_stable`: when "
                 "both ends fall the same side of 0.50 the gate holds no "
                 "matter how the in-play concepts land.",
    },
}

VERDICT_GLOSSARY: dict[str, str] = {
    "absent": "The codes carry no more signal about the concept than the floor -- "
              "either the permutation null (the signal is not real) or, where a "
              "random-model SAE control was measured, that control (the signal is "
              "real but NOT LEARNED: a dictionary trained on an untrained network "
              "recovers the concept just as well). Not a capacity problem: change "
              "the hook, or go E2E / supervised.",
    "captured": "Recovered by a small, low-dimensional set of latents. The SAE already "
                "has this concept cleanly; nothing to fix.",
    "spread": "Recoverable from the dictionary, and demonstrably better than both "
              "floors, but NOT concentrated: no single latent carries most of it. "
              "This is the GEOMETRIC verdict and the H10 outcome -- the information "
              "is present and the flat dictionary is not presenting it in one atom, "
              "so the fix is an architecture change rather than a dead end. "
              "Replaces the former diluted/tiled pair as of rule 3A.3: the test "
              "that separated them fired only on 2-latent communities with a single "
              "negative edge, and the underlying statistic does not discriminate "
              "captured from spread at any scope (see `classify`).",
    # Retired, retained so a pre-3A.3 report is still readable.
    "diluted": "RETIRED at rule 3A.3 -- now reported as `spread`. Meant: recoverable "
               "only by aggregating many mutually-overlapping latents.",
    "tiled": "RETIRED at rule 3A.3 -- now reported as `spread`. Meant: recoverable "
             "but spread over near-disjoint, competing latents. The test for it was "
             "not measuring that; see `classify`.",
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def phi_association(
    firing: np.ndarray, y: np.ndarray, freq: np.ndarray | None = None
) -> np.ndarray:
    """Signed phi coefficient (== MCC for 2x2) of each column of ``firing``
    against binary ``y``. Shape (d_dict,). Constant columns give 0.

    firing: (N, d_dict) in {0,1}, any float dtype; y: (N,) in {0,1}.
    freq:   optional precomputed column means (firing rates) -- they do not
            depend on ``y``, so a caller diagnosing many concepts against the
            same codes should compute them once.

    The joint term is a single BLAS matrix-vector product rather than a
    broadcast product, which avoids allocating a second (N, d_dict) array per
    concept -- that allocation dominated the runtime when this is called once
    per BSP over a 296k x 4096 code cache.
    """
    N = y.shape[0]
    y = y.astype(firing.dtype, copy=False)
    py = float(y.mean())
    pf = firing.mean(axis=0) if freq is None else freq
    cov = (firing.T @ y) / N - pf * py
    var_f = pf * (1.0 - pf)
    var_y = py * (1.0 - py)
    denom = np.sqrt(np.clip(var_f * var_y, 0.0, None))
    phi = np.zeros(pf.shape, dtype=np.float64)
    ok = denom > 0
    phi[ok] = cov[ok] / denom[ok]
    return phi


def select_concept_features(
    firing: np.ndarray, y: np.ndarray, cfg: DilutionConfig,
    freq: np.ndarray | None = None, alive: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, phi) of the top-k alive latents by |phi| with y.

    Indices are ordered by descending |phi| (the association ranking used by
    the restricted-R2 curve). ``phi`` is the signed association for those
    indices, same order. ``freq``/``alive`` may be supplied precomputed; they
    depend only on the codes, not on the concept.
    """
    if freq is None:
        freq = firing.mean(axis=0)
    if alive is None:
        alive = (freq >= cfg.min_freq) & (freq <= cfg.max_freq)
    phi = phi_association(firing, y, freq=freq)
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
    """Mean PAIRWISE JACCARD overlap of the firing supports of community latents.

    For each unordered pair (a, b) of community latents, Jaccard is
    ``|rows where both fire| / |rows where either fires|``; the result is the
    mean over all pairs. It is NOT "how many latents fire per position".

    High -> the latents fire on the same rows: redundant/overlapping (dilution).
    Low  -> they fire on disjoint rows: shattered (tiling).

    There is no "good" value: this metric picks *which* geometric failure mode
    is present, it does not measure quality. Note the degenerate case -- a
    single-latent community has no pairs and returns 1.0 by convention, which is
    numerically identical to "perfectly redundant duplicates". Always read it
    together with ``community_size`` (or the ``singleton_community`` flag).

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


class _NestedOLS:
    """Held-out R2 for every nested prefix X[:, :k], k = 1..m, in one pass.

    OLS with intercept, identical to fitting ``LinearRegression`` per prefix,
    but the (m x m) train Gram matrix and cross-product are formed ONCE and each
    prefix is a k x k solve. The naive version refits from scratch m times per
    split and, at m=64 with five splits and a permutation null, was the dominant
    cost of the whole diagnostic.

    The permutation null reuses the same Gram: shuffling the labels changes only
    the cross-product ``b``, so a null draw costs one mat-vec, not a refit.
    """

    def __init__(self, Xtr, ytr, Xte, yte, jitter: float = 1e-10):
        self.mu_x = Xtr.mean(axis=0)
        self.Xtrc = (Xtr - self.mu_x).astype(np.float64, copy=False)
        self.Xtec = (Xte - self.mu_x).astype(np.float64, copy=False)
        self.yte = yte.astype(np.float64, copy=False)
        self.m = Xtr.shape[1]
        self.G = self.Xtrc.T @ self.Xtrc
        # Scale-relative jitter keeps rank-deficient prefixes solvable (latents
        # in a community are often near-collinear) without biasing the fit.
        self.G.flat[:: self.m + 1] += jitter * max(np.trace(self.G), 1.0) / self.m
        ss_tot = float(((self.yte - self.yte.mean()) ** 2).sum())
        self.ss_tot = ss_tot if ss_tot > 0 else None

    def _r2_for(self, ytr, ks) -> list[float]:
        if self.ss_tot is None:
            return [0.0] * len(ks)
        mu_y = float(ytr.mean())
        b = self.Xtrc.T @ (ytr.astype(np.float64, copy=False) - mu_y)
        out = []
        for k in ks:
            try:
                w = np.linalg.solve(self.G[:k, :k], b[:k])
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(self.G[:k, :k], b[:k], rcond=None)[0]
            resid = self.yte - (self.Xtec[:, :k] @ w + mu_y)
            out.append(1.0 - float((resid ** 2).sum()) / self.ss_tot)
        return out

    def curve(self, ytr) -> list[float]:
        return self._r2_for(ytr, range(1, self.m + 1))

    def full_support(self, ytr) -> float:
        return self._r2_for(ytr, [self.m])[0]


def split_train_test(
    N: int, n_te: int, rng: np.random.Generator, groups: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(test_idx, train_idx). With ``groups``, whole groups go to one side.

    Positions are deduplicated by exact bytes, so the 8 board symmetries survive
    as separate rows. They are legitimately distinct inputs (the CNN is not
    rotation equivariant) but they are statistically DEPENDENT, and for a
    rotation-invariant concept a position and its rotation are a near-duplicate
    (x, y) pair. Splitting by orbit instead of by row keeps such a pair on the
    same side, so the held-out R2 is not inflated by that dependence.

    Group IDs come from ``scripts/compute_orbit_ids.py``.
    """
    if groups is None:
        perm = rng.permutation(N)
        return perm[:n_te], perm[n_te:]

    order = np.argsort(groups, kind="stable")
    _, counts = np.unique(groups[order], return_counts=True)
    gperm = rng.permutation(counts.size)
    # Take whole groups until the test budget is reached.
    n_test_groups = int(np.searchsorted(np.cumsum(counts[gperm]), n_te) + 1)
    is_test = np.zeros(counts.size, dtype=bool)
    is_test[gperm[:n_test_groups]] = True
    mask = np.repeat(is_test, counts)          # aligned to the sorted order
    return order[mask], order[~mask]


def restricted_r2_curve(
    codes: np.ndarray, y: np.ndarray, order: np.ndarray, cfg: DilutionConfig,
    groups: np.ndarray | None = None,
) -> dict:
    """Held-out restricted-R2 as latents are added in ``order``.

    The curve is averaged over ``cfg.n_splits`` independent train/test resamples
    rather than a single split. A single split makes every downstream number --
    and especially ``knee_k``, which reads off a threshold crossing -- a hostage
    to one draw: for a base-rate-0.02 concept a single 30% test split holds only
    a few hundred positives, so split noise alone can move the crossing by many
    latents. ``asymptote_r2_std`` reports the across-split spread so that
    instability is visible in the report instead of silently priced in.

    Returns the mean curve (k=1..m), its asymptote (last point) and spread, the
    knee (smallest k reaching knee_frac * asymptote), and a permutation-null
    asymptote (mean over cfg.n_perm label shuffles per split, at full support).
    """
    rng = np.random.default_rng(cfg.seed)
    N = codes.shape[0]
    m = len(order)
    X_all = codes[:, order]
    n_te = max(1, int(cfg.test_frac * N))

    split_curves, null_vals, asymptotes = [], [], []
    for _ in range(max(1, cfg.n_splits)):
        te_idx, tr_idx = split_train_test(N, n_te, rng, groups)
        ytr, yte = y[tr_idx], y[te_idx]
        ols = _NestedOLS(X_all[tr_idx], ytr, X_all[te_idx], yte)

        c = ols.curve(ytr)
        split_curves.append(c)
        asymptotes.append(c[-1] if c else 0.0)

        for _ in range(cfg.n_perm):
            yp = ytr.copy()
            rng.shuffle(yp)
            null_vals.append(ols.full_support(yp))

    curve = np.mean(split_curves, axis=0).tolist() if split_curves else []
    asymptote = curve[-1] if curve else 0.0
    asym_std = float(np.std(asymptotes)) if len(asymptotes) > 1 else 0.0

    knee = m
    if asymptote > 0:
        target = cfg.knee_frac * asymptote
        for k, v in enumerate(curve, start=1):
            if v >= target:
                knee = k
                break

    null_r2 = float(np.mean(null_vals)) if null_vals else 0.0

    # Share of the total recoverable signal already carried by the single
    # best-associated latent. This is the metric that separates "one latent
    # does the job, the rest is noise-level creep" from real dilution; knee_k
    # alone cannot, because a 2% upward drift in the tail of the curve pushes
    # the 90%-of-asymptote crossing arbitrarily far to the right.
    solo_r2 = curve[0] if curve else 0.0
    solo_frac = (solo_r2 / asymptote) if asymptote > 0 else 0.0

    return {
        "curve": [round(float(v), 4) for v in curve],
        "asymptote_r2": round(float(asymptote), 4),
        "asymptote_r2_std": round(asym_std, 4),
        "solo_r2": round(float(solo_r2), 4),
        "solo_frac": round(float(np.clip(solo_frac, 0.0, 1.0)), 4),
        "knee_k": int(knee),
        "null_r2": round(float(null_r2), 4),
        "n_curve_rows": int(N),
        "n_curve_positives": int(y.sum()),
    }


# ---------------------------------------------------------------------------
# Per-concept diagnosis
# ---------------------------------------------------------------------------


def solo_frac_of(metrics: dict) -> float:
    """``solo_frac`` for a metrics dict, back-filling it from ``curve`` when the
    dict predates rule 3A.2. Lets ``classify`` re-run on stored JSON reports
    without recomputing anything (see the ``reclassify`` CLI command)."""
    if "solo_frac" in metrics:
        return float(metrics["solo_frac"])
    curve, asym = metrics.get("curve") or [], metrics.get("asymptote_r2", 0.0)
    if not curve or asym <= 0:
        return 0.0
    return float(min(max(curve[0] / asym, 0.0), 1.0))


def classify(metrics: dict, cfg: DilutionConfig) -> str:
    """Map raw metrics to {absent, captured, diluted, tiled}. Pure function of
    the numbers in ``metrics`` and the thresholds in ``cfg`` -- auditable, and
    re-runnable on a stored report without touching the SAE codes.

    Rule 3A.3. Two changes from 3A.2, both recorded in
    ``docs/diary/2026-08-16_rule-3A3.md``:

    **(1) A random-model floor.** ``absent`` now also fires when the concept is
    no more recoverable from this dictionary than from one trained on an
    UNTRAINED network's activations. The permutation null cannot do this job: it
    shuffles the labels but leaves the dictionary's structure intact, so it
    tests "is this signal real?", not "is this signal LEARNED?". Measured
    2026-08-14, the random-model conv2 dictionary returns ``diluted`` on every
    real gorilla threat concept (R2 0.025-0.034, above ``absent_floor``), i.e.
    an untrained network passed gate G-3A at 100%. The floor only applies when
    ``random_asymptote_r2`` is present; when it is absent the verdict is
    provisional and ``random_control_applied`` records that.

    **(2) ``diluted`` and ``tiled`` are collapsed into ``spread``.** The tiling
    test (``support_overlap < tau AND neg_coupling_frac > 0.5``) fired 42 times
    in 1,100 verdicts and *every* firing had ``community_size == 2`` with
    exactly one negative edge -- a Bernoulli on the sign of a single partial
    correlation, not a measurement of tiling. Rescoping the same statistics to
    the full candidate list does not rescue it: mean pairwise Jaccard is
    ~0.10 for CAPTURED gorilla concepts and ~0.10 for spread tiger concepts, so
    the statistic does not discriminate at either scope. Separating "diluted"
    from "tiled" needs a new statistic validated on a planted positive control
    (y = OR of k disjoint latents must come out tiled); until that exists,
    claiming the distinction would be reporting noise as a finding.

    The ``captured`` branch has two *sufficient* conditions:

      (a) knee_k <= captured_k AND community_size <= captured_size   [3A.1]
      (b) solo_frac >= captured_solo_frac
          AND (intrinsic_dim <= captured_idim OR knee_k <= captured_k)

    In (b), ``solo_frac`` is the necessary term -- the concept must actually be
    concentrated in ONE latent -- and the second term confirms the carrier is
    low-dimensional by either available measure. Either measure suffices because
    both are one-sided: ``intrinsic_dim`` is inflated by redundant community
    members that add no information, and ``knee_k`` is inflated by noise creep;
    a concept that is concentrated *and* clean on either one is not diluted.

    (b) was added after the 2026-07-27 run, where (a) alone put the *supervised*
    anchored positive control at ``diluted`` on 22/23 concepts despite
    community_size 2 and intrinsic_dim 1.01 -- i.e. it failed the calibration
    check pre-registered in the 3A method spec. Two independent causes:
    ``knee_k`` is measured over the association-ranked candidate list while
    ``community_size`` is measured over the community (mixing scopes), and
    ``community_size`` counts co-firing *redundancy*, which is not the same as
    the dimensionality actually needed. (b) is scope-consistent: both of its
    terms describe how concentrated the signal is, not how many latents happen
    to co-fire. (a) is retained because it is correct whenever it fires, so no
    previously-``captured`` verdict changes.

    Rationale and the full before/after tally:
    ``docs/diary/2026-07-27_3A-dilution-results.md``.
    """
    gap = metrics["asymptote_r2"] - metrics["null_r2"]
    if gap < cfg.absent_margin or metrics["asymptote_r2"] < cfg.absent_floor:
        return "absent"
    # The learned-signal floor. Only applies where a random-model control was
    # actually measured -- a control named in metadata but never computed (the
    # fc1 case, where the control SAE has zero alive latents) must NOT be
    # silently treated as a pass.
    rand = metrics.get("random_asymptote_r2")
    if rand is not None and metrics["asymptote_r2"] - rand < cfg.random_margin:
        return "absent"
    if metrics["knee_k"] <= cfg.captured_k and metrics["community_size"] <= cfg.captured_size:
        return "captured"
    if solo_frac_of(metrics) >= cfg.captured_solo_frac and (
            metrics["intrinsic_dim"] <= cfg.captured_idim
            or metrics["knee_k"] <= cfg.captured_k):
        return "captured"
    return "spread"


# Quantities ``classify`` thresholds on that carry a measurable uncertainty.
# Perturbing the METRIC rather than the threshold means one entry here covers
# every rule that reads it -- ``asymptote_r2`` alone feeds three separate
# comparisons in ``classify`` (the permutation gap, the absolute floor and the
# learned-signal floor), and they must move together.
#
# name -> how to get its sd from a metrics dict
_BAND_SOURCES: tuple[tuple[str, str], ...] = (
    # Measured WITHIN the run. NOTE the sqrt(n_splits) in ``_band_sd``:
    # ``asymptote_r2`` is the MEAN over n_splits resamples while
    # ``asymptote_r2_std`` is the sd ACROSS them, so the uncertainty OF THE
    # STORED NUMBER is sd/sqrt(n_splits), not sd. Banding the mean at the
    # across-split sd made K04's conv2 verdicts 46% undecided against a
    # measured 17% flip rate -- a scale error, not a finding.
    ("asymptote_r2", "asymptote_r2_std"),
    # Measured ACROSS SEEDS; not estimable from one run, so it comes from cfg.
    ("solo_frac", "@cfg.solo_frac_seed_sd"),
)


def _band_sd(key: str, source: str, metrics: dict, cfg: DilutionConfig) -> float:
    """Uncertainty of the STORED value of ``key``, in its own units."""
    if source.startswith("@cfg."):
        return float(getattr(cfg, source[len("@cfg."):]))
    sd = float(metrics.get(source) or 0.0)
    if key == "asymptote_r2" and cfg.n_splits > 1:
        # sd of a mean of n_splits draws...
        sd /= math.sqrt(cfg.n_splits)
        # ...floored at the MEASURED cross-seed movement, because a split sd
        # never retrains the SAE and so cannot see seed variation at all.
        sd = max(sd, cfg.asymptote_r2_seed_sd)
    return sd


def classify_with_stability(metrics: dict, cfg: DilutionConfig) -> tuple[str, dict]:
    """``(verdict, stability)`` -- the point verdict plus whether it survives.

    Rule 3A.4. ``classify`` maps a point estimate to a label; this asks whether
    that label is an artefact of where the point happens to sit. Every quantity
    in ``_BAND_SOURCES`` is pushed to +/- ``band_sds`` sd and ``classify`` is
    re-run at each corner of the resulting box. If every corner agrees the
    verdict is ``confident``; otherwise it is ``undecided`` and
    ``flips_on`` names the quantities that, moved alone, change it.

    Why this is not optional, and why it bands EVERY threshold rather than the
    ``captured`` one:

      * `solo_frac` is not bimodal (retraction, 2026-08-21 §4) -- 0.70 cuts a
        continuum, so the point verdict near it carries no information without
        a band.
      * Direct seed replication (§2) measured per-concept verdicts flipping
        12-17% on seed alone. Decomposing those flips: K03's 9 flips were 8 via
        `solo_frac` crossing 0.70 and 1 via another term -- but **all 4 of
        K04's flips were `absent <-> spread`, with `solo_frac` never crossing
        anything**. A band on `solo_frac` alone is structurally blind to half
        the measured instability, which is why the box covers `asymptote_r2`
        (and therefore the random-model floor) as well.

    The two sds are not the same KIND of number, and that limit is part of the
    result: ``asymptote_r2``'s band is a within-run SPLIT sd (row resampling
    only -- it does not retrain the SAE, so it is a LOWER BOUND on seed-to-seed
    movement), while ``solo_frac``'s is a true cross-seed sd but a single
    pooled constant rather than per-concept. The band is therefore the best
    available uncertainty for each quantity, not one calibrated interval:
    read ``undecided`` as "near a boundary relative to how much this number is
    known to move", never as a significance test. Where seeds actually exist,
    the MEASURED flip rate beats this estimate and should be quoted instead.
    """
    verdict = classify(metrics, cfg)

    deltas: dict[str, float] = {}
    for key, source in _BAND_SOURCES:
        sd = _band_sd(key, source, metrics, cfg)
        if sd <= 0:
            continue
        # solo_frac may be absent on a pre-3A.2 report; back-fill it so the
        # perturbed copy has something to move.
        base = solo_frac_of(metrics) if key == "solo_frac" else metrics.get(key)
        if base is None:
            continue
        deltas[key] = sd * cfg.band_sds

    def _base(key: str) -> float:
        return solo_frac_of(metrics) if key == "solo_frac" else float(metrics[key])

    keys = sorted(deltas)

    # All 2^k corners of the box: two quantities can jointly cross a boundary
    # that neither crosses alone (asymptote_r2 down AND solo_frac down both push
    # toward a different label), so a one-at-a-time sweep can miss a flip.
    stable = True
    for corner in itertools.product((-1, 1), repeat=len(keys)):
        probe = dict(metrics)
        for key, sign in zip(keys, corner):
            probe[key] = _base(key) + sign * deltas[key]
        if classify(probe, cfg) != verdict:
            stable = False
            break

    # ...but ATTRIBUTION is one-at-a-time, because "which threshold is doing the
    # work" is the actionable half of the answer.
    flips_on: list[str] = []
    for key in keys:
        for sign in (-1, 1):
            probe = dict(metrics)
            probe[key] = _base(key) + sign * deltas[key]
            if classify(probe, cfg) != verdict:
                flips_on.append(key)
                break

    stability = {
        "stability": "confident" if stable else "undecided",
        "flips_on": flips_on,
        "band_sds": cfg.band_sds,
    }
    return verdict, stability


class RankingCache:
    """Per-run stage-A state: the binarized firing subsample and its rates.

    None of this depends on the concept, so building it once per run instead of
    once per BSP removes an (max_rows x d_dict) materialisation from every
    concept. Pass the same instance to every ``diagnose_concept`` call for a
    given ``h``; it is safe to omit, in which case it is rebuilt per call.
    """

    def __init__(self, h: torch.Tensor, cfg: DilutionConfig):
        N_full = h.shape[0]
        if N_full > cfg.max_rows:
            rng = np.random.default_rng(cfg.seed)
            rows = rng.choice(N_full, size=cfg.max_rows, replace=False)
            rows.sort()
            self.rows = rows
            firing_src = h[rows]
        else:
            self.rows = None
            firing_src = h
        # float32 (not float64): phi is computed through a BLAS product, and the
        # precision is irrelevant for a ranking over {0,1} columns.
        self.firing = (_to_numpy(firing_src) > cfg.fire_threshold).astype(np.float32)
        self.freq = self.firing.mean(axis=0, dtype=np.float64)
        self.alive = (self.freq >= cfg.min_freq) & (self.freq <= cfg.max_freq)


def diagnose_concept(
    h: torch.Tensor,
    y: torch.Tensor,
    cfg: DilutionConfig,
    h_random: torch.Tensor | None = None,
    cache: RankingCache | None = None,
    cache_random: RankingCache | None = None,
    orbit_ids: np.ndarray | None = None,
) -> dict:
    """Full 3A diagnosis of a single concept. Returns a JSON-ready dict.

    h:        (N, d_dict) SAE codes (post-activation feature values).
    y:        (N,) binary concept labels, row-aligned to h.
    h_random: optional (N, d_dict) codes from a random-model SAE, row-aligned;
              adds a random-control asymptote alongside the permutation null.
    """
    y_full = _to_numpy(y).astype(np.float64).ravel()
    base_rate = float(y_full.mean())
    N_full = h.shape[0]

    # --- Stage A: rank candidates. Needs every column, so rows are capped at
    # max_rows to bound the (rows x d_dict) materialisation. Ranking by |phi| is
    # stable well below the full N. The base rate above is taken on the full
    # data so the reported value is exact.
    if cache is None:
        cache = RankingCache(h, cfg)
    rows = cache.rows
    y_rank = y_full if rows is None else y_full[rows]
    order, phi = select_concept_features(
        cache.firing, y_rank, cfg, freq=cache.freq, alive=cache.alive)

    if order.size == 0:
        metrics = {
            "base_rate": round(base_rate, 4),
            "n_candidates": 0,
            "asymptote_r2": 0.0, "asymptote_r2_std": 0.0,
            "solo_r2": 0.0, "solo_frac": 0.0,
            "null_r2": 0.0, "knee_k": 0,
            "community_size": 0, "neg_coupling_frac": 0.0,
            "support_overlap": 0.0, "singleton_community": True,
            "intrinsic_dim": 0.0, "n_curve_rows": 0, "n_curve_positives": 0,
            "top_phi": 0.0, "curve": [], "orbit_aware_split": False,
            "random_control_applied": False,
        }
        # No candidates at all -- absent with nothing to be uncertain about.
        metrics["verdict"] = "absent"
        metrics["verdict_stability"] = "confident"
        metrics["verdict_flips_on"] = []
        return metrics

    # --- Stage B: everything else runs on the top_k selected columns only, so
    # rows are cheap (n x top_k floats). Use more of them -- and enough of them
    # that a rare concept still has min_positives positive rows -- because the
    # curve, the community and the intrinsic dimension are all estimated here.
    n_curve = cfg.curve_rows
    if base_rate > 0:
        n_curve = max(n_curve, int(np.ceil(cfg.min_positives / base_rate)))
    n_curve = min(N_full, max(n_curve, cfg.max_rows))

    codes_cols = _to_numpy(h[:, torch.as_tensor(order, dtype=torch.long)])
    codes_cols = codes_cols.astype(np.float32)
    if N_full > n_curve:
        rng_b = np.random.default_rng(cfg.seed + 1)
        rows_b = rng_b.choice(N_full, size=n_curve, replace=False)
        rows_b.sort()
        codes_sub = codes_cols[rows_b]
        y_np = y_full[rows_b]
    else:
        rows_b = None
        codes_sub, y_np = codes_cols, y_full
    del codes_cols
    groups = None
    if orbit_ids is not None:
        groups = orbit_ids if rows_b is None else orbit_ids[rows_b]
    firing_sub = (codes_sub > cfg.fire_threshold).astype(np.float64)

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

    r2 = restricted_r2_curve(codes_sub, y_np, np.arange(len(order)), cfg, groups=groups)

    metrics = {
        "base_rate": round(base_rate, 4),
        "n_candidates": int(order.size),
        "top_phi": round(float(phi[0]), 4),
        "asymptote_r2": r2["asymptote_r2"],
        "asymptote_r2_std": r2["asymptote_r2_std"],
        "solo_r2": r2["solo_r2"],
        "solo_frac": r2["solo_frac"],
        "null_r2": r2["null_r2"],
        "knee_k": r2["knee_k"],
        "community_size": int(len(community)),
        "neg_coupling_frac": round(neg_frac, 4),
        "support_overlap": round(overlap, 4),
        "singleton_community": bool(len(community) <= 1),
        "intrinsic_dim": round(float(idim), 4),
        "knee_over_idim": round(r2["knee_k"] / max(1.0, idim), 4),
        "n_curve_rows": r2["n_curve_rows"],
        "n_curve_positives": r2["n_curve_positives"],
        "orbit_aware_split": bool(orbit_ids is not None),
        "curve": r2["curve"],
    }

    # False until a control number is actually produced. A control that was
    # supplied but yielded nothing -- the fc1 case, where the random-model SAE
    # has ZERO alive latents so `select_concept_features` returns empty -- must
    # not read as "tested and passed". Rule 3A.3 keys the learned-signal floor
    # on the presence of the number, and the gate reports the coverage.
    metrics["random_control_applied"] = False
    if h_random is not None:
        # Same two-stage treatment, on the SAME rows, so the control is
        # row-aligned with the real measurement.
        if cache_random is None:
            cache_random = RankingCache(h_random, cfg)
        ro, _ = select_concept_features(
            cache_random.firing, y_rank, cfg,
            freq=cache_random.freq, alive=cache_random.alive)
        if ro.size:
            rcols = _to_numpy(
                h_random[:, torch.as_tensor(ro, dtype=torch.long)]).astype(np.float32)
            rcols = rcols if rows_b is None else rcols[rows_b]
            rr = restricted_r2_curve(rcols, y_np, np.arange(len(ro)), cfg, groups=groups)
            metrics["random_asymptote_r2"] = rr["asymptote_r2"]
            metrics["random_control_applied"] = True

    verdict, stability = classify_with_stability(metrics, cfg)
    metrics["verdict"] = verdict
    metrics["verdict_stability"] = stability["stability"]
    metrics["verdict_flips_on"] = stability["flips_on"]
    return metrics
