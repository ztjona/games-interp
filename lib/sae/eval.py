"""SAE evaluation metrics: coverage and board reconstruction.

Coverage measures how well SAE features align with known Board State Properties
(BSPs).  Board reconstruction measures whether high-precision features can
recover the full board state.

All functions operate on pre-computed tensors and are game-agnostic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Animal / BSP-set naming
# ---------------------------------------------------------------------------

# Per project convention: an animal name is ``{basis}{ChampionSuffix?}`` where
# basis is all-lowercase (``gorilla``, ``hawk``, ``fox``) and the champion
# suffix (if any) starts with an uppercase letter (``S4``, ``Ta``, ``Aa``).
# The BSP schema is identical across distributions of the same basis, so the
# schema file is keyed by basis alone; only the label tensor differs per
# champion. See CLAUDE.md § "Domain conventions".
# An animal is `{basis}{Suffix?}`. Two suffix kinds exist (CLAUDE.md, Domain
# conventions): a CHAMPION tag, which is upper-case-initial (`gorillaVe`), and a
# POOL-SIZE tag, which is `<digits>k` (`gorilla677k`, from unify_positions.py).
# The numeric form was missing here, so `gorilla677k` resolved to itself, the
# basis fallback in resolve_schema_path never fired, and unify_positions.py had
# to write a suffixed schema copy per pool size just to be findable -- which is
# what produced the duplicate, convention-violating schemas on disk.
_ANIMAL_BASIS_RE = re.compile(r"^([a-z]+)(\d+k|[A-Z].*)?$")


def animal_to_basis(animal: str) -> str:
    """Return the basis name of an animal, e.g. ``gorillaS4`` → ``gorilla``.

    If ``animal`` is already a basis (no uppercase suffix) it is returned
    unchanged. If the name does not match the convention, returns the input
    unchanged so callers can use it as-is.
    """
    m = _ANIMAL_BASIS_RE.match(animal)
    return m.group(1) if m else animal


def resolve_schema_path(data_dir: Path, animal: str) -> Path | None:
    """Resolve ``bsp_schema-{animal}_*.json``, falling back to the basis.

    Lookup order:
      1. ``bsp_schema-{animal}_[0-9]*.json`` (legacy per-champion file, if any)
      2. ``bsp_schema-{basis}_[0-9]*.json``  where basis = animal_to_basis(animal)

    Returns ``None`` if neither exists.
    """
    matches = sorted(data_dir.glob(f"bsp_schema-{animal}_[0-9]*.json"))
    if matches:
        return matches[0]
    basis = animal_to_basis(animal)
    if basis != animal:
        matches = sorted(data_dir.glob(f"bsp_schema-{basis}_[0-9]*.json"))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Concept families (cross-basis grouping)
# ---------------------------------------------------------------------------
#
# A BSP *basis* (gorilla / hawk / tiger) is a packaging convention; a *concept
# family* is the underlying game fact that several bases each describe in their
# own vocabulary. The mapping is stamped onto the schema by
# ``compute_bsp_labels.py`` (source: the game module's ``CONCEPT_FAMILIES``), so
# these readers stay game-agnostic: they only read what the data declares and
# never carry a mapping of their own.


# Ascending agent-relativity; mirrors the game module's FAMILY_ROLE_ORDER.
FAMILY_ROLE_ORDER: tuple[str, ...] = ("state", "state_any", "agent_relative")


def category_families(schema: dict) -> dict[str, dict[str, str]]:
    """Return ``{category: {"concept_family", "family_role"}}`` from a schema.

    Reads the schema-level ``category_families`` rollup when present, and
    otherwise reconstructs it from the per-BSP stamps. Returns ``{}`` for a
    schema written before concept families existed, which callers should treat
    as "this schema predates families -- re-stamp it with
    ``scripts/stamp_concept_families.py``" rather than as "no families".
    """
    rollup = schema.get("category_families")
    if isinstance(rollup, dict) and rollup:
        return rollup
    out: dict[str, dict[str, str]] = {}
    for bsp in schema.get("bsps", []):
        family = bsp.get("concept_family")
        if not family:
            continue
        out[bsp.get("category", "unknown")] = {
            "concept_family": family,
            "family_role": bsp.get("family_role", "state"),
        }
    return out


def family_of_category(schema: dict, category: str) -> str | None:
    """Concept family for one category, or ``None`` if the schema declares none."""
    entry = category_families(schema).get(category)
    return entry.get("concept_family") if entry else None


def derive_triads(schemas: dict[str, dict]) -> dict[str, dict[str, str]]:
    """Derive ``{family: {basis: category}}`` across several bases' schemas.

    ``schemas`` maps a basis label (``"gorilla"``) to its loaded schema dict.
    Within one (family, basis) the category kept is the one at the highest
    agent-relativity that basis offers, so a triad contrasts each basis at its
    most agent-relative phrasing -- the contrast the 2026-05-22 reframing audit
    was actually making.

    Families present in only one basis are dropped: they are not a cross-basis
    comparison, and reporting them as one would be misleading.
    """
    rank = {role: i for i, role in enumerate(FAMILY_ROLE_ORDER)}
    best: dict[str, dict[str, tuple[int, str]]] = {}
    for basis, schema in schemas.items():
        for category, entry in category_families(schema).items():
            family = entry.get("concept_family")
            if not family:
                continue
            score = rank.get(entry.get("family_role", ""), -1)
            slot = best.setdefault(family, {})
            if basis not in slot or score > slot[basis][0]:
                slot[basis] = (score, category)
    return {
        family: {basis: cat for basis, (_, cat) in sorted(per_basis.items())}
        for family, per_basis in sorted(best.items())
        if len(per_basis) > 1
    }


def aggregate_per_category_by_family(
    per_category: dict[str, dict],
    schema: dict,
    metric_keys: tuple[str, ...] = ("mean_mcc", "mean_youden_j",
                                    "mean_mcc_at_pref", "mean_base_rate",
                                    "mean_f1"),
) -> dict[str, dict]:
    """Roll a ``per_category`` block up into concept families.

    Category means are re-weighted by BSP ``count`` so the family mean is the
    mean over BSPs, not the mean over categories -- categories differ in size by
    an order of magnitude (``global`` has 1 BSP, ``cell_attribute`` has 64), so
    an unweighted mean would silently be a different statistic.
    """
    fams = category_families(schema)
    out: dict[str, dict] = {}
    for category, vals in per_category.items():
        entry = fams.get(category)
        if not entry:
            continue
        family = entry["concept_family"]
        n = vals.get("count") or 0
        if not n:
            continue
        acc = out.setdefault(family, {"count": 0, "categories": []})
        acc["count"] += n
        acc["categories"].append(category)
        for key in metric_keys:
            v = vals.get(key)
            if v is None:
                continue
            acc.setdefault(f"_{key}", 0.0)
            acc[f"_{key}"] += float(v) * n
            acc.setdefault(f"_{key}_n", 0)
            acc[f"_{key}_n"] += n
    for family, acc in out.items():
        acc["categories"] = sorted(acc["categories"])
        for key in metric_keys:
            total, n = acc.pop(f"_{key}", None), acc.pop(f"_{key}_n", None)
            if total is not None and n:
                acc[key] = round(total / n, 4)
    return out


# ---------------------------------------------------------------------------
# Feature–BSP matching
# ---------------------------------------------------------------------------


@dataclass
class FeatureBSPMatching:
    """Result of matching SAE features to BSPs.

    Attributes:
        precision: (d_dict, num_bsps) float tensor
        recall:    (d_dict, num_bsps) float tensor
        f1:        (d_dict, num_bsps) float tensor
        mcc:       (d_dict, num_bsps) float tensor — Matthews correlation coef.
        best_f1_per_bsp:       (num_bsps,) best F1 for each BSP
        best_feature_per_bsp:  (num_bsps,) index of best (F1) feature per BSP
        best_mcc_per_bsp:      (num_bsps,) best MCC for each BSP
        best_feature_per_bsp_mcc: (num_bsps,) index of best (MCC) feature per BSP
        youden_j:  (d_dict, num_bsps) TPR - FPR, prevalence-INVARIANT
        best_j_per_bsp:        (num_bsps,) best Youden J for each BSP
        best_feature_per_bsp_j: (num_bsps,) index of best (J) feature per BSP
        base_rates:            (num_bsps,) base rate (positive-class freq) per BSP
        f1_lift_per_bsp:       (num_bsps,) best_f1 minus trivial-baseline F1, clipped at 0
        trivial_f1_per_bsp:    (num_bsps,) the "always positive" F1 = 2p/(1+p)
    """

    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    best_f1_per_bsp: torch.Tensor
    best_feature_per_bsp: torch.Tensor
    # Added 2026-05-11. Optional for back-compat with legacy caches.
    mcc: torch.Tensor | None = None
    best_mcc_per_bsp: torch.Tensor | None = None
    best_feature_per_bsp_mcc: torch.Tensor | None = None

    # Youden's J = TPR - FPR: the prevalence-INVARIANT companion to MCC.
    youden_j: torch.Tensor | None = None
    best_j_per_bsp: torch.Tensor | None = None
    best_feature_per_bsp_j: torch.Tensor | None = None

    # MCC standardised to a reference prevalence -- comparable across
    # populations with different base rates. p_ref is stored so a standardised
    # number is never ambiguous.
    mcc_at_pref: torch.Tensor | None = None
    best_mcc_at_pref_per_bsp: torch.Tensor | None = None
    best_feature_per_bsp_mcc_at_pref: torch.Tensor | None = None
    p_ref: float | None = None
    base_rates: torch.Tensor | None = None
    f1_lift_per_bsp: torch.Tensor | None = None
    trivial_f1_per_bsp: torch.Tensor | None = None


# Reference prevalence for standardised MCC. Chosen on evidence (rank stability
# across p_ref = 0.002..0.100 with zero flips; median observed tiger base rate
# 0.0233; inside the measured range so nothing is extrapolated) -- see
# docs/methods-reference.md S1.4 and scripts/choose_p_ref.py. FROZEN: changing it
# changes every standardised number, so it is stored in each registry row.
P_REF_DEFAULT = 0.025


def mcc_from_rates(tpr, tnr, p):
    """MCC a classifier of quality (tpr, tnr) would score at prevalence ``p``.

    MCC is a deterministic function of (TPR, TNR, prevalence), so a feature's
    measured sensitivity and specificity can be re-expressed at a canonical
    prevalence with NO rows discarded and NO sampling noise. This is the
    analytic, noise-free equivalent of subsample matching (verified to agree to
    4 decimal places), and it is what makes MCC comparable across populations
    whose base rates differ.
    """
    eps = 1e-12
    tp = tpr * p
    fn = (1.0 - tpr) * p
    tn = tnr * (1.0 - p)
    fp = (1.0 - tnr) * (1.0 - p)
    num = tp * tn - fp * fn
    den = torch.sqrt(
        ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp(min=eps)
    )
    return num / den


def match_features_to_bsps(
    h: torch.Tensor,
    bsp_labels: torch.Tensor,
    p_ref: float = P_REF_DEFAULT,
) -> FeatureBSPMatching:
    """Compute precision, recall, F1, MCC between every SAE feature and every BSP.

    Also computes per-BSP base rates and the F1 lift over the trivial
    "always positive" baseline, which is the source of the F1 ≈ 0.667 artifact
    for high-base-rate BSPs.

    Args:
        h:          (N, d_dict) SAE hidden activations (non-negative).
        bsp_labels: (N, num_bsps) binary BSP labels (0.0 or 1.0).

    Returns:
        FeatureBSPMatching with all pairwise metrics and best-feature assignments.
    """
    if h.shape[0] != bsp_labels.shape[0]:
        raise ValueError(
            f"Sample count mismatch: h has {h.shape[0]}, "
            f"bsp_labels has {bsp_labels.shape[0]}"
        )

    with torch.no_grad():
        # Binarize feature activations
        fires = (h > 0).float()  # (N, d_dict)
        labels = bsp_labels.float()  # (N, num_bsps)
        N = float(fires.shape[0])

        # Pairwise counts via matrix multiplication
        # TP[i,j] = sum over samples where feature i fires AND BSP j is true
        tp = fires.T @ labels  # (d_dict, num_bsps)
        fp = fires.T @ (1.0 - labels)  # (d_dict, num_bsps)
        fn = (1.0 - fires).T @ labels  # (d_dict, num_bsps)
        tn = N - tp - fp - fn

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        # Numerically: clamp denominator and treat 0 → 0 (constant predictor).
        mcc_num = tp * tn - fp * fn
        mcc_den = torch.sqrt(
            (tp + fp).clamp(min=eps)
            * (tp + fn).clamp(min=eps)
            * (tn + fp).clamp(min=eps)
            * (tn + fn).clamp(min=eps)
        )
        mcc = mcc_num / mcc_den

        # Youden's J = TPR - FPR = sensitivity + specificity - 1.
        # THE prevalence-invariant companion to MCC. F1, F1-lift, MCC and
        # precision all move with the base rate for a detector of FIXED quality
        # (audited in scripts/prevalence_audit.py: 44-76% swing over
        # p in [0.005, 0.100]); TPR, TNR and J do not move at all. J is
        # literally the (a + b - 1) factor in the MCC numerator, i.e. MCC with
        # the prevalence term stripped out, so MCC-vs-J divergence is a direct
        # read on how much of a difference is prevalence rather than quality.
        # Report the PAIR: J alone is misleading in the other direction, since a
        # latent firing on half the dataset can score J = 0.4 while being
        # useless at base rate 0.02.
        tpr = tp / (tp + fn + eps)
        fpr = fp / (fp + tn + eps)
        youden_j = tpr - fpr

        best_f1_per_bsp, best_feature_per_bsp = f1.max(dim=0)
        best_mcc_per_bsp, best_feature_per_bsp_mcc = mcc.max(dim=0)
        best_j_per_bsp, best_feature_per_bsp_j = youden_j.max(dim=0)

        # Standardised MCC: what each feature would score at P_REF_DEFAULT.
        # Comparable across populations/champions with different base rates.
        mcc_at_pref = mcc_from_rates(tpr, 1.0 - fpr, p_ref)
        best_mcc_at_pref_per_bsp, best_feature_per_bsp_mcc_at_pref = mcc_at_pref.max(dim=0)

        # Base rates and trivial-baseline F1 ("always predict positive")
        base_rates = labels.mean(dim=0)  # (num_bsps,)
        trivial_f1 = (2.0 * base_rates) / (1.0 + base_rates + eps)
        f1_lift = (best_f1_per_bsp - trivial_f1).clamp(min=0.0)

    return FeatureBSPMatching(
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        youden_j=youden_j,
        best_j_per_bsp=best_j_per_bsp,
        best_feature_per_bsp_j=best_feature_per_bsp_j,
        mcc_at_pref=mcc_at_pref,
        best_mcc_at_pref_per_bsp=best_mcc_at_pref_per_bsp,
        best_feature_per_bsp_mcc_at_pref=best_feature_per_bsp_mcc_at_pref,
        p_ref=p_ref,
        best_f1_per_bsp=best_f1_per_bsp,
        best_feature_per_bsp=best_feature_per_bsp,
        best_mcc_per_bsp=best_mcc_per_bsp,
        best_feature_per_bsp_mcc=best_feature_per_bsp_mcc,
        base_rates=base_rates,
        f1_lift_per_bsp=f1_lift,
        trivial_f1_per_bsp=trivial_f1,
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def compute_coverage(matching: FeatureBSPMatching) -> dict[str, float]:
    """Compute coverage: mean of best-F1 across BSPs (plus MCC and F1-lift variants).

    Coverage definitions:
        - coverage:           mean of best-F1 per BSP (literature standard, but
                              base-rate-sensitive — high-base-rate BSPs score
                              F1 ≈ 2p/(1+p) trivially).
        - coverage_mcc:       mean of best-MCC per BSP (base-rate-invariant;
                              0 for any constant predictor).
        - coverage_f1_lift:   mean of (best_f1 − trivial_f1)+, where trivial_f1
                              is the "always predict positive" baseline.

    Args:
        matching: Output of match_features_to_bsps().

    Returns:
        Dict with keys: coverage, coverage_above_50, coverage_above_75,
        coverage_mcc, coverage_mcc_above_25, coverage_mcc_above_50,
        coverage_f1_lift, coverage_f1_lift_above_10,
        num_bsps, min_f1, max_f1, median_f1.
    """
    best_f1 = matching.best_f1_per_bsp
    num_bsps = best_f1.shape[0]

    # `coverage` (mean best-F1) is retained as the SINGLE F1 number, for
    # comparison with the published literature at write-up. Its derivatives
    # (above_50/above_75/min/max/median and the whole F1-lift family) were
    # removed on 2026-08-11: all are prevalence-dependent, none is used to
    # select, rank, gate or conclude, and their presence made the registry read
    # as if the project optimises F1. MCC equivalents cover every real use.
    out: dict[str, float] = {
        "coverage": round(best_f1.mean().item(), 4),
        "num_bsps": num_bsps,
    }

    if matching.best_mcc_per_bsp is not None:
        best_mcc = matching.best_mcc_per_bsp
        out.update(
            {
                "coverage_mcc": round(best_mcc.mean().item(), 4),
                "coverage_mcc_above_25": round(
                    (best_mcc > 0.25).float().mean().item(), 4
                ),
                "coverage_mcc_above_50": round(
                    (best_mcc > 0.50).float().mean().item(), 4
                ),
                "median_mcc": round(best_mcc.median().item(), 4),
                "min_mcc": round(best_mcc.min().item(), 4),
                "max_mcc": round(best_mcc.max().item(), 4),
            }
        )

    if matching.best_j_per_bsp is not None:
        best_j = matching.best_j_per_bsp
        out.update(
            {
                # Prevalence-invariant. Compare THIS across populations or
                # champions with different base rates; coverage_mcc is not safe
                # for that without matching (see docs/methods-reference.md S1).
                "coverage_youden_j": round(best_j.mean().item(), 4),
                "median_youden_j": round(best_j.median().item(), 4),
                "min_youden_j": round(best_j.min().item(), 4),
                "max_youden_j": round(best_j.max().item(), 4),
            }
        )

    if matching.base_rates is not None:
        br = matching.base_rates
        out.update(
            {
                # Prevalence belongs in the registry: it decides how hard each
                # concept is to score and which comparisons are legitimate at
                # all. Without it a reader cannot tell a weak SAE from a rare
                # concept. See docs/methods-reference.md S1.1.
                "mean_base_rate": round(br.mean().item(), 4),
                "median_base_rate": round(br.median().item(), 4),
                "min_base_rate": round(br.min().item(), 4),
                "max_base_rate": round(br.max().item(), 4),
                # How much of the menu sits where metrics misbehave.
                "frac_bsps_very_rare": round((br < 0.01).float().mean().item(), 4),
                "frac_bsps_trivial_f1": round((br > 0.40).float().mean().item(), 4),
            }
        )

    if matching.best_mcc_at_pref_per_bsp is not None:
        best_std = matching.best_mcc_at_pref_per_bsp
        out.update(
            {
                # Prevalence-STANDARDISED. This is the number to compare across
                # champions/populations whose base rates differ; raw
                # coverage_mcc is not safe for that (see methods-reference S1.4).
                "coverage_mcc_at_pref": round(best_std.mean().item(), 4),
                "median_mcc_at_pref": round(best_std.median().item(), 4),
                "p_ref": matching.p_ref,
            }
        )



    return out


def compute_per_category_coverage(
    matching: FeatureBSPMatching,
    bsp_schema: dict | list | None,
) -> dict[str, dict[str, float]] | None:
    """Compute coverage broken down by BSP category.

    Args:
        matching:   Output of match_features_to_bsps().
        bsp_schema: Either a list of BSP dicts (each with 'category' key),
                    or the full schema dict with a 'bsps' key containing
                    that list. Index must align with matching.best_f1_per_bsp.

    Returns:
        Dict mapping category name to {count, mean_f1, min_f1, max_f1, median_f1},
        or None if bsp_schema is not available.
    """
    if bsp_schema is None:
        return None

    # Accept both the full schema dict and bare list
    if isinstance(bsp_schema, dict):
        bsp_list = bsp_schema.get("bsps", [])
    else:
        bsp_list = bsp_schema

    if not bsp_list:
        return None

    best_f1 = matching.best_f1_per_bsp
    best_mcc = matching.best_mcc_per_bsp
    f1_lift = matching.f1_lift_per_bsp
    base_rates = matching.base_rates

    # Group BSP indices by category
    categories: dict[str, list[int]] = {}
    for i, bsp in enumerate(bsp_list):
        cat = bsp.get("category", "unknown")
        categories.setdefault(cat, []).append(i)

    result = {}
    for cat, indices in sorted(categories.items()):
        # ONE F1 number only, for the literature comparison at write-up. The
        # F1 derivatives (min/max/median/lift) were dropped on 2026-08-11: they
        # are all prevalence-dependent, none is used to select or conclude, and
        # a registry full of F1 columns reads as if the project optimises F1.
        entry = {
            "count": len(indices),
            "mean_f1": round(best_f1[indices].mean().item(), 4),
        }
        if best_mcc is not None:
            cat_mcc = best_mcc[indices]
            entry["mean_mcc"] = round(cat_mcc.mean().item(), 4)
            entry["median_mcc"] = round(cat_mcc.median().item(), 4)
            entry["min_mcc"] = round(cat_mcc.min().item(), 4)
            entry["max_mcc"] = round(cat_mcc.max().item(), 4)
            # Within-category spread. Reporting Standard clause 3: a category is
            # many independent BSPs and the mean can hide a structural split
            # (champYb's anchored line_winnable: rows/cols 0.66-0.83, the two
            # DIAGONALS 0.20, invisible in the 0.60 mean).
            entry["sd_mcc"] = round(cat_mcc.std(unbiased=False).item(), 4)
            entry["heterogeneous"] = bool(
                (cat_mcc.max() - cat_mcc.min()).item() > 0.30
            )
        if matching.best_j_per_bsp is not None:
            entry["mean_youden_j"] = round(
                matching.best_j_per_bsp[indices].mean().item(), 4)
        if matching.best_mcc_at_pref_per_bsp is not None:
            entry["mean_mcc_at_pref"] = round(
                matching.best_mcc_at_pref_per_bsp[indices].mean().item(), 4)
        if base_rates is not None:
            entry["mean_base_rate"] = round(base_rates[indices].mean().item(), 4)
        result[cat] = entry

    return result


def compute_feature_sharing(matching: FeatureBSPMatching) -> dict[str, float | int]:
    """Summarize reuse of SAE features under independent best-BSP matching."""
    num_bsps = int(matching.best_feature_per_bsp.shape[0])
    num_features = int(matching.f1.shape[0])

    if num_bsps == 0 or num_features == 0:
        return {
            "num_features_used_by_best_matches": 0,
            "fraction_features_used_by_best_matches": 0.0,
            "mean_bsps_per_used_feature": 0.0,
            "max_bsps_per_feature": 0,
            "num_shared_features": 0,
            "fraction_shared_features": 0.0,
            "num_bsps_with_unique_best_feature": 0,
            "fraction_bsps_with_unique_best_feature": 0.0,
            "num_bsps_with_shared_best_feature": 0,
            "fraction_bsps_with_shared_best_feature": 0.0,
        }

    reuse_counts = torch.bincount(matching.best_feature_per_bsp, minlength=num_features)
    used_counts = reuse_counts[reuse_counts > 0]

    num_features_used = int(used_counts.shape[0])
    num_shared_features = int((used_counts > 1).sum().item())
    num_unique_bsp = int((used_counts == 1).sum().item())
    num_shared_bsp = num_bsps - num_unique_bsp

    return {
        "num_features_used_by_best_matches": num_features_used,
        "fraction_features_used_by_best_matches": round(
            num_features_used / num_features, 4
        ),
        "mean_bsps_per_used_feature": round(used_counts.float().mean().item(), 4),
        "max_bsps_per_feature": int(used_counts.max().item()),
        "num_shared_features": num_shared_features,
        "fraction_shared_features": round(
            num_shared_features / max(num_features_used, 1), 4
        ),
        "num_bsps_with_unique_best_feature": num_unique_bsp,
        "fraction_bsps_with_unique_best_feature": round(num_unique_bsp / num_bsps, 4),
        "num_bsps_with_shared_best_feature": num_shared_bsp,
        "fraction_bsps_with_shared_best_feature": round(num_shared_bsp / num_bsps, 4),
    }


# ---------------------------------------------------------------------------
# Board Reconstruction
# ---------------------------------------------------------------------------


def compute_board_reconstruction(
    matching: FeatureBSPMatching,
    h: torch.Tensor,
    bsp_labels: torch.Tensor,
    precision_threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute board reconstruction from high-precision SAE features.

    For each BSP, select the feature with highest precision above the threshold.
    Then compute classification accuracy using that feature's firing pattern as
    the BSP prediction.

    Args:
        matching:             Output of match_features_to_bsps().
        h:                    (N, d_dict) SAE hidden activations.
        bsp_labels:           (N, num_bsps) binary BSP labels.
        precision_threshold:  Minimum precision to qualify a feature (default 0.9).

    Returns:
        Dict with keys: board_reconstruction, num_reconstructable_bsps,
        fraction_reconstructable, mean_accuracy, per_bsp_accuracy (list).
    """
    num_bsps = bsp_labels.shape[1]

    with torch.no_grad():
        fires = (h > 0).float()  # (N, d_dict)
        labels = bsp_labels.float()  # (N, num_bsps)

        # For each BSP, find the feature with highest precision above threshold
        precision = matching.precision  # (d_dict, num_bsps)

        # Mask features below threshold
        above_thresh = precision >= precision_threshold  # (d_dict, num_bsps)

        # For each BSP: best precision feature among those above threshold
        # Set below-threshold precisions to -1 so they can't win argmax
        masked_precision = precision.clone()
        masked_precision[~above_thresh] = -1.0
        best_prec_per_bsp, best_prec_feature = masked_precision.max(dim=0)

        # Which BSPs have at least one qualifying feature?
        reconstructable = best_prec_per_bsp >= precision_threshold  # (num_bsps,)
        num_reconstructable = int(reconstructable.sum().item())

        if num_reconstructable == 0:
            return {
                "board_reconstruction": 0.0,
                "num_reconstructable_bsps": 0,
                "fraction_reconstructable": 0.0,
                "mean_accuracy": 0.0,
                "per_bsp_accuracy": [],
            }

        # Compute accuracy for each reconstructable BSP
        per_bsp_accuracy = []
        accuracy_sum = 0.0

        for j in range(num_bsps):
            if not reconstructable[j]:
                per_bsp_accuracy.append(None)
                continue

            feat_idx = int(best_prec_feature[j].item())
            pred = fires[:, feat_idx]  # (N,)
            true = labels[:, j]  # (N,)
            correct = (pred == true).float().mean().item()
            per_bsp_accuracy.append(round(correct, 4))
            accuracy_sum += correct

        mean_acc = accuracy_sum / num_reconstructable

    return {
        "board_reconstruction": round(mean_acc, 4),
        "num_reconstructable_bsps": num_reconstructable,
        "fraction_reconstructable": round(num_reconstructable / num_bsps, 4),
        "mean_accuracy": round(mean_acc, 4),
        "per_bsp_accuracy": per_bsp_accuracy,
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate_sae(
    sae,
    activations: torch.Tensor,
    bsp_labels: torch.Tensor,
    precision_threshold: float = 0.9,
    batch_size: int = 4096,
) -> dict:
    """Run full Layer 1 evaluation: coverage + board reconstruction.

    Args:
        sae:                  A BaseSAE instance.
        activations:          (N, d_input) raw activation tensor.
        bsp_labels:           (N, num_bsps) binary BSP labels.
        precision_threshold:  Threshold for board reconstruction (default 0.9).
        batch_size:           Process activations in chunks to control memory.

    Returns:
        Dict combining coverage and reconstruction metrics plus structural
        metrics (FVU, L0, dead_features_pct, mse).

    Note on BatchTopK:
        BatchTopK is evaluated in *training mode* (batch-level sparsity) so
        that L0 matches the training-time sparsity target.  The calibrated
        per-feature inference thresholds lose accuracy on out-of-distribution
        batch sizes and are therefore not used for evaluation.
    """
    from .architectures import BatchTopKSAE
    from .train import compute_metrics

    # BatchTopK must run in training mode to enforce batch-level sparsity;
    # all other architectures use eval mode.
    is_batchtopk = isinstance(sae, BatchTopKSAE)
    if is_batchtopk:
        sae.train()
    else:
        sae.eval()

    N = activations.shape[0]
    device = next(sae.parameters()).device

    # Encode all activations to get hidden representations
    h_parts = []
    n_batches = (N + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, N, batch_size), total=n_batches, desc="Encoding", unit="batch"
        ):
            batch = activations[i : i + batch_size].to(device)
            result = sae(batch)
            h_parts.append(result["h"].cpu())

    h = torch.cat(h_parts, dim=0)  # (N, d_dict) on CPU
    bsp_labels_cpu = bsp_labels.cpu()

    # Structural metrics (on a sample, like training)
    eval_sample = activations[: min(N, batch_size)].to(device)
    structural = compute_metrics(sae, eval_sample)

    # Feature-BSP matching
    matching = match_features_to_bsps(h, bsp_labels_cpu)

    # Coverage
    coverage = compute_coverage(matching)

    # Feature sharing / reuse under the current many-to-one matching policy
    feature_sharing = compute_feature_sharing(matching)

    # Board reconstruction
    reconstruction = compute_board_reconstruction(
        matching, h, bsp_labels_cpu, precision_threshold=precision_threshold
    )

    return {
        **structural,
        **coverage,
        "feature_sharing": feature_sharing,
        **reconstruction,
    }
