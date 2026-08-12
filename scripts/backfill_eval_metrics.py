"""Bring every eval_registry.json row up to the CURRENT metric schema.

Walks ``saes/<game>/eval_registry.json`` and, for each row, adds the metrics
introduced after it was written and removes the ones since retired, so old and
new rows are directly comparable:

  ADDS     coverage_mcc*        MCC family (headline metric)
           coverage_youden_j    Youden's J -- prevalence-INVARIANT
           coverage_mcc_at_pref MCC restated at p_ref = 0.025
           mean/median/min/max_base_rate, frac_bsps_very_rare
  REMOVES  coverage_above_50/75, min/max/median_f1, and the whole F1-lift
           family -- prevalence-dependent F1 derivatives that nothing selects,
           ranks, gates or concludes on.

Youden's J and mcc_at_pref are functions of the full ``(d_dict, num_bsps)`` rate
tensors, so -- unlike ``f1_lift`` -- they CANNOT be derived from the stored
per-BSP bests. Any row whose matching cache predates them is rebuilt from the
``_h`` codes; a row whose ``_h`` cache is gone is reported by name with the
exact ``sae_eval`` command that regenerates it, never silently left stale.

For each entry the script looks for:

  saes/<game>/cache/{run_id}_matching-{bsp_set}.pt    (required)
  saes/<game>/cache/{run_id}_h.pt                     (required for J / MCC)
  data/<game>/bsp_labels-<bsp_set>_*.pt               (required)
  data/<game>/bsp_schema-<bsp_set>_*.json             (required for per-cat)

The registry is written with a ``.bak`` sibling kept on disk.

Usage:
    backfill_eval_metrics.py [--game=<g>] [--dry-run] [--force-recompute]
    backfill_eval_metrics.py (-h | --help)

Options:
    --game=<g>           Game name [default: quarto].
    --dry-run            Show what would change, do not write anything.
    --force-recompute    Re-derive metrics even for rows that already carry the
                         current schema.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from docopt import docopt

# Local imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.sae.eval import (  # noqa: E402
    FeatureBSPMatching,
    compute_coverage,
    compute_per_category_coverage,
    match_features_to_bsps,
    resolve_schema_path,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# --- path resolution helpers (mirror sae_eval.py) ---------------------------


def _resolve_bsp_paths(
    game: str,
    animal: str,
    expected_count: int | None = None,
) -> tuple[Path | None, Path | None]:
    data_dir = ROOT / "data" / game
    label_matches = sorted(data_dir.glob(f"bsp_labels-{animal}_*.pt"))

    # If we know how many BSPs the matching cache holds, prefer the file
    # whose filename count matches exactly.
    def _pick(paths: list[Path]) -> Path | None:
        if not paths:
            return None
        if expected_count is not None:
            for p in paths:
                # filename pattern: bsp_labels-<animal>_<N>.{pt,json}
                stem = p.stem  # drops extension
                try:
                    n = int(stem.rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                if n == expected_count:
                    return p
        return paths[-1]

    # Schema lookup falls back from per-champion to basis (gorillaS4 → gorilla)
    schema_path = resolve_schema_path(data_dir, animal)
    return _pick(label_matches), schema_path


def _load_bsp_labels(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("labels", "bsp_labels", "tensor"):
            if key in obj:
                return obj[key].float()
        # fallback: first tensor value
        for v in obj.values():
            if torch.is_tensor(v) and v.ndim == 2:
                return v.float()
        raise ValueError(f"Cannot find label tensor in {path}")
    return obj.float()


def _load_schema(path: Path) -> list | dict | None:
    if not path or not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# Fields a *current* matching cache must carry. Youden's J and mcc_at_pref are
# functions of the full (d_dict, num_bsps) rate tensors, so unlike f1_lift they
# CANNOT be derived from the stored per-BSP bests -- they need the `_h` codes.
# Listing them here (rather than testing one field ad hoc) is what stops the
# next added metric from silently not being backfilled.
REQUIRED_MATCHING_FIELDS = (
    "mcc",
    "best_mcc_per_bsp",
    "best_feature_per_bsp_mcc",
    "youden_j",
    "best_j_per_bsp",
    "best_feature_per_bsp_j",
    "mcc_at_pref",
    "best_mcc_at_pref_per_bsp",
    "best_feature_per_bsp_mcc_at_pref",
    "base_rates",
)

# Registry metric keys retired on 2026-08-11. They are prevalence-dependent F1
# derivatives that nothing selects, ranks, gates or concludes on, and leaving
# them on old rows means old and new rows do not share one schema -- so a reader
# cannot tell "this run scored badly" from "this row is from before the change".
RETIRED_METRIC_KEYS = (
    "coverage_above_50",
    "coverage_above_75",
    "min_f1",
    "max_f1",
    "median_f1",
    "coverage_f1_lift",
    "coverage_f1_lift_above_10",
    "coverage_f1_lift_above_25",
    "median_f1_lift",
)

# Same, one level down, inside per_category.
RETIRED_PER_CATEGORY_KEYS = ("min_f1", "max_f1", "median_f1", "mean_f1_lift")

# Registry keys a row must have to count as current. Keep this keyed on the
# NEWEST metrics: a row is re-opened automatically whenever a metric is added.
CURRENT_METRIC_KEYS = (
    "coverage_mcc",
    "coverage_youden_j",
    "coverage_mcc_at_pref",
    "mean_base_rate",
)


def _matching_is_current(cache: dict) -> bool:
    return all(cache.get(f) is not None for f in REQUIRED_MATCHING_FIELDS)


def _save_matching(cache_path: Path, m: FeatureBSPMatching) -> None:
    """Persist every field the current FeatureBSPMatching carries.

    The previous version wrote a hand-listed subset that predated J and
    mcc_at_pref, so a rebuilt matching lost exactly the fields the rebuild had
    been run to produce -- and the next run rebuilt it again.
    """
    torch.save(
        {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "best_f1_per_bsp": m.best_f1_per_bsp,
            "best_feature_per_bsp": m.best_feature_per_bsp,
            "mcc": m.mcc,
            "best_mcc_per_bsp": m.best_mcc_per_bsp,
            "best_feature_per_bsp_mcc": m.best_feature_per_bsp_mcc,
            "youden_j": m.youden_j,
            "best_j_per_bsp": m.best_j_per_bsp,
            "best_feature_per_bsp_j": m.best_feature_per_bsp_j,
            "mcc_at_pref": m.mcc_at_pref,
            "best_mcc_at_pref_per_bsp": m.best_mcc_at_pref_per_bsp,
            "best_feature_per_bsp_mcc_at_pref": m.best_feature_per_bsp_mcc_at_pref,
            "p_ref": m.p_ref,
            "base_rates": m.base_rates,
            "f1_lift_per_bsp": m.f1_lift_per_bsp,
            "trivial_f1_per_bsp": m.trivial_f1_per_bsp,
        },
        cache_path,
    )


def _augment_matching(
    cache_path: Path,
    h_path: Path | None,
    bsp_labels: torch.Tensor,
) -> tuple[FeatureBSPMatching, bool]:
    """Load (or rebuild) a matching cache so every current field is populated.

    Returns ``(matching, cache_was_rewritten)``. Raises ``FileNotFoundError``
    when a rebuild is required but the ``_h`` code cache is absent, so the
    caller can report that specific reason rather than silently emitting a row
    with the new metrics missing.
    """
    c = torch.load(cache_path, map_location="cpu", weights_only=False)

    # A cache missing ANY current field must be rebuilt from the codes. The old
    # logic rebuilt only when MCC was absent, so the 497 rows that already had
    # MCC were never revisited and never gained J or mcc_at_pref -- neither a
    # plain run nor --force-recompute produced them.
    if not _matching_is_current(c):
        if h_path is None or not h_path.exists():
            missing = [f for f in REQUIRED_MATCHING_FIELDS if c.get(f) is None]
            raise FileNotFoundError(
                f"needs rebuild (missing {', '.join(missing)}) but no _h cache "
                f"at {h_path}")
        h = torch.load(h_path, map_location="cpu", weights_only=False)
        log.info("    rebuilding full matching from h.pt (%s)", tuple(h.shape))
        matching = match_features_to_bsps(h, bsp_labels)
        _save_matching(cache_path, matching)
        return matching, True

    matching = FeatureBSPMatching(
        precision=c["precision"],
        recall=c["recall"],
        f1=c["f1"],
        best_f1_per_bsp=c["best_f1_per_bsp"],
        best_feature_per_bsp=c["best_feature_per_bsp"],
        mcc=c.get("mcc"),
        best_mcc_per_bsp=c.get("best_mcc_per_bsp"),
        best_feature_per_bsp_mcc=c.get("best_feature_per_bsp_mcc"),
        youden_j=c.get("youden_j"),
        best_j_per_bsp=c.get("best_j_per_bsp"),
        best_feature_per_bsp_j=c.get("best_feature_per_bsp_j"),
        mcc_at_pref=c.get("mcc_at_pref"),
        best_mcc_at_pref_per_bsp=c.get("best_mcc_at_pref_per_bsp"),
        best_feature_per_bsp_mcc_at_pref=c.get("best_feature_per_bsp_mcc_at_pref"),
        p_ref=c.get("p_ref"),
        base_rates=c.get("base_rates"),
        f1_lift_per_bsp=c.get("f1_lift_per_bsp"),
        trivial_f1_per_bsp=c.get("trivial_f1_per_bsp"),
    )

    rewrote = False
    # f1_lift / base_rates are derivable from the labels alone, so they never
    # justify re-encoding.
    if matching.f1_lift_per_bsp is None:
        base_rates = bsp_labels.float().mean(dim=0)
        trivial_f1 = (2.0 * base_rates) / (1.0 + base_rates + 1e-8)
        matching.base_rates = base_rates
        matching.trivial_f1_per_bsp = trivial_f1
        matching.f1_lift_per_bsp = (matching.best_f1_per_bsp - trivial_f1).clamp(
            min=0.0
        )
        _save_matching(cache_path, matching)
        rewrote = True

    return matching, rewrote


def _trim_retired_keys(metrics: dict) -> int:
    """Drop retired F1-derivative keys from a registry row. Returns n removed."""
    n = 0
    for key in RETIRED_METRIC_KEYS:
        if metrics.pop(key, None) is not None:
            n += 1
    for vals in (metrics.get("per_category") or {}).values():
        if isinstance(vals, dict):
            for key in RETIRED_PER_CATEGORY_KEYS:
                if vals.pop(key, None) is not None:
                    n += 1
    return n


# --- main -------------------------------------------------------------------


def main() -> int:
    args = docopt(__doc__)
    game: str = args["--game"]
    dry_run: bool = bool(args["--dry-run"])
    force: bool = bool(args["--force-recompute"])

    registry_path = ROOT / "saes" / game / "eval_registry.json"
    if not registry_path.exists():
        log.error("Registry not found: %s", registry_path)
        return 1

    cache_dir = ROOT / "saes" / game / "cache"
    legacy_cache_dir = ROOT / "saes" / game  # pre-cache/ layout
    with open(registry_path, "r") as f:
        registry = json.load(f)

    # Memoize loaded labels/schemas per (bsp_set, count)
    labels_cache: dict[tuple[str, int], torch.Tensor] = {}
    schema_cache: dict[tuple[str, int], object] = {}

    n_total = len(registry)
    n_updated = 0
    n_skipped_no_cache = 0
    n_skipped_no_labels = 0
    n_skipped_no_h = 0
    n_already_done = 0
    n_trimmed = 0
    needs_h: list[str] = []

    log.info("Scanning %d registry entries (game=%s)", n_total, game)
    for key, entry in registry.items():
        bsp_set = entry.get("bsp_set", "gorilla")
        run_id = key.split(":", 1)[0]

        metrics = entry.setdefault("metrics", {})
        # Up-to-date means "carries every CURRENT metric", tested against the
        # newest field. The old condition tested `coverage_mcc` and
        # `coverage_f1_lift` -- both of which 497/523 rows already had -- so the
        # backfill skipped almost the whole registry and never added J or
        # mcc_at_pref. Keying on the newest field means adding a metric in
        # future automatically re-opens the rows that lack it.
        if not force and all(k in metrics for k in CURRENT_METRIC_KEYS):
            # An up-to-date row can still carry retired keys (it was
            # re-evaluated rather than backfilled), so trim before skipping.
            if _trim_retired_keys(metrics):
                n_trimmed += 1
            n_already_done += 1
            continue

        # 1. matching cache
        cache_path = cache_dir / f"{run_id}_matching-{bsp_set}.pt"
        if not cache_path.exists():
            cache_path = legacy_cache_dir / f"{run_id}_matching-{bsp_set}.pt"
        if not cache_path.exists():
            log.warning("  [%s] no matching cache → skip", key)
            n_skipped_no_cache += 1
            continue

        # 2. labels — need count from the matching cache first to disambiguate
        # legacy hawk_92 vs current hawk_173.
        try:
            c_head = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            log.warning("  [%s] could not read cache (%s) → skip", key, e)
            continue
        expected_count = int(c_head["best_f1_per_bsp"].shape[0])

        if (bsp_set, expected_count) not in labels_cache:
            label_path, schema_path = _resolve_bsp_paths(
                game, bsp_set, expected_count=expected_count
            )
            if not label_path:
                log.warning("  [%s] no bsp_labels for %s → skip", key, bsp_set)
                n_skipped_no_labels += 1
                continue
            labels = _load_bsp_labels(label_path)
            if labels.shape[1] != expected_count:
                log.warning(
                    "  [%s] label-count mismatch (cache=%d, labels=%d) → skip",
                    key,
                    expected_count,
                    labels.shape[1],
                )
                n_skipped_no_labels += 1
                continue
            labels_cache[(bsp_set, expected_count)] = labels
            schema_cache[(bsp_set, expected_count)] = (
                _load_schema(schema_path) if schema_path else None
            )
        bsp_labels = labels_cache[(bsp_set, expected_count)]
        schema = schema_cache.get((bsp_set, expected_count))

        h_path = cache_dir / f"{run_id}_h.pt"
        if not h_path.exists():
            h_path_alt = legacy_cache_dir / f"{run_id}_h.pt"
            h_path = (
                h_path_alt if h_path_alt.exists() else h_path
            )  # may still not exist

        if dry_run:
            # Report the SAME outcome the real run would reach. `c_head` is
            # already loaded, so the check is free -- and a dry-run that
            # reported "487 updated" while the real run could only manage a
            # fraction would be worse than no dry-run at all.
            if _matching_is_current(c_head):
                log.info("  [%s] cache current -> registry refresh only", key)
                n_updated += 1
            elif h_path.exists():
                log.info("  [%s] would rebuild matching from _h", key)
                n_updated += 1
            else:
                log.warning("  [%s] needs rebuild but no _h cache", key)
                needs_h.append(key)
                n_skipped_no_h += 1
            continue

        log.info("  [%s] backfilling …", key)
        try:
            matching, rewrote = _augment_matching(cache_path, h_path, bsp_labels)
        except FileNotFoundError as e:
            # Named separately from a generic failure: this row is recoverable,
            # it just needs its codes re-encoded first. Silently lumping it in
            # with "failed" is how 240 rows stayed stale unnoticed.
            log.warning("  [%s] %s", key, e)
            needs_h.append(key)
            n_skipped_no_h += 1
            continue
        except Exception as e:  # noqa: BLE001
            log.warning("  [%s] failed (%s) → skip", key, e)
            continue

        cov_extra = compute_coverage(matching)
        per_cat = compute_per_category_coverage(matching, schema) if schema else None

        # Merge: keep existing keys, overwrite the ones we compute.
        for k, v in cov_extra.items():
            metrics[k] = v
        if per_cat:
            metrics["per_category"] = per_cat
        # Trim AFTER merging: compute_* no longer emits the retired keys, but a
        # row that had them keeps them unless they are removed explicitly.
        if _trim_retired_keys(metrics):
            n_trimmed += 1

        n_updated += 1
        if rewrote:
            log.info("    cache rewritten with augmented fields")

    log.info("-" * 60)
    log.info("Total entries:        %d", n_total)
    log.info("  Already up-to-date: %d", n_already_done)
    log.info("  Updated:            %d", n_updated)
    log.info("  Retired keys trimmed in: %d row(s)", n_trimmed)
    log.info("  No matching cache:  %d", n_skipped_no_cache)
    log.info("  No bsp_labels:      %d", n_skipped_no_labels)
    log.info("  Needs _h re-encode: %d", n_skipped_no_h)

    if needs_h:
        log.info("")
        log.info("These rows need J / mcc_at_pref but their _h code cache is gone.")
        log.info("Re-encode with (one call per checkpoint, all bases at once):")
        by_run: dict[str, list[str]] = {}
        for key in needs_h:
            rid, _, bsp = key.partition(":")
            by_run.setdefault(rid, []).append(bsp or "gorilla")
        for rid, bsps in sorted(by_run.items())[:10]:
            log.info("  python sae_eval.py evaluate saes/%s/%s.pt --bsps=%s --force",
                     game, rid, ",".join(sorted(set(bsps))))
        if len(by_run) > 10:
            log.info("  ... and %d more checkpoint(s)", len(by_run) - 10)

    if dry_run:
        log.info("DRY-RUN: registry NOT written")
        return 0

    if n_updated == 0 and n_trimmed == 0:
        log.info("Nothing to write.")
        return 0

    backup = registry_path.with_suffix(".json.bak")
    shutil.copyfile(registry_path, backup)
    log.info("Backup written: %s", backup)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    log.info("Registry updated: %s", registry_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
