"""Backfill MCC and F1-lift metrics into eval_registry.json.

Walks every entry in ``saes/<game>/eval_registry.json`` and recomputes the
extended coverage metrics (MCC-based + F1-lift over the base-rate baseline)
introduced after the registry was first written.

For each entry the script looks for:

  saes/<game>/cache/{run_id}_matching-{bsp_set}.pt    (required)
  saes/<game>/cache/{run_id}_h.pt                     (optional, enables MCC)
  data/<game>/bsp_labels-<bsp_set>_*.pt               (required)
  data/<game>/bsp_schema-<bsp_set>_*.json             (required for per-cat)

If the matching cache predates MCC, the script:
  * always derives ``base_rates``, ``trivial_f1`` and ``f1_lift`` from labels;
  * recomputes MCC from the activation cache ``_h.pt`` when it is available;
  * rewrites the matching cache with the augmented fields.

The registry is written atomically with a ``.bak`` sibling kept on disk.

Usage:
    backfill_eval_metrics.py [--game=<g>] [--dry-run] [--force-recompute]
    backfill_eval_metrics.py (-h | --help)

Options:
    --game=<g>           Game name [default: quarto].
    --dry-run            Show what would change, do not write anything.
    --force-recompute    Re-derive metrics even if the registry already has
                         ``coverage_mcc`` / ``coverage_f1_lift``.
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
    schema_matches = sorted(data_dir.glob(f"bsp_schema-{animal}_*.json"))

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

    return _pick(label_matches), _pick(schema_matches)


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


def _augment_matching(
    cache_path: Path,
    h_path: Path | None,
    bsp_labels: torch.Tensor,
) -> tuple[FeatureBSPMatching, bool]:
    """Load (or rebuild) a matching cache so MCC + F1-lift are populated.

    Returns ``(matching, cache_was_rewritten)``.
    """
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    matching = FeatureBSPMatching(
        precision=c["precision"],
        recall=c["recall"],
        f1=c["f1"],
        best_f1_per_bsp=c["best_f1_per_bsp"],
        best_feature_per_bsp=c["best_feature_per_bsp"],
        mcc=c.get("mcc"),
        best_mcc_per_bsp=c.get("best_mcc_per_bsp"),
        best_feature_per_bsp_mcc=c.get("best_feature_per_bsp_mcc"),
        base_rates=c.get("base_rates"),
        f1_lift_per_bsp=c.get("f1_lift_per_bsp"),
        trivial_f1_per_bsp=c.get("trivial_f1_per_bsp"),
    )

    rewrote = False

    # Always make sure f1_lift / base_rates exist (cheap; needs only labels).
    if matching.f1_lift_per_bsp is None:
        base_rates = bsp_labels.float().mean(dim=0)
        trivial_f1 = (2.0 * base_rates) / (1.0 + base_rates + 1e-8)
        matching.base_rates = base_rates
        matching.trivial_f1_per_bsp = trivial_f1
        matching.f1_lift_per_bsp = (matching.best_f1_per_bsp - trivial_f1).clamp(
            min=0.0
        )
        rewrote = True

    # MCC needs the activation cache.  If we have it, rebuild full matching.
    if matching.best_mcc_per_bsp is None and h_path is not None and h_path.exists():
        h = torch.load(h_path, map_location="cpu", weights_only=False)
        log.info("    rebuilding full matching from h.pt (%s)", tuple(h.shape))
        matching = match_features_to_bsps(h, bsp_labels)
        rewrote = True

    if rewrote:
        torch.save(
            {
                "precision": matching.precision,
                "recall": matching.recall,
                "f1": matching.f1,
                "best_f1_per_bsp": matching.best_f1_per_bsp,
                "best_feature_per_bsp": matching.best_feature_per_bsp,
                "mcc": matching.mcc,
                "best_mcc_per_bsp": matching.best_mcc_per_bsp,
                "best_feature_per_bsp_mcc": matching.best_feature_per_bsp_mcc,
                "base_rates": matching.base_rates,
                "f1_lift_per_bsp": matching.f1_lift_per_bsp,
                "trivial_f1_per_bsp": matching.trivial_f1_per_bsp,
            },
            cache_path,
        )

    return matching, rewrote


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
    n_already_done = 0

    log.info("Scanning %d registry entries (game=%s)", n_total, game)
    for key, entry in registry.items():
        bsp_set = entry.get("bsp_set", "gorilla")
        run_id = key.split(":", 1)[0]

        metrics = entry.setdefault("metrics", {})
        if not force and "coverage_mcc" in metrics and "coverage_f1_lift" in metrics:
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

        log.info("  [%s] backfilling …", key)
        if dry_run:
            n_updated += 1
            continue

        try:
            matching, rewrote = _augment_matching(cache_path, h_path, bsp_labels)
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

        n_updated += 1
        if rewrote:
            log.info("    cache rewritten with augmented fields")

    log.info("-" * 60)
    log.info("Total entries:        %d", n_total)
    log.info("  Already up-to-date: %d", n_already_done)
    log.info("  Updated:            %d", n_updated)
    log.info("  No matching cache:  %d", n_skipped_no_cache)
    log.info("  No bsp_labels:      %d", n_skipped_no_labels)

    if dry_run:
        log.info("DRY-RUN: registry NOT written")
        return 0

    if n_updated == 0:
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
