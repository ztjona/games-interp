"""SAE evaluation script: coverage, board reconstruction, comparison.

Usage:
    sae_eval.py evaluate <checkpoint> [options]
    sae_eval.py compare [<run_ids>...] [options]
    sae_eval.py history [--game=<game>] [options]
    sae_eval.py -h | --help

Commands:
    evaluate    Run Layer 1 evaluation (coverage + board reconstruction)
    compare     Side-by-side comparison of evaluated runs
    history     List all evaluated runs from the registry

Arguments:
    <checkpoint>    Path to trained SAE .pt file
    <run_ids>       One or more eval registry keys (checkpoint stems)

Options:
    --bsps=<name>           BSP set animal name (e.g. gorilla, fox). Comma-
                            separated for multiple sets in one call
                            (e.g. ``gorillaS4,hawkS4``); the activation tensor
                            and per-feature ``h`` are encoded once and reused
                            across sets. Incompatible with ``--bsp-labels`` /
                            ``--bsp-schema``. [default: gorilla]
    --data=<path>           Activation data path [default: auto]
    --bsp-labels=<path>     BSP label tensor path [default: auto]
    --bsp-schema=<path>     BSP schema JSON path [default: auto]
    --precision-thresh=<f>  Board reconstruction precision threshold [default: 0.9]
    --batch-size=<n>        Evaluation batch size [default: 4096]
    --tag=<str>             Human-readable tag for this evaluation
    --device=<dev>          Device (cuda|cpu|auto) [default: auto]
    --force                 Re-evaluate even if results exist in the registry
    --game=<game>           Game filter for history command [default: quarto]
    --config=<path>         YAML config listing run_ids for compare
    --verbose               Enable debug-level logging
    -h --help               Show this help

Auto-resolution (from checkpoint metadata):
    data    -> data/{game}/{hook}_amalgam_activations.pt
    bsps    -> data/{game}/bsp_labels-{animal}_{count}.pt  (by glob)
    schema  -> data/{game}/bsp_schema-{animal}_{count}.json

Examples:
    # Evaluate with full BSP set (gorilla)
    python sae_eval.py evaluate saes/quarto/arnold-topk-k32-exp8-fc1.pt

    # Re-evaluate (force recompute)
    python sae_eval.py evaluate saes/quarto/arnold-topk-k32-exp8-fc1.pt --force

    # Compare two runs by ID
    python sae_eval.py compare arnold-topk-k32-exp8-fc1 arnold-topk-k64-exp8-fc1

    # Compare runs from a config file
    python sae_eval.py compare --config configs/compare-arnold.yaml

"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from docopt import docopt

from tqdm import tqdm

from lib.sae import load_checkpoint, load_activation_data
from lib.sae.architectures import BatchTopKSAE
from lib.sae.eval import (
    FeatureBSPMatching,
    match_features_to_bsps,
    compute_coverage,
    compute_feature_sharing,
    compute_per_category_coverage,
    compute_board_reconstruction,
    resolve_schema_path,
)
from lib.sae.train import compute_metrics

log = logging.getLogger("sae_eval")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_data_path(game: str, hook: str, metadata: dict | None = None) -> Path:
    """Auto-resolve activation data from game and hook.

    Prefers data_path stored in checkpoint metadata (set during training),
    falling back to the convention data/{game}/{hook}_amalgam_activations.pt.
    """
    # Prefer the exact path used during training
    if metadata and "data_path" in metadata:
        meta_path = Path(metadata["data_path"])
        if meta_path.exists():
            return meta_path
        log.warning("  data_path from checkpoint metadata not found: %s", meta_path)
        log.warning("  Falling back to convention-based resolution.")

    path = Path(f"data/{game}/{hook}_amalgam_activations.pt")
    if not path.exists():
        log.error("Activation data not found: %s", path)
        log.error("  Expected: data/%s/%s_amalgam_activations.pt", game, hook)
        sys.exit(1)
    return path


def _resolve_bsp_paths(game: str, animal: str) -> tuple[Path, Path]:
    """Resolve BSP label and schema paths from animal name.

    Labels are per-distribution (e.g. ``bsp_labels-gorillaS4_164.pt``); the
    schema is per-basis (e.g. ``bsp_schema-gorilla_164.json``) and is shared
    across champions because the concept menu is distribution-independent.
    The numeric ``_[0-9]*`` guard on the label glob prevents ``gorilla`` from
    accidentally matching ``bsp_labels-gorilla_s4_164.pt`` (which shares the
    same concept menu but a different position distribution).
    """
    data_dir = Path(f"data/{game}")
    label_matches = sorted(data_dir.glob(f"bsp_labels-{animal}_[0-9]*.pt"))

    if not label_matches:
        available = sorted(data_dir.glob("bsp_labels-*.pt"))
        available_names = [p.stem.split("-", 1)[1].rsplit("_", 1)[0] for p in available]
        log.error("No BSP labels found for animal '%s' in data/%s/", animal, game)
        if available_names:
            log.error("  Available BSP sets: %s", ", ".join(available_names))
        sys.exit(1)

    schema_path = resolve_schema_path(data_dir, animal)
    if schema_path is None:
        log.error(
            "BSP schema not found for animal '%s' in data/%s/ (tried suffixed and basis)",
            animal,
            game,
        )
        sys.exit(1)

    return label_matches[0], schema_path


def _extract_checkpoint_info(metadata: dict) -> dict[str, str]:
    """Extract game, hook, architecture from checkpoint metadata."""
    return {
        "game": metadata.get("game", "quarto"),
        "hook": metadata.get("hook", "fc1"),
        "architecture": metadata.get("architecture", "unknown"),
        "experiment": metadata.get("experiment", "unknown"),
    }


# ---------------------------------------------------------------------------
# Eval registry
# ---------------------------------------------------------------------------


def _registry_path(game: str) -> Path:
    return Path(f"saes/{game}/eval_registry.json")


def _load_registry(game: str) -> dict[str, Any]:
    path = _registry_path(game)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(game: str, registry: dict[str, Any]) -> None:
    path = _registry_path(game)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def _register_eval(
    run_id: str,
    game: str,
    checkpoint_path: str,
    info: dict[str, str],
    bsp_set: str,
    metrics: dict,
    tag: str | None,
) -> None:
    """Save evaluation results to the eval registry.

    Registry keys use the format ``run_id:bsp_set`` so the same checkpoint can
    be evaluated against multiple BSP sets without overwriting previous results.
    """
    registry = _load_registry(game)
    registry[f"{run_id}:{bsp_set}"] = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "checkpoint": checkpoint_path,
        "architecture": info["architecture"],
        "experiment": info["experiment"],
        "game": game,
        "hook": info["hook"],
        "bsp_set": bsp_set,
        "tag": tag or "",
        "metrics": metrics,
    }
    _save_registry(game, registry)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_evaluate(args: dict) -> None:
    """Run Layer 1 evaluation on a checkpoint, for one or more BSP sets."""
    checkpoint_path = args["<checkpoint>"]
    animals = [a.strip() for a in args["--bsps"].split(",") if a.strip()]
    if not animals:
        log.error("--bsps is empty")
        sys.exit(1)
    if len(animals) != len(set(animals)):
        log.error("--bsps contains duplicates: %s", animals)
        sys.exit(1)

    bsp_label_arg = args["--bsp-labels"]
    bsp_schema_arg = args["--bsp-schema"]
    explicit_paths = (bsp_label_arg and bsp_label_arg != "auto") or (
        bsp_schema_arg and bsp_schema_arg != "auto"
    )
    if explicit_paths and len(animals) > 1:
        log.error(
            "--bsp-labels / --bsp-schema cannot be combined with multi-bsp "
            "(--bsps=%s); pass one BSP set at a time when overriding paths.",
            args["--bsps"],
        )
        sys.exit(1)

    precision_thresh = float(args["--precision-thresh"])
    batch_size = int(args["--batch-size"])
    tag = args["--tag"]
    force = args["--force"]

    device_arg = args["--device"]
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    run_id = Path(checkpoint_path).stem

    # Peek at metadata to find game before deciding on registry
    log.info("Loading checkpoint: %s", checkpoint_path)
    sae, metadata = load_checkpoint(Path(checkpoint_path), device=device)
    info = _extract_checkpoint_info(metadata)
    game, hook = info["game"], info["hook"]

    # ---- Cache check (per animal) ----
    registry = _load_registry(game)
    pending: list[str] = []
    for animal in animals:
        cache_key = f"{run_id}:{animal}"
        if cache_key in registry and not force:
            cached = registry[cache_key]
            log.info(
                "Found cached evaluation for '%s' on '%s' (use --force to recompute)",
                run_id,
                animal,
            )
            output = {
                "run_id": run_id,
                "checkpoint": cached.get("checkpoint", checkpoint_path),
                "architecture": cached.get("architecture", "?"),
                "experiment": cached.get("experiment", "?"),
                "game": game,
                "hook": cached.get("hook", hook),
                "bsp_set": cached.get("bsp_set", animal),
                "cached": True,
                "metrics": cached.get("metrics", {}),
            }
            print(json.dumps(output, indent=2))
            _print_summary(
                run_id,
                cached.get("bsp_set", animal),
                cached.get("metrics", {}),
                game,
                cached=True,
            )
        else:
            pending.append(animal)

    if not pending:
        return

    log.info(
        "  Architecture: %s, Game: %s, Hook: %s, d_input=%d, d_dict=%d",
        info["architecture"],
        game,
        hook,
        sae.d_input,
        sae.d_dict,
    )

    # Resolve shared activation path
    data_arg = args["--data"]
    if data_arg and data_arg != "auto":
        data_path = Path(data_arg)
    else:
        data_path = _resolve_data_path(game, hook, metadata)
    log.info("  Activations: %s", data_path)

    activations = load_activation_data(str(data_path), device)

    act_dim = activations.shape[-1]
    if act_dim != sae.d_input:
        log.error(
            "Shape mismatch: activations have d=%d but SAE expects d_input=%d",
            act_dim,
            sae.d_input,
        )
        log.error("  Activation file: %s", data_path)
        if metadata and "data_path" in metadata:
            log.error("  Hint: checkpoint was trained on '%s'", metadata["data_path"])
        sys.exit(1)

    if isinstance(sae, BatchTopKSAE):
        sae.train()  # BatchTopK needs train() for batch-level sparsity
    else:
        sae.eval()

    N = activations.shape[0]
    saes_dir = Path(f"saes/{game}")
    cache_dir = saes_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Encode activations → h  (cached to disk; shared across animals) ──
    h_cache = cache_dir / f"{run_id}_h.pt"
    if h_cache.exists() and not force:
        h = torch.load(h_cache, map_location="cpu", weights_only=True)
        log.info("  h loaded from cache: %s", h_cache)
    else:
        h_parts = []
        n_batches = (N + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in tqdm(
                range(0, N, batch_size), total=n_batches, desc="Encoding", unit="batch"
            ):
                batch = activations[i : i + batch_size].to(device)
                result = sae(batch)
                h_parts.append(result["h"].cpu())
        h = torch.cat(h_parts, dim=0)  # (N, d_dict) float32 CPU
        torch.save(h, h_cache)
        log.info("  h cached → %s", h_cache)

    eval_sample = activations[: min(N, batch_size)].to(device)
    structural = compute_metrics(sae, eval_sample)

    # ── 2-3. Per-animal matching + coverage + register ──────────────────────
    for animal in pending:
        if bsp_label_arg and bsp_label_arg != "auto":
            bsp_label_path = Path(bsp_label_arg)
            bsp_schema_path = (
                Path(bsp_schema_arg)
                if bsp_schema_arg and bsp_schema_arg != "auto"
                else None
            )
        else:
            bsp_label_path, bsp_schema_path = _resolve_bsp_paths(game, animal)

        log.info("[bsps=%s]", animal)
        log.info("  BSP labels:  %s", bsp_label_path)
        if bsp_schema_path:
            log.info("  BSP schema:  %s", bsp_schema_path)

        bsp_labels = torch.load(bsp_label_path, map_location="cpu", weights_only=True)

        bsp_schema = None
        if bsp_schema_path and bsp_schema_path.exists():
            with open(bsp_schema_path, "r") as f:
                bsp_schema = json.load(f)

        log.info("  Samples: %d, BSPs: %d", activations.shape[0], bsp_labels.shape[1])

        matching_cache = cache_dir / f"{run_id}_matching-{animal}.pt"
        if matching_cache.exists() and not force:
            c = torch.load(matching_cache, map_location="cpu", weights_only=False)
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
            log.info("  matching loaded from cache: %s", matching_cache)
            if matching.f1_lift_per_bsp is None:
                base_rates = bsp_labels.float().mean(dim=0)
                trivial_f1 = (2.0 * base_rates) / (1.0 + base_rates + 1e-8)
                matching.base_rates = base_rates
                matching.trivial_f1_per_bsp = trivial_f1
                matching.f1_lift_per_bsp = (
                    matching.best_f1_per_bsp - trivial_f1
                ).clamp(min=0.0)
                log.info("  legacy cache: derived f1_lift from labels")
        else:
            matching = match_features_to_bsps(h, bsp_labels)
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
                matching_cache,
            )
            log.info("  matching cached → %s", matching_cache)

        coverage_metrics = compute_coverage(matching)
        feature_sharing = compute_feature_sharing(matching)
        per_category = compute_per_category_coverage(matching, bsp_schema)
        reconstruction = compute_board_reconstruction(
            matching, h, bsp_labels, precision_threshold=precision_thresh
        )

        metrics = {**structural, **coverage_metrics, **reconstruction}
        metrics.pop("per_bsp_accuracy", None)
        metrics["feature_sharing"] = feature_sharing
        if per_category:
            metrics["per_category"] = per_category

        # Anchored diagonal: per-anchor F1/MCC/lift on the (feature, bsp)
        # pair that was supervised at training time.  Read constructor_kwargs
        # from the checkpoint metadata to know which slots are anchored.
        ctor_kwargs = metadata.get("constructor_kwargs", {}) or {}
        anchor_feature_idx = ctor_kwargs.get("anchor_feature_idx")
        anchor_bsp_idx = ctor_kwargs.get("anchor_bsp_idx")
        if anchor_feature_idx is not None and anchor_bsp_idx is not None:
            anchor_meta = metadata.get("anchor", {}) or {}
            afi = torch.as_tensor(anchor_feature_idx, dtype=torch.long)
            abi = torch.as_tensor(anchor_bsp_idx, dtype=torch.long)
            num_anchored = int(afi.numel())
            num_bsps_eval = matching.f1.shape[1]
            in_range = abi < num_bsps_eval
            n_skipped = int((~in_range).sum().item())
            if n_skipped:
                log.warning(
                    "Anchored diagonal: %d/%d anchor BSP indices fall outside "
                    "the current eval BSP set (%s, n=%d) and will be skipped.",
                    n_skipped, num_anchored, animal, num_bsps_eval,
                )
            afi_e = afi[in_range]
            abi_e = abi[in_range]
            if afi_e.numel() > 0:
                diag_f1 = matching.f1[afi_e, abi_e]
                diag_mcc = (
                    matching.mcc[afi_e, abi_e]
                    if matching.mcc is not None
                    else torch.zeros_like(diag_f1)
                )
                if matching.trivial_f1_per_bsp is not None:
                    diag_lift = (
                        diag_f1 - matching.trivial_f1_per_bsp[abi_e]
                    ).clamp(min=0.0)
                else:
                    diag_lift = torch.zeros_like(diag_f1)
                anchored_diagonal: dict[str, Any] = {
                    "num_anchored": num_anchored,
                    "num_evaluated": int(afi_e.numel()),
                    "mean_f1": float(diag_f1.mean()),
                    "mean_mcc": float(diag_mcc.mean()),
                    "mean_f1_lift": float(diag_lift.mean()),
                    "per_anchor_f1": diag_f1.tolist(),
                    "per_anchor_mcc": diag_mcc.tolist(),
                    "per_anchor_f1_lift": diag_lift.tolist(),
                    "anchor_feature_idx": afi_e.tolist(),
                    "anchor_bsp_idx": abi_e.tolist(),
                }
                # Tier breakdown (high vs medium) using anchor_meta + schema
                high_cats = set(anchor_meta.get("anchor_high_categories") or [])
                if high_cats and bsp_schema is not None:
                    bsps_list = (
                        bsp_schema.get("bsps", bsp_schema)
                        if isinstance(bsp_schema, dict)
                        else bsp_schema
                    )
                    high_mask = torch.tensor(
                        [
                            (
                                bsps_list[int(j)].get("category", "") in high_cats
                                if int(j) < len(bsps_list)
                                else False
                            )
                            for j in abi_e.tolist()
                        ],
                        dtype=torch.bool,
                    )
                    med_mask = ~high_mask
                    tier: dict[str, dict[str, float]] = {}
                    if high_mask.any():
                        tier["high"] = {
                            "n": int(high_mask.sum().item()),
                            "mean_f1": float(diag_f1[high_mask].mean()),
                            "mean_mcc": float(diag_mcc[high_mask].mean()),
                            "mean_f1_lift": float(diag_lift[high_mask].mean()),
                        }
                    if med_mask.any():
                        tier["medium"] = {
                            "n": int(med_mask.sum().item()),
                            "mean_f1": float(diag_f1[med_mask].mean()),
                            "mean_mcc": float(diag_mcc[med_mask].mean()),
                            "mean_f1_lift": float(diag_lift[med_mask].mean()),
                        }
                    if tier:
                        anchored_diagonal["per_tier"] = tier
                metrics["anchored_diagonal"] = anchored_diagonal

        _register_eval(run_id, game, checkpoint_path, info, animal, metrics, tag)

        output = {
            "run_id": run_id,
            "checkpoint": checkpoint_path,
            "architecture": info["architecture"],
            "experiment": info["experiment"],
            "game": game,
            "hook": hook,
            "bsp_set": animal,
            "num_samples": activations.shape[0],
            "cached": False,
            "metrics": metrics,
        }
        print(json.dumps(output, indent=2))
        _print_summary(
            run_id, animal, metrics, game, num_bsps_label=bsp_labels.shape[1]
        )


def _print_summary(
    run_id: str,
    animal: str,
    metrics: dict,
    game: str,
    num_bsps_label: int | None = None,
    cached: bool = False,
) -> None:
    """Print a human-readable summary to stderr."""
    log.info("")
    log.info("=" * 60)
    if cached:
        log.info("  %s  (cached)", run_id)
    else:
        log.info("  %s", run_id)
    bsp_count = num_bsps_label or metrics.get("num_bsps", "?")
    log.info("  BSP set: %s (%s BSPs)", animal, bsp_count)
    log.info("-" * 60)
    log.info("  FVU:                  %.6f", metrics.get("fvu", 0))
    log.info("  L0:                   %.1f", metrics.get("l0", 0))
    log.info("  Dead features:        %.1f%%", metrics.get("dead_features_pct", 0))
    log.info("  Coverage (F1):        %.4f", metrics.get("coverage", 0))
    log.info("  Coverage F1 >50%%:     %.4f", metrics.get("coverage_above_50", 0))
    log.info("  Coverage F1 >75%%:     %.4f", metrics.get("coverage_above_75", 0))
    if "coverage_mcc" in metrics:
        log.info("  Coverage (MCC):       %.4f", metrics["coverage_mcc"])
        log.info(
            "  Coverage MCC >25%%:    %.4f",
            metrics.get("coverage_mcc_above_25", 0),
        )
        log.info(
            "  Coverage MCC >50%%:    %.4f",
            metrics.get("coverage_mcc_above_50", 0),
        )
    if "coverage_f1_lift" in metrics:
        log.info("  Coverage F1-lift:     %.4f", metrics["coverage_f1_lift"])
        log.info(
            "  F1-lift >10%%:         %.4f",
            metrics.get("coverage_f1_lift_above_10", 0),
        )
    feature_sharing = metrics.get("feature_sharing", {})
    if feature_sharing:
        log.info(
            "  Features used:         %d",
            feature_sharing.get("num_features_used_by_best_matches", 0),
        )
        log.info(
            "  BSPs on shared feats:  %.4f",
            feature_sharing.get("fraction_bsps_with_shared_best_feature", 0),
        )
        log.info(
            "  Max BSPs / feature:    %d",
            feature_sharing.get("max_bsps_per_feature", 0),
        )
    log.info("  Board reconstruction: %.4f", metrics.get("board_reconstruction", 0))
    log.info(
        "  Reconstructable BSPs: %d / %s (%.1f%%)",
        metrics.get("num_reconstructable_bsps", 0),
        metrics.get("num_bsps", "?"),
        metrics.get("fraction_reconstructable", 0) * 100,
    )
    per_cat = metrics.get("per_category")
    if per_cat:
        log.info("-" * 60)
        log.info("  Per-category coverage (F1 / MCC / F1-lift):")
        for cat, vals in sorted(per_cat.items()):
            log.info(
                "    %-22s  F1=%.3f  MCC=%.3f  lift=%.3f  (%d BSPs, base=%.3f)",
                cat,
                vals.get("mean_f1", 0.0),
                vals.get("mean_mcc", 0.0),
                vals.get("mean_f1_lift", 0.0),
                vals.get("count", 0),
                vals.get("mean_base_rate", 0.0),
            )
    log.info("=" * 60)
    log.info("  Saved to eval registry: %s", _registry_path(game))


def cmd_history(args: dict) -> None:
    """List all evaluated runs from the registry."""
    game = args["--game"]

    # If no game specified, scan all game directories
    games = [game] if game else [d.name for d in Path("saes").iterdir() if d.is_dir()]

    all_entries = []
    for g in games:
        registry = _load_registry(g)
        for reg_key, entry in registry.items():
            if reg_key in ("experiments", "version", "note"):
                continue
            # Support both new-style keys (run_id:bsp_set) and legacy plain run_id keys
            run_id = entry.get("run_id", reg_key)
            all_entries.append((run_id, g, entry))

    if not all_entries:
        print("No evaluated runs found.")
        return

    # Sort by timestamp
    all_entries.sort(key=lambda x: x[2].get("timestamp", ""))

    # Define columns
    metrics_cols = [
        ("fvu", ".4f"),
        ("l0", ".0f"),
        ("dead_features_pct", ".0f"),
        ("coverage", ".3f"),
        ("coverage_above_50", ".3f"),
        ("coverage_mcc", ".3f"),
        ("coverage_f1_lift", ".3f"),
        ("board_reconstruction", ".3f"),
    ]

    # Header
    header = f"{'Run ID':<45} {'Arch':<10} {'Hook':<6} {'BSPs':<8}"
    for col, _ in metrics_cols:
        short = (
            col.replace("_features_pct", "%")
            .replace("_above_50", ">50")
            .replace("board_reconstruction", "board_rec")
        )
        header += f" {short:>10}"
    # Per-category columns (if any entry has them)
    has_categories = any(
        "per_category" in e.get("metrics", {}) for _, _, e in all_entries
    )
    category_cols = []
    if has_categories:
        # Collect all category names across entries
        for _, _, e in all_entries:
            for cat in e.get("metrics", {}).get("per_category", {}):
                if cat not in category_cols:
                    category_cols.append(cat)
        for cat in category_cols:
            header += f" {cat[:12]:>12}"

    print(header)
    print("-" * len(header))

    for run_id, g, entry in all_entries:
        m = entry.get("metrics", {})
        row = (
            f"{run_id:<45} "
            f"{entry.get('architecture', '?'):<10} "
            f"{entry.get('hook', '?'):<6} "
            f"{entry.get('bsp_set', '?'):<8}"
        )
        for col, fmt in metrics_cols:
            val = m.get(col)
            if val is not None:
                row += f" {val:>10{fmt}}"
            else:
                row += f" {'—':>10}"
        if has_categories:
            per_cat = m.get("per_category", {})
            for cat in category_cols:
                cat_data = per_cat.get(cat, {})
                val = cat_data.get("mean_f1")
                if val is not None:
                    row += f" {val:>12.3f}"
                else:
                    row += f" {'—':>12}"
        print(row)

    print(f"\nTotal: {len(all_entries)} evaluated run(s)")


def cmd_compare(args: dict) -> None:
    """Compare multiple evaluated runs side by side."""
    run_ids = args["<run_ids>"] or []
    config_path = args["--config"]

    # Load run_ids from config if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            log.error("Config file not found: %s", config_path)
            sys.exit(1)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or "run_ids" not in config:
            log.error("Config must be a YAML with a 'run_ids' list.")
            sys.exit(1)
        config_runs = config["run_ids"]
        if not isinstance(config_runs, list):
            log.error("'run_ids' must be a list in the config file.")
            sys.exit(1)
        # Each entry can be a string or a dict {id: ..., label: ...}
        for item in config_runs:
            if isinstance(item, str):
                run_ids.append(item)
            elif isinstance(item, dict) and "id" in item:
                run_ids.append(item["id"])
            else:
                log.warning("Skipping invalid entry in config run_ids: %s", item)

    if not run_ids:
        log.error("No run IDs specified. Use positional args or --config.")
        sys.exit(1)

    # Try each game's registry to find the runs
    registries: dict[str, dict] = {}
    for game_dir in Path("saes").iterdir():
        if game_dir.is_dir():
            reg = _load_registry(game_dir.name)
            registries.update(reg)

    # Collect entries — support both "run_id:bsp_set" exact keys and plain run_ids
    entries = []
    for rid in run_ids:
        if rid in registries:
            entries.append((rid, registries[rid]))
        else:
            # Fall back: find most recent entry whose stored run_id field matches
            matches = [
                (k, v) for k, v in registries.items() if v.get("run_id", k) == rid
            ]
            if matches:
                matches.sort(key=lambda x: x[1].get("timestamp", ""))
                _, entry = matches[-1]
                entries.append((rid, entry))
            else:
                log.warning("Run '%s' not found in any eval registry", rid)

    if len(entries) < 2:
        log.error("Need at least 2 valid runs to compare.")
        sys.exit(1)

    # Define which metrics to compare and their display format
    metric_rows = [
        ("architecture", None),
        ("bsp_set", None),
        ("fvu", ".6f"),
        ("l0", ".1f"),
        ("dead_features_pct", ".1f"),
        ("coverage", ".4f"),
        ("coverage_above_50", ".4f"),
        ("coverage_above_75", ".4f"),
        ("board_reconstruction", ".4f"),
        ("num_reconstructable_bsps", "d"),
        ("fraction_reconstructable", ".1%"),
    ]

    # Print table
    col_width = max(len(rid) for rid, _ in entries)
    col_width = max(col_width, 24)
    label_width = 26

    header = f"{'Metric':<{label_width}}"
    for rid, _ in entries:
        header += f"  {rid:>{col_width}}"
    print(header)
    print("-" * len(header))

    for key, fmt in metric_rows:
        row = f"{key:<{label_width}}"
        for _, entry in entries:
            if key in ("architecture", "bsp_set"):
                val = entry.get(key, "?")
                row += f"  {val:>{col_width}}"
            else:
                metrics = entry.get("metrics", {})
                val = metrics.get(key)
                if val is None:
                    row += f"  {'—':>{col_width}}"
                elif fmt:
                    row += f"  {val:{col_width}{fmt}}"
                else:
                    row += f"  {str(val):>{col_width}}"
        print(row)

    # Delta row for key metrics if exactly 2 runs
    if len(entries) == 2:
        print("-" * len(header))
        delta_keys = [
            ("coverage", ".4f", True),  # higher is better
            ("board_reconstruction", ".4f", True),
            ("fvu", ".6f", False),  # lower is better
            ("dead_features_pct", ".1f", False),
        ]
        m0 = entries[0][1].get("metrics", {})
        m1 = entries[1][1].get("metrics", {})
        row = f"{'Δ (B - A)':<{label_width}}"
        for key, fmt, higher_better in delta_keys:
            v0, v1 = m0.get(key), m1.get(key)
            if v0 is not None and v1 is not None:
                delta = v1 - v0
                sign = "+" if delta > 0 else ""
                marker = "✓" if (delta > 0) == higher_better else "✗"
                row += f"  {sign}{delta:{fmt}} {marker}"
        print(row)

    # Per-category coverage comparison (if any entry has per_category)
    all_cats: list[str] = []
    for _, entry in entries:
        for cat in entry.get("metrics", {}).get("per_category", {}):
            if cat not in all_cats:
                all_cats.append(cat)

    if all_cats:
        print()
        cat_header = f"{'Category Coverage':<{label_width}}"
        for rid, _ in entries:
            cat_header += f"  {rid:>{col_width}}"
        print(cat_header)
        print("-" * len(cat_header))

        for cat in sorted(all_cats):
            row = f"{cat:<{label_width}}"
            for _, entry in entries:
                per_cat = entry.get("metrics", {}).get("per_category", {})
                cat_data = per_cat.get(cat, {})
                val = cat_data.get("mean_f1")
                if val is not None:
                    row += f"  {val:{col_width}.4f}"
                else:
                    row += f"  {'—':>{col_width}}"
            print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = docopt(__doc__)

    # Configure logging
    level = logging.DEBUG if args["--verbose"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )

    if args["evaluate"]:
        cmd_evaluate(args)
    elif args["compare"]:
        cmd_compare(args)
    elif args["history"]:
        cmd_history(args)


if __name__ == "__main__":
    main()
