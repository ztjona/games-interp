"""Query the SAE evaluation registry from the command line.

A thin, deterministic wrapper around ``saes/<game>/eval_registry.json``
so that humans (and LLM agents) don't have to grep raw JSON or write
ad-hoc one-liners every time they need a ranking, comparison, or
per-category breakdown. All output is plain text by default and
machine-readable JSON when ``--json`` is passed.

Usage:
    registry_query.py top [--game=<g>] [--bsps=<set>] [--metric=<m>] [--limit=<n>] [--include=<pat>] [--exclude=<pat>] [--hook=<h>] [--arch=<a>] [--json]
    registry_query.py get <run_id> [--game=<g>] [--bsps=<set>] [--json]
    registry_query.py category <run_id> [--game=<g>] [--bsps=<set>] [--metric=<m>] [--json]
    registry_query.py family <run_id> [--game=<g>] [--bsps=<set>] [--json]
    registry_query.py triads [--game=<g>] [--json]
    registry_query.py compare <run_id_a> <run_id_b> [--game=<g>] [--bsps=<set>] [--bsps-b=<set>] [--json]
    registry_query.py list [--game=<g>] [--bsps=<set>] [--limit=<n>] [--json]
    registry_query.py bsps [--game=<g>] [--json]
    registry_query.py (-h | --help)

Subcommands:
    top        Top-N runs ranked by a metric.
    get        Full headline metrics for one run.
    category   Per-category breakdown for one run (e.g. threat_line, cell_attribute).
    family     Per-concept-family breakdown: categories rolled up by the game
               fact they describe (line_threat, square_threat, ...), read from
               the schema's ``concept_family`` stamp. This is the grouping that
               makes gorilla/hawk/tiger numbers comparable.
    triads     Cross-basis triads: which category in each basis describes the
               same family, derived from the schemas on disk.
    compare    Side-by-side headline metrics for two runs (delta = a - b).
    list       Compact listing of all runs for a BSP set.
    bsps       List available BSP sets in the registry.

Options:
    --game=<g>       Game subdirectory under ``saes/``. [default: quarto]
    --bsps=<set>     BSP set: ``gorilla``, ``hawk``, etc. [default: gorilla]
    --metric=<m>     Metric key on which to rank or report.
                     Common: ``coverage``, ``coverage_mcc``,
                     ``coverage_youden_j``, ``coverage_mcc_at_pref``,
                     ``board_reconstruction``, ``fvu``, ``l0``,
                     ``dead_features_pct``. [default: coverage_mcc]
                     NOTE: MCC is the project's headline metric (base-rate
                     invariant). F1 / F1-lift are retained for comparison with
                     the literature only -- do not rank on them.
    --limit=<n>      Max rows to show. [default: 20]
    --include=<pat>  Substring or regex to keep (run_id must match).
    --exclude=<pat>  Substring or regex to drop.
    --hook=<h>       Filter by hook (matches if run_id ends with ``-<hook>``).
    --arch=<a>       Filter by architecture (matches if ``-<arch>-`` in run_id).
    --json           Emit JSON instead of a text table.
    -h --help        Show this message.

Examples:
    # Top 10 conv2 trained-model runs by F1-lift on gorilla
    #   registry_query.py top --bsps=gorilla --hook=conv2 \\
    #     --exclude=random --limit=10
    #
    # Per-category for the headline winner on hawk
    #   registry_query.py category C01-c2arch-s42-topk-k16-exp8-conv2 --bsps=hawk
    #
    # Side-by-side a trained run vs its random sibling
    #   registry_query.py compare C01-c2arch-s42-topk-k16-exp8-conv2 \\
    #     G01-c2random-s42-topk-k16-exp8-conv2 --bsps=gorilla
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from docopt import docopt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lib.sae.eval import (  # noqa: E402
    aggregate_per_category_by_family,
    category_families,
    derive_triads,
    resolve_schema_path,
)


# ──────────────────────────────────────────────────────────────────────
# Registry I/O
# ──────────────────────────────────────────────────────────────────────


def load_registry(game: str) -> dict[str, dict[str, Any]]:
    path = PROJECT_DIR / "saes" / game / "eval_registry.json"
    if not path.exists():
        sys.exit(f"error: registry not found at {path}")
    return json.loads(path.read_text())


def split_key(key: str) -> tuple[str, str | None]:
    """Registry keys are ``run_id`` or ``run_id:bsp_set``."""
    if ":" in key:
        rid, bsp = key.rsplit(":", 1)
        return rid, bsp
    return key, None


def entries_for(
    registry: dict[str, dict[str, Any]], bsps: str
) -> dict[str, dict[str, Any]]:
    """Return ``{run_id: entry}`` for a given BSP set.

    Handles both new (``run_id:bsp_set``) and legacy (plain ``run_id``)
    keys; legacy entries default to gorilla.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, entry in registry.items():
        rid, bsp = split_key(key)
        if bsp is None:
            bsp = "gorilla"  # legacy default
        if bsp == bsps:
            out[rid] = entry
    return out


# ──────────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────────


def matches(rid: str, pattern: str | None) -> bool:
    if pattern is None:
        return True
    try:
        return re.search(pattern, rid) is not None
    except re.error:
        return pattern in rid


def apply_filters(
    runs: dict[str, dict[str, Any]],
    include: str | None,
    exclude: str | None,
    hook: str | None,
    arch: str | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rid, entry in runs.items():
        if include is not None and not matches(rid, include):
            continue
        if exclude is not None and matches(rid, exclude):
            continue
        if hook is not None and not rid.endswith(f"-{hook}"):
            continue
        if arch is not None and f"-{arch}-" not in rid:
            continue
        out[rid] = entry
    return out


# ──────────────────────────────────────────────────────────────────────
# Subcommands
# ──────────────────────────────────────────────────────────────────────


# Order matters. MCC leads (headline) and Youden's J sits beside it as the
# prevalence-INVARIANT companion; MCC-vs-J divergence reads off how much of a
# difference is base rate rather than quality.
#
# The F1 family is DEMOTED (2026-08-11): F1 is retained only for comparison with
# the published literature at write-up, and F1-lift is retained only so older
# registry rows stay readable. Neither is used to select, rank, gate or
# conclude, and neither belongs in a headline table -- both move with prevalence
# (F1 54%, F1-lift 44% over p in [0.005, 0.100]; audit in
# scripts/prevalence_audit.py), so they are not comparable across concepts,
# populations or champions.
HEADLINE_METRICS = (
    "coverage_mcc",
    "coverage_youden_j",
    "board_reconstruction",
    "fraction_reconstructable",
    "fvu",
    "l0",
    "dead_features_pct",
    # --- demoted: the SINGLE F1 number, literature comparison only ---
    "coverage",
)


def fmt(v: Any, width: int = 8, digits: int = 4) -> str:
    if v is None:
        return f"{'-':>{width}}"
    if isinstance(v, float):
        return f"{v:>{width}.{digits}f}"
    return f"{str(v):>{width}}"


def cmd_top(args: dict[str, Any]) -> None:
    registry = load_registry(args["--game"])
    runs = entries_for(registry, args["--bsps"])
    runs = apply_filters(
        runs,
        args["--include"],
        args["--exclude"],
        args["--hook"],
        args["--arch"],
    )
    metric = args["--metric"]
    limit = int(args["--limit"])

    rows = []
    for rid, entry in runs.items():
        m = entry.get("metrics", {})
        v = m.get(metric)
        if v is None:
            continue
        rows.append((rid, v, m))

    # Sort: lower-is-better for fvu, l0, dead_features_pct
    lower_better = metric in ("fvu", "l0", "dead_features_pct")
    rows.sort(key=lambda r: r[1], reverse=not lower_better)
    rows = rows[:limit]

    if args["--json"]:
        print(
            json.dumps(
                {
                    "game": args["--game"],
                    "bsps": args["--bsps"],
                    "metric": metric,
                    "results": [
                        {"run_id": rid, metric: v, "metrics": m} for rid, v, m in rows
                    ],
                },
                indent=2,
            )
        )
        return

    print(
        f"# top by {metric} | game={args['--game']} | bsps={args['--bsps']}"
        f" | n={len(rows)}"
    )
    header = (
        f"{'rank':>4} {'run_id':60s} {metric:>10s} "
        f"{'MCC':>6} {'J':>6} {'FVU':>6} {'L0':>5}   {'(F1)':>6}"
    )
    print(header)
    for i, (rid, v, m) in enumerate(rows, 1):
        # MCC and J are the decision metrics. F1 is parenthesised because it is
        # for literature comparison only and must not be ranked on.
        print(
            f"{i:>4} {rid:60s} {fmt(v, 10):>10s} "
            f"{fmt(m.get('coverage_mcc'), 6, 3)} "
            f"{fmt(m.get('coverage_youden_j'), 6, 3)} "
            f"{fmt(m.get('fvu'), 6, 3)} "
            f"{fmt(m.get('l0'), 5, 1)}   "
            f"{fmt(m.get('coverage'), 6, 3)}"
        )
    print("MCC = headline. J = Youden (prevalence-invariant); MCC-vs-J "
          "divergence flags base-rate effects.")
    print("(F1) = literature comparison only -- never rank on it. J is blank "
          "until a run is re-evaluated.")
    print()
    print("!! These are WHOLE-BASIS means: a within-run health/ranking signal,")
    print("   NOT a basis for comparing two runs. A gorilla mean is 39%")
    print("   cell_attribute and 46% threats, so a gap between two runs here can")
    print("   sit entirely in families you do not care about -- which is exactly")
    print("   how the 2026-08-15 campaign-K architecture comparison went wrong.")
    print("   To COMPARE runs:  registry_query.py family <run_id> --bsps=<set>")


def cmd_get(args: dict[str, Any]) -> None:
    registry = load_registry(args["--game"])
    runs = entries_for(registry, args["--bsps"])
    rid = args["<run_id>"]
    if rid not in runs:
        sys.exit(f"error: '{rid}' not found for bsps={args['--bsps']}")
    entry = runs[rid]
    if args["--json"]:
        print(json.dumps({rid: entry}, indent=2))
        return
    m = entry.get("metrics", {})
    print(f"# {rid} | bsps={args['--bsps']}")
    for k in HEADLINE_METRICS:
        if k in m:
            print(f"  {k:30s} {fmt(m[k], 10, 4)}")
    extras = sorted(set(m) - set(HEADLINE_METRICS) - {"per_category"})
    if extras:
        print("  --- other ---")
        for k in extras:
            v = m[k]
            if isinstance(v, (int, float)) or v is None:
                print(f"  {k:30s} {fmt(v, 10, 4)}")


def cmd_category(args: dict[str, Any]) -> None:
    registry = load_registry(args["--game"])
    runs = entries_for(registry, args["--bsps"])
    rid = args["<run_id>"]
    if rid not in runs:
        sys.exit(f"error: '{rid}' not found for bsps={args['--bsps']}")
    pc = runs[rid].get("metrics", {}).get("per_category")
    if pc is None:
        sys.exit(
            f"error: '{rid}:{args['--bsps']}' has no per_category breakdown "
            "(older eval). Re-run sae_eval.py evaluate --force to regenerate."
        )
    if args["--json"]:
        print(json.dumps({rid: pc}, indent=2))
        return
    print(f"# {rid} | bsps={args['--bsps']} | per-category breakdown")
    print(
        f"  {'category':30s} {'n':>4} {'MCC':>7} {'sd':>7} {'J':>7} "
        f"{'base':>6} {'(F1)':>7}"
    )
    for cat, vals in sorted(pc.items()):
        print(
            f"  {cat:30s} {fmt(vals.get('count'), 4, 0)} "
            f"{fmt(vals.get('mean_mcc'), 7, 3)} "
            f"{fmt(vals.get('sd_mcc'), 7, 3)} "
            f"{fmt(vals.get('mean_youden_j'), 7, 3)} "
            f"{fmt(vals.get('mean_base_rate'), 6, 2)} "
            f"{fmt(vals.get('mean_f1'), 7, 3)}"
        )


def _load_schema_for(game: str, bsps: str) -> dict | None:
    """Load the BSP schema for a set, or ``None`` if there isn't one on disk."""
    path = resolve_schema_path(PROJECT_DIR / "data" / game, bsps)
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_family(args: dict[str, Any]) -> None:
    """Per-CONCEPT-FAMILY breakdown: categories rolled up by the game fact.

    A basis is a packaging convention, a family is the concept. Grouping by
    family is what makes gorilla/hawk/tiger numbers comparable, and it is read
    straight off the schema's ``concept_family`` stamp -- no mapping lives here.
    """
    registry = load_registry(args["--game"])
    runs = entries_for(registry, args["--bsps"])
    rid = args["<run_id>"]
    if rid not in runs:
        sys.exit(f"error: '{rid}' not found for bsps={args['--bsps']}")
    pc = runs[rid].get("metrics", {}).get("per_category")
    if pc is None:
        sys.exit(
            f"error: '{rid}:{args['--bsps']}' has no per_category breakdown "
            "(older eval). Re-run sae_eval.py evaluate --force to regenerate."
        )
    schema = _load_schema_for(args["--game"], args["--bsps"])
    if schema is None:
        sys.exit(f"error: no BSP schema on disk for bsps={args['--bsps']}")

    fams = aggregate_per_category_by_family(pc, schema)
    if not fams:
        sys.exit(
            f"error: schema for '{args['--bsps']}' carries no concept_family "
            "stamp. Run: python scripts/stamp_concept_families.py"
        )
    if args["--json"]:
        print(json.dumps({rid: fams}, indent=2, sort_keys=True))
        return

    print(f"# {rid} | bsps={args['--bsps']} | per-concept-family breakdown")
    print("# Family means are weighted by BSP count, so they are means over "
          "BSPs, not over categories.")
    print(
        f"  {'concept_family':24s} {'n':>4} {'MCC':>7} {'J':>7} "
        f"{'MCC@pref':>9} {'base':>6} {'(F1)':>7}  categories"
    )
    for family, vals in sorted(fams.items()):
        print(
            f"  {family:24s} {fmt(vals.get('count'), 4, 0)} "
            f"{fmt(vals.get('mean_mcc'), 7, 3)} "
            f"{fmt(vals.get('mean_youden_j'), 7, 3)} "
            f"{fmt(vals.get('mean_mcc_at_pref'), 9, 4)} "
            f"{fmt(vals.get('mean_base_rate'), 6, 2)} "
            f"{fmt(vals.get('mean_f1'), 7, 3)}  "
            f"{', '.join(vals.get('categories', []))}"
        )
    print("MCC@pref is the number to compare ACROSS bases/champions; raw MCC "
          "moves with the base rate.")


def cmd_compare(args: dict[str, Any]) -> None:
    registry = load_registry(args["--game"])
    bsps_a = args["--bsps"]
    bsps_b = args.get("--bsps-b") or bsps_a
    runs_a = entries_for(registry, bsps_a)
    runs_b = entries_for(registry, bsps_b) if bsps_b != bsps_a else runs_a
    ra, rb = args["<run_id_a>"], args["<run_id_b>"]
    if ra not in runs_a:
        sys.exit(f"error: '{ra}' not found for bsps={bsps_a}")
    if rb not in runs_b:
        sys.exit(f"error: '{rb}' not found for bsps={bsps_b}")
    ma = runs_a[ra].get("metrics", {})
    mb = runs_b[rb].get("metrics", {})
    keys = [k for k in HEADLINE_METRICS if k in ma or k in mb]

    if args["--json"]:
        print(
            json.dumps(
                {
                    "a": ra,
                    "b": rb,
                    "bsps": bsps_a,
                    "bsps_b": bsps_b,
                    "metrics": {
                        k: {
                            "a": ma.get(k),
                            "b": mb.get(k),
                            "delta": (
                                ma.get(k) - mb.get(k)
                                if isinstance(ma.get(k), (int, float))
                                and isinstance(mb.get(k), (int, float))
                                else None
                            ),
                        }
                        for k in keys
                    },
                },
                indent=2,
            )
        )
        return

    bsps_label = bsps_a if bsps_a == bsps_b else f"{bsps_a} vs {bsps_b}"
    print(f"# compare on bsps={bsps_label}")
    print(f"  a = {ra}  (bsps={bsps_a})")
    print(f"  b = {rb}  (bsps={bsps_b})")
    print(f"  {'metric':30s} {'a':>10} {'b':>10} {'a - b':>10}")
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        d = (
            va - vb
            if isinstance(va, (int, float)) and isinstance(vb, (int, float))
            else None
        )
        print(f"  {k:30s} {fmt(va, 10, 4)} {fmt(vb, 10, 4)} {fmt(d, 10, 4)}")


def cmd_list(args: dict[str, Any]) -> None:
    registry = load_registry(args["--game"])
    runs = entries_for(registry, args["--bsps"])
    limit = int(args["--limit"]) if args["--limit"] else len(runs)

    items = []
    for rid in sorted(runs):
        m = runs[rid].get("metrics", {})
        items.append(
            {
                "run_id": rid,
                "coverage": m.get("coverage"),
                "coverage_mcc": m.get("coverage_mcc"),
                "coverage_youden_j": m.get("coverage_youden_j"),
            }
        )

    if args["--json"]:
        print(
            json.dumps(
                {
                    "game": args["--game"],
                    "bsps": args["--bsps"],
                    "n": len(items),
                    "runs": items[:limit],
                },
                indent=2,
            )
        )
        return

    print(f"# list | game={args['--game']} | bsps={args['--bsps']} | n={len(items)}")
    print(f"  {'run_id':60s} {'cov':>6} {'MCC':>6} {'lift':>6}")
    for it in items[:limit]:
        print(
            f"  {it['run_id']:60s} "
            f"{fmt(it['coverage'], 6, 3)} "
            f"{fmt(it['coverage_mcc'], 6, 3)} "
            f"{fmt(it['coverage_youden_j'], 6, 3)}"
        )


def cmd_bsps(args: dict[str, Any]) -> None:
    registry = load_registry(args["--game"])
    seen: dict[str, int] = {}
    for key in registry:
        _, bsp = split_key(key)
        bsp = bsp or "gorilla"
        seen[bsp] = seen.get(bsp, 0) + 1
    out = sorted(seen.items())
    if args["--json"]:
        print(json.dumps({"game": args["--game"], "bsp_sets": dict(out)}, indent=2))
        return
    print(f"# bsp sets in saes/{args['--game']}/eval_registry.json")
    for bsp, n in out:
        print(f"  {bsp:20s} {n:>5} entries")


def cmd_triads(args: dict[str, Any]) -> None:
    """Show which category in each basis describes the same concept family."""
    data_dir = PROJECT_DIR / "data" / args["--game"]
    schemas: dict[str, dict] = {}
    for path in sorted(data_dir.glob("bsp_schema-*_[0-9]*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        basis = schema.get("animal")
        # Keep one schema per basis; the champion/pool-suffixed copies are
        # duplicates of the basis schema by construction.
        if basis and basis not in schemas:
            schemas[basis] = schema
    if not schemas:
        sys.exit(f"error: no BSP schemas under {data_dir}")

    triads = derive_triads(schemas)
    if args["--json"]:
        print(json.dumps({"bases": sorted(schemas), "triads": triads},
                         indent=2, sort_keys=True))
        return
    if not triads:
        sys.exit("error: schemas carry no concept_family stamp. "
                 "Run: python scripts/stamp_concept_families.py")
    print(f"# cross-basis triads | bases={', '.join(sorted(schemas))}")
    print("# Each row: one game fact, named in each basis' own vocabulary.")
    bases = sorted(schemas)
    print(f"  {'concept_family':22s} " + " ".join(f"{b:30s}" for b in bases))
    for family, per_basis in sorted(triads.items()):
        print(f"  {family:22s} "
              + " ".join(f"{per_basis.get(b, '-'):30s}" for b in bases))
    singles = {
        c["concept_family"]
        for s in schemas.values() for c in category_families(s).values()
    } - set(triads)
    if singles:
        print(f"\n  single-basis families (no cross-basis comparison exists): "
              f"{', '.join(sorted(singles))}")


# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = docopt(__doc__)
    if args["top"]:
        cmd_top(args)
    elif args["get"]:
        cmd_get(args)
    elif args["category"]:
        cmd_category(args)
    elif args["family"]:
        cmd_family(args)
    elif args["triads"]:
        cmd_triads(args)
    elif args["compare"]:
        cmd_compare(args)
    elif args["list"]:
        cmd_list(args)
    elif args["bsps"]:
        cmd_bsps(args)


if __name__ == "__main__":
    main()
