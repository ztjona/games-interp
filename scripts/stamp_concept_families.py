"""Stamp ``concept_family`` / ``family_role`` onto BSP schemas already on disk.

``compute_bsp_labels.py`` writes these fields for every schema it generates, but
schemas written before concept families existed do not have them, and
re-running the label computation just to add two string fields would burn hours
recomputing labels that are already correct.

This script rewrites the schema JSON **in place** and touches nothing else: no
label tensor is read, recomputed or modified. It is idempotent -- running it
twice is a no-op -- and it refuses to change the BSP list, so a stamped schema
still matches its existing ``bsp_labels-*.pt`` row-for-row.

Usage:
    stamp_concept_families.py [options] [<schema>...]
    stamp_concept_families.py (-h | --help)

Arguments:
    <schema>           Schema JSON files. Default: every
                       ``data/<game>/bsp_schema-*.json``.

Options:
    -h --help          Show this help message.
    --game <name>      Game data directory under data/ [default: quarto]
    --game-module <m>  Game module supplying CONCEPT_FAMILIES [default: quarto]
    --dry-run          Report what would change; write nothing.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def stamp(schema: dict, family_of) -> tuple[dict, list[str], set[str]]:
    """Return ``(schema, changed_categories, unclassified_categories)``."""
    changed: list[str] = []
    unclassified: set[str] = set()
    family_summary: dict[str, int] = {}
    category_families: dict[str, dict[str, str]] = {}

    for bsp in schema.get("bsps", []):
        category = bsp.get("category", "unknown")
        fam = family_of(category)
        if fam is None:
            unclassified.add(category)
            # Drop any stale stamp rather than leaving a mapping that the game
            # module no longer declares.
            if bsp.pop("concept_family", None) is not None:
                changed.append(category)
            bsp.pop("family_role", None)
            continue
        family, role = fam
        if bsp.get("concept_family") != family or bsp.get("family_role") != role:
            changed.append(category)
        bsp["concept_family"] = family
        bsp["family_role"] = role
        family_summary[family] = family_summary.get(family, 0) + 1
        category_families[category] = {"concept_family": family,
                                       "family_role": role}

    if schema.get("concept_families") != family_summary:
        changed.append("<rollup>")
    schema["concept_families"] = family_summary
    schema["category_families"] = category_families
    return schema, changed, unclassified


def main() -> int:
    args = docopt(__doc__)
    data_dir = Path("data") / args["--game"]

    module = importlib.import_module(f"scripts.games.{args['--game-module']}")
    family_of = getattr(module, "concept_family_of", None)
    if family_of is None:
        print(f"error: {module.__name__} has no concept_family_of()",
              file=sys.stderr)
        return 1

    paths = [Path(p) for p in args["<schema>"]]
    if not paths:
        paths = sorted(data_dir.glob("bsp_schema-*.json"))
    if not paths:
        print(f"error: no schemas found in {data_dir}", file=sys.stderr)
        return 1

    n_changed = 0
    all_unclassified: set[str] = set()
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        n_bsps_before = len(schema.get("bsps", []))
        schema, changed, unclassified = stamp(schema, family_of)
        assert len(schema.get("bsps", [])) == n_bsps_before, (
            f"{path}: BSP count changed -- refusing to write")
        all_unclassified |= unclassified

        if not changed:
            print(f"  [ ok ] {path.name}: already stamped")
            continue
        n_changed += 1
        cats = sorted(set(c for c in changed if c != "<rollup>"))
        print(f"  [{'DRY' if args['--dry-run'] else 'writ'}] {path.name}: "
              f"{len(cats)} category/categories stamped")
        if not args["--dry-run"]:
            # sort_keys matches compute_bsp_labels.py, so a re-generated schema
            # and a stamped one are byte-comparable and git churn stays honest.
            path.write_text(
                json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")

    if all_unclassified:
        print(f"\nWARNING: no concept_family declared for "
              f"{sorted(all_unclassified)}.\nAdd them to CONCEPT_FAMILIES in "
              f"scripts/games/quarto.py; they are excluded from family "
              f"rollups until then.", file=sys.stderr)
    print(f"\n{n_changed} schema(s) "
          f"{'would be' if args['--dry-run'] else ''} updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
