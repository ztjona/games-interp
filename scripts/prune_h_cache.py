"""Delete SAE code caches (`_h.pt`) that nothing downstream still needs.

`saes/<game>/cache/{run_id}_h.pt` is the full ``(N, d_dict)`` code matrix. It is
a *cache* -- always regenerable with ``sae_eval.py evaluate --force`` -- but it
is also the single biggest thing on disk: 4.4 GB for an exp8 dictionary at ~290k
positions, 35 GB for exp64, and the directory reached 755 GB on 2026-08-13 with
32 GB free.

A cache is KEPT if any of these still needs it:

  1. its registry row lacks a current metric, so ``backfill_eval_metrics.py``
     must re-read the codes to compute Youden's J / MCC@pref;
  2. it is a 3A panel run or a ``basis_comparison`` input (both read ``_h``
     directly, not through the registry);
  3. it is a random-model control (the 3A null; without it the diagnostic
     silently degrades to the permutation null).

Everything else is deletable. In particular a champion that has been fully
re-evaluated with the current ``sae_eval`` -- champTa after its 2026-08-12
rebuild -- has current metrics on every row, so only its panel members are
still needed.

The keep-set is DERIVED at run time from the registry and the runners, never
hardcoded, so it stays correct as rows are backfilled.

Usage:
    prune_h_cache.py [options]
    prune_h_cache.py (-h | --help)

Options:
    -h --help        Show this help message.
    --game <name>    Game under saes/ [default: quarto]
    --apply          Actually delete. Without it, this is a dry run.
    --min-gb <f>     Only consider caches at least this large [default: 0]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docopt import docopt

# A row counts as current only if it carries every one of these.
CURRENT_METRIC_KEYS = (
    "coverage_mcc",
    "coverage_youden_j",
    "coverage_mcc_at_pref",
    "mean_base_rate",
)


def runs_needing_backfill(registry: dict) -> set[str]:
    return {
        key.split(":")[0]
        for key, entry in registry.items()
        if not all(k in (entry.get("metrics") or {}) for k in CURRENT_METRIC_KEYS)
    }


def runs_read_directly(runners_dir: Path) -> tuple[set[str], set[str]]:
    """Return ``(run_ids, champion_tags)`` that read ``_h`` outside the registry.

    Parsed from the runners rather than duplicated here: 3A's ``$RUNS`` entries
    and its ``$RANDOM_CONTROLS`` values are the things that read the cache
    without going through the registry, so a hardcoded copy would silently rot
    the moment the panel changed.
    """
    out: set[str] = set()
    for name in ("3A-dilution.ps1", "basis-verdict.ps1"):
        path = runners_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # '<run_id>|<bsps>|<random>'  and  = '<control run_id>'
        out |= set(re.findall(r"'([A-Z]\d\w*-champ[\w.\-]+?)\|", text))
        out |= set(re.findall(r"=\s*'(R[12]-champ\w+random-[\w.\-]+)'", text))
        # basis-verdict picks F04/E05/I04 per champion by prefix; keep all three
        # prefixes for every champion that appears in the panel.
    champs = {m for r in out for m in re.findall(r"champ(\w\w)", r)}
    return out, champs


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    cache_dir = Path("saes") / game / "cache"
    registry_path = Path("saes") / game / "eval_registry.json"
    if not cache_dir.is_dir():
        sys.exit(f"error: no cache dir at {cache_dir}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    need_backfill = runs_needing_backfill(registry)
    direct, champs = runs_read_directly(Path("runners"))
    # basis_comparison picks the first available F04/E05/I04 per champion+hook;
    # keep every candidate so the pick cannot be pruned out from under it.
    for rid in list(registry):
        stem = rid.split(":")[0]
        if stem[:3] in ("F04", "E05", "I04", "I03", "E01") and any(
            f"champ{c}" in stem for c in champs
        ):
            direct.add(stem)

    keep = need_backfill | direct
    min_bytes = float(args["--min-gb"]) * 2 ** 30

    kept = dropped = 0.0
    victims: list[Path] = []
    for path in sorted(cache_dir.glob("*_h.pt")):
        rid = path.name[: -len("_h.pt")]
        size = path.stat().st_size
        if rid in keep or size < min_bytes:
            kept += size
            continue
        victims.append(path)
        dropped += size

    print(f"  keep  : {kept / 2**30:>8.1f} GB")
    print(f"  delete: {dropped / 2**30:>8.1f} GB  ({len(victims)} files)")
    print(f"    ({len(need_backfill)} run(s) still need backfill; "
          f"{len(direct)} read _h directly)")
    print()
    for path in victims:
        print(f"    {path.name}  ({path.stat().st_size / 2**30:.1f} GB)")

    if not args["--apply"]:
        print("\n  DRY RUN -- nothing deleted. Re-run with --apply.")
        return 0

    freed = 0.0
    for path in victims:
        freed += path.stat().st_size
        path.unlink()
    print(f"\n  Deleted {len(victims)} file(s), freed {freed / 2**30:.1f} GB.")
    print("  Regenerate any of them with: "
          "python sae_eval.py evaluate saes/%s/<run_id>.pt --bsps=<sets> --force"
          % game)
    return 0


if __name__ == "__main__":
    sys.exit(main())
