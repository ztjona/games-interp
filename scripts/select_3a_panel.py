"""Select the 3A panel: top-N distinct CONDITIONS per (champion, hook, basis).

Replaces the single hand-pinned panel member per cell. Two design points, both
learned the hard way:

1. **Rank on THREAT families, not whole-basis `coverage_mcc`.** 3A only
   diagnoses threat categories, and a whole-basis mean is 39% `cell_attribute`
   on gorilla -- ranking on it picks dictionaries that are good at cells. Same
   error the campaign-K comparison made (CLAUDE.md, "never compare two runs on a
   whole-basis scalar").

2. **Deduplicate by CONDITION, not run.** Seeds of one condition are not three
   panel members: a well-replicated recipe would monopolise the panel and the
   robustness question -- "does the conclusion depend on which SAE we picked?"
   -- would go unanswered. The seed with the best threat score represents its
   condition.

Health gate: a run whose alive-latent count is below `top_k` cannot fill the
diagnostic's candidate list, so it is excluded here rather than silently
measured at a smaller support (see scripts/check_sae_usable.py).

Usage:
    select_3a_panel.py [options]
    select_3a_panel.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game=<name>      Game name [default: quarto]
    --champs=<list>    Comma-separated champion tags [default: Ta,Ve,Yb]
    --hooks=<list>     Comma-separated hooks [default: s4.fc1,s4.conv2]
    --bases=<list>     Comma-separated bases [default: gorilla,hawk,tiger]
    --top=<n>          Conditions to keep per cell [default: 3]
    --top-k=<n>        Diagnostic candidate count; the health floor [default: 64]
    --include-anchored Include anchored runs (normally excluded: they are the
                       supervised positive control, not panel members).
    --output=<path>    Write the panel as JSON [default: none]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.sae.eval import aggregate_per_category_by_family, resolve_schema_path  # noqa: E402

THREAT_FAMILIES = ("line_threat", "square_threat", "global_threat",
                   "offered_completion")
SEED_RE = re.compile(r"-s\d+")


def condition_of(run_id: str) -> str:
    """Run id with the seed stripped -- `F04-champYb-s42-topk-...` -> the recipe."""
    return SEED_RE.sub("-sSEED", run_id)


def threat_score(metrics: dict, schema: dict) -> float | None:
    """Mean MCC over THREAT families only, BSP-count weighted."""
    pc = metrics.get("per_category")
    if not pc:
        return None
    fams = aggregate_per_category_by_family(pc, schema)
    num = den = 0
    for fam, v in fams.items():
        if fam in THREAT_FAMILIES and v.get("mean_mcc") is not None:
            num += v["mean_mcc"] * v["count"]
            den += v["count"]
    return num / den if den else None


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    champs = args["--champs"].split(",")
    hooks = args["--hooks"].split(",")
    bases = args["--bases"].split(",")
    top_n, top_k = int(args["--top"]), int(args["--top-k"])
    data_dir = ROOT / "data" / game

    reg = json.loads((ROOT / "saes" / game / "eval_registry.json").read_text())
    schemas = {}
    for b in bases:
        p = resolve_schema_path(data_dir, b)
        if p:
            schemas[b] = json.loads(Path(p).read_text(encoding="utf-8"))

    best: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for key, entry in reg.items():
        run_id, _, bsp = key.partition(":")
        champ = next((c for c in champs if f"champ{c}" in run_id), None)
        basis = next((b for b in bases if bsp == f"{b}{champ}"), None) if champ else None
        hook = entry.get("hook")
        if not (champ and basis and hook in hooks):
            continue
        if "random" in run_id:
            continue
        if "anchored" in run_id and not args["--include-anchored"]:
            continue
        m = entry.get("metrics") or {}
        dead = m.get("dead_features_pct")
        if dead is None:
            continue
        d_dict = 512 * next((int(e) for e in (64, 32, 16, 8) if f"exp{e}" in run_id), 8)
        if d_dict * (1 - dead / 100) < top_k:      # health gate
            continue
        score = threat_score(m, schemas[basis])
        if score is None:
            continue
        cond = condition_of(run_id)
        cur = best[(champ, hook, basis)].get(cond)
        if cur is None or score > cur[0]:
            best[(champ, hook, basis)][cond] = (score, run_id, round(d_dict * (1 - dead/100)))

    panel, needed = {}, set()
    print(f"{'cell':<26}{'top-' + str(top_n) + ' CONDITIONS by threat-family MCC':<62}")
    for cell in sorted(best):
        top = sorted(best[cell].values(), reverse=True)[:top_n]
        panel["/".join(cell)] = [
            {"run_id": r, "threat_mcc": round(s, 4), "alive": a} for s, r, a in top]
        needed.update(r for _, r, _ in top)
        print(f"{'/'.join(cell):<26}" +
              "  ".join(f"{r.split('-')[0]}:{s:.3f}" for s, r, _ in top))

    cache = ROOT / "saes" / game / "cache"
    missing = sorted(r for r in needed if not (cache / f"{r}_h.pt").exists())
    print(f"\ndistinct runs: {len(needed)}   _h cached: {len(needed)-len(missing)}   "
          f"missing: {len(missing)}")
    for r in missing:
        print("   needs encode:", r)

    out = args["--output"]
    if out and out != "none":
        Path(out).write_text(json.dumps(
            {"panel": panel, "needs_encode": missing,
             "selected_on": "mean MCC over threat families, BSP-weighted",
             "top_n": top_n, "health_floor_alive": top_k}, indent=2, sort_keys=True),
            encoding="utf-8")
        print("\nSaved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
