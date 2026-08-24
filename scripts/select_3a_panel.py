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

**"Alive" has to mean the same thing here as it does in the diagnostic**, and
on 2026-08-21 it did not. This script estimated it from the registry's
`dead_features_pct` -- which counts a latent that fires on EVERY row as alive --
while `RankingCache` (and therefore `check_sae_usable.py` and the diagnostic
itself) requires `min_freq < freq <= max_freq`, excluding always-on latents
because a constant column carries no information and cannot be a candidate.
`E06-champYb-s42-batchtopk-k64-exp8-s4.conv2` estimated 85 alive and measured
**38**: 47 of its latents fire on >99.9% of rows. It was selected, and the
panel run then died at the runner's gate having produced nothing.

So the health check is now two-tier:

  * where the `_h` cache exists, MEASURE alive with the diagnostic's own
    `RankingCache` -- identical definition, no drift possible;
  * where it does not, fall back to the registry estimate, which is an
    UPPER bound (it cannot see always-on latents), and say so.

and selection walks further down the ranked list to backfill a cell whose
top candidate fails, rather than returning a short cell.

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
    --reserve=<n>      Extra ranked candidates to consider per cell when a
                       selection fails the measured health check [default: 5]
    --no-measure       Skip the _h-based measurement (registry estimate only).
                       Faster, but restores the 2026-08-21 failure mode.
    --canonical-only   Keep only runs trained with their architecture's own
                       specified machinery (see `is_canonical`). champTa and
                       champVe have NO canonical batchtopk/jumprelu runs, so
                       this collapses them to TopK-only -- use it per champion.
    --merge-into=<p>   Update only the cells this invocation selects, inside an
                       EXISTING panel JSON, leaving every other cell untouched.
                       For fixing one champion without re-running the panel.
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
EXP_RE = re.compile(r"exp(\d+)")


def condition_of(run_id: str, entry: dict | None = None) -> str:
    """The CONDITION a run instantiates: architecture, sparsity, expansion, hook.

    Keyed on the HYPERPARAMETERS, not on the run-id string. Stripping the seed
    from the id is not enough, because the campaign letter is part of the id:
    `C01-champYb-s42-topk-k16-exp8-s4.conv2` and
    `K07-champYb-s42-topk-k16-exp8-s4.conv2` are the SAME recipe re-trained
    after the 2026-08-15 conformance fix, and an id-keyed dedup counted them as
    two of the three "distinct conditions" in `Yb/s4.conv2/tiger` — so that
    cell's 0.65 verdict spread was substantially a legacy-vs-canonical
    comparison, not three conditions disagreeing.

    Falls back to the seed-stripped id when the registry entry is unavailable.
    """
    if entry is None:
        return SEED_RE.sub("-sSEED", run_id)
    exp = EXP_RE.search(run_id)
    return "|".join(str(x) for x in (
        entry.get("architecture"),
        entry.get("k") if entry.get("k") is not None else entry.get("l0_target"),
        exp.group(1) if exp else "?",
        entry.get("hook"),
    ))


def is_canonical(entry: dict) -> bool:
    """Was this SAE trained with its OWN architecture's specified machinery?

    The 2026-08-15 audit found dead-feature revival implemented for `TopKSAE`
    only, and JumpReLU's tied init missing — so a panel that mixes pre- and
    post-fix runs is partly comparing training machinery rather than
    dictionaries. Under `--canonical-only` those runs are excluded.

      topk       -- always carried the Gao aux loss; needs `aux_loss_weight`.
      batchtopk  -- needs the aux loss AND the tied init.
      jumprelu   -- aux loss is measured null for it; needs the tied init.

    Note this is a *training-conformance* test, not a quality test: a canonical
    run can still be a bad dictionary, which is what the health gate is for.
    """
    arch = entry.get("architecture")
    if arch == "topk":
        return entry.get("aux_loss_weight") is not None
    if arch == "batchtopk":
        return (entry.get("aux_loss_weight") is not None
                and entry.get("init_mode") is not None)
    if arch == "jumprelu":
        return entry.get("init_mode") is not None
    return False


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


def measured_alive(cache_dir: Path, run_id: str, cfg) -> int | None:
    """Alive-latent count under the DIAGNOSTIC's own definition, or None.

    Uses `RankingCache`, the same class `dilution_diagnostic` and
    `check_sae_usable` use, so the number cannot drift from the one the run
    will be gated on. Returns None when the `_h` cache is absent.
    """
    import torch
    from lib.sae.dilution import RankingCache

    path = cache_dir / f"{run_id}_h.pt"
    if not path.exists():
        return None
    h = torch.load(path, map_location="cpu", weights_only=True)
    try:
        return int(RankingCache(h, cfg).alive.sum())
    finally:
        del h


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    champs = args["--champs"].split(",")
    hooks = args["--hooks"].split(",")
    bases = args["--bases"].split(",")
    top_n, top_k = int(args["--top"]), int(args["--top-k"])
    data_dir = ROOT / "data" / game

    reg = json.loads((ROOT / "saes" / game / "eval_registry.json").read_text())
    # Conformance and the recipe key are TRAINING properties, so they come from
    # the training registry, not the eval one.
    treg = json.loads((ROOT / "saes" / game / "training_registry.json").read_text())
    canon_only = args["--canonical-only"]
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
        tentry = treg.get(run_id)
        if canon_only and not (tentry and is_canonical(tentry)):
            continue
        cond = condition_of(run_id, tentry)
        cur = best[(champ, hook, basis)].get(cond)
        if cur is None or score > cur[0]:
            best[(champ, hook, basis)][cond] = (score, run_id, round(d_dict * (1 - dead/100)))

    from lib.sae.dilution import DilutionConfig  # noqa: E402
    cfg = DilutionConfig(top_k=top_k)
    cache = ROOT / "saes" / game / "cache"
    reserve = int(args["--reserve"])
    measure = not args["--no-measure"]
    seen_alive: dict[str, int | None] = {}

    panel, needed, rejected = {}, set(), []
    print(f"{'cell':<26}{'top-' + str(top_n) + ' CONDITIONS by threat-family MCC':<62}")
    for cell in sorted(best):
        ranked = sorted(best[cell].values(), reverse=True)[:top_n + reserve]
        kept = []
        for score, run_id, est in ranked:
            if len(kept) == top_n:
                break
            alive, how = est, "estimate"
            if measure:
                if run_id not in seen_alive:
                    seen_alive[run_id] = measured_alive(cache, run_id, cfg)
                m = seen_alive[run_id]
                if m is not None:
                    alive, how = m, "measured"
            # Only a MEASURED count can reject: the registry estimate is an
            # upper bound (blind to always-on latents), so rejecting on it
            # would drop runs that are actually fine.
            if how == "measured" and alive < top_k:
                rejected.append((("/".join(cell)), run_id, est, alive))
                continue
            kept.append({"run_id": run_id, "threat_mcc": round(score, 4),
                         "alive": alive, "alive_source": how})
        panel["/".join(cell)] = kept
        needed.update(k["run_id"] for k in kept)
        print(f"{'/'.join(cell):<26}" +
              "  ".join(f"{k['run_id'].split('-')[0]}:{k['threat_mcc']:.3f}"
                        f"{'' if k['alive_source'] == 'measured' else '~'}"
                        for k in kept)
              + ("" if len(kept) == top_n else f"   << ONLY {len(kept)}"))

    if rejected:
        print(f"\n{len(rejected)} candidate(s) REJECTED on measured alive < top_k={top_k} "
              f"(the registry estimate cannot see always-on latents):")
        for cell, run_id, est, alive in rejected:
            print(f"   {cell:<24}{run_id[:48]:<50}est {est:>5}  measured {alive:>5}")

    est_only = sorted(k["run_id"] for ks in panel.values() for k in ks
                      if k["alive_source"] == "estimate")
    if est_only:
        print(f"\n{len(est_only)} selected run(s) have NO _h cache, so their health is an "
              f"UPPER-BOUND estimate and the runner's gate is the real check:")
        for r in est_only:
            print("   ", r)

    missing = sorted(r for r in needed if not (cache / f"{r}_h.pt").exists())
    print(f"\ndistinct runs: {len(needed)}   _h cached: {len(needed)-len(missing)}   "
          f"missing: {len(missing)}")
    for r in missing:
        print("   needs encode:", r)

    merge = args["--merge-into"]
    if merge:
        target = Path(merge)
        base = json.loads(target.read_text(encoding="utf-8"))
        before = {c: [m["run_id"] for m in ms] for c, ms in base["panel"].items()}
        base["panel"].update(panel)
        base.setdefault("merged_selections", []).append({
            "cells": sorted(panel),
            "canonical_only": bool(canon_only),
            "top_k": top_k,
        })
        base["needs_encode"] = sorted(
            r for ms in base["panel"].values() for m in ms
            for r in [m["run_id"]] if not (cache / f"{r}_h.pt").exists())
        target.write_text(json.dumps(base, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nMerged {len(panel)} cell(s) into {target}; "
              f"{len(base['panel']) - len(panel)} cell(s) untouched.")
        changed = 0
        for c, ms in sorted(panel.items()):
            now = [m["run_id"] for m in ms]
            was = before.get(c, [])
            if set(now) != set(was):
                changed += 1
                print(f"  {c}")
                print(f"     was: {', '.join(r.split('-')[0] for r in was)}")
                print(f"     now: {', '.join(r.split('-')[0] for r in now)}")
        print(f"  {changed} of {len(panel)} selected cell(s) changed membership.")
        return 0

    out = args["--output"]
    if out and out != "none":
        Path(out).write_text(json.dumps(
            {"panel": panel, "needs_encode": missing,
             "selected_on": "mean MCC over threat families, BSP-weighted",
             "top_n": top_n, "health_floor_alive": top_k,
             "health_measured_with": ("lib.sae.dilution.RankingCache "
                                      "(same definition as the diagnostic)"),
             "rejected_on_measured_health": [
                 {"cell": c, "run_id": r, "alive_estimate": e, "alive_measured": a}
                 for c, r, e, a in rejected]}, indent=2, sort_keys=True),
            encoding="utf-8")
        print("\nSaved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
