"""Analyze anchored SAE feature slots against their target BSP set.

Usage:
    anchor_analysis.py <checkpoint> [options]
    anchor_analysis.py --run-id=<id> [options]
    anchor_analysis.py -h | --help

Arguments:
    <checkpoint>        Path to anchored SAE checkpoint (.pt)

Options:
    --run-id=<id>       Run ID (checkpoint stem) instead of path
    --game=<game>       Game name [default: quarto]
    --bsps=<name>       Anchor BSP set to analyze [default: auto]
    --cross-bsps=<list> Comma-separated additional BSP sets for cross-set
                        analysis (e.g. gorillaTa,hawkTa) [default: auto]
    --output=<path>     Output JSON path [default: auto]
    --quiet             Suppress stdout summary
    -h --help           Show this help
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from docopt import docopt

from lib.sae.eval import resolve_schema_path


def load_matching_cache(cache_dir, run_id, bsp_set):
    path = cache_dir / f"{run_id}_matching-{bsp_set}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def analyze_anchor_slots(run_id, game, anchor_bsp_set, cross_bsp_sets, quiet=False):
    cache_dir = Path(f"saes/{game}/cache")
    h_path = cache_dir / f"{run_id}_h.pt"
    if not h_path.exists():
        print(f"Error: h cache not found: {h_path}", file=sys.stderr)
        print("Run sae_eval.py evaluate first.", file=sys.stderr)
        sys.exit(1)

    h = torch.load(h_path, map_location="cpu", weights_only=True)
    N, d_dict = h.shape

    m = load_matching_cache(cache_dir, run_id, anchor_bsp_set)
    if m is None:
        print(f"Error: matching cache not found for bsps={anchor_bsp_set}", file=sys.stderr)
        sys.exit(1)

    n_anchors = m["f1"].shape[1]

    schema_path = resolve_schema_path(Path(f"data/{game}"), anchor_bsp_set)
    if schema_path is None:
        print(f"Error: no BSP schema found for animal={anchor_bsp_set}", file=sys.stderr)
        sys.exit(1)

    with open(schema_path) as f:
        schema = json.load(f)
    bsps = schema["bsps"]

    ckpt_path = Path(f"saes/{game}/{run_id}.pt")
    anchor_config = {}
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {})
        anchor_meta = meta.get("anchor", meta)
        for k in ["anchor_lambda_high", "anchor_lambda_medium", "anchor_loss",
                   "anchor_index_mode", "anchor_high_categories", "anchor_tier_counts"]:
            if k in anchor_meta:
                anchor_config[k] = anchor_meta[k]

    high_cats = anchor_config.get("anchor_high_categories", [])

    firing = (h > 0).float()
    feat_freq = firing.mean(dim=0)
    feat_alive = feat_freq > 0

    # --- Per-slot analysis ---
    slots = []
    for i in range(n_anchors):
        bsp = bsps[i]
        cat = bsp["category"]
        tier = "high" if cat in high_cats else "medium"
        alive = bool(feat_alive[i].item())
        freq = float(feat_freq[i].item())
        f1_val = float(m["f1"][i, i].item())
        p_val = float(m["precision"][i, i].item())
        r_val = float(m["recall"][i, i].item())
        mcc_val = float(m["mcc"][i, i].item()) if m["mcc"] is not None else 0.0
        best_feat = int(m["best_feature_per_bsp"][i].item())
        best_f1 = float(m["best_f1_per_bsp"][i].item())
        is_best = best_feat == i

        non_target = []
        for j in range(n_anchors):
            if j != i and m["f1"][i, j].item() > 0.3:
                non_target.append({
                    "bsp_index": j,
                    "bsp_id": bsps[j]["id"],
                    "f1": round(float(m["f1"][i, j].item()), 4),
                })

        slot = {
            "slot": i,
            "bsp_id": bsp["id"],
            "category": cat,
            "tier": tier,
            "alive": alive,
            "firing_rate": round(freq, 6),
            "f1": round(f1_val, 4),
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mcc": round(mcc_val, 4),
            "is_best_feature": is_best,
            "best_feature_index": best_feat,
            "best_feature_f1": round(best_f1, 4),
        }
        if non_target:
            slot["cross_bsp_matches"] = non_target
        slots.append(slot)

    # --- Category summary ---
    cat_map = {}
    for s in slots:
        cat = s["category"]
        if cat not in cat_map:
            cat_map[cat] = {"tier": s["tier"], "n": 0, "alive": 0, "best": 0,
                            "f1s": [], "mccs": []}
        cat_map[cat]["n"] += 1
        cat_map[cat]["f1s"].append(s["f1"])
        cat_map[cat]["mccs"].append(s["mcc"])
        if s["alive"]:
            cat_map[cat]["alive"] += 1
        if s["is_best_feature"]:
            cat_map[cat]["best"] += 1

    categories = {}
    for cat, d in sorted(cat_map.items()):
        categories[cat] = {
            "tier": d["tier"],
            "n": d["n"],
            "alive": d["alive"],
            "best_match": d["best"],
            "mean_f1": round(sum(d["f1s"]) / len(d["f1s"]), 4),
            "mean_mcc": round(sum(d["mccs"]) / len(d["mccs"]), 4),
        }

    # --- Anchor vs free summary ---
    anchor_alive = int(feat_alive[:n_anchors].sum().item())
    free_alive = int(feat_alive[n_anchors:].sum().item())
    free_total = d_dict - n_anchors

    n_best = sum(1 for s in slots if s["is_best_feature"])
    n_polysemantic = sum(1 for s in slots if "cross_bsp_matches" in s)
    n_free_beats = sum(1 for s in slots
                       if not s["is_best_feature"] and s["best_feature_index"] >= n_anchors)

    summary = {
        "run_id": run_id,
        "anchor_bsp_set": anchor_bsp_set,
        "n_anchors": n_anchors,
        "d_dict": d_dict,
        "n_samples": N,
        "anchor_config": anchor_config,
        "anchor_alive": anchor_alive,
        "anchor_alive_pct": round(anchor_alive / n_anchors * 100, 1),
        "free_alive": free_alive,
        "free_alive_pct": round(free_alive / free_total * 100, 1),
        "anchor_mean_freq": round(float(feat_freq[:n_anchors].mean().item()) * 100, 3),
        "free_mean_freq": round(float(feat_freq[n_anchors:].mean().item()) * 100, 3),
        "best_match_count": n_best,
        "polysemantic_count": n_polysemantic,
        "free_beats_anchor_count": n_free_beats,
    }

    # --- Cross-BSP analysis ---
    cross_results = {}
    for cross_set in cross_bsp_sets:
        cm = load_matching_cache(cache_dir, run_id, cross_set)
        if cm is None:
            continue
        n_cross_bsps = cm["f1"].shape[1]

        cross_schema = None
        cross_schema_path = resolve_schema_path(Path(f"data/{game}"), cross_set)
        if cross_schema_path is not None:
            with open(cross_schema_path) as f:
                cross_schema = json.load(f)

        anchor_hits = []
        for i in range(n_anchors):
            for j in range(n_cross_bsps):
                f1_val = cm["f1"][i, j].item()
                if f1_val > 0.3:
                    bsp_name = cross_schema["bsps"][j]["id"] if cross_schema else f"bsp_{j}"
                    anchor_hits.append({
                        "anchor_slot": i,
                        "anchor_bsp": bsps[i]["id"],
                        "cross_bsp_index": j,
                        "cross_bsp_id": bsp_name,
                        "f1": round(f1_val, 4),
                    })

        cross_bsps_covered = len(set(h["cross_bsp_index"] for h in anchor_hits))
        cross_results[cross_set] = {
            "n_bsps": n_cross_bsps,
            "anchor_hits_above_03": len(anchor_hits),
            "cross_bsps_covered": cross_bsps_covered,
            "cross_bsps_covered_pct": round(cross_bsps_covered / n_cross_bsps * 100, 1),
            "hits": anchor_hits,
        }

    result = {
        "summary": summary,
        "categories": categories,
        "slots": slots,
    }
    if cross_results:
        result["cross_bsp_analysis"] = cross_results

    # --- Print summary ---
    if not quiet:
        s = summary
        print(f"{'='*80}")
        print(f"Anchor analysis: {run_id}")
        print(f"{'='*80}")
        print(f"Anchors: {s['n_anchors']}/{s['d_dict']}  "
              f"Alive: {s['anchor_alive']}/{s['n_anchors']} ({s['anchor_alive_pct']}%)  "
              f"Free alive: {s['free_alive']}/{d_dict - n_anchors} ({s['free_alive_pct']}%)")
        print(f"Anchor mean freq: {s['anchor_mean_freq']}%  "
              f"Free mean freq: {s['free_mean_freq']}%")
        print(f"Best match: {s['best_match_count']}/{s['n_anchors']}  "
              f"Polysemantic: {s['polysemantic_count']}/{s['n_anchors']}  "
              f"Free beats anchor: {s['free_beats_anchor_count']}/{s['n_anchors']}")
        print()
        print(f"{'category':<40} {'tier':>4} {'n':>3} {'alive':>5} "
              f"{'best':>4} {'F1':>6} {'MCC':>6}")
        print("-" * 80)
        for cat, d in sorted(categories.items()):
            print(f"{cat:<40} {d['tier']:>4} {d['n']:>3} {d['alive']:>5} "
                  f"{d['best_match']:>4} {d['mean_f1']:>6.3f} {d['mean_mcc']:>6.3f}")

        if cross_results:
            print()
            for cs, cr in cross_results.items():
                print(f"Cross-BSP ({cs}): {cr['cross_bsps_covered']}/{cr['n_bsps']} "
                      f"BSPs covered by anchor features ({cr['cross_bsps_covered_pct']}%)")

    return result


def main():
    args = docopt(__doc__)

    game = args["--game"]
    quiet = args["--quiet"]

    if args["<checkpoint>"]:
        run_id = Path(args["<checkpoint>"]).stem
    else:
        run_id = args["--run-id"]

    if args["--bsps"] == "auto":
        ckpt_path = Path(f"saes/{game}/{run_id}.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            anchor_meta = meta.get("anchor", meta)
            anchor_bsps_path = anchor_meta.get("anchor_bsps", "")
            stem = Path(anchor_bsps_path).stem
            if stem.startswith("bsp_labels-"):
                anchor_bsp_set = stem.split("-", 1)[1].rsplit("_", 1)[0]
            else:
                print("Error: cannot auto-detect anchor BSP set. Use --bsps=<name>.",
                      file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: checkpoint not found at {ckpt_path}. Use --bsps=<name>.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        anchor_bsp_set = args["--bsps"]

    if args["--cross-bsps"] == "auto":
        all_animals = set()
        cache_dir = Path(f"saes/{game}/cache")
        for p in cache_dir.glob(f"{run_id}_matching-*.pt"):
            animal = p.stem.split("matching-")[1]
            all_animals.add(animal)
        cross_bsp_sets = sorted(all_animals - {anchor_bsp_set})
    else:
        cross_bsp_sets = [s.strip() for s in args["--cross-bsps"].split(",") if s.strip()]

    result = analyze_anchor_slots(run_id, game, anchor_bsp_set, cross_bsp_sets, quiet)

    out_dir = Path(f"saes/{game}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args["--output"] == "auto":
        out_path = out_dir / f"{run_id}_anchor-{anchor_bsp_set}.json"
    else:
        out_path = Path(args["--output"])

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if not quiet:
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
