"""Phase 3A dilution diagnostic CLI.

Classify each threat/relational BSP as absent / captured / diluted / tiled in a
trained SAE, from its cached codes. Writes a parseable JSON report and prints
the Gate G-3A tally (>= half of threat BSPs diluted-or-tiled -> 3C proceeds).

Method + thresholds: docs/diary/2026-07-21_3A-dilution-diagnostic.md
Core (game-agnostic): lib/sae/dilution.py

Usage:
    dilution_diagnostic.py <checkpoint> [options]
    dilution_diagnostic.py --run-id=<id> [options]
    dilution_diagnostic.py -h | --help

Arguments:
    <checkpoint>          Path to the SAE checkpoint (.pt); stem == run_id.

Options:
    --run-id=<id>         Run ID (checkpoint stem) instead of a path.
    --game=<game>        Game name [default: quarto].
    --bsps=<name>        BSP set to diagnose (e.g. tigerTa, gorillaVe). Required.
    --categories=<list>  Comma-separated categories to include
                         [default: auto] (auto = relational/threat categories).
    --random-run-id=<id> Random-model SAE run_id for the absent control
                         [default: none].
    --top-k=<n>          Candidate latents per concept [default: 64].
    --output=<path>      Output JSON path [default: auto].
    --quiet              Suppress the stdout summary.
    -h --help            Show this help.

Inputs (all must be row-aligned, N positions):
    saes/<game>/cache/<run_id>_h.pt          full (N, d_dict) SAE codes
    data/<game>/bsp_labels-<bsps>_<count>.pt (N, num_bsps) binary labels
    data/<game>/bsp_schema-<basis>_<count>.json
Optional:
    saes/<game>/cache/<random_run_id>_h.pt   random-model SAE codes
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from docopt import docopt

from lib.sae.eval import resolve_schema_path
from lib.sae.dilution import DilutionConfig, diagnose_concept

# categories treated as relational/threat by --categories=auto. Substring match
# on the category name; covers gorilla (threat_*), hawk (reframed_*), tiger
# (*_winnable, *_completing_attr).
AUTO_MARKERS = ("threat", "winnable", "completable", "completing",
                "reframed_count", "reframed_sq_count", "any_threat")

VERDICTS = ("absent", "captured", "diluted", "tiled")


def load_h(cache_dir: Path, run_id: str) -> torch.Tensor:
    path = cache_dir / f"{run_id}_h.pt"
    if not path.exists():
        print(f"Error: SAE code cache not found: {path}", file=sys.stderr)
        print("  Run `sae_eval.py evaluate <ckpt> --bsps=<set>` first to build it.",
              file=sys.stderr)
        sys.exit(1)
    return torch.load(path, map_location="cpu", weights_only=True)


def resolve_labels(data_dir: Path, bsps: str) -> Path:
    matches = sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))
    if not matches:
        avail = sorted(p.stem.split("-", 1)[1].rsplit("_", 1)[0]
                       for p in data_dir.glob("bsp_labels-*.pt"))
        print(f"Error: no BSP labels for '{bsps}' in {data_dir}", file=sys.stderr)
        if avail:
            print(f"  Available: {', '.join(avail)}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def is_auto_category(cat: str) -> bool:
    return any(mark in cat for mark in AUTO_MARKERS)


def run(run_id, game, bsps, categories, random_run_id, top_k, quiet):
    cache_dir = Path(f"saes/{game}/cache")
    data_dir = Path(f"data/{game}")

    h = load_h(cache_dir, run_id)
    N, d_dict = h.shape

    labels_path = resolve_labels(data_dir, bsps)
    y_all = torch.load(labels_path, map_location="cpu", weights_only=False)
    if y_all.shape[0] != N:
        print(f"Error: row mismatch h N={N} vs labels N={y_all.shape[0]} "
              f"({labels_path.name}). Codes and labels are not aligned.",
              file=sys.stderr)
        sys.exit(1)

    schema_path = resolve_schema_path(data_dir, bsps)
    if schema_path is None:
        print(f"Error: no BSP schema for '{bsps}' in {data_dir}", file=sys.stderr)
        sys.exit(1)
    with open(schema_path) as f:
        schema = json.load(f)
    bsp_defs = schema["bsps"]
    if len(bsp_defs) != y_all.shape[1]:
        print(f"Error: schema has {len(bsp_defs)} bsps but labels have "
              f"{y_all.shape[1]} columns.", file=sys.stderr)
        sys.exit(1)

    h_random = None
    if random_run_id and random_run_id != "none":
        rpath = cache_dir / f"{random_run_id}_h.pt"
        if rpath.exists():
            h_random = torch.load(rpath, map_location="cpu", weights_only=True)
            if h_random.shape[0] != N:
                print(f"Warning: random control N mismatch; ignoring it.",
                      file=sys.stderr)
                h_random = None
        else:
            print(f"Warning: random control cache not found: {rpath}; "
                  f"using permutation null only.", file=sys.stderr)

    if categories == "auto":
        selected = [i for i, b in enumerate(bsp_defs) if is_auto_category(b["category"])]
    else:
        wanted = {c.strip() for c in categories.split(",") if c.strip()}
        selected = [i for i, b in enumerate(bsp_defs) if b["category"] in wanted]
    if not selected:
        print(f"Error: no BSPs matched categories='{categories}'.", file=sys.stderr)
        sys.exit(1)

    cfg = DilutionConfig(top_k=top_k)

    concepts = []
    for idx in selected:
        b = bsp_defs[idx]
        y = y_all[:, idx]
        if float(y.float().mean()) == 0.0 or float(y.float().mean()) == 1.0:
            m = {"verdict": "absent", "base_rate": round(float(y.float().mean()), 4),
                 "degenerate": True}
        else:
            m = diagnose_concept(h, y, cfg, h_random=h_random)
        concepts.append({"bsp_index": idx, "bsp_id": b["id"],
                         "category": b["category"], **m})

    # category + gate summaries
    cat_summary = {}
    for c in concepts:
        cat = c["category"]
        d = cat_summary.setdefault(cat, {v: 0 for v in VERDICTS})
        d["n"] = d.get("n", 0) + 1
        d[c["verdict"]] += 1
    for cat, d in cat_summary.items():
        rec = [c["asymptote_r2"] for c in concepts
               if c["category"] == cat and "asymptote_r2" in c]
        d["mean_asymptote_r2"] = round(sum(rec) / len(rec), 4) if rec else 0.0

    n_threat = len(concepts)
    n_geo = sum(1 for c in concepts if c["verdict"] in ("diluted", "tiled"))
    geo_frac = n_geo / n_threat if n_threat else 0.0
    gate = {
        "n_threat_bsps": n_threat,
        "n_diluted": sum(1 for c in concepts if c["verdict"] == "diluted"),
        "n_tiled": sum(1 for c in concepts if c["verdict"] == "tiled"),
        "n_captured": sum(1 for c in concepts if c["verdict"] == "captured"),
        "n_absent": sum(1 for c in concepts if c["verdict"] == "absent"),
        "geometric_frac": round(geo_frac, 4),
        "verdict": "3C-proceeds" if geo_frac >= 0.5 else "3C-deprioritized",
    }

    result = {
        "summary": {
            "run_id": run_id, "game": game, "bsp_set": bsps,
            "n_samples": int(N), "d_dict": int(d_dict),
            "categories": sorted({c["category"] for c in concepts}),
            "random_control": (random_run_id if h_random is not None else None),
            "config": cfg.to_dict(),
        },
        "gate_g3a": gate,
        "category_summary": cat_summary,
        "concepts": concepts,
    }

    if not quiet:
        _print_summary(result)
    return result


def _print_summary(result):
    s, g = result["summary"], result["gate_g3a"]
    print("=" * 78)
    print(f"3A dilution diagnostic: {s['run_id']}  bsps={s['bsp_set']}")
    print(f"N={s['n_samples']}  d_dict={s['d_dict']}  "
          f"random_control={s['random_control']}")
    print("=" * 78)
    print(f"{'category':<34}{'n':>3}{'capt':>5}{'dilu':>5}{'tile':>5}"
          f"{'abs':>5}{'meanR2':>8}")
    print("-" * 78)
    for cat in sorted(result["category_summary"]):
        d = result["category_summary"][cat]
        print(f"{cat:<34}{d['n']:>3}{d['captured']:>5}{d['diluted']:>5}"
              f"{d['tiled']:>5}{d['absent']:>5}{d['mean_asymptote_r2']:>8.3f}")
    print("-" * 78)
    print(f"Gate G-3A: {g['n_diluted']} diluted + {g['n_tiled']} tiled "
          f"of {g['n_threat_bsps']} threat BSPs -> geometric_frac="
          f"{g['geometric_frac']:.2f} -> {g['verdict']}")


def main():
    args = docopt(__doc__)
    game = args["--game"]

    if args["<checkpoint>"]:
        run_id = Path(args["<checkpoint>"]).stem
    else:
        run_id = args["--run-id"]
    if not run_id:
        print("Error: provide <checkpoint> or --run-id.", file=sys.stderr)
        sys.exit(1)
    if not args["--bsps"]:
        print("Error: --bsps=<name> is required.", file=sys.stderr)
        sys.exit(1)

    result = run(
        run_id=run_id, game=game, bsps=args["--bsps"],
        categories=args["--categories"], random_run_id=args["--random-run-id"],
        top_k=int(args["--top-k"]), quiet=args["--quiet"],
    )

    out_dir = Path(f"saes/{game}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args["--output"] == "auto":
        out_path = out_dir / f"{run_id}_dilution-{args['--bsps']}.json"
    else:
        out_path = Path(args["--output"])
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    if not args["--quiet"]:
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
