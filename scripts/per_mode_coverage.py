"""Per-opponent-mode SAE coverage: is the dictionary describing real play?

A champion's amalgam mixes four opponent modes, and `random_v_random` is a large
slice of it (~30% of champYb). Those are positions a strong agent would rarely
reach, so an SAE trained on the amalgam may be describing what the network does
OFF-policy. The counter-argument is coverage: random play carries ~2x the threat
density, so dropping it would halve the positive examples for exactly the
conjunctive concepts the programme is about.

This script settles it by measurement instead of argument. It tags every row of
a champion's amalgam with the opponent mode(s) it came from, then recomputes the
feature-to-BSP matching SEPARATELY on each mode's rows and reports coverage per
mode. If coverage on self-play rows matches coverage on random rows, the concern
is moot; if they diverge, that is a finding and it changes the training design.

MCC is the reported metric throughout: the modes have different base rates, so
F1 is not comparable across them (see docs/methods-reference.md).

Usage:
    per_mode_coverage.py --run-id=<id> --bsps=<name> [options]
    per_mode_coverage.py (-h | --help)

Options:
    -h --help            Show this help message.
    --run-id=<id>        SAE run id (checkpoint stem); needs its _h cache.
    --bsps=<name>        BSP set, e.g. tigerYb.
    --game=<name>        Game name [default: quarto]
    --positions=<path>   Champion amalgam positions file [default: auto]
    --categories=<list>  Restrict to these categories (comma-separated), or
                         'auto' for the threat/relational families, or 'all'
                         [default: auto]
    --output=<path>      Output JSON [default: auto]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt

from lib.sae.eval import match_features_to_bsps, resolve_schema_path

AUTO_MARKERS = ("threat", "winnable", "completable", "completing",
                "reframed_count", "reframed_sq_count", "any_threat")


def position_keys(data: dict) -> list[tuple[bytes, bytes]]:
    """The same (board, piece) byte key deduplicate_positions.py uses."""
    boards, pieces = data["boards"], data["pieces"]
    return [(np.asarray(b).tobytes(), np.asarray(p).tobytes())
            for b, p in zip(boards, pieces)]


def tag_modes(amalgam_path: Path) -> tuple[dict[str, np.ndarray], int]:
    """{mode: boolean mask over amalgam rows}. A position can belong to several
    modes (the early game is shared), so masks may overlap -- that is real, and
    is why they are reported as separate populations rather than a partition."""
    amalgam = torch.load(amalgam_path, map_location="cpu", weights_only=False)
    keys = position_keys(amalgam)
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    sources = amalgam.get("provenance", {}).get("source_files", [])
    masks: dict[str, np.ndarray] = {}
    for src in sources:
        src_path = Path(str(src).replace("\\", "/"))
        if not src_path.exists():
            print(f"  [WARN] source not found, skipping: {src_path}", file=sys.stderr)
            continue
        mode = next((m for m in ("random_v_random", "model_v_random",
                                 "random_v_model", "model_v_model")
                     if m in src_path.name), src_path.stem)
        raw = torch.load(src_path, map_location="cpu", weights_only=False)
        mask = np.zeros(n, dtype=bool)
        hits = 0
        for k in position_keys(raw):
            i = index.get(k)
            if i is not None:
                mask[i] = True
                hits += 1
        masks[mode] = mask
        print(f"  {mode:<18} {mask.sum():>7} amalgam rows "
              f"({100.0 * mask.sum() / n:5.1f}% of the dataset)")
    return masks, n


def coverage_on(h: torch.Tensor, labels: torch.Tensor, rows: np.ndarray) -> dict:
    idx = torch.from_numpy(np.flatnonzero(rows))
    m = match_features_to_bsps(h[idx], labels[idx])
    best_mcc = m.best_mcc_per_bsp
    base = labels[idx].float().mean(dim=0)
    return {
        "n_rows": int(idx.numel()),
        "coverage_mcc": round(float(best_mcc.mean()), 4),
        "median_mcc": round(float(best_mcc.median()), 4),
        "mean_base_rate": round(float(base.mean()), 4),
        "per_bsp_mcc": [round(float(v), 4) for v in best_mcc],
    }


def main():
    args = docopt(__doc__)
    game = args["--game"]
    run_id, bsps = args["--run-id"], args["--bsps"]
    data_dir = Path(f"data/{game}")

    h_path = Path(f"saes/{game}/cache/{run_id}_h.pt")
    if not h_path.exists():
        sys.exit(f"error: {h_path} not found (run sae_eval.py evaluate first)")
    h = torch.load(h_path, map_location="cpu", weights_only=True)

    label_files = sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))
    if not label_files:
        sys.exit(f"error: no BSP labels for '{bsps}'")
    labels = torch.load(label_files[0], map_location="cpu", weights_only=False)

    schema_path = resolve_schema_path(data_dir, bsps)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    defs = schema["bsps"]

    if args["--categories"] == "auto":
        keep = [i for i, b in enumerate(defs)
                if any(mk in b["category"] for mk in AUTO_MARKERS)]
    elif args["--categories"] == "all":
        keep = list(range(len(defs)))
    else:
        wanted = {c.strip() for c in args["--categories"].split(",")}
        keep = [i for i, b in enumerate(defs) if b["category"] in wanted]
    labels = labels[:, keep]
    kept_ids = [defs[i]["id"] for i in keep]

    pos_path = args["--positions"]
    if pos_path == "auto":
        tag = "".join(ch for ch in bsps if ch.isupper() or ch.isdigit())
        cand = data_dir / f"positions-amalgam_{tag.lower()}_unique.pt"
        if not cand.exists():
            sys.exit(f"error: could not auto-resolve positions ({cand}); "
                     f"pass --positions")
        pos_path = cand
    pos_path = Path(pos_path)

    print(f"run_id  : {run_id}")
    print(f"bsps    : {bsps}  ({len(keep)} concepts)")
    print(f"h       : {tuple(h.shape)}")
    print(f"positions: {pos_path}\n")
    if h.shape[0] != labels.shape[0]:
        sys.exit(f"error: h N={h.shape[0]} != labels N={labels.shape[0]}")

    print("Tagging rows by opponent mode (masks overlap: the early game is shared)")
    masks, n = tag_modes(pos_path)
    if h.shape[0] != n:
        sys.exit(f"error: positions N={n} != codes N={h.shape[0]}")

    results = {"all": coverage_on(h, labels, np.ones(n, dtype=bool))}
    for mode, mask in masks.items():
        if mask.sum() < 100:
            print(f"  [SKIP] {mode}: only {mask.sum()} rows")
            continue
        results[mode] = coverage_on(h, labels, mask)

    # Rows reached ONLY by agent play -- the on-policy population, i.e. the
    # positions the champion actually meets when not paired with a random bot.
    rvr = masks.get("random_v_random")
    if rvr is not None:
        onpolicy = ~rvr
        if onpolicy.sum() >= 100:
            results["NOT_random_v_random"] = coverage_on(h, labels, onpolicy)

    print(f"\n{'population':<24}{'rows':>9}{'base':>8}{'MCC':>8}{'medMCC':>9}")
    print("-" * 60)
    for k in ["all"] + [k for k in results if k != "all"]:
        r = results[k]
        print(f"{k:<24}{r['n_rows']:>9}{r['mean_base_rate']:>8.4f}"
              f"{r['coverage_mcc']:>8.4f}{r['median_mcc']:>9.4f}")
    print("-" * 60)
    ref = results["all"]["coverage_mcc"]
    worst = max((abs(r["coverage_mcc"] - ref), k) for k, r in results.items())
    print(f"Largest deviation from the pooled estimate: {worst[1]} "
          f"({worst[0]:+.4f} MCC)")
    print("Interpretation: a small spread means the dictionary describes all "
          "modes\nalike and the off-policy fraction is harmless. A large spread "
          "means the\ntraining mix matters and should be revisited.")

    out = args["--output"]
    if out == "auto":
        out = Path(f"saes/{game}/analysis/{run_id}_per-mode-{bsps}.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run_id": run_id, "bsps": bsps, "bsp_ids": kept_ids,
         "positions_file": str(pos_path), "results": results}, indent=2),
        encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
