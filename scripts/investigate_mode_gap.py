"""Is the on-policy coverage drop real, or a base-rate / phase artefact?

`per_mode_coverage.py` found tiger coverage ~38% lower on `model_v_model` rows
than on `random_v_random` rows. Three confounds could produce that WITHOUT the
concept being any harder to read:

  A. PREVALENCE. MCC is NOT base-rate invariant. For a detector of fixed quality
     (TPR=0.50, TNR=0.99), MCC falls 25% from p=0.050 to p=0.013 -- most of the
     observed gap. This is the big one.
  B. POPULATION OVERLAP. The mode masks are not a partition (the early game is
     shared), so "self-play rows" may largely be random-reachable rows.
  C. GAME PHASE. If self-play rows sit later in the game (more pieces placed),
     the boards are more crowded and the concept may be harder for reasons that
     have nothing to do with who was playing.

This script measures all three, then runs the decisive test: compare the two
populations with the number of rows AND the number of positives matched exactly
per concept, so prevalence cannot differ. Any residual gap is a real difficulty
difference.

Usage:
    investigate_mode_gap.py --run-id=<id> --bsps=<name> --positions=<path> [options]
    investigate_mode_gap.py (-h | --help)

Options:
    -h --help          Show this help message.
    --run-id=<id>      SAE run id (needs its _h cache).
    --bsps=<name>      BSP set, e.g. tigerYb.
    --positions=<path> Champion amalgam positions file.
    --game=<name>      Game name [default: quarto]
    --repeats=<n>      Resamples for the matched comparison [default: 5]
    --seed=<n>         RNG seed [default: 0]
    --output=<path>    Output JSON [default: auto]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt

from lib.sae.eval import resolve_schema_path
from scripts.per_mode_coverage import AUTO_MARKERS, tag_modes


def best_mcc(fires: torch.Tensor, y: torch.Tensor) -> float:
    """Max MCC over features for one binary concept, on the given rows."""
    n = float(y.shape[0])
    yf = y.float()
    tp = fires.T @ yf
    fp = fires.sum(0) - tp
    fn = yf.sum() - tp
    tn = n - tp - fp - fn
    num = tp * tn - fp * fn
    den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp(min=1e-12)
    return float((num / den).max())


def matched_compare(fires, labels, mask_a, mask_b, repeats, rng):
    """Per-concept best-MCC with rows AND positives matched exactly.

    For each concept we draw the same number of positives and the same number of
    negatives from both populations, so prevalence and N are identical by
    construction and confound A is removed.
    """
    rows_a, rows_b = np.flatnonzero(mask_a), np.flatnonzero(mask_b)
    out = []
    for j in range(labels.shape[1]):
        col = labels[:, j].numpy()
        pa, na = rows_a[col[rows_a] > 0], rows_a[col[rows_a] == 0]
        pb, nb = rows_b[col[rows_b] > 0], rows_b[col[rows_b] == 0]
        n_pos, n_neg = min(len(pa), len(pb)), min(len(na), len(nb))
        if n_pos < 50:
            out.append({"bsp": j, "skipped": True, "n_pos": int(n_pos)})
            continue
        a_vals, b_vals = [], []
        for _ in range(repeats):
            sel_a = np.concatenate([rng.choice(pa, n_pos, replace=False),
                                    rng.choice(na, n_neg, replace=False)])
            sel_b = np.concatenate([rng.choice(pb, n_pos, replace=False),
                                    rng.choice(nb, n_neg, replace=False)])
            ta, tb = torch.from_numpy(sel_a), torch.from_numpy(sel_b)
            a_vals.append(best_mcc(fires[ta], labels[ta, j]))
            b_vals.append(best_mcc(fires[tb], labels[tb, j]))
        out.append({
            "bsp": j, "skipped": False,
            "n_pos": int(n_pos), "n_neg": int(n_neg),
            "matched_prevalence": round(n_pos / (n_pos + n_neg), 4),
            "mcc_a": round(float(np.mean(a_vals)), 4),
            "mcc_b": round(float(np.mean(b_vals)), 4),
        })
    return out


def main():
    args = docopt(__doc__)
    game, run_id, bsps = args["--game"], args["--run-id"], args["--bsps"]
    rng = np.random.default_rng(int(args["--seed"]))
    data_dir = Path(f"data/{game}")

    h = torch.load(Path(f"saes/{game}/cache/{run_id}_h.pt"),
                   map_location="cpu", weights_only=True)
    fires = (h > 0).float()
    del h

    labels = torch.load(sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))[0],
                        map_location="cpu", weights_only=False)
    schema = json.loads(Path(resolve_schema_path(data_dir, bsps)).read_text(encoding="utf-8"))
    keep = [i for i, b in enumerate(schema["bsps"])
            if any(m in b["category"] for m in AUTO_MARKERS)]
    labels = labels[:, keep]
    names = [schema["bsps"][i]["id"] for i in keep]

    pos_path = Path(args["--positions"])
    print(f"run: {run_id}   bsps: {bsps}   concepts: {len(keep)}\n")
    print("Tagging rows by opponent mode")
    masks, n = tag_modes(pos_path)
    rvr, mm = masks["random_v_random"], masks["model_v_model"]

    # --- B: population overlap ------------------------------------------
    inter = int((rvr & mm).sum())
    print(f"\n[B] POPULATION OVERLAP")
    print(f"  random_v_random rows      : {int(rvr.sum())}")
    print(f"  model_v_model rows        : {int(mm.sum())}")
    print(f"  in BOTH                   : {inter}"
          f"  ({100.0 * inter / max(1, int(mm.sum())):.1f}% of self-play rows are"
          f" also reachable by random play)")
    mm_only = mm & ~rvr
    print(f"  self-play ONLY            : {int(mm_only.sum())}")

    # --- C: game phase ---------------------------------------------------
    data = torch.load(pos_path, map_location="cpu", weights_only=False)
    meta = data.get("metadata")
    print(f"\n[C] GAME PHASE (pieces already on the board)")
    phase = None
    if meta and "n_pieces" in meta[0]:
        phase = np.array([m["n_pieces"] for m in meta])
        for label, msk in (("random_v_random", rvr), ("model_v_model", mm),
                           ("self-play only", mm_only)):
            v = phase[msk]
            print(f"  {label:<20} mean {v.mean():5.2f}  median {np.median(v):5.1f}"
                  f"  p10 {np.percentile(v, 10):4.1f}  p90 {np.percentile(v, 90):4.1f}")
    else:
        print("  metadata has no n_pieces; skipping")

    # --- A: prevalence-matched comparison (the decisive test) ------------
    print(f"\n[A] PREVALENCE-MATCHED COMPARISON "
          f"({args['--repeats']} resamples, rows and positives matched exactly)")
    res = matched_compare(fires, labels, rvr, mm, int(args["--repeats"]), rng)
    used = [r for r in res if not r["skipped"]]
    if not used:
        print("  no concept had >=50 positives in both populations")
        return
    ma = float(np.mean([r["mcc_a"] for r in used]))
    mb = float(np.mean([r["mcc_b"] for r in used]))
    print(f"  concepts compared        : {len(used)} / {len(res)}")
    print(f"  matched prevalence (mean): "
          f"{np.mean([r['matched_prevalence'] for r in used]):.4f}")
    print(f"  random_v_random  MCC     : {ma:.4f}")
    print(f"  model_v_model    MCC     : {mb:.4f}")
    print(f"  residual gap             : {mb - ma:+.4f} "
          f"({100.0 * (mb - ma) / ma:+.1f}%)")

    # Same test against the self-play-only rows, which removes overlap too.
    res2 = matched_compare(fires, labels, rvr, mm_only, int(args["--repeats"]), rng)
    used2 = [r for r in res2 if not r["skipped"]]
    if used2:
        ma2 = float(np.mean([r["mcc_a"] for r in used2]))
        mb2 = float(np.mean([r["mcc_b"] for r in used2]))
        print(f"\n  vs SELF-PLAY-ONLY rows (overlap removed as well):")
        print(f"  random_v_random  MCC     : {ma2:.4f}")
        print(f"  self-play only   MCC     : {mb2:.4f}")
        print(f"  residual gap             : {mb2 - ma2:+.4f} "
              f"({100.0 * (mb2 - ma2) / ma2:+.1f}%)")

    out = args["--output"]
    if out == "auto":
        out = Path(f"saes/{game}/analysis/{run_id}_mode-gap-{bsps}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "run_id": run_id, "bsps": bsps, "bsp_ids": names,
        "overlap": {"rvr": int(rvr.sum()), "mm": int(mm.sum()),
                    "both": inter, "mm_only": int(mm_only.sum())},
        "phase": None if phase is None else {
            "rvr_mean": float(phase[rvr].mean()),
            "mm_mean": float(phase[mm].mean()),
            "mm_only_mean": float(phase[mm_only].mean())},
        "matched_vs_mm": res, "matched_vs_mm_only": res2,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
