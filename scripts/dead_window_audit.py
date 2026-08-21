"""How much of `dead_features_pct` is a per-batch MEASUREMENT artefact?

`compute_metrics` and (until rule 3A.3) the revival loss both call a feature dead
when it does not fire in ONE batch. Gao et al. 2024 instead call it dead when it
has not fired in the last N tokens (~10M). The two differ mechanically: at
batch 4096 with k = 32 only B*k of B*d_dict slots CAN fire, so a healthy but
rare feature reads dead in most batches.

This measures the gap directly on cached codes -- no retraining, no GPU. For
window sizes W it reports the share of latents that fire in NO batch of a
W-batch window, which is the Gao definition evaluated on the same data the
per-batch number came from.

Usage:
    dead_window_audit.py <checkpoint>... [options]
    dead_window_audit.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game=<name>      Game name [default: quarto]
    --batch=<n>        Batch size the training loop used [default: 4096]
    --windows=<list>   Comma-separated window sizes in batches [default: 1,2,5,10,25,50]
    --output=<path>    Write results as JSON [default: none]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent


def audit(h: torch.Tensor, batch: int, windows: list[int]) -> dict:
    n, d = h.shape
    n_batches = n // batch
    # (n_batches, d) boolean: did latent j fire anywhere in batch i?
    fired = np.zeros((n_batches, d), dtype=bool)
    for i in range(n_batches):
        fired[i] = (h[i * batch:(i + 1) * batch] > 0).any(dim=0).numpy()
    out = {"n_batches": n_batches, "d_dict": d, "windows": {}}
    for w in windows:
        if w > n_batches:
            continue
        # mean over disjoint windows of "fired in NO batch of the window"
        k = n_batches // w
        dead = [(~fired[i * w:(i + 1) * w].any(axis=0)).mean() for i in range(k)]
        out["windows"][w] = round(float(np.mean(dead)) * 100, 2)
    out["never_fires_anywhere"] = round(float((~fired.any(axis=0)).mean()) * 100, 2)
    return out


def main() -> int:
    args = docopt(__doc__)
    batch = int(args["--batch"])
    windows = [int(w) for w in args["--windows"].split(",")]
    cache = ROOT / "saes" / args["--game"] / "cache"

    results = {}
    hdr = f"{'run':<48}" + "".join(f"W={w:<7}" for w in windows) + f"{'never':>8}"
    print(hdr)
    print("-" * len(hdr))
    for raw in args["<checkpoint>"]:
        run_id = Path(raw).stem
        p = cache / f"{run_id}_h.pt"
        if not p.exists():
            print(f"{run_id[:47]:<48}no _h cache")
            continue
        h = torch.load(p, map_location="cpu", weights_only=True)
        r = audit(h, batch, windows)
        del h
        results[run_id] = r
        row = "".join(f"{r['windows'].get(w, float('nan')):<9.1f}" for w in windows)
        print(f"{run_id[:47]:<48}{row}{r['never_fires_anywhere']:>8.1f}")

    print(f"\nW=1 is the per-batch number the metric reports; `never` is the share of")
    print("latents that fire on NO row of the whole dataset -- the only unambiguous")
    print("'dead'. The gap between them is the measurement artefact.")
    out = args["--output"]
    if out and out != "none":
        Path(out).write_text(json.dumps(results, indent=2, sort_keys=True),
                             encoding="utf-8")
        print("Saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
