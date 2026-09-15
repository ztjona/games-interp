"""Freshness filter for 3B-causal Wave 1b (pre-registration 2026-09-14, S4.3).

Drops every position whose BOARD, up to the 8 board symmetries (the piece in
hand ignored), is the board of any pair a pilot run used -- any concept, any
pair kind. Matching is the seed-0 board-only canonical hash of
``scripts/compute_orbit_ids.py`` (``board_keys``).

The pilot's boards come from its per-pair records, R7 keys only: every
representation of a run shares one pair set per (concept, kind), and R7 (the
gate's DAS-1) is the only representation whose results are unsealed. The R7
records are checked complete against the run's power table before use.

``interchange_3b.py`` imports :func:`pilot_board_keys` and :func:`fresh_mask`
to re-check a filtered set at run time, so the filter and the check share one
implementation.

Usage:
    freshness_filter.py <positions_file> --pilot-run=<json> --output=<path>
    freshness_filter.py (-h | --help)

Options:
    -h --help           Show this help message.
    --pilot-run=<json>  Pilot run summary, e.g.
                        saes/quarto/analysis/3B-causal_champYb_wave1.json
                        (its _pairs.pt must sit next to it).
    --output=<path>     Filtered positions file; its provenance gains a
                        "freshness" block with the counts removed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_orbit_ids import board_keys  # noqa: E402


def pilot_board_keys(run_json: str | Path) -> tuple[np.ndarray, dict]:
    """Sorted unique board keys of every pair in a pilot run, and a provenance dict."""
    run_json = ROOT / run_json
    run = json.loads(run_json.read_text(encoding="utf-8"))
    rec = torch.load(str(run_json).replace(".json", "_pairs.pt"), map_location="cpu",
                     weights_only=False)
    cfg = yaml.safe_load((ROOT / run["config"]).read_text(encoding="utf-8"))
    boards = torch.load(ROOT / cfg["positions"], map_location="cpu", weights_only=False)["boards"]
    bases = []
    for bsp_id, per_kind in run["power"].items():
        for kind, pw in per_kind.items():
            if pw["n"] == 0:
                continue
            x = rec.get(f"{bsp_id}|R7|{kind}")
            if x is None or int(x["base"].numel()) != pw["n"]:
                raise SystemExit(f"pilot records incomplete for {bsp_id}|{kind}: the power "
                                 f"table says {pw['n']} pairs")
            bases.append(x["base"].long())
    rows = torch.unique(torch.cat(bases))
    keys = np.unique(board_keys(boards[rows]))
    info = {"pilot_run": (run_json.relative_to(ROOT) if run_json.is_relative_to(ROOT)
                          else run_json).as_posix(),
            "pilot_positions": cfg["positions"],
            "pilot_pair_rows": int(rows.numel()),
            "pilot_board_orbits": int(keys.size),
            "hash": "scripts/compute_orbit_ids.py board_keys, seed 0"}
    return keys, info


def fresh_mask(boards: torch.Tensor, pilot_keys: np.ndarray) -> torch.Tensor:
    """(N,) bool: True where the board matches no pilot board up to symmetry."""
    return torch.from_numpy(~np.isin(board_keys(boards), pilot_keys))


def main() -> int:
    args = docopt(__doc__)
    src = Path(args["<positions_file>"])
    data = torch.load(src, map_location="cpu", weights_only=False)
    keys, info = pilot_board_keys(args["--pilot-run"])
    keep = fresh_mask(data["boards"], keys)
    n, n_keep = int(keep.numel()), int(keep.sum())
    idx = torch.nonzero(keep).flatten()
    prov = dict(data.get("provenance", {}))
    prov["freshness"] = {**info, "source": src.as_posix(), "n_in": n, "n_removed": n - n_keep,
                         "n_out": n_keep}
    out = {"boards": data["boards"][idx], "pieces": data["pieces"][idx],
           "metadata": [data["metadata"][i] for i in idx.tolist()], "provenance": prov}
    dest = Path(args["--output"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    print(json.dumps(prov["freshness"], indent=1))
    print(f"removed {n - n_keep:,} of {n:,} positions ({(n - n_keep) / max(n, 1):.2%}); "
          f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
