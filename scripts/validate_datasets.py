"""Validate position datasets against their declared champion, by PROVENANCE.

Why this exists: champTa's `positions-amalgam_ta_unique.pt` was aggregated from
`positions-random_v_random_raw.pt` alone -- it contained no model-generated
positions at all -- and champS4's file is byte-identical to champAa's. Neither
was caught for months, because a row count of 88,524 looks merely small rather
than wrong. A count check cannot find this; a provenance check can.

Checks per champion amalgam:
  1. every expected opponent mode is present in `provenance.source_files`;
  2. the model-generated sources name THIS champion's tag, not another's;
  3. `provenance.game` matches the champion's game module;
  4. the board tensor is not byte-identical to another champion's.

Exit code 0 if every dataset is OK or explicitly QUARANTINED in
`data/<game>/_dataset_status.json`; 1 if an unflagged problem is found, so a
runner can gate on it.

Usage:
    validate_datasets.py [options]
    validate_datasets.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game <name>      Game data directory under data/ [default: quarto]
    --strict           Also fail on datasets that are flagged QUARANTINED
                       (use in a runner that must not touch them at all).
    --json             Emit JSON instead of a table.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_MODES = ("random_v_random", "model_v_random",
                  "random_v_model", "model_v_model")

# champion tag -> (amalgam filename, expected game module, expected model tag)
CHAMPIONS = {
    "Aa": ("positions-amalgam_unique.pt", "quarto", "Aa_replay"),
    "S4": ("positions-amalgam_s4_unique.pt", "quarto_s4", "Sa_archScan"),
    "Ta": ("positions-amalgam_ta_unique.pt", "quarto_s4", "Ta_minimaxSelect"),
    "Ve": ("positions-amalgam_ve_unique.pt", "quarto_s4", "Ve_oracleAblation"),
    "Yb": ("positions-amalgam_yb_unique.pt", "quarto_s4_hot", "Yb_hotChamp"),
}


def board_fingerprint(path: Path) -> str:
    """Cheap content hash of the board tensor, to spot two champions sharing
    one dataset (champS4's file is byte-identical to champAa's)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    b = d["boards"]
    if not isinstance(b, torch.Tensor):
        b = torch.stack(list(b))
    h = hashlib.sha256()
    h.update(str(tuple(b.shape)).encode())
    # Sample rather than hash gigabytes: head, tail and a stride are ample to
    # distinguish datasets while staying fast.
    h.update(b[:512].numpy().tobytes())
    h.update(b[-512:].numpy().tobytes())
    h.update(b[::997].numpy().tobytes())
    return h.hexdigest()[:16]


def check_champion(data_dir: Path, tag: str) -> dict:
    fname, exp_game, exp_model = CHAMPIONS[tag]
    path = data_dir / fname
    res = {"champion": tag, "file": fname, "problems": [], "n_positions": None}
    if not path.exists():
        res["problems"].append("file missing")
        return res

    d = torch.load(path, map_location="cpu", weights_only=False)
    prov = d.get("provenance", {}) or {}
    res["n_positions"] = prov.get("n_positions") or len(d.get("boards", []))
    res["game"] = prov.get("game")
    sources = [Path(s).name for s in prov.get("source_files", [])]
    res["source_files"] = sources

    # 1 + 2: every mode present, and model modes name THIS champion.
    for mode in EXPECTED_MODES:
        hits = [s for s in sources if mode in s]
        if not hits:
            res["problems"].append(f"missing opponent mode '{mode}'")
        elif mode != "random_v_random" and not any(exp_model in s for s in hits):
            res["problems"].append(
                f"'{mode}' source does not name {exp_model}: {hits}")

    # 3: game module.
    if res["game"] != exp_game:
        res["problems"].append(f"game='{res['game']}' but expected '{exp_game}'")

    res["fingerprint"] = board_fingerprint(path)
    return res


def main():
    args = docopt(__doc__)
    data_dir = Path("data") / args["--game"]
    status_path = data_dir / "_dataset_status.json"
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8")).get("datasets", {})

    results = [check_champion(data_dir, t) for t in CHAMPIONS]

    # 4: two champions sharing one dataset.
    by_fp: dict[str, list[str]] = {}
    for r in results:
        if r.get("fingerprint"):
            by_fp.setdefault(r["fingerprint"], []).append(r["champion"])
    for fp, champs in by_fp.items():
        if len(champs) > 1:
            for r in results:
                if r["champion"] in champs:
                    others = [c for c in champs if c != r["champion"]]
                    r["problems"].append(
                        f"board tensor identical to champion(s) {others}")

    failed = []
    for r in results:
        entry = status.get(r["file"], {})
        flag = entry.get("status", "OK")
        r["declared_status"] = flag
        # `accepted_problems` waives specific known findings by substring, so a
        # sound dataset (champAa) is not failed by a defect that belongs to the
        # champion which copied it (champS4).
        waivers = entry.get("accepted_problems", [])
        unwaived = [p for p in r["problems"]
                    if not any(w.lower() in p.lower() for w in waivers)]
        r["unwaived_problems"] = unwaived
        if unwaived and flag == "OK":
            failed.append(r["champion"])
        elif r["problems"] and args["--strict"]:
            failed.append(r["champion"])

    if args["--json"]:
        print(json.dumps({"results": results, "failed": failed}, indent=2))
    else:
        print(f"{'champ':<6}{'status':<13}{'N':>9}  {'game':<14}file / problems")
        print("-" * 92)
        for r in results:
            mark = "OK" if not r["problems"] else r["declared_status"]
            print(f"{r['champion']:<6}{mark:<13}{str(r['n_positions']):>9}  "
                  f"{str(r.get('game')):<14}{r['file']}")
            for p in r["problems"]:
                waived = p not in r["unwaived_problems"]
                print(f"{'':<29}  {'(waived)' if waived else '!!'} {p}")
        print("-" * 92)
        if failed:
            print(f"UNFLAGGED PROBLEMS in: {', '.join(failed)}")
            print("Either fix the dataset or record it in "
                  f"{status_path} with a status and reason.")
        else:
            print("No unflagged problems. (Datasets with problems are declared "
                  "in _dataset_status.json.)")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
