"""Expand `analysis/3A_panel.json` into the exact entry list 3A must run.

The runner only sequences; every decision that can silently break a run lives
here, where it is testable. Four things this settles that a PowerShell array
cannot:

1. **The anchored positive control is carried through.** `select_3a_panel.py`
   excludes anchored runs on purpose -- they are supervised, so they are not
   panel members. But a gate whose other arm never fires is not a test
   (CLAUDE.md), and the anchored runs ARE that arm: rule 3A.2 exists only
   because the anchored control came out `diluted` 22/23 times under 3A.1.
   Dropping them from the panel run would remove the calibration check while
   leaving the gate looking healthier.

2. **`hen` is run on hawk's own dictionaries.** The polarity hypothesis is a
   MATCHED-PAIR question -- "is the negative-pole half of tiger less captured
   than the positive-pole half, in the same dictionary?" -- so hen entries
   mirror the hawk selections rather than being selected independently (hen has
   no eval-registry history to rank on anyway).

3. **The random-model control mapping**, previously duplicated between the
   runner's pre-flight and its exec loop as two copies of one regex.

4. **Which runs still need an `_h` encode**, so the runner can encode BEFORE
   the usability gate. That ordering is not cosmetic: `check_sae_usable.py`
   exits 2 on a missing `_h` and the runner throws on a non-zero exit, so
   gating first makes a first run on a freshly-pruned cache impossible.

Usage:
    build_3a_panel_runlist.py [options]
    build_3a_panel_runlist.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game=<name>      Game name [default: quarto]
    --panel=<path>     Panel JSON [default: auto] (auto = analysis/3A_panel.json)
    --with-hen         Mirror every hawk entry onto the `hen` basis.
    --no-anchored      Drop the anchored positive controls (NOT recommended).
    --output=<path>    Write the runlist JSON [default: auto]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent

# The supervised positive control, one per champion. Not selected by
# `select_3a_panel.py` (it excludes anchored runs); named here so the
# calibration arm cannot go missing when the panel is regenerated.
ANCHORED_CONTROLS: tuple[tuple[str, str], ...] = (
    ("I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1", "tigerTa"),
    ("I04-champVe-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1", "tigerVe"),
    ("I03-champVe-lh030-s43-anchored-jumprelu-t64-exp8-s4.fc1", "gorillaVe"),
    ("I04-champYb-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1", "tigerYb"),
)

# Recipe-matched SAEs trained on an UNTRAINED network, keyed "<champ>|<hook>".
# R3 (not R1) on fc1: R1 had zero alive latents and silently contributed
# nothing to 13 of 17 reports. See runners/3A-prep.ps1.
RANDOM_CONTROLS: dict[str, str] = {
    "Ta|s4.fc1": "R3-champTarandom-s42-jumprelu-t64-exp8-s4.fc1",
    "Ve|s4.fc1": "R3-champVerandom-s42-jumprelu-t64-exp8-s4.fc1",
    "Yb|s4.fc1": "R3-champYbrandom-s42-jumprelu-t64-exp8-s4.fc1",
    "Ta|s4.conv2": "R2-champTarandom-s42-batchtopk-k32-exp8-s4.conv2",
    "Ve|s4.conv2": "R2-champVerandom-s42-batchtopk-k32-exp8-s4.conv2",
    "Yb|s4.conv2": "R2-champYbrandom-s42-batchtopk-k32-exp8-s4.conv2",
}

_RUN_RE = re.compile(r"champ(\w\w)[-\w.]*?-(s4\.\w+|fc1|conv2)$")


def resolve_control(run_id: str) -> str:
    m = _RUN_RE.search(run_id)
    if not m:
        return "none"
    return RANDOM_CONTROLS.get(f"{m.group(1)}|{m.group(2)}", "none")


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    analysis = ROOT / "saes" / game / "analysis"
    cache = ROOT / "saes" / game / "cache"
    data = ROOT / "data" / game

    panel_path = (analysis / "3A_panel.json" if args["--panel"] == "auto"
                  else Path(args["--panel"]))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))

    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(run_id: str, bsps: str, role: str) -> None:
        if (run_id, bsps) in seen:
            return
        seen.add((run_id, bsps))
        entries.append({"run_id": run_id, "bsps": bsps, "role": role,
                        "random_run_id": resolve_control(run_id)})

    for cell, members in panel["panel"].items():
        champ, _hook, basis = cell.split("/")
        for m in members:
            add(m["run_id"], f"{basis}{champ}", "panel")
            if args["--with-hen"] and basis == "hawk":
                add(m["run_id"], f"hen{champ}", "panel-hen")

    if not args["--no-anchored"]:
        for run_id, bsps in ANCHORED_CONTROLS:
            add(run_id, bsps, "anchored-positive-control")

    # Group consecutive entries by checkpoint. Each entry is a separate process
    # that loads that run's `_h` (1.4-4.8 GB) plus its control's, and a
    # checkpoint appears once per basis -- up to four times with `hen`. Running
    # its bases back-to-back means the OS page cache still holds the tensor, so
    # only the first load hits disk. Pure I/O ordering: the diagnostic writes
    # one file per (run_id, bsps) and shares no state between entries, so the
    # order cannot change any number.
    entries.sort(key=lambda e: (e["run_id"], e["bsps"]))

    # --- what the runner has to do before it can gate ----------------------
    # One encode per CHECKPOINT (not per basis): `sae_eval --bsps` takes a comma
    # list and `_h` does not depend on the BSP set, so encoding per basis
    # rewrites a 1.4-4.8 GB tensor for nothing.
    need_encode: dict[str, set[str]] = {}
    missing_labels: list[str] = []
    for e in entries:
        if not (cache / f"{e['run_id']}_h.pt").exists():
            need_encode.setdefault(e["run_id"], set()).add(e["bsps"])
        if not list(data.glob(f"bsp_labels-{e['bsps']}_[0-9]*.pt")):
            missing_labels.append(e["bsps"])

    controls = sorted({e["random_run_id"] for e in entries
                       if e["random_run_id"] != "none"})
    missing_controls = [c for c in controls
                        if not (cache / f"{c}_h.pt").exists()]

    runlist = {
        "generated_from": str(panel_path.relative_to(ROOT)).replace("\\", "/"),
        "n_entries": len(entries),
        "entries": entries,
        "controls": controls,
        "needs_encode": {k: sorted(v) for k, v in sorted(need_encode.items())},
        "missing_control_h": missing_controls,
        "missing_labels": sorted(set(missing_labels)),
    }

    out = (analysis / "3A_panel_runlist.json" if args["--output"] == "auto"
           else Path(args["--output"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runlist, indent=2), encoding="utf-8")

    by_role: dict[str, int] = {}
    for e in entries:
        by_role[e["role"]] = by_role.get(e["role"], 0) + 1
    print(f"{len(entries)} entries  ({', '.join(f'{k}={v}' for k, v in sorted(by_role.items()))})")
    print(f"  distinct checkpoints : {len({e['run_id'] for e in entries})}")
    print(f"  need an _h encode    : {len(need_encode)}")
    print(f"  random-model controls: {len(controls)}"
          + (f"  MISSING _h: {missing_controls}" if missing_controls else ""))
    if runlist["missing_labels"]:
        print(f"  !! missing BSP labels: {runlist['missing_labels']}")
    print(f"Wrote {out}")
    return 1 if runlist["missing_labels"] else 0


if __name__ == "__main__":
    sys.exit(main())
