"""Re-evaluate SAEs against the unified cross-champion position pool.

``scripts/unify_positions.py`` builds the shared pool
(``positions-amalgam_all_unique.pt``), the per-champion activations on it
(``<hook>_amalgam_all_<tag>[_random]_activations.pt``) and the unified labels
(``bsp_labels-<basis><N>k_*.pt``). This driver then re-runs ``sae_eval.py`` for
each SAE against the matching unified activations + labels with ``--force``, so
every SAE's eval caches (``saes/<game>/cache/<run_id>_h.pt`` and
``<run_id>_matching-<basis><N>k.pt``) land on the SAME positions. After that:

    python scripts/export_viz_data.py --game=<g> --bsps=<basis><N>k \\
        --positions=data/<g>/positions-amalgam_all_unique.pt

no longer hits the h/labels row-count mismatch, and coverage is comparable
across champions (no per-distribution artifact).

Per checkpoint, derived from the run id alone:
    tag    -> champS4/Ta/Ve else Aa, lower-cased   (aa/s4/ta/ve)
    hook   -> run_id.rsplit('-', 1)[-1]            (fc1 / conv2 / s4.fc1 / s4.conv2)
    random -> 'random-control' or 'c2random' in run_id
    data   -> data/<game>/<hook>_amalgam_all_<tag>[_random]_activations.pt
Each SAE is evaluated against all requested bases in one call (h encoded once
and reused across BSP sets). SAEs whose unified activation file is absent (e.g.
an fc2 hook unify did not collect) are skipped with a warning.

Usage:
    eval_unified.py [options]
    eval_unified.py (-h | --help)

Options:
    --game=<name>     Game id [default: quarto].
    --suffix=<s>      Unified pool suffix from unify_positions, e.g. 156k.
                      'auto' infers it from the bsp_labels-<basis><N>k file
                      [default: auto].
    --bases=<list>    Comma-separated BSP bases; one sae_eval call per SAE
                      covers all of them [default: gorilla,hawk,tiger].
    --saes=<list>     Comma-separated run-ids, or 'all' for every
                      saes/<game>/*.pt checkpoint [default: all].
    --device=<dev>    Device passed to sae_eval (cuda|cpu|auto) [default: cuda].
    --dry-run         Print the sae_eval commands without running them.
    --keep-going      Continue if an individual eval fails (default: stop).
    -h --help         Show this help.

Examples:
    python scripts/eval_unified.py --suffix=156k --dry-run
    python scripts/eval_unified.py --suffix=156k
    python scripts/eval_unified.py --suffix=156k --bases=gorilla \\
        --saes=E05-champTa-s42-batchtopk-k32-exp8-s4.conv2
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from docopt import docopt

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
KNOWN_HOOKS = {"fc1", "conv2", "s4.fc1", "s4.conv2"}

# Run-id -> champion / random-control inference. Mirrors the canonical logic in
# scripts/export_viz_data.py (_infer_champion / _infer_kind) and
# src/lib/champions.ts -- keep in sync if champion naming changes.
DEFAULT_CHAMPION = "Aa"
_CHAMPION_PATTERNS = [
    ("S4", re.compile(r"champS4", re.IGNORECASE)),
    ("Ta", re.compile(r"champTa", re.IGNORECASE)),
    ("Ve", re.compile(r"champVe", re.IGNORECASE)),
]
_RANDOM_CONTROL_RE = re.compile(r"random-control|c2random", re.IGNORECASE)


def _data_dir(game: str) -> Path:
    return PROJECT_DIR / "data" / game


def _saes_dir(game: str) -> Path:
    return PROJECT_DIR / "saes" / game


def _champion_tag(run_id: str) -> str:
    for cid, pat in _CHAMPION_PATTERNS:
        if pat.search(run_id):
            return cid.lower()
    return DEFAULT_CHAMPION.lower()


def _detect_suffix(game: str, basis: str) -> str | None:
    """Infer the unified suffix (e.g. '156k') from bsp_labels-<basis><N>k_*.pt."""
    pat = re.compile(rf"^bsp_labels-{re.escape(basis)}(\d+k)_\d+\.pt$")
    found = set()
    for p in _data_dir(game).glob(f"bsp_labels-{basis}*k_*.pt"):
        m = pat.match(p.name)
        if m:
            found.add(m.group(1))
    if len(found) == 1:
        return found.pop()
    return None


def _activation_path(game: str, run_id: str):
    """(path, tag, hook, is_random) for a run-id's unified activation file."""
    tag = _champion_tag(run_id)
    hook = run_id.rsplit("-", 1)[-1]
    is_random = bool(_RANDOM_CONTROL_RE.search(run_id))
    rnd = "_random" if is_random else ""
    name = f"{hook}_amalgam_all_{tag}{rnd}_activations.pt"
    return _data_dir(game) / name, tag, hook, is_random


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    bases = [b.strip() for b in args["--bases"].split(",") if b.strip()]
    if not bases:
        print("ERROR: --bases is empty.", file=sys.stderr)
        return 1

    suffix = args["--suffix"]
    if suffix == "auto":
        suffix = _detect_suffix(game, bases[0])
        if suffix is None:
            print(
                f"ERROR: could not auto-detect a unique "
                f"bsp_labels-{bases[0]}<N>k_*.pt under {_data_dir(game)}; "
                f"pass --suffix=<N>k explicitly.",
                file=sys.stderr,
            )
            return 1
    animals = ",".join(f"{b}{suffix}" for b in bases)

    if args["--saes"] == "all":
        run_ids = sorted(p.stem for p in _saes_dir(game).glob("*.pt"))
    else:
        run_ids = [s.strip() for s in args["--saes"].split(",") if s.strip()]
    if not run_ids:
        print(f"ERROR: no SAE checkpoints under {_saes_dir(game)}.", file=sys.stderr)
        return 1

    eval_script = PROJECT_DIR / "sae_eval.py"
    dry = args["--dry-run"]
    keep_going = args["--keep-going"]

    print(f"Unified eval | game={game} | bsps={animals} | {len(run_ids)} SAE(s)")
    n_ok = n_skip = n_fail = 0

    for run_id in run_ids:
        ckpt = _saes_dir(game) / f"{run_id}.pt"
        if not ckpt.exists():
            print(f"  SKIP {run_id}: checkpoint missing")
            n_skip += 1
            continue

        data_path, tag, hook, is_random = _activation_path(game, run_id)
        if hook not in KNOWN_HOOKS:
            print(f"  SKIP {run_id}: unrecognized hook '{hook}'")
            n_skip += 1
            continue
        if not data_path.exists():
            print(f"  SKIP {run_id}: unified activations missing ({data_path.name})")
            n_skip += 1
            continue

        cmd = [
            PYTHON, str(eval_script), "evaluate", str(ckpt),
            f"--bsps={animals}",
            f"--data={data_path}",
            f"--device={args['--device']}",
            "--force",
        ]
        label = f"[{tag}{'/rnd' if is_random else ''} {hook}]"
        if dry:
            print(f"  DRY  {run_id} {label}")
            print("       $ " + " ".join(cmd))
            n_ok += 1
            continue

        print(f"  EVAL {run_id} {label}")
        rc = subprocess.run(cmd, cwd=str(PROJECT_DIR)).returncode
        if rc != 0:
            n_fail += 1
            print(f"  FAIL {run_id}: sae_eval exit {rc}", file=sys.stderr)
            if not keep_going:
                print("Stopping (pass --keep-going to continue).", file=sys.stderr)
                break
        else:
            n_ok += 1

    print(f"Done | {n_ok} ok, {n_skip} skipped, {n_fail} failed.")
    return 2 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
