"""Build gold position sets from their configs' recipes (3B-causal Wave 1b
pre-registration 2026-09-14, S4; Wave 1c pre-registration 2026-09-15, S4).

Per config: generate (gold<k> self-play, scripts/generate_positions.py) ->
dedup within the set -> freshness filter against earlier runs -> BSP labels
on every basis the run uses -> orbit IDs. Generation runs in parallel across
configs, the label and orbit jobs in parallel across (config, basis). A step
whose output exists is skipped, so a rerun resumes -- but an existing raw or
final positions file is reused only if its provenance matches the recipe
(opponents, seed, games): two sets drawn with the same protocol share a raw
file NAME, and reusing the other draw's data would be silent. Every step is the
repo's own CLI, so each file carries the same provenance as any other set.

Usage:
    build_gold_sets.py --config=<yaml>... [--dry-run]
    build_gold_sets.py (-h | --help)

Options:
    -h --help        Show this help message.
    --config=<yaml>  A gold config, e.g. configs/3B-causal/champYb-gold3.yaml
                     (repeatable).
    --dry-run        Print the commands in order; run nothing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGES = ("generate", "dedup", "fresh", "labels")   # "labels" also holds the orbit-ID job


def steps(cfg: dict) -> list[tuple[str, list[str], str]]:
    """(stage, argv without the interpreter, output path) for one gold config."""
    from scripts.generate_positions import extract_model_name

    r = cfg["positions_recipe"]
    champ = yaml.safe_load((ROOT / cfg["champion"]).read_text(encoding="utf-8"))
    raw = f"{r['raw_dir']}/positions-{r['opponents']}-{extract_model_name(champ['path'])}_raw.pt"
    dedup = f"{r['raw_dir']}/{Path(cfg['positions']).stem.replace('_unique', '_dedup')}.pt"
    pos = cfg["positions"]
    out = [
        ("generate", ["scripts/generate_positions.py", "--game", champ["game"],
                      "--opponents", r["opponents"], "--model", champ["path"],
                      "--num-games", str(r["games"]), "--seed", str(r["seed"]),
                      "--device", r["device"], "--output-dir", r["raw_dir"]], raw),
        ("dedup", ["scripts/deduplicate_positions.py", raw, "--output", dedup], dedup),
        ("fresh", ["scripts/freshness_filter.py", dedup, f"--output={pos}",
                   *(f"--pilot-run={run}" for run in pilot_runs(cfg))], pos),
    ]
    for basis in cfg["wave1"]["bases"]:
        n = sorted(ROOT.glob(f"data/quarto/bsp_schema-{basis}_*.json"))[0].stem.rsplit("_", 1)[-1]
        lab = f"data/quarto/bsp_labels-{basis}{cfg['labels_suffix']}_{n}.pt"
        out.append(("labels", ["scripts/compute_bsp_labels.py", pos, "--game", "quarto",
                               "--name", basis, "--output", lab, "--schema-out", "none"], lab))
    out.append(("labels", ["scripts/compute_orbit_ids.py", pos, "--output", cfg["orbit_ids"]],
                cfg["orbit_ids"]))
    return out


def pilot_runs(cfg: dict) -> list[str]:
    """Earlier runs whose pair boards the set must exclude: ``pilot_runs`` (a
    list, Wave 1c) or the single ``pilot_run`` of a Wave-1b config."""
    runs = cfg.get("pilot_runs") or ([cfg["pilot_run"]] if cfg.get("pilot_run") else [])
    if not runs:
        raise SystemExit("a gold config needs pilot_runs (or pilot_run) for the freshness filter")
    return list(runs)


def recipe_provenance(path: Path) -> dict | None:
    """(opponents, seed, num_games) recorded in a raw file, or in the raw source
    of a deduplicated / filtered one."""
    import torch

    prov = torch.load(path, map_location="cpu", weights_only=False).get("provenance") or {}
    src = (prov.get("source_provenances") or [prov])[0]
    return {k: src.get(k) for k in ("opponents", "seed", "num_games")}


def check_reuse(stage: str, out: str, cfg: dict) -> None:
    """Refuse to reuse an existing raw / positions file drawn with another recipe."""
    if stage not in ("generate", "fresh"):
        return
    r = cfg["positions_recipe"]
    want = {"opponents": r["opponents"], "seed": r["seed"], "num_games": r["games"]}
    got = recipe_provenance(ROOT / out)
    if got != want:
        raise SystemExit(f"[{stage}] {out} exists but was drawn with {got}, not the recipe's "
                         f"{want}. Give this set its own raw_dir / positions name.")


def main() -> int:
    from scripts.interchange_3b import load_config

    args = docopt(__doc__)
    cfgs = {c: load_config(c) for c in args["--config"]}
    plans = {c: steps(cfg) for c, cfg in cfgs.items()}
    (ROOT / "logs").mkdir(exist_ok=True)
    for stage in STAGES:
        jobs = [(c, argv, out) for c, plan in plans.items() for st, argv, out in plan if st == stage]
        procs = []
        for c, argv, out in jobs:
            if (ROOT / out).exists():
                check_reuse(stage, out, cfgs[c])
                checked = ", recipe matches" if stage in ("generate", "fresh") else ""
                print(f"[{stage}] exists{checked}, skipped: {out}", flush=True)
                continue
            print(f"[{stage}] python {' '.join(argv)}", flush=True)
            if args["--dry-run"]:
                continue
            log = ROOT / "logs" / f"gold_{Path(out).stem}.log"
            procs.append((out, log, subprocess.Popen([sys.executable, *argv], cwd=ROOT,
                                                     stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        t0 = time.time()
        failed = [(out, log) for out, log, p in procs if p.wait() != 0]
        if failed:
            raise SystemExit(f"[{stage}] failed: " + "; ".join(f"{o} (log {lg})" for o, lg in failed))
        if procs:
            print(f"[{stage}] {len(procs)} job(s) done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
