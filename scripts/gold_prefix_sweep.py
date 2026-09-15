"""Descriptive prefix sweep for the gold position sets (3B-causal Wave 1b
pre-registration, 2026-09-14, S4.2).

For each prefix length k, plays gold<k> self-play games (k random placements,
then the champion's legal argmax for both sides) and reports what the data look
like: distinct games, recorded positions per game, the pieces-on-board
distribution, threat prevalence, and the design-stage Wave-1b pair counts
(within-set dedup and the freshness filter applied first, exactly as for the
gold sets). DESCRIPTIVE ONLY: the pre-registration fixes k at 3 and 5 and this
sweep cannot change it. Self-play and unpatched forward passes only (the
switch-off filter needs the base decision); no interchange is computed.

The pair and power code is the run's own (interchange_3b.design_pairs,
lib.sae.interchange.feasibility), so the counts are the ones a dry run on a set
generated with that k would print.

Usage:
    gold_prefix_sweep.py --config=<yaml> [options]
    gold_prefix_sweep.py (-h | --help)

Options:
    -h --help           Show this help message.
    --config=<yaml>     configs/3B-causal/champ<Tag>.yaml (champion, concepts)
    --ks=<list>         Prefix lengths [default: 1,2,3,4,5,6,8]
    --games=<n>         Games per k [default: 2000]
    --seed-base=<n>     Game seed is seed-base + k [default: 4000]
    --pilot-run=<json>  Pilot whose pair boards are excluded
                        [default: saes/quarto/analysis/3B-causal_champYb_wave1.json]
    --device=<d>        [default: cpu]
    --cap=<n>           Pairs per (concept, kind), as the run [default: 1000]
    --seed=<n>          Pair-sampling seed, as the run [default: 0]
    --output=<json>     [default: auto] (auto = analysis/3B-causal_<champ>_gold-prefix-sweep.json)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.sae import interchange as ix  # noqa: E402
from scripts.freshness_filter import fresh_mask, pilot_board_keys  # noqa: E402
from scripts.games import get_game_module  # noqa: E402
from scripts.games import quarto_counterfactuals as qc  # noqa: E402
from scripts.interchange_3b import (  # noqa: E402
    Champion, design_pairs, load_config, load_schemas, log, wave_concepts)

GLOSSARY = {
    "games": {"range": "count", "meaning": "games played for this k"},
    "games_recorded": {"range": "[0, games]", "meaning": "games that reached their (k+1)-th placement (a random prefix can end in a win when k >= 4)"},
    "distinct_games": {"range": "[1, games_recorded]", "ideal": "high", "meaning": "distinct (board, piece) at the (k+1)-th placement; the continuation is deterministic, so this is the number of different games"},
    "positions_per_game": {"range": "[0, 16-k]", "meaning": "recorded placement decisions per game, before dedup"},
    "unique": {"range": "count", "meaning": "positions after within-set dedup on exact (board, piece)"},
    "removed_fresh": {"range": "[0, unique]", "meaning": "unique positions dropped because their board, up to symmetry, is a pilot pair's board"},
    "any_threat": {"range": "[0, 1]", "meaning": "fraction of positions with a line/square of 3 sharing a value and one empty cell"},
    "win_available": {"range": "[0, 1]", "meaning": "fraction where the piece in hand can win now (tiger_win_now_exists)"},
    "powered_in_every_arm": {"range": "[0, concepts]", "ideal": ">= 50% of concepts", "meaning": "concepts whose switch-on, specificity and switch-off pair counts all reach n_min (100/100/50)"},
}


def family(spec: qc.ConceptSpec) -> str:
    return f"{spec.basis} pinned" if spec.kind == "pinned" else f"tiger {spec.kind}"


def dedup(boards: torch.Tensor, pieces: torch.Tensor) -> torch.Tensor:
    """Row indices of the first occurrence of each exact (board, piece)."""
    key = torch.cat([boards.reshape(len(boards), -1), pieces], dim=1).to(torch.uint8).numpy()
    _, first = np.unique(key, axis=0, return_index=True)
    return torch.from_numpy(np.sort(first))


def describe(k, boards, pieces, meta, n_games, ch, chosen, pilot_keys, cap, seed, dev) -> dict:
    rec_games = {m["game_idx"] for m in meta}
    firsts = {(boards[i].numpy().tobytes(), pieces[i].numpy().tobytes())
              for i, m in enumerate(meta) if m["turn"] == k}
    u = dedup(boards, pieces)
    bu, pu = boards[u], pieces[u]
    keep = fresh_mask(bu, pilot_keys)
    bf, pf = bu[keep], pu[keep]
    T = qc.board_tables(bf, pf)
    natural = T.offered_poles()
    specs = [s for s, _ in chosen]
    prev = {s.bsp_id: float(qc.concept_value(T, s, natural).float().mean()) for s in specs}

    Z = ch.hook_values(bf, pf)
    SW = qc.all_source_wins(T)
    raw = {s.bsp_id: qc.build_pairs(T, s, SW, seed=seed) for s in specs}
    _, power = design_pairs(ch, specs, raw, bf, pf, Z, cap, seed, dev)
    under = ix.underpowered_arms(power)
    fams = defaultdict(lambda: {"concepts": 0, "powered_in_every_arm": 0})
    for s in specs:
        f = fams[family(s)]
        f["concepts"] += 1
        f["powered_in_every_arm"] += not under[s.bsp_id]

    n_on = bf.sum((1, 2, 3)).long().tolist()
    return {
        "k": k, "games": n_games, "games_recorded": len(rec_games),
        "distinct_games": len(firsts),
        "positions_raw": len(meta), "positions_per_game": len(meta) / n_games,
        "unique": int(len(u)), "removed_fresh": int((~keep).sum()), "fresh": int(keep.sum()),
        "pieces_on_board": dict(sorted(Counter(n_on).items())),
        "any_threat": float(T.threat.any(-1).any(-1).float().mean()),
        "win_available": float(qc.win_cells(T, natural).any(-1).float().mean()),
        "concept_prevalence": {"median": float(np.median(list(prev.values()))),
                               "n_zero": sum(v == 0 for v in prev.values())},
        "feasibility": ix.feasibility(power),
        "powered_by_family": dict(sorted(fams.items())),
        "median_pairs_per_arm": {kind: float(np.median([pw[kind]["n"] for pw in power.values()]))
                                 for kind in qc.PAIR_KINDS},
        "power": power,
    }


def main() -> int:
    args = docopt(__doc__)
    cfg = load_config(args["--config"])
    champ = yaml.safe_load((ROOT / cfg["champion"]).read_text(encoding="utf-8"))
    dev = args["--device"]
    ks = [int(x) for x in args["--ks"].split(",")]
    n_games, seed_base = int(args["--games"]), int(args["--seed-base"])
    cap, seed = int(args["--cap"]), int(args["--seed"])

    ch = Champion(cfg, False, dev)
    gm = get_game_module(champ["game"])
    chosen = wave_concepts(cfg, load_schemas(cfg))
    pilot_keys, pilot_info = pilot_board_keys(args["--pilot-run"])
    log(f"{ch.name}: {len(chosen)} concepts; pilot boards {pilot_info['pilot_board_orbits']:,} orbits")

    rows = []
    for k in ks:
        t0 = time.time()
        b, p, meta = gm.generate_positions(n_games, seed=seed_base + k, opponents=f"gold{k}",
                                           model_path=str(ROOT / champ["path"]), device=dev)
        boards = torch.from_numpy(np.stack(b)).float()
        pieces = torch.from_numpy(np.stack(p)).float()
        r = describe(k, boards, pieces, meta, n_games, ch, chosen, pilot_keys, cap, seed, dev)
        r["seed"] = seed_base + k
        rows.append(r)
        f = r["feasibility"]
        log(f"k={k}: {r['distinct_games']:,} distinct games, {r['positions_per_game']:.2f} pos/game, "
            f"{r['unique']:,} unique, {r['removed_fresh']:,} not fresh, threat {r['any_threat']:.3f}, "
            f"powered in every arm {f['powered_in_every_arm']}/{f['concepts']} "
            f"({f['fraction']:.0%})  [{time.time() - t0:.0f}s]")

    out = {
        "status": ("DESCRIPTIVE ONLY (Wave 1b pre-registration S4.2): k is fixed at 3 and 5 "
                   "by the pre-registration; this sweep cannot change it. Self-play and "
                   "unpatched forward passes only; no interchange computed."),
        "champion": ch.name, "config": args["--config"], "games_per_k": n_games,
        "seed_rule": f"{seed_base} + k", "device": dev, "cap": cap, "pair_seed": seed,
        "pilot": pilot_info, "glossary": {**GLOSSARY, **ix.GLOSSARY}, "rows": rows,
    }
    dest = (ROOT / "saes/quarto/analysis" / f"3B-causal_{ch.name}_gold-prefix-sweep.json"
            if args["--output"] == "auto" else ROOT / args["--output"])
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
