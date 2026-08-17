"""Is a random-model control SAE usable as a 3A learned-signal floor?

A control that trains without error can still contribute nothing. `R1-champ*random`
(JumpReLU t64, kaiming init) has 4032/4096 latents that never fire and 64 that
fire on >99.98% of rows, so ZERO latents fall inside the diagnostic's alive band
and `select_concept_features` returns an empty candidate list -- the control was
named in 13 of 17 reports and produced no number in any of them.

This applies the diagnostic's OWN alive definition, so "usable" here means
exactly what rule 3A.3 needs and nothing looser.

Usage:
    check_control_usable.py <checkpoint>... [options]
    check_control_usable.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game=<name>      Game name [default: quarto]
    --top-k=<n>        Candidate latents the diagnostic asks for [default: 64]
    --rows=<n>         Rows to sample when measuring firing rates [default: 40000]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.sae.dilution import DilutionConfig, RankingCache  # noqa: E402


def main() -> int:
    args = docopt(__doc__)
    cfg = DilutionConfig(top_k=int(args["--top-k"]), max_rows=int(args["--rows"]))
    cache_dir = ROOT / "saes" / args["--game"] / "cache"

    worst = 0
    print(f"{'control':<52}{'alive':>7}{'dead%':>8}{'fire>99.9%':>12}  verdict")
    for raw in args["<checkpoint>"]:
        run_id = Path(raw).stem
        h_path = cache_dir / f"{run_id}_h.pt"
        if not h_path.exists():
            print(f"{run_id[:51]:<52}{'-':>7}{'-':>8}{'-':>12}  NO _h CACHE "
                  f"(run sae_eval first)")
            worst = max(worst, 2)
            continue
        h = torch.load(h_path, map_location="cpu", weights_only=True)
        rc = RankingCache(h, cfg)
        alive = int(rc.alive.sum())
        never = int((rc.freq == 0).sum())
        always = int((rc.freq > cfg.max_freq).sum())
        ok = alive >= cfg.top_k
        # Below top_k the diagnostic silently measures a smaller support than it
        # does for every other run, which is not comparable -- so that is a fail
        # too, not just the zero case.
        verdict = ("USABLE" if ok else
                   "UNUSABLE: alive < top_k, the control cannot fill the "
                   "candidate list")
        print(f"{run_id[:51]:<52}{alive:>7}{100*never/h.shape[1]:>8.1f}"
              f"{always:>12}  {verdict}")
        worst = max(worst, 0 if ok else 1)
        del h, rc
    return worst


if __name__ == "__main__":
    sys.exit(main())
