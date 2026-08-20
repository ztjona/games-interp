"""Is an SAE usable by the 3A diagnostic -- as a panel member OR as a control?

An SAE that trains without error can still be unusable, and this has bitten
BOTH roles:

  * as a CONTROL -- `R1-champ*random` (JumpReLU t64, kaiming init) had 4032/4096
    latents that never fire and 64 firing on >99.98% of rows, so ZERO fell in the
    alive band. It was named in 13 of 17 reports and produced no number in any.
  * as a PANEL MEMBER -- `E05-champYb` had 41 alive latents, FEWER than
    `top_k = 64`, so its candidate list could not be filled and its
    `asymptote_r2` was measured at a smaller support than every other cell. Its
    verdict described the collapsed dictionary, not the champion.

The bar is the same in both roles and is not arbitrary: the diagnostic ranks
`top_k` candidates, so a dictionary with fewer alive latents than `top_k`
cannot answer the question being asked of it. FVU is printed alongside for
judgement (E05 sat at 0.110 against 0.043/0.051 for the same recipe on its
sibling champions) but is NOT gated on, because a defensible absolute threshold
would differ per hook and architecture.

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

import json
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
    reg_path = ROOT / "saes" / args["--game"] / "eval_registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}

    def fvu_of(run_id):
        for k, v in reg.items():
            if k.split(":")[0] == run_id:
                return (v.get("metrics") or {}).get("fvu")
        return None

    print(f"{'sae':<52}{'alive':>7}{'dead%':>8}{'fire>99.9%':>12}{'FVU':>8}  verdict")
    for raw in args["<checkpoint>"]:
        run_id = Path(raw).stem
        h_path = cache_dir / f"{run_id}_h.pt"
        if not h_path.exists():
            print(f"{run_id[:51]:<52}{'-':>7}{'-':>8}{'-':>12}{'-':>8}  NO _h CACHE "
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
        fvu = fvu_of(run_id)
        print(f"{run_id[:51]:<52}{alive:>7}{100*never/h.shape[1]:>8.1f}"
              f"{always:>12}{'     -  ' if fvu is None else f'{fvu:>8.4f}'}  {verdict}")
        worst = max(worst, 0 if ok else 1)
        del h, rc
    return worst


if __name__ == "__main__":
    sys.exit(main())
