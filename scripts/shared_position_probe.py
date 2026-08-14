"""Compare champions on the positions they ALL share -- a matched-position control.

Every cross-champion number in this project is computed on each champion's own
self-play distribution, and those distributions are largely disjoint: measured
2026-08-14, champTa/champVe/champYb hold ~290k positions each but share only
88,970 (Jaccard 0.185, ~31% of the smaller set), and the shared core is
essentially the random-play portion -- the self-play regions barely overlap.

``coverage_mcc_at_pref`` fixes the MARGINAL prevalence confound. It cannot fix
the DISTRIBUTIONAL one: two champions can agree on a concept's base rate while
their positions differ in how separable that concept is. The linear probe shows
this directly -- fc1 ``square_threat`` availability is 0.774 / 0.838 / 0.938 for
Ta / Ve / Yb before any SAE is involved, which confounds "champYb's model
represents it better" with "champYb's positions make it easier".

This script settles that cheaply. It evaluates each champion's SAE on the
INTERSECTION of the position sets -- literally the same boards, same offered
pieces, same labels -- using only caches already on disk. No GPU, no new
activations, no unified pool.

It exists to DECIDE whether the expensive unified pool is needed:

  * champion RANKINGS on the shared subset match those on own-distribution
        -> the distributional confound is immaterial; skip the pool.
  * rankings DIVERGE
        -> the confound is real; the pool is justified, and the divergent cells
           are exactly the claims to re-examine.

Its honest limitation: the shared core is mostly random-play, so it is
off-distribution for every champion. That makes it a good decision procedure and
a poor final answer -- the unified pool covers each champion's own region, this
covers none of them. Read it as "does the distribution change the ordering?",
not as "here are the true champion scores".

Usage:
    shared_position_probe.py [options]
    shared_position_probe.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game <name>      Game name [default: quarto]
    --champs <list>    Comma-separated champion tags [default: Ta,Ve,Yb]
    --hook <h>         Activation hook [default: s4.fc1]
    --bases <list>     Comma-separated BSP bases [default: gorilla,hawk,tiger]
    --prefixes <list>  Run-id prefixes to prefer, in order [default: F04,E05,I04]
    --min-shared <n>   Refuse to run below this many shared positions [default: 5000]
    --output <path>    Output JSON [default: auto]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Minimum spread in mean_mcc_at_pref before a champion ORDERING is worth
# interpreting. Below this the champions are not separated and their order is
# noise: on 2026-08-14 two tiger cells 'flipped' at spreads of 0.004 and 0.025
# (Ta and Ve tied to four decimals) while every cell with a real gap
# (gorilla/hawk threats, spreads 0.1-0.5) kept its order exactly.
MIN_SPREAD = 0.05

from lib.sae.eval import (  # noqa: E402
    aggregate_per_category_by_family,
    compute_coverage,
    compute_per_category_coverage,
    match_features_to_bsps,
    resolve_schema_path,
)


def position_keys(path: Path) -> np.ndarray:
    """One opaque byte-key per position: board tensor + offered piece.

    Keys are what define "the same position", so both the board AND the offered
    piece must be included -- every agent-relative concept (tiger) is a function
    of the pair, and two rows with equal boards but different offered pieces are
    different positions for those concepts.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    boards, pieces = d["boards"], d["pieces"]
    if not isinstance(boards, torch.Tensor):
        boards = torch.stack(list(boards))
    if not isinstance(pieces, torch.Tensor):
        pieces = torch.stack(list(pieces))
    flat = torch.cat(
        [
            boards.reshape(boards.shape[0], -1).to(torch.int8),
            pieces.reshape(pieces.shape[0], -1).to(torch.int8),
        ],
        dim=1,
    ).numpy()
    return np.ascontiguousarray(flat).view(
        np.dtype((np.void, flat.shape[1]))).ravel()


def pick_run(cache_dir: Path, champ: str, hook: str,
             prefixes: list[str]) -> str | None:
    """Panel run for a champion+hook, by the runners' preference order."""
    cached = sorted(p.name[: -len("_h.pt")]
                    for p in cache_dir.glob(f"*champ{champ}*{hook}_h.pt"))
    for prefix in prefixes:
        hits = [c for c in cached if c.startswith(prefix + "-")]
        if hits:
            return hits[0]
    return None


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    champs = [c.strip() for c in args["--champs"].split(",") if c.strip()]
    hook = args["--hook"]
    bases = [b.strip() for b in args["--bases"].split(",") if b.strip()]
    prefixes = [p.strip() for p in args["--prefixes"].split(",") if p.strip()]
    data_dir, cache_dir = Path("data") / game, Path("saes") / game / "cache"

    # --- 1. intersect the position sets --------------------------------------
    print("Hashing positions...", file=sys.stderr)
    keys: dict[str, np.ndarray] = {}
    for champ in champs:
        path = data_dir / f"positions-amalgam_{champ.lower()}_unique.pt"
        if not path.exists():
            sys.exit(f"error: no positions file for champ{champ}: {path}")
        keys[champ] = position_keys(path)
        print(f"  champ{champ}: {len(keys[champ])} positions", file=sys.stderr)

    shared = keys[champs[0]]
    for champ in champs[1:]:
        shared = np.intersect1d(shared, keys[champ], assume_unique=False)
    n_shared = len(shared)
    print(f"  shared by all {len(champs)}: {n_shared}", file=sys.stderr)
    if n_shared < int(args["--min-shared"]):
        sys.exit(f"error: only {n_shared} shared positions; too few to compare.")

    # Row indices of the shared positions, in ONE canonical order (sorted key
    # order) for every champion -- so row i means the same board everywhere and
    # the label-agreement check below is a real check, not a coincidence.
    idx: dict[str, np.ndarray] = {}
    for champ in champs:
        order = np.argsort(keys[champ], kind="stable")
        pos = np.searchsorted(keys[champ][order], shared)
        idx[champ] = order[pos]
        assert np.array_equal(keys[champ][idx[champ]], shared), champ

    # --- 2. run the comparison, per basis ------------------------------------
    out: dict = {"n_shared": int(n_shared), "hook": hook, "champs": champs,
                 "per_basis": {}, "label_agreement": {}}
    runs = {c: pick_run(cache_dir, c, hook, prefixes) for c in champs}
    for champ, rid in runs.items():
        if rid is None:
            sys.exit(f"error: no _h cache for champ{champ}/{hook} "
                     f"(prefixes {prefixes})")
        print(f"  champ{champ}: {rid}", file=sys.stderr)
    out["runs"] = runs

    for basis in bases:
        print(f"\n=== {basis} ===", file=sys.stderr)
        schema_path = resolve_schema_path(data_dir, f"{basis}{champs[0]}")
        if schema_path is None:
            print(f"  [SKIP] no schema for {basis}", file=sys.stderr)
            continue
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

        # Labels are a pure function of the position, so on shared rows they
        # MUST be identical across champions. If they are not, the position
        # keys or the row indexing are wrong and nothing below means anything.
        ref = None
        agree = True
        subsets: dict[str, torch.Tensor] = {}
        for champ in champs:
            lbl = sorted(data_dir.glob(f"bsp_labels-{basis}{champ}_[0-9]*.pt"))
            if not lbl:
                print(f"  [SKIP] no labels for {basis}{champ}", file=sys.stderr)
                subsets = {}
                break
            y = torch.load(lbl[0], map_location="cpu", weights_only=True)
            sub = y[torch.as_tensor(idx[champ], dtype=torch.long)].float()
            subsets[champ] = sub
            if ref is None:
                ref = sub
            elif not torch.equal(ref, sub):
                agree = False
        if not subsets:
            continue
        out["label_agreement"][basis] = bool(agree)
        if not agree:
            print("  ERROR: labels disagree across champions on shared "
                  "positions -- indexing is wrong; skipping this basis.",
                  file=sys.stderr)
            continue
        print(f"  label agreement across champions: OK ({ref.shape})",
              file=sys.stderr)

        per_basis: dict[str, dict] = {}
        for champ in champs:
            h = torch.load(cache_dir / f"{runs[champ]}_h.pt",
                           map_location="cpu", weights_only=True)
            h_sub = h[torch.as_tensor(idx[champ], dtype=torch.long)]
            del h
            m = match_features_to_bsps(h_sub, subsets[champ])
            del h_sub
            cov = compute_coverage(m)
            pc = compute_per_category_coverage(m, schema)
            per_basis[champ] = {
                "coverage_mcc": cov.get("coverage_mcc"),
                "coverage_mcc_at_pref": cov.get("coverage_mcc_at_pref"),
                "coverage_youden_j": cov.get("coverage_youden_j"),
                "mean_base_rate": cov.get("mean_base_rate"),
                "per_family": aggregate_per_category_by_family(pc, schema),
            }
            print(f"    champ{champ}: mcc={cov.get('coverage_mcc'):.4f} "
                  f"mcc@pref={cov.get('coverage_mcc_at_pref'):.4f}",
                  file=sys.stderr)
        out["per_basis"][basis] = per_basis

    # --- 3. the decision: do rankings change? --------------------------------
    registry = json.loads(
        (Path("saes") / game / "eval_registry.json").read_text(encoding="utf-8"))

    def rank(scores: dict[str, float | None]) -> list[str]:
        ok = {c: s for c, s in scores.items() if s is not None}
        return [c for c, _ in sorted(ok.items(), key=lambda kv: -kv[1])]

    flips: list[dict] = []
    ties: list[dict] = []
    print("\n" + "=" * 78)
    print("RANKING: own distribution vs shared positions")
    print("A ranking that survives the switch is not an artifact of which")
    print("positions each champion happens to play.")
    print("=" * 78)
    print(f"  {'basis':9s} {'family':17s} {'own-distribution':26s} "
          f"{'shared positions':26s} same?")
    for basis, per_basis in out["per_basis"].items():
        fams = sorted({f for c in per_basis.values() for f in c["per_family"]})
        for fam in fams:
            own, shr = {}, {}
            for champ in champs:
                shr[champ] = per_basis[champ]["per_family"].get(
                    fam, {}).get("mean_mcc_at_pref")
                entry = registry.get(f"{runs[champ]}:{basis}{champ}")
                pc = (entry or {}).get("metrics", {}).get("per_category") or {}
                schema_path = resolve_schema_path(data_dir, f"{basis}{champ}")
                sch = json.loads(Path(schema_path).read_text(encoding="utf-8"))
                own[champ] = aggregate_per_category_by_family(pc, sch).get(
                    fam, {}).get("mean_mcc_at_pref")
            r_own, r_shr = rank(own), rank(shr)
            # A ranking is only worth interpreting when the champions are
            # actually separated. Rank-only comparison flagged two cells as
            # 'flipped' at spreads of 0.004 and 0.025 -- Ta and Ve tied to
            # four decimals -- which is reshuffling noise, not a
            # distribution effect. Require MIN_SPREAD before a flip counts.
            vals = [v for v in shr.values() if v is not None]
            spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
            same = r_own == r_shr
            meaningful = (not same) and spread >= MIN_SPREAD
            if meaningful:
                flips.append({"basis": basis, "family": fam,
                              "own": r_own, "shared": r_shr,
                              "spread": spread,
                              "own_scores": own, "shared_scores": shr})
            elif not same:
                ties.append({"basis": basis, "family": fam,
                             "spread": spread})
            mark = "yes" if same else ("NO" if meaningful else "tied")
            print(f"  {basis:9s} {fam:17s} {' > '.join(r_own):26s} "
                  f"{' > '.join(r_shr):26s} {mark:5s} spread={spread:.4f}")

    out["ranking_flips"] = flips
    out["ties"] = ties
    out["min_spread"] = MIN_SPREAD
    n_cells = sum(len({f for c in pb.values() for f in c["per_family"]})
                  for pb in out["per_basis"].values())
    out["n_cells"] = n_cells
    print("\n" + "-" * 78)
    print(f"  {n_cells - len(flips)}/{n_cells} family cells keep the same "
          f"champion ordering on shared positions.")
    if ties:
        print(f"  {len(ties)} cell(s) reordered but are TIED "
              f"(spread < {MIN_SPREAD}) -- ranking noise, not a "
              f"distribution effect:")
        for t_ in ties:
            print(f"    {t_['basis']}/{t_['family']}: spread {t_['spread']:.4f}")
    if flips:
        print(f"\n  {len(flips)} MEANINGFUL flip(s) -- separated "
              f"champions reorder when the positions change:")
        for f in flips:
            print(f"    {f['basis']}/{f['family']} (spread {f['spread']:.4f}): "
                  f"own {' > '.join(f['own'])}  ->  shared {' > '.join(f['shared'])}")
        print("\n  => The distributional confound is REAL where it "
              "matters. The unified pool is justified; re-examine those cells.")
    else:
        print("\n  => No ordering change among SEPARATED champions; "
              "every cell that reordered was a statistical tie.")
        print("     The distribution does not alter any defensible "
              "cross-champion conclusion, so the unified pool can be "
              "skipped for these claims.")

    path = args["--output"]
    if path == "auto" or path is None:
        path = f"saes/{game}/analysis/shared-position-probe-{hook}.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2, sort_keys=True),
                          encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
