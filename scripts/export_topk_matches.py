"""Export the top-K candidate features per BSP -- the input to the causal track.

``lib/sae/eval.py`` matches each BSP to ONE feature, the argmax. That is a
DECODABILITY ranking, and reporting standard 4 forbids resting a causal claim on
it: the argmax feature can be a spectator while the feature the network actually
uses ranks #2 or lower. Phase 3B-causal re-ranks a shortlist by intervention
effect, so it needs the shortlist.

This script writes that shortlist. It reads the ``_matching-<animal>.pt`` cache
that ``sae_eval.py`` already wrote -- which holds the full ``(d_dict, num_bsps)``
metric matrices -- so it needs **no ``_h`` cache, no GPU and no re-encode**, and
runs on every checkpoint already evaluated.

Ranking metric is ``mcc`` by default: reporting standard 1 makes MCC the
headline, and standard 4 prefers it for the rare threat concepts this track is
about (base rate ~0.02, where F1 moves with prevalence through precision).
The export records, per BSP, where the F1-argmax feature lands inside the MCC
shortlist (``argmax_f1_rank``, -1 when it falls outside) -- reporting standard
4's "F1-vs-MCC disagreement is a robustness flag", measured rather than asserted.

The ordering here is still decodability. It is a CANDIDATE LIST, not a causal
result; nothing in this file decides which feature the network uses.

Usage:
    export_topk_matches.py --run-id=<id> --bsps=<animal> [options]
    export_topk_matches.py --all [options]
    export_topk_matches.py (-h | --help)

Options:
    -h --help          Show this help message.
    --run-id=<id>      Checkpoint stem, e.g. F04-champYb-s42-jumprelu-t64-exp8-s4.fc1
    --bsps=<animal>    BSP set, e.g. tigerYb / hawkYb / gorillaYb.
    --all              Every cached (run_id, animal) pair found on disk.
    --filter=<glob>    With --all: only cache stems matching this glob.
    --game=<name>      Game name [default: quarto]
    --top-k=<n>        Candidates per BSP [default: 16]
    --metric=<m>       Ranking metric: mcc | mcc_at_pref | youden_j | f1
                       [default: mcc]
    --output=<path>    Output JSON. Default: auto
                       (saes/<game>/analysis/<run_id>_topk-<animal>.json)
    --dry-run          Report what would be written; write nothing.
"""

from __future__ import annotations

import json
import sys
from fnmatch import fnmatch
from pathlib import Path

import torch
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.sae.eval import (  # noqa: E402
    RANKABLE_METRICS,
    FeatureBSPMatching,
    resolve_schema_path,
    top_k_features_per_bsp,
)

# Machine-readable term definitions travel WITH the artefact, so a report is
# self-describing (reporting standard 6). Same contract as lib/sae/dilution.py.
GLOSSARY: dict[str, dict[str, str]] = {
    "rank_metric": {
        "range": "see per-metric entries",
        "meaning": "Metric the shortlist is ordered by. Decodability, not "
                   "causality -- 3B-causal re-ranks by intervention effect.",
    },
    "mcc": {
        "range": "[-1, 1], ideal 1",
        "meaning": "Matthews correlation between a feature firing and the BSP "
                   "holding. Headline metric (reporting standard 1). Not "
                   "base-rate invariant.",
    },
    "mcc_at_pref": {
        "range": "[-1, 1], ideal 1",
        "meaning": "MCC restated at p_ref = 0.025. The number comparable "
                   "across populations with different base rates.",
    },
    "youden_j": {
        "range": "[-1, 1], ideal 1",
        "meaning": "TPR - FPR. Prevalence-INVARIANT companion to MCC; report "
                   "the pair, since J alone flatters a latent that fires on "
                   "half the dataset.",
    },
    "f1": {
        "range": "[0, 1], ideal 1",
        "meaning": "Harmonic mean of precision and recall. Stored for "
                   "literature comparison only -- never used to select or "
                   "conclude (reporting standard 1).",
    },
    "base_rate": {
        "range": "[0, 1]",
        "meaning": "Positive-class frequency of the BSP on this position set. "
                   "Below ~0.01 an F1-selected feature is not trustworthy.",
    },
    "argmax_f1_rank": {
        "range": "integer >= -1",
        "meaning": "0-based position of the F1-argmax feature inside this "
                   "shortlist; -1 means it is not in the shortlist at all. "
                   "Non-zero = the two metrics disagree on the best candidate "
                   "(reporting standard 4's robustness flag).",
    },
}


def cache_dir(game: str) -> Path:
    return Path("saes") / game / "cache"


def load_matching(path: Path) -> FeatureBSPMatching:
    """Rebuild a FeatureBSPMatching from a cache written by sae_eval.py."""
    c = torch.load(path, map_location="cpu", weights_only=False)
    return FeatureBSPMatching(
        precision=c["precision"],
        recall=c["recall"],
        f1=c["f1"],
        best_f1_per_bsp=c["best_f1_per_bsp"],
        best_feature_per_bsp=c["best_feature_per_bsp"],
        mcc=c.get("mcc"),
        best_mcc_per_bsp=c.get("best_mcc_per_bsp"),
        best_feature_per_bsp_mcc=c.get("best_feature_per_bsp_mcc"),
        youden_j=c.get("youden_j"),
        best_j_per_bsp=c.get("best_j_per_bsp"),
        best_feature_per_bsp_j=c.get("best_feature_per_bsp_j"),
        mcc_at_pref=c.get("mcc_at_pref"),
        best_mcc_at_pref_per_bsp=c.get("best_mcc_at_pref_per_bsp"),
        best_feature_per_bsp_mcc_at_pref=c.get(
            "best_feature_per_bsp_mcc_at_pref"),
        p_ref=c.get("p_ref"),
        base_rates=c.get("base_rates"),
        f1_lift_per_bsp=c.get("f1_lift_per_bsp"),
        trivial_f1_per_bsp=c.get("trivial_f1_per_bsp"),
    )


def build_report(
    run_id: str, animal: str, game: str, top_k: int, metric: str,
) -> dict:
    path = cache_dir(game) / f"{run_id}_matching-{animal}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"no matching cache at {path}\n"
            f"  run: python sae_eval.py evaluate saes/{game}/{run_id}.pt "
            f"--bsps={animal}"
        )
    matching = load_matching(path)
    tk = top_k_features_per_bsp(matching, k=top_k, metric=metric)

    schema_path = resolve_schema_path(Path("data") / game, animal)
    bsps = (json.loads(schema_path.read_text(encoding="utf-8"))["bsps"]
            if schema_path else [])
    if bsps and len(bsps) != int(tk.indices.shape[0]):
        raise ValueError(
            f"schema has {len(bsps)} BSPs but the cache has "
            f"{int(tk.indices.shape[0])} -- stale cache or wrong schema"
        )

    def col(t: torch.Tensor | None, b: int) -> list | None:
        return None if t is None else [round(float(v), 6) for v in t[b]]

    base = matching.base_rates
    entries = []
    for b in range(int(tk.indices.shape[0])):
        meta = bsps[b] if bsps else {}
        entries.append({
            "bsp_index": b,
            "bsp_id": meta.get("id"),
            "category": meta.get("category"),
            "concept_family": meta.get("concept_family"),
            "family_role": meta.get("family_role"),
            "base_rate": round(float(base[b]), 6) if base is not None else None,
            "argmax_f1_rank": (int(tk.argmax_f1_rank[b])
                               if tk.argmax_f1_rank is not None else None),
            "candidates": {
                "feature": [int(i) for i in tk.indices[b]],
                metric: col(tk.values, b),
                "mcc": col(tk.mcc, b),
                "mcc_at_pref": col(tk.mcc_at_pref, b),
                "youden_j": col(tk.youden_j, b),
                "f1": col(tk.f1, b),
                "precision": col(tk.precision, b),
                "recall": col(tk.recall, b),
            },
        })

    disagree = [e for e in entries if e["argmax_f1_rank"] not in (0, None)]
    outside = [e for e in entries if e["argmax_f1_rank"] == -1]
    return {
        "run_id": run_id,
        "bsp_set": animal,
        "game": game,
        "rank_metric": metric,
        "top_k": int(tk.k),
        "d_dict": int(matching.f1.shape[0]),
        "num_bsps": len(entries),
        "p_ref": matching.p_ref,
        "source_cache": str(path).replace("\\", "/"),
        "summary": {
            "n_argmax_f1_disagrees": len(disagree),
            "n_argmax_f1_outside_topk": len(outside),
            "frac_argmax_f1_disagrees": (
                round(len(disagree) / len(entries), 4) if entries else None),
        },
        "glossary": GLOSSARY,
        "bsps": entries,
    }


def discover(game: str, pattern: str | None) -> list[tuple[str, str]]:
    pairs = []
    for p in sorted(cache_dir(game).glob("*_matching-*.pt")):
        stem = p.name[: -len(".pt")]
        run_id, _, animal = stem.rpartition("_matching-")
        if not run_id:
            continue
        if pattern and not fnmatch(stem, pattern):
            continue
        pairs.append((run_id, animal))
    return pairs


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    metric = args["--metric"]
    top_k = int(args["--top-k"])
    if metric not in RANKABLE_METRICS:
        sys.exit(f"error: --metric must be one of {', '.join(RANKABLE_METRICS)}")

    if args["--all"]:
        pairs = discover(game, args["--filter"])
        if not pairs:
            sys.exit("error: no matching caches found")
    else:
        if not (args["--run-id"] and args["--bsps"]):
            sys.exit("error: give --run-id and --bsps, or --all")
        pairs = [(args["--run-id"], args["--bsps"])]

    out_dir = Path("saes") / game / "analysis"
    written = 0
    for run_id, animal in pairs:
        try:
            report = build_report(run_id, animal, game, top_k, metric)
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP {run_id}:{animal} -- {exc}")
            continue
        out = (Path(args["--output"]) if args["--output"] and not args["--all"]
               else out_dir / f"{run_id}_topk-{animal}.json")
        s = report["summary"]
        print(f"{run_id}:{animal}  d_dict={report['d_dict']} "
              f"bsps={report['num_bsps']} k={report['top_k']} "
              f"argmax-F1 disagrees on {s['n_argmax_f1_disagrees']}"
              f" ({s['n_argmax_f1_outside_topk']} outside top-{report['top_k']})"
              + ("  [dry-run]" if args["--dry-run"] else f" -> {out}"))
        if not args["--dry-run"]:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=1), encoding="utf-8")
            written += 1
    if not args["--dry-run"]:
        print(f"\nwrote {written} report(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
