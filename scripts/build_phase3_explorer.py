"""Build the phase-3 explorer: a single self-contained HTML page.

Reads only artefacts that already exist on disk -- the 3A dilution reports, the
SAE-vs-LP efficiency reports, the basis comparisons, the shared-position probe,
the eval registry and the linear-probe reports -- and splices them into
``tools/phase3_explorer_template.html`` at the ``__DATA__`` marker.

No GPU, no model loading, no recomputation: the page is a *view* over the
committed analysis JSONs, so re-running it after a new panel run is the whole
update procedure.

Usage:
    build_phase3_explorer.py [options]
    build_phase3_explorer.py (-h | --help)

Options:
    -h --help            Show this help message.
    --game=<name>        Game name [default: quarto]
    --template=<path>    HTML template with a __DATA__ marker
                         [default: tools/phase3_explorer_template.html]
    --output=<path>      Output HTML [default: tools/phase3_explorer.html]
    --data-only=<path>   Also write the raw JSON bundle here [default: none]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.sae.eval import resolve_schema_path  # noqa: E402

BASES = ("gorilla", "hawk", "tiger")

# Per-concept fields the page reads. Everything else in a dilution report is
# either recomputable in the page or not used, and dropping it keeps the
# embedded bundle small enough to stay a single file.
CONCEPT_FIELDS = (
    "verdict", "base_rate", "top_phi", "asymptote_r2", "asymptote_r2_std",
    "null_r2", "solo_r2", "solo_frac", "knee_k", "community_size",
    "neg_coupling_frac", "support_overlap", "singleton_community",
    "intrinsic_dim", "knee_over_idim", "n_candidates", "n_curve_positives",
    "random_asymptote_r2",
)


def r4(v):
    return None if v is None else round(float(v), 4)


def basis_of(animal: str) -> str:
    for b in BASES:
        if animal.startswith(b):
            return b
    return animal


def champ_of(run_id: str) -> str:
    for tag in ("Ta", "Ve", "Yb"):
        if f"champ{tag}" in run_id:
            return tag
    return "?"


def build(game: str) -> dict:
    data_dir = ROOT / "data" / game
    analysis = ROOT / "saes" / game / "analysis"
    bundle: dict = {}

    # --- schemas: the concept_family stamp, so the page never re-derives it ---
    bundle["families"] = {}
    for basis in BASES:
        path = resolve_schema_path(data_dir, basis)
        if path is None:
            continue
        schema = json.loads(Path(path).read_text(encoding="utf-8"))
        bundle["families"][basis] = {
            b["id"]: {"c": b["category"], "f": b.get("concept_family"),
                      "r": b.get("family_role")}
            for b in schema["bsps"]
        }

    # --- 3A dilution reports -------------------------------------------------
    runs = []
    glossary = None
    for p in sorted(analysis.glob("*_dilution-*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d["summary"]
        glossary = glossary or d.get("glossary")
        fam_map = bundle["families"].get(basis_of(s["bsp_set"]), {})
        concepts = []
        for c in d["concepts"]:
            if "asymptote_r2" not in c:
                concepts.append({"id": c["bsp_id"], "cat": c["category"],
                                 "verdict": c["verdict"], "degenerate": True})
                continue
            row = {"id": c["bsp_id"], "cat": c["category"],
                   "fam": fam_map.get(c["bsp_id"], {}).get("f")}
            for k in CONCEPT_FIELDS:
                if k in c:
                    row[k] = r4(c[k]) if isinstance(c[k], (int, float)) else c[k]
            row["curve"] = [round(float(v), 3) for v in c.get("curve", [])]
            concepts.append(row)
        runs.append({
            "run_id": s["run_id"], "bsp_set": s["bsp_set"],
            "basis": basis_of(s["bsp_set"]), "champ": champ_of(s["run_id"]),
            "hook": "s4.fc1" if s["run_id"].endswith("fc1") else "s4.conv2",
            "family": s["run_id"].split("-")[0],
            "anchored": "anchored" in s["run_id"],
            "n_samples": s["n_samples"], "d_dict": s["d_dict"],
            "orbit_aware": s.get("orbit_aware_split"),
            "random_control": s.get("random_control"),
            "n_with_random": sum(1 for c in d["concepts"]
                                 if "random_asymptote_r2" in c),
            "config": s["config"], "gate": d["gate_g3a"],
            "category_summary": d["category_summary"], "concepts": concepts,
        })
    bundle["runs_3a"] = runs
    bundle["glossary_3a"] = glossary or {"metrics": {}, "verdicts": {}}

    # --- basis verdict -------------------------------------------------------
    bundle["efficiency"] = [
        {k: json.loads(p.read_text(encoding="utf-8"))[k]
         for k in ("run_id", "champ", "hook", "metric", "lp_floor",
                   "per_basis", "verdict", "triads", "excluded")}
        for p in sorted(analysis.glob("*_sae-lp-efficiency.json"))
    ]
    bundle["basis_comparison"] = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(analysis.glob("*_basis-comparison.json"))
    ]

    probe = analysis / "shared-position-probe-s4.fc1.json"
    bundle["shared_probe"] = (json.loads(probe.read_text(encoding="utf-8"))
                              if probe.exists() else None)

    # --- eval registry (champion era only) -----------------------------------
    reg = json.loads((ROOT / "saes" / game / "eval_registry.json")
                     .read_text(encoding="utf-8"))
    rows = []
    for key, v in sorted(reg.items()):
        run_id, _, bsp = key.partition(":")
        if champ_of(run_id) == "?":
            continue
        m = v.get("metrics", {})
        rows.append({
            "run_id": run_id, "bsp_set": bsp, "arch": v.get("architecture"),
            "hook": v.get("hook"), "champ": champ_of(run_id),
            "random": "random" in run_id, "anchored": "anchored" in run_id,
            "family": run_id.split("-")[0],
            "l0": r4(m.get("l0")), "fvu": r4(m.get("fvu")),
            "dead": r4(m.get("dead_features_pct")),
            "mcc": r4(m.get("coverage_mcc")),
            "j": r4(m.get("coverage_youden_j")),
            "mccp": r4(m.get("coverage_mcc_at_pref")),
            "f1": r4(m.get("coverage")),
            "base_rate": r4(m.get("mean_base_rate")),
            "current_schema": all(
                k in m for k in ("coverage_mcc", "coverage_youden_j",
                                 "coverage_mcc_at_pref", "mean_base_rate")),
            "per_category": {
                cat: {**{k: r4(cv.get(k)) for k in
                         ("mean_mcc", "mean_youden_j", "mean_mcc_at_pref",
                          "mean_base_rate", "mean_f1", "sd_mcc")
                         if cv.get(k) is not None},
                      "count": cv.get("count")}
                for cat, cv in (m.get("per_category") or {}).items()},
        })
    bundle["registry"] = rows

    # --- linear-probe reports (the efficiency denominators) ------------------
    lp = []
    pattern = str(data_dir / "linear_probe_*_s4.*_amalgam_*_activations_results.json")
    for p in sorted(glob.glob(pattern)):
        name = os.path.basename(p)
        if "_random_" in name:
            continue
        tag = name.split("_amalgam_")[1].split("_")[0]
        if tag not in ("ta", "ve", "yb"):
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        cfg = d["config"]
        n_train = cfg.get("n_train") or 0
        lp.append({
            "champ": tag.capitalize(),
            "hook": "s4.fc1" if "s4.fc1" in name else "s4.conv2",
            "basis": basis_of((cfg.get("bsp_set") or "").split("_")[0]),
            "n_train": n_train, "n_test": cfg.get("n_test"),
            "subsampled": cfg.get("train_subsampled"),
            "overall": {k: r4(v) for k, v in d["overall"].items()
                        if isinstance(v, (int, float))},
            "per_category": {
                cat: {**{k: r4(cv.get(k)) for k in
                         ("mean_mcc", "mean_youden_j", "mean_mcc_at_pref",
                          "mean_base_rate", "mean_f1")},
                      "count": cv["count"]}
                for cat, cv in d["per_category"].items()},
            "per_family": d.get("per_family") or {},
            # Rare BSPs are where a capped training split actually costs
            # something; name how many so a thin denominator is visible.
            "thin_bsps": sum(1 for r in d["per_bsp"]
                             if r.get("train_pos_rate", 1) * n_train < 200),
        })
    bundle["lp"] = lp
    return bundle


def main() -> int:
    args = docopt(__doc__)
    game = args["--game"]
    template = ROOT / args["--template"]
    output = ROOT / args["--output"]

    if not template.exists():
        print(f"error: template not found: {template}", file=sys.stderr)
        return 1

    bundle = build(game)
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    # The bundle rides inside <script type="application/json">, so the only
    # sequence that can break out of it is a literal "</script". Escaping the
    # slash keeps the JSON byte-identical after parsing.
    payload = payload.replace("</", "<\\/")

    html = template.read_text(encoding="utf-8")
    if "__DATA__" not in html:
        print("error: template has no __DATA__ marker", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace("__DATA__", payload), encoding="utf-8")

    if args["--data-only"] not in (None, "none"):
        Path(args["--data-only"]).write_text(payload, encoding="utf-8")

    print(f"runs_3a={len(bundle['runs_3a'])} "
          f"concepts={sum(len(r['concepts']) for r in bundle['runs_3a'])} "
          f"efficiency={len(bundle['efficiency'])} "
          f"registry={len(bundle['registry'])} lp={len(bundle['lp'])}")
    print(f"wrote {output}  ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
