"""Build the LaTeX tables for the LXAI @ NeurIPS 2026 paper.

Every number in the paper comes from here, so the tables cannot drift from the
artefacts on disk. Family rollups use the project's own
``aggregate_per_category_by_family`` (which reads ``concept_family`` off the BSP
schema) rather than re-deriving the mapping -- see CLAUDE.md.

Usage:
    make_tables.py [--out=DIR] [--metric=KEY]
    make_tables.py (-h | --help)

Options:
    --out=DIR       Directory for generated .tex / .json [default: .]
    --metric=KEY    Family metric [default: mean_mcc_at_pref]
    -h --help       Show this screen.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from docopt import docopt

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from lib.sae.eval import aggregate_per_category_by_family  # noqa: E402
from lib.sae.eval import resolve_schema_path  # noqa: E402

GAME = "quarto"
CHAMPS = ("Ta", "Ve", "Yb")

# Paper-facing agent names, ordered by head-to-head strength (weakest first).
# Internal champion tags are project jargon and are also identifying, so the
# double-blind submission uses these instead.
AGENT_LABEL = {"Ta": "M1", "Ve": "M2", "Yb": "M3"}

# Internal basis codenames are project jargon; the paper names them by what
# they are. A basis is a packaging of the concept menu, not a concept axis.
# `hen` is hawk's shape on the four NEGATIVE attribute poles, so the paper names
# it as such. These two maps are the SINGLE home for the anonymisation --
# make_3a_tables.py imports them rather than restating them.
BASIS_LABEL = {"gorilla": "state", "hawk": "count", "hen": "count$^{-}$",
               "tiger": "agent-rel."}

# Three recipes trained at BOTH hooks -> the hook is the only thing that varies.
# (fc1_exp, conv2_exp, recipe label)
RECIPES = (
    ("F01", "E01", "topk-k32-exp8"),
    ("F02", "E03", "topk-k64-exp8"),
    ("F03", "E05", "batchtopk-k32-exp8"),
)

# Concept families, grouped by the partition the paper reports.
PER_CELL_FAMILIES = ("board_occupancy", "board_attribute", "offered_piece_attr")
RELATIONAL_FAMILIES = (
    "line_threat", "square_threat", "global_threat",
    "game_phase", "pool_reasoning", "offered_completion",
)
# Base rate far from p_ref = 0.025; standardised MCC is a ~20x extrapolation.
EXTRAPOLATED = ("pool_reasoning", "global_threat")

BASES = ("gorilla", "hawk", "tiger")


def run_id(exp: str, champ: str, recipe: str, hook: str) -> str:
    return f"{exp}-champ{champ}-s42-{recipe}-s4.{hook}"


def load_registry() -> dict:
    path = PROJECT_DIR / "saes" / GAME / "eval_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema(bsps: str) -> dict:
    path = resolve_schema_path(PROJECT_DIR / "data" / GAME, bsps)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def families_for(registry: dict, rid: str, bsps: str, schema: dict) -> dict:
    row = registry.get(f"{rid}:{bsps}")
    if row is None:
        raise KeyError(f"missing registry row: {rid}:{bsps}")
    per_category = row["metrics"].get("per_category")
    if not per_category:
        raise KeyError(f"row has no per_category: {rid}:{bsps}")
    return aggregate_per_category_by_family(per_category, schema)


def collect(metric: str) -> dict:
    """Per (basis, family): fc1 vs conv2 over the nine matched cells."""
    registry = load_registry()
    schemas = {b: load_schema(b) for b in BASES}
    acc: dict[tuple[str, str], dict] = {}

    for basis in BASES:
        schema = schemas[basis]
        for champ in CHAMPS:
            bsps = f"{basis}{champ}"
            for fc1_exp, conv2_exp, recipe in RECIPES:
                fam_fc1 = families_for(
                    registry, run_id(fc1_exp, champ, recipe, "fc1"), bsps, schema)
                fam_conv2 = families_for(
                    registry, run_id(conv2_exp, champ, recipe, "conv2"), bsps, schema)
                for family in set(fam_fc1) & set(fam_conv2):
                    a, b = fam_fc1[family], fam_conv2[family]
                    if metric not in a or metric not in b:
                        continue
                    entry = acc.setdefault(
                        (basis, family),
                        {"n_bsps": a["count"], "fc1": [], "conv2": [],
                         "delta": [], "base_rate": [], "cells": []},
                    )
                    entry["fc1"].append(a[metric])
                    entry["conv2"].append(b[metric])
                    entry["delta"].append(a[metric] - b[metric])
                    entry["base_rate"].append(a.get("mean_base_rate", 0.0))
                    entry["cells"].append(f"{champ}/{recipe}")

    out = {}
    for (basis, family), e in acc.items():
        deltas = e["delta"]
        out[f"{basis}/{family}"] = {
            "basis": basis,
            "family": family,
            "n_bsps": e["n_bsps"],
            "n_cells": len(deltas),
            "base_rate": round(statistics.fmean(e["base_rate"]), 4),
            "fc1": round(statistics.fmean(e["fc1"]), 4),
            "conv2": round(statistics.fmean(e["conv2"]), 4),
            "delta": round(statistics.fmean(deltas), 4),
            "sd_delta": round(statistics.stdev(deltas), 4) if len(deltas) > 1 else 0.0,
            "fc1_wins": sum(1 for d in deltas if d > 0),
            "extrapolated": family in EXTRAPOLATED,
        }
    return out


def collect_availability(repair: bool = True) -> list[dict]:
    """Availability (LP) vs isolation (best latent) from the efficiency reports.

    The champYb ``s4.conv2`` member of those reports (``E05``) is a degenerate
    dictionary -- FVU 0.110, 99.0% dead -- because BatchTopK lacked the Gao
    et al. dead-feature auxiliary loss until 2026-08-15. With ``repair`` we
    substitute the canonical retrain (``K04``, FVU 0.0066) so conv2 is
    represented by its BEST dictionary, not its worst. Availability (the LP
    column) is read off raw activations and is identical either way.

    SINCE 2026-09-04 the substitution lives upstream: the project generated the
    ``K04`` efficiency report and moved ``E05``'s to ``analysis/superseded/``,
    because patching it here fixed one consumer and left the defective report on
    disk for every other one (docs/methods-reference.md S8). ``repair`` is now an
    idempotent GUARD -- it re-corrects if anyone regenerates the ``E05`` report
    -- and is a no-op on a clean tree.
    """
    analysis = PROJECT_DIR / "saes" / GAME / "analysis"
    rows = []
    for path in sorted(analysis.glob("*_sae-lp-efficiency.json")):
        rep = json.loads(path.read_text(encoding="utf-8"))
        for basis, block in rep["per_basis"].items():
            for family, vals in block["families"].items():
                rows.append({
                    "champ": rep["champ"],
                    "hook": rep["hook"].replace("s4.", ""),
                    "basis": basis,
                    "family": family,
                    "n_bsps": vals["n_bsps"],
                    "base_rate": vals["base_rate"],
                    "lp": vals["lp_mcc_at_pref"],
                    "sae": vals["sae_mcc_at_pref"],
                    "efficiency": vals["efficiency"],
                    "run_id": rep["run_id"],
                })
    if repair:
        rows = apply_conv2_repair(rows)
    return rows


CONV2_REPAIR = ("K04", "champYb", "batchtopk-k32-exp8")


def apply_conv2_repair(rows: list[dict]) -> list[dict]:
    """Swap champYb's conv2 isolation column onto the canonical K04 dictionary.

    Idempotent guard -- a no-op once the upstream artefact is the K04 report.
    """
    registry = load_registry()
    exp, champ, recipe = CONV2_REPAIR
    rid = run_id(exp, champ.replace("champ", ""), recipe, "conv2")
    for basis in BASES:
        schema = load_schema(basis)
        try:
            fams = families_for(registry, rid, f"{basis}Yb", schema)
        except KeyError:
            continue
        for r in rows:
            if not (r["champ"] == "Yb" and r["hook"] == "conv2"
                    and r["basis"] == basis):
                continue
            new = fams.get(r["family"], {}).get("mean_mcc_at_pref")
            if new is None:
                continue
            r["sae_legacy"] = r["sae"]
            r["efficiency_legacy"] = r["efficiency"]
            r["sae"] = round(new, 4)
            r["efficiency"] = round(new / r["lp"], 4) if r["lp"] else 0.0
            r["run_id"] = rid
            r["repaired"] = True
    return rows


def robustness_check(metric: str = "mean_mcc") -> dict:
    """Does substituting the canonical conv2 dictionary flip any family sign?"""
    registry = load_registry()
    out = {}
    for basis in BASES:
        schema = load_schema(basis)
        bsps = f"{basis}Yb"
        fc1 = families_for(
            registry, run_id("F03", "Yb", "batchtopk-k32-exp8", "fc1"), bsps, schema)
        for label, exp in (("legacy_E05", "E05"), ("canonical_K04", "K04")):
            try:
                conv2 = families_for(
                    registry, run_id(exp, "Yb", "batchtopk-k32-exp8", "conv2"),
                    bsps, schema)
            except KeyError:
                continue
            for family in set(fc1) & set(conv2):
                if metric not in fc1[family] or metric not in conv2[family]:
                    continue
                key = f"{basis}/{family}"
                out.setdefault(key, {})[label] = round(
                    fc1[family][metric] - conv2[family][metric], 4)
    return out


def fmt(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}"


def table_split(data: dict) -> str:
    """Table 1: the family partition."""
    order = [(b, f) for b in BASES
             for f in PER_CELL_FAMILIES + RELATIONAL_FAMILIES
             if f"{b}/{f}" in data]
    lines = [
        r"\begin{tabular}{llrrrrrc}",
        r"\toprule",
        r"basis & concept family & $n$ & base rate & fc1 & conv2 & $\Delta$ & fc1 wins \\",
        r"\midrule",
    ]
    group = None
    for basis, family in order:
        d = data[f"{basis}/{family}"]
        kind = "per-cell" if family in PER_CELL_FAMILIES else "relational"
        if group is not None and kind != group:
            lines.append(r"\midrule")
        group = kind
        mark = r"$\ddagger$" if d["extrapolated"] else ""
        name = family.replace("_", r"\_")
        lines.append(
            f"{BASIS_LABEL[basis]} & \\texttt{{{name}}}{mark} & {d['n_bsps']} & "
            f"{fmt(d['base_rate'], 4)} & {fmt(d['fc1'])} & {fmt(d['conv2'])} & "
            f"\\textbf{{{d['delta']:+.3f}}} & {d['fc1_wins']}/{d['n_cells']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_availability(rows: list[dict]) -> str:
    """Table 2: availability vs isolation on the state-threat families."""
    keep = [r for r in rows
            if r["basis"] in ("gorilla", "hawk")
            and r["family"] in ("line_threat", "square_threat")]
    by_key: dict[tuple, dict] = {}
    for r in keep:
        by_key[(r["basis"], r["family"], r["champ"], r["hook"])] = r

    lines = [
        r"\begin{tabular}{llrrcrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{availability (LP)} & \multicolumn{2}{c}{isolation (SAE)} & \multicolumn{2}{c}{efficiency} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"basis / family & agent & fc1 & conv2 & higher & fc1 & conv2 & fc1 & conv2 \\",
        r"\midrule",
    ]
    for basis in ("gorilla", "hawk"):
        for family in ("line_threat", "square_threat"):
            for champ in CHAMPS:
                f = by_key.get((basis, family, champ, "fc1"))
                c = by_key.get((basis, family, champ, "conv2"))
                if not f or not c:
                    continue
                higher = r"\textbf{conv2}" if c["lp"] > f["lp"] else "fc1"
                name = f"{BASIS_LABEL[basis]}/{family}".replace("_", r"\_")
                lines.append(
                    f"\\texttt{{{name}}} & {AGENT_LABEL[champ]} & {fmt(f['lp'])} & {fmt(c['lp'])} & "
                    f"{higher} & {fmt(f['sae'])} & {fmt(c['sae'])} & "
                    f"{fmt(f['efficiency'], 2)} & {fmt(c['efficiency'], 2)} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    args = docopt(__doc__)
    out_dir = Path(args["--out"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = args["--metric"]

    split = collect(metric)
    avail = collect_availability()
    robust = robustness_check()

    (out_dir / "table_split.tex").write_text(table_split(split) + "\n", encoding="utf-8")
    (out_dir / "table_availability.tex").write_text(
        table_availability(avail) + "\n", encoding="utf-8")
    (out_dir / "numbers.json").write_text(
        json.dumps({"metric": metric, "split": split, "availability": avail,
                    "robustness_conv2_repair": robust},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_cell = [d for d in split.values() if d["family"] in PER_CELL_FAMILIES]
    relational = [d for d in split.values() if d["family"] in RELATIONAL_FAMILIES]
    print(f"metric = {metric}")
    print(f"families: {len(per_cell)} per-cell, {len(relational)} relational")
    print("per-cell rows won by fc1  :",
          [f"{d['basis']}/{d['family']} {d['fc1_wins']}/{d['n_cells']}" for d in per_cell])
    print("relational rows won by fc1:",
          [f"{d['basis']}/{d['family']} {d['fc1_wins']}/{d['n_cells']}" for d in relational])

    st = [r for r in avail if r["basis"] in ("gorilla", "hawk")
          and r["family"] in ("line_threat", "square_threat")]
    pairs = {}
    for r in st:
        pairs.setdefault((r["basis"], r["family"], r["champ"]), {})[r["hook"]] = r
    conv2_more_available = sum(
        1 for v in pairs.values()
        if "fc1" in v and "conv2" in v and v["conv2"]["lp"] > v["fc1"]["lp"])
    print(f"state-threat cells where conv2 is MORE linearly available: "
          f"{conv2_more_available}/{len(pairs)}")

    flips = {k: v for k, v in robust.items()
             if "legacy_E05" in v and "canonical_K04" in v
             and (v["legacy_E05"] > 0) != (v["canonical_K04"] > 0)}
    print(f"conv2-repair robustness: {len(flips)} sign flips out of {len(robust)} "
          f"families" + (f" -> {flips}" if flips else " (partition unchanged)"))
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
