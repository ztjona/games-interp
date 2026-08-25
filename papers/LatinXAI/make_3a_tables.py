"""Build the concentration-diagnostic (Phase 3A) tables for the LXAI paper.

Reads the frozen dilution reports under ``saes/quarto/analysis/`` -- nothing is
transcribed by hand, so the paper cannot drift from the artefacts. Companion to
``make_tables.py`` (probe-vs-dictionary tables); this file covers the second
instrument: how the recoverable signal is DISTRIBUTED over the dictionary.

Source of record: docs/diary/2026-08-25_3A-final-report.md.

Usage:
    make_3a_tables.py [--out=DIR]
    make_3a_tables.py (-h | --help)

Options:
    --out=DIR       Directory for generated .tex / .json [default: .]
    -h --help       Show this screen.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

from docopt import docopt

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

ANALYSIS = PROJECT_DIR / "saes" / "quarto" / "analysis"

# Paper-facing names (double-blind: internal codenames are identifying jargon).
# Imported, not restated -- a label that lives in two files is free to drift.
from make_tables import AGENT_LABEL, BASIS_LABEL  # noqa: E402

HOOK_LABEL = {"s4.conv2": "conv2", "s4.fc1": "fc1"}
CHAMPS = ("Ta", "Ve", "Yb")
BASES = ("gorilla", "hawk", "hen", "tiger")

# Number of disjuncts each concept category ORs over, known BY CONSTRUCTION from
# the BSP definitions (scripts/games/quarto.py), not fitted to anything:
#   pinned              -- one location AND one attribute pole        -> 1
#   *_any_threat        -- OR over the 4 attributes at one location   -> 4
#   tiger *_winnable    -- OR over 8 poles (tiger == OR(hawk u hen),
#                          verified with zero violations 2026-08-21)  -> 8
# `tiger_offered_completing_attr` is excluded: it ORs over locations rather than
# poles, so its disjunct count is not fixed by the category.
DISJUNCTS = {
    "threat_line": 1, "threat_square_2x2": 1,
    "reframed_count": 1, "reframed_completable": 1,
    "reframed_sq_count": 1, "reframed_sq_completable": 1,
    "neg_count": 1, "neg_completable": 1,
    "neg_sq_count": 1, "neg_sq_completable": 1,
    "reframed_any_threat": 4, "reframed_sq_any_threat": 4,
    "neg_any_threat": 4, "neg_sq_any_threat": 4,
    "tiger_line_winnable": 8, "tiger_square_winnable": 8,
}
SHAPE_LABEL = {1: "pinned", 4: "OR over 4 attributes", 8: "OR over 8 poles"}


def pct(x: float, places: int = 1) -> str:
    """Percent, LaTeX-escaped."""
    return f"{100 * x:.{places}f}" + r"\%"


def report_path(run_id: str, bsp_set: str) -> Path:
    return ANALYSIS / f"{run_id}_dilution-{bsp_set}.json"


def is_unsupervised(run_id: str) -> bool:
    """Anchored (I-series) dictionaries are the supervised positive control."""
    return not run_id.startswith("I")


def load_concepts(run_id: str, bsp_set: str) -> list[dict]:
    path = report_path(run_id, bsp_set)
    if not path.exists():
        raise FileNotFoundError(path)
    rep = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in rep["concepts"]:
        c = dict(c)
        c["run_id"] = run_id
        c["bsp_set"] = bsp_set
        out.append(c)
    return out


def all_reports() -> list[tuple[str, str]]:
    pairs = []
    for path in sorted(ANALYSIS.glob("*_dilution-*.json")):
        run_id, rest = path.name.split("_dilution-")
        pairs.append((run_id, rest[: -len(".json")]))
    return pairs


def strip_champ(bsp_set: str) -> str:
    for b in BASES:
        if bsp_set.startswith(b):
            return b
    raise ValueError(bsp_set)


def load_roles() -> dict[str, str]:
    from lib.sae.eval import resolve_schema_path
    roles = {}
    for b in BASES:
        path = resolve_schema_path(PROJECT_DIR / "data" / "quarto", b)
        schema = json.loads(Path(path).read_text(encoding="utf-8"))
        for cat, v in schema["category_families"].items():
            roles[cat] = v["family_role"]
    return roles


def target_cell_concepts() -> list[dict]:
    """Every unsupervised dictionary on the strongest agent's bottleneck.

    That is the one (agent, layer) where the dictionary has resolved anything
    into atoms at all, so it is the only place a disjunction could be built out
    of atoms -- the precondition the mechanism result is conditional on.
    """
    rows = []
    for run_id, bsp_set in all_reports():
        if "champYb" not in run_id or "s4.fc1" not in run_id:
            continue
        if not is_unsupervised(run_id):
            continue
        rows += load_concepts(run_id, bsp_set)
    return rows


def frac(rows: list[dict], verdict: str) -> float:
    return sum(1 for r in rows if r["verdict"] == verdict) / len(rows)


def med(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals)


# --------------------------------------------------------------------------
# Table 3: the controlled comparison -- shape vs agent-relativity, within basis
# --------------------------------------------------------------------------

def table_disjunction(rows: list[dict]) -> tuple[str, dict]:
    """Rows grouped by (basis, shape, agent-relativity).

    The comparison that isolates logical FORM is within a single basis, because
    ``count`` contains both shapes and both roles. Splitting the pinned rows by
    ``family_role`` puts the 2x2 (agent-relative x disjunctive) on the page.
    """
    schema_role = load_roles()
    groups: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        n_disj = DISJUNCTS.get(r["category"])
        if n_disj is None:
            continue
        agent = schema_role.get(r["category"], "?") == "agent_relative"
        groups[(strip_champ(r["bsp_set"]), n_disj, agent)].append(r)

    order = [
        ("gorilla", 1, False), ("hawk", 1, False), ("hen", 1, False),
        ("hawk", 1, True), ("hen", 1, True),
        ("hawk", 4, False), ("hen", 4, False), ("tiger", 8, True),
    ]
    lines = [
        r"\begin{tabular}{llcrrrrr}",
        r"\toprule",
        r"basis & concept shape & agent-rel. & $n$ & captured & "
        r"solo & $k_{90}$ & $R^2$ \\",
        r"\midrule",
    ]
    out = {}
    prev = None
    for basis, n_disj, agent in order:
        g = groups.get((basis, n_disj, agent))
        if not g:
            continue
        if prev is not None and n_disj != prev:
            lines.append(r"\midrule")
        prev = n_disj
        rec = {
            "basis": basis, "n_disjuncts": n_disj, "agent_relative": agent,
            "n": len(g),
            "captured": round(frac(g, "captured"), 4),
            "spread": round(frac(g, "spread"), 4),
            "solo_frac": round(med(g, "solo_frac"), 3),
            "knee_k": med(g, "knee_k"),
            "asymptote_r2": round(med(g, "asymptote_r2"), 3),
            "base_rate": round(med(g, "base_rate"), 4),
        }
        out[f"{basis}/{n_disj}/{'agent' if agent else 'state'}"] = rec
        emph = (lambda s: r"\textbf{" + s + "}") if n_disj > 1 else (lambda s: s)
        lines.append(
            emph(BASIS_LABEL[basis]) + " & " + emph(SHAPE_LABEL[n_disj]) + " & "
            + ("yes" if agent else "no") + " & " + str(rec["n"]) + " & "
            + emph(pct(rec["captured"]))
            + " & " + f"{rec['solo_frac']:.3f}" + " & " + f"{rec['knee_k']:.0f}"
            + " & " + f"{rec['asymptote_r2']:.3f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), out


# --------------------------------------------------------------------------
# The role split: does agent-relativity or disjunction cost the capture?
# --------------------------------------------------------------------------

def role_split(rows: list[dict]) -> dict:
    schema_role = load_roles()
    groups: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        n_disj = DISJUNCTS.get(r["category"])
        if n_disj is None:
            continue
        agent = schema_role.get(r["category"], "?") == "agent_relative"
        groups[(agent, n_disj > 1)].append(r)
    out = {}
    for (agent, disj), g in sorted(groups.items()):
        out[f"agent={agent}/disjunctive={disj}"] = {
            "n": len(g), "captured": round(frac(g, "captured"), 4),
            "spread": round(frac(g, "spread"), 4),
            "solo_frac": round(med(g, "solo_frac"), 3),
            "knee_k": med(g, "knee_k"),
            "asymptote_r2": round(med(g, "asymptote_r2"), 3),
            "base_rate": round(med(g, "base_rate"), 4),
        }
    return out


def prevalence_matched(rows: list[dict], lo: float = 0.015,
                       hi: float = 0.050) -> dict:
    """The same contrast restricted to a common base-rate window."""
    sel = [r for r in rows if lo <= r["base_rate"] <= hi
           and DISJUNCTS.get(r["category"]) is not None]
    out = {"window": [lo, hi]}
    for disj in (False, True):
        g = [r for r in sel if (DISJUNCTS[r["category"]] > 1) == disj]
        if not g:
            continue
        out["disjunctive" if disj else "pinned"] = {
            "n": len(g), "captured": round(frac(g, "captured"), 4),
            "spread": round(frac(g, "spread"), 4),
            "solo_frac": round(med(g, "solo_frac"), 3),
            "base_rate": round(med(g, "base_rate"), 4),
        }
    return out


# --------------------------------------------------------------------------
# Table 4: the retention / concentration ladder over the top-3 panel
# --------------------------------------------------------------------------

def table_ladder() -> tuple[str, list[dict]]:
    """Retention and concentration per (agent, layer), over the top-3 panel.

    Pooled over the three panel bases: the row is a property of the (agent,
    layer) pair, and the per-basis breakdown is what Table 3 is for.
    """
    panel = json.loads((ANALYSIS / "3A_panel.json").read_text(
        encoding="utf-8"))["panel"]
    per_cell, pooled = [], collections.defaultdict(list)
    for cell, members in sorted(panel.items()):
        champ, hook, basis = cell.split("/")
        cs = []
        for m in members:
            try:
                cs += load_concepts(m["run_id"], f"{basis}{champ}")
            except FileNotFoundError:
                continue
        if not cs:
            continue
        pooled[(champ, hook)] += cs
        r2, rand = med(cs, "asymptote_r2"), med(cs, "random_asymptote_r2")
        per_cell.append({
            "champ": champ, "hook": hook, "basis": basis, "n": len(cs),
            "n_runs": len(members),
            "r2": round(r2, 3), "random_r2": round(rand, 3),
            "retention": round(r2 / rand, 2) if rand else None,
            "solo_frac": round(med(cs, "solo_frac"), 3),
            "top_phi": round(med(cs, "top_phi"), 3),
            "captured": round(frac(cs, "captured"), 4),
            "spread": round(frac(cs, "spread"), 4),
            "absent": round(frac(cs, "absent"), 4),
        })

    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"agent & layer & $n$ & $R^2$ & $R^2_{\text{rand}}$ & retention & solo \\",
        r"\midrule",
    ]
    for champ in CHAMPS:
        for hook in ("s4.conv2", "s4.fc1"):
            cs = pooled.get((champ, hook))
            if not cs:
                continue
            r2, rand = med(cs, "asymptote_r2"), med(cs, "random_asymptote_r2")
            hot = champ == "Yb" and hook == "s4.fc1"
            emph = (lambda s: r"\textbf{" + s + "}") if hot else (lambda s: s)
            lines.append(
                emph(AGENT_LABEL[champ]) + " & " + emph(HOOK_LABEL[hook]) + " & "
                + str(len(cs)) + " & " + f"{r2:.3f}" + " & " + f"{rand:.3f}"
                + " & " + emph(f"{r2 / rand:.1f}" + r"$\times$")
                + " & " + emph(f"{med(cs, 'solo_frac'):.2f}") + r" \\"
            )
        if champ != CHAMPS[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), per_cell


# --------------------------------------------------------------------------
# knee_k distribution -- the tiling signature
# --------------------------------------------------------------------------

def knee_distribution(rows: list[dict]) -> dict:
    out = {}
    for n_disj in (1, 4, 8):
        g = [r for r in rows if DISJUNCTS.get(r["category"]) == n_disj]
        if not g:
            continue
        kk = [r["knee_k"] for r in g]
        counts = collections.Counter(kk)
        out[str(n_disj)] = {
            "n": len(kk),
            "mode": counts.most_common(1)[0][0],
            "mode_share": round(counts.most_common(1)[0][1] / len(kk), 4),
            "median": statistics.median(kk),
            "min": min(kk),
            "at_or_below_predicted": round(
                sum(1 for x in kk if x <= n_disj) / len(kk), 4),
            "below_predicted": round(
                sum(1 for x in kk if x < n_disj) / len(kk), 4),
            "histogram": {str(k): v for k, v in sorted(counts.items())},
            "solo_frac": round(med(g, "solo_frac"), 3),
            "asymptote_r2": round(med(g, "asymptote_r2"), 3),
        }
    return out


# --------------------------------------------------------------------------
# Gate + capacity + agreement headline numbers
# --------------------------------------------------------------------------

def gate_summary() -> dict:
    runs = json.loads((ANALYSIS / "3A_gate_summary.json").read_text(
        encoding="utf-8"))["runs"]
    verdicts = collections.Counter(r["verdict"] for r in runs)
    n_sp = sum(r["n_spread"] for r in runs)
    n_cap = sum(r["n_captured"] for r in runs)
    n_abs = sum(r["n_absent"] for r in runs)
    tot = n_sp + n_cap + n_abs
    return {
        "n_cells": len(runs),
        "n_concept_verdicts": tot,
        "cell_verdicts": dict(verdicts),
        "spread": round(n_sp / tot, 4),
        "captured": round(n_cap / tot, 4),
        "absent": round(n_abs / tot, 4),
        "n_undecided": sum(r["n_undecided"] for r in runs),
        "undecided_frac": round(sum(r["n_undecided"] for r in runs) / tot, 4),
        "n_unstable_cells": sum(
            1 for r in runs if not r["gate_verdict_is_stable"]),
        "provisional": sum(1 for r in runs if r["gate_is_provisional"]),
        "min_random_control_coverage": min(
            r["random_control_coverage"] for r in runs),
        "rule_versions": sorted({r["rule_version"] for r in runs}),
    }


def capacity_check() -> dict:
    """Doubling the dictionary twice: does the spread fraction move at all?"""
    runs = json.loads((ANALYSIS / "3A_gate_summary.json").read_text(
        encoding="utf-8"))["runs"]
    keep: dict[str, dict] = {}
    for r in runs:
        # One seed (s42) across the capacity ladder, so the only thing that
        # varies is d_dict: exp8 (4096) -> exp32 (16384) -> exp64 (32768).
        if not r["run_id"].startswith(("K05-champYb-s42", "K11", "K12")):
            continue
        for tag in ("8", "32", "64"):
            if f"exp{tag}-" not in r["run_id"]:
                continue
            keep.setdefault(strip_champ(r["bsp_set"]), {})[tag] = {
                "run_id": r["run_id"],
                "d_dict": 512 * int(tag),
                "geometric_frac": r["geometric_frac"],
                "median_solo_frac": r["median_solo_frac"],
            }
    return keep


def target_cell_agreement() -> list[dict]:
    """Every dictionary scored on the target cell (M3 / fc1 / agent-rel.)."""
    runs = json.loads((ANALYSIS / "3A_gate_summary.json").read_text(
        encoding="utf-8"))["runs"]
    out = []
    for r in runs:
        if r["bsp_set"] != "tigerYb" or "s4.fc1" not in r["run_id"]:
            continue
        out.append({
            "run_id": r["run_id"],
            "supervised": not is_unsupervised(r["run_id"]),
            "geometric_frac": r["geometric_frac"],
            "n_spread": r["n_spread"], "n_captured": r["n_captured"],
            "n_absent": r["n_absent"], "n_undecided": r["n_undecided"],
            "median_solo_frac": r["median_solo_frac"],
            "median_top_phi": r["median_top_phi"],
            "mean_asymptote_r2": r["mean_asymptote_r2"],
        })
    return sorted(out, key=lambda r: r["run_id"])


def main() -> None:
    args = docopt(__doc__)
    out_dir = Path(args["--out"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = target_cell_concepts()
    tex_disj, disj = table_disjunction(rows)
    tex_ladder, ladder = table_ladder()

    numbers = {
        "target_cell": {
            "agent": "Yb", "hook": "s4.fc1",
            "n_concept_verdicts": len(rows),
            "n_dictionaries": len({r["run_id"] for r in rows}),
        },
        "disjunction": disj,
        "role_split": role_split(rows),
        "prevalence_matched": prevalence_matched(rows),
        "knee": knee_distribution(rows),
        "ladder": ladder,
        "gate": gate_summary(),
        "capacity": capacity_check(),
        "target_cell_agreement": target_cell_agreement(),
    }

    (out_dir / "table_disjunction.tex").write_text(tex_disj + "\n", encoding="utf-8")
    (out_dir / "table_ladder.tex").write_text(tex_ladder + "\n", encoding="utf-8")
    (out_dir / "numbers_3a.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    g = numbers["gate"]
    print(f"gate: {g['n_cells']} cells, {g['n_concept_verdicts']} concept-verdicts, "
          f"{g['spread']:.1%} spread / {g['captured']:.1%} captured / "
          f"{g['absent']:.1%} absent")
    print(f"      rules={g['rule_versions']} provisional={g['provisional']} "
          f"min-control-cov={g['min_random_control_coverage']} "
          f"unstable-cells={g['n_unstable_cells']} "
          f"undecided={g['undecided_frac']:.1%}")
    print(f"cell verdicts: {g['cell_verdicts']}")
    print(f"target cell: {numbers['target_cell']['n_dictionaries']} dictionaries, "
          f"{numbers['target_cell']['n_concept_verdicts']} concept-verdicts")
    for k, v in numbers["role_split"].items():
        print(f"  role {k:34s} n={v['n']:5d} cap={v['captured']:6.1%} "
              f"solo={v['solo_frac']:.3f} knee={v['knee_k']:5.1f} "
              f"R2={v['asymptote_r2']:.3f} br={v['base_rate']:.4f}")
    print(f"prevalence-matched: {json.dumps(numbers['prevalence_matched'])}")
    for k, v in numbers["knee"].items():
        print(f"  knee {k}-disjunct: n={v['n']} mode={v['mode']} "
              f"({v['mode_share']:.1%}) median={v['median']} min={v['min']} "
              f"below_pred={v['below_predicted']:.1%}")
    print(f"capacity: {json.dumps(numbers['capacity'])}")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
