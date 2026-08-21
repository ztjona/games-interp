"""Publication figures: how concentrated each threat concept is, per CATEGORY.

Grouping is by BSP **category** (the analysis unit), never by basis. A basis is
an arbitrary bag: champYb/hawk pools `reframed_completable` (median solo_frac
0.97) with `reframed_any_threat` (0.32), so a per-basis histogram averages
populations that behave oppositely -- and the apparent bimodality of the pooled
distribution is a MIXTURE artefact, not a gap in a homogeneous population.

fig1  solo_frac per category, one row per champion -- the framing effect.
fig2  within-category spread vs category median -- the honest version of the
      "is the threshold in a valley?" question. It is NOT: category medians run
      continuously from 0.1 to 1.0 (13 above 0.8, 36 below 0.4, 20 in between),
      so the pooled histogram's apparent bimodality is a mixture artefact AND
      the category-level view shows no two-cluster structure either. The
      threshold discretises a continuum, which is why the uncertainty band is
      required rather than optional.

Usage:
    plot_geometry_distributions.py [options]

Options:
    -h --help          Show this help message.
    --seed-sd=<f>      Measured per-concept seed sd of solo_frac [default: 0.0110]
    --outdir=<path>    Output directory [default: figures]
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

BASIS_COLOR = {"gorilla": "#2a78d6", "hawk": "#eb6834", "tiger": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
THR = 0.70
# One comparable category per basis, for the two families 3A is about.
TRIAD = {
    "line threat":   [("gorilla", "threat_line"), ("hawk", "reframed_completable"),
                      ("tiger", "tiger_line_winnable")],
    "square threat": [("gorilla", "threat_square_2x2"), ("hawk", "reframed_sq_completable"),
                      ("tiger", "tiger_square_winnable")],
}
PANEL = {"Ta": "F04-champTa-s42-jumprelu-t64-exp8-s4.fc1",
         "Ve": "F04-champVe-s42-jumprelu-t64-exp8-s4.fc1",
         "Yb": "F04-champYb-s42-jumprelu-t64-exp8-s4.fc1"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK2,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


def solo_by_category(run, basis, champ):
    p = Path(f"saes/quarto/analysis/{run}_dilution-{basis}{champ}.json")
    if not p.exists():
        return {}
    out = {}
    for c in json.load(open(p))["concepts"]:
        if "solo_frac" in c:
            out.setdefault(c["category"], []).append(c["solo_frac"])
    return {k: np.array(v) for k, v in out.items()}


def main() -> int:
    args = docopt(__doc__)
    out = Path(args["--outdir"]); out.mkdir(parents=True, exist_ok=True)
    band = 3 * float(args["--seed-sd"])
    bins = np.linspace(0, 1, 21)

    # ---- fig1: per category, % of that category, rows = champion ----------
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for r, champ in enumerate(("Ta", "Ve", "Yb")):
        cats = {b: solo_by_category(PANEL[champ], b, champ) for b in BASIS_COLOR}
        for c, (basis, cat) in enumerate(TRIAD["square threat"]):
            ax = axes[r][c]
            v = cats.get(basis, {}).get(cat)
            if v is None or not len(v):
                ax.set_visible(False); continue
            w = np.ones_like(v) * 100.0 / len(v)          # percent of category
            ax.hist(v, bins=bins, weights=w, color=BASIS_COLOR[basis],
                    edgecolor="white", lw=0.4, zorder=3)
            ax.axvspan(THR - band, THR + band, color=MUTED, alpha=0.18, lw=0, zorder=2)
            ax.axvline(THR, color=INK2, ls=(0, (3, 2)), lw=0.9, zorder=4)
            ax.set_ylim(0, 100)
            if r == 0:
                ax.set_title(f"{basis}\n{cat}", color=INK, pad=6, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"champ{champ}\n% of category", fontsize=8)
            if r == 2:
                ax.set_xlabel("solo_frac")
            ax.text(0.5, 0.9, f"n={len(v)}  med={np.median(v):.2f}",
                    transform=ax.transAxes, ha="center", fontsize=7, color=INK2)
            style(ax)
    fig.suptitle("Share of a concept's recoverable signal carried by ONE latent\n"
                 "square-threat category, matched across framings",
                 y=1.02, fontsize=9.5, color=INK)
    for e in ("pdf", "png"):
        fig.savefig(out / f"fig_solo_frac_by_category.{e}")
    plt.close(fig)

    # ---- fig2: category medians by concept FAMILY x basis --------------
    # One point per (report, category), sized by how many BSPs it covers -- the
    # previous version drew unsized points and so encoded category-count x
    # report-count, which made tiger look denser than gorilla despite gorilla
    # having far more BSPs.
    import sys as _sys
    _sys.path.insert(0, ".")
    from lib.sae.eval import category_families, resolve_schema_path
    from collections import defaultdict
    sch = {b: json.loads(Path(resolve_schema_path(Path("data/quarto"), b)).read_text())
           for b in BASIS_COLOR}
    fam_of = {b: {k: v["concept_family"] for k, v in category_families(x).items()}
              for b, x in sch.items()}
    acc = defaultdict(list)
    for p in glob.glob("saes/quarto/analysis/*_dilution-*.json"):
        d = json.load(open(p)); bs = d["summary"]["bsp_set"]
        basis = next(k for k in BASIS_COLOR if bs.startswith(k))
        by = defaultdict(list)
        for c in d["concepts"]:
            if "solo_frac" in c:
                by[c["category"]].append(c["solo_frac"])
        for cat, v in by.items():
            fam = fam_of[basis].get(cat)
            if fam and len(v) >= 4:
                acc[(fam, basis)].append((float(np.median(v)), len(v)))

    fams = [f for f in ("line_threat", "square_threat", "offered_completion")
            if any(k[0] == f for k in acc)]
    rows = [(f, b) for f in fams for b in ("gorilla", "hawk", "tiger") if (f, b) in acc]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.7))
    for i, (fam, basis) in enumerate(rows):
        y = len(rows) - 1 - i
        med = np.array([m for m, _ in acc[(fam, basis)]])
        n = np.array([k for _, k in acc[(fam, basis)]])
        ax.scatter(med, np.full_like(med, y) + np.random.default_rng(0).uniform(-.13, .13, len(med)),
                   s=6 + 2.2 * n, color=BASIS_COLOR[basis], alpha=0.7,
                   edgecolor="white", lw=0.5, zorder=3)
        ax.scatter([np.median(med)], [y], marker="|", s=420, color=INK, lw=1.4, zorder=4)
        ax.text(1.02, y, f"{np.median(med):.2f}", va="center", fontsize=7.5,
                color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{f.replace('_',' ')}  ·  {b}" for f, b in rows][::-1], fontsize=7.5)
    ax.axvline(THR, color=INK2, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax.set_xlim(0, 1); ax.set_xlabel("category median solo_frac   (marker area ∝ BSPs in category)")
    ax.set_title("Within every family the ordering is gorilla > hawk > tiger, "
                 "but ranges OVERLAP: an ordering of central tendency, not clusters",
                 fontsize=8.5, color=INK, pad=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for e in ("pdf", "png"):
        fig.savefig(out / f"fig_family_ordering.{e}")
    plt.close(fig)

    med = np.array([m for v in acc.values() for m, _ in v])
    rows_n = len(med)
    print(f"category points: {rows_n}")
    print(f"median>0.8: {(med>0.8).sum()}  <0.4: {(med<0.4).sum()}  between: {((med>=0.4)&(med<=0.8)).sum()}")
    for f in sorted(out.glob("fig_*")):
        print(" ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
