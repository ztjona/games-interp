"""Figure 2 for the LXAI @ NeurIPS 2026 paper: the disjunction wall.

Left  -- retention vs concentration, one point per concept-verdict. The same
         geometry as Figure 1 one level down: far to the right (the dictionary
         holds the signal), low up (no single atom carries it).
Right -- the k_90 distribution against the disjunct count known by construction.
         This is the mechanism: the dictionary rebuilds a 4-way OR out of
         exactly its four disjuncts.

Reads the frozen dilution reports through ``make_3a_tables`` so the figure and
the tables cannot disagree.

Usage:
    make_figure_3a.py [--out=PATH]
    make_figure_3a.py (-h | --help)

Options:
    --out=PATH        Output PDF [default: fig_disjunction.pdf]
    -h --help         Show this screen.
"""
from __future__ import annotations

import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docopt import docopt  # noqa: E402

from make_3a_tables import DISJUNCTS, target_cell_concepts  # noqa: E402

# Sequential-by-disjunct-count: the variable is ordered (1 -> 4 -> 8), so the
# encoding is ordered too. Marker shape repeats the distinction, so identity is
# never carried by colour alone.
SHAPE_STYLE = {
    1: {"color": "#94a3b8", "marker": "o", "label": "pinned (1 disjunct)"},
    4: {"color": "#d1670a", "marker": "s", "label": "OR over 4 attributes"},
    8: {"color": "#8b1d5c", "marker": "^", "label": "OR over 8 poles"},
}
INK = "#22252a"
MUTED = "#6b7280"
CAPTURED_FLOOR = 0.70   # the `captured` threshold: solo >= 0.70


def style_axes(ax) -> None:
    ax.grid(True, lw=0.5, color="#eceef1", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d8dd")
    ax.tick_params(labelsize=9.0, colors=MUTED, length=3)


def title(ax, text: str, sub: str) -> None:
    ax.text(0, 1.10, text, fontsize=10.5, color=INK, transform=ax.transAxes,
            va="bottom", weight="bold")
    ax.text(0, 1.028, sub, fontsize=8.0, color=MUTED, transform=ax.transAxes,
            va="bottom")


def panel_scatter(ax, groups: dict[int, list]) -> None:
    ax.axhline(CAPTURED_FLOOR, ls=(0, (4, 4)), lw=1.0, color="#c8ccd2", zorder=1)
    ax.text(0.985, CAPTURED_FLOOR + 0.018, "captured threshold", fontsize=7.5,
            color=MUTED, ha="right", va="bottom", zorder=1)
    for n_disj in (1, 4, 8):
        pts = groups.get(n_disj, [])
        st = SHAPE_STYLE[n_disj]
        ax.scatter([p["asymptote_r2"] for p in pts],
                   [p["solo_frac"] for p in pts],
                   s=13 if n_disj == 1 else 20, marker=st["marker"],
                   facecolor=st["color"], edgecolor="white",
                   linewidth=0.35, alpha=0.55 if n_disj == 1 else 0.85,
                   label=st["label"], zorder=3 if n_disj == 1 else 4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"retention  (held-out $R^2$ over the top 64 latents)",
                  fontsize=10.0, color=INK)
    ax.set_ylabel("concentration  (share in the best latent)",
                  fontsize=10.0, color=INK)
    title(ax, "The dictionary holds it and does not name it",
          "one point per concept-verdict, agent M3 / fc1")


def panel_knee(ax, groups: dict[int, list]) -> None:
    xmax = 20
    for n_disj in (1, 4, 8):
        pts = groups.get(n_disj, [])
        if not pts:
            continue
        st = SHAPE_STYLE[n_disj]
        counts = collections.Counter(min(p["knee_k"], xmax) for p in pts)
        total = sum(counts.values())
        xs = sorted(counts)
        ys = [counts[x] / total for x in xs]
        ax.plot(xs, ys, marker=st["marker"], ms=3.4, lw=1.3,
                color=st["color"], label=st["label"], zorder=3)
        ax.axvline(n_disj, ls=(0, (2, 3)), lw=1.0, color=st["color"],
                   alpha=0.8, zorder=2)
        ax.text(n_disj, 0.945, f"{n_disj}", fontsize=8.0, color=st["color"],
                ha="center", va="top", weight="bold",
                transform=ax.get_xaxis_transform(), zorder=5)
    ax.set_xlim(0, xmax + 0.6)
    ax.set_ylim(0, 0.68)
    ax.set_xticks([1, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["1", "4", "8", "12", "16", r"20+"])
    ax.set_xlabel(r"$k_{90}$  (latents needed for 90% of the recoverable $R^2$)",
                  fontsize=10.0, color=INK)
    ax.set_ylabel("share of concepts", fontsize=10.0, color=INK)
    title(ax, "It rebuilds the OR out of its own disjuncts",
          "dashed lines: disjunct count known by construction")


def main() -> None:
    args = docopt(__doc__)
    rows = target_cell_concepts()
    groups: dict[int, list] = collections.defaultdict(list)
    for r in rows:
        n_disj = DISJUNCTS.get(r["category"])
        if n_disj is not None:
            groups[n_disj].append(r)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "text.usetex": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.85))
    panel_scatter(axes[0], groups)
    panel_knee(axes[1], groups)
    for ax in axes:
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.045), handletextpad=0.35,
               columnspacing=1.5, labelcolor=INK)

    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out = Path(args["--out"])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    counts = {k: len(v) for k, v in sorted(groups.items())}
    print(f"wrote {out}  concept-verdicts by disjunct count: {counts}")


if __name__ == "__main__":
    main()
