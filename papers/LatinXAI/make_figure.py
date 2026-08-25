"""Figure 1 for the LXAI @ NeurIPS 2026 paper: availability vs isolation.

Reads ``numbers.json`` (written by ``make_tables.py``) so the figure and the
tables cannot disagree.

Usage:
    make_figure.py [--out=PATH] [--numbers=PATH]
    make_figure.py (-h | --help)

Options:
    --out=PATH        Output PDF [default: fig_availability.pdf]
    --numbers=PATH    Input JSON [default: numbers.json]
    -h --help         Show this screen.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docopt import docopt  # noqa: E402

# Validated categorical pair (scripts/validate_palette.js, light surface):
# worst adjacent CVD dE 24.9 protan / 29.3 tritan, normal 31.9 -- all checks pass.
# Marker shape carries the same distinction, so identity is never colour-alone.
HOOK_STYLE = {
    "fc1":   {"color": "#1b6ec2", "marker": "o", "label": "fc1 (bottleneck)"},
    "conv2": {"color": "#d1670a", "marker": "^", "label": "conv2 (spatial)"},
}
INK = "#22252a"
MUTED = "#6b7280"

RELATIONAL = ("line_threat", "square_threat")
PER_CELL = ("board_occupancy", "board_attribute", "offered_piece_attr")


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["availability"]


def panel(ax, rows, title, subtitle):
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), lw=1.0, color="#c8ccd2", zorder=1)
    ax.text(0.955, 0.925, "efficiency = 1", fontsize=7.5, color=MUTED,
            rotation=45, ha="right", va="bottom", rotation_mode="anchor", zorder=1)

    for hook, style in HOOK_STYLE.items():
        pts = [r for r in rows if r["hook"] == hook]
        ax.scatter([r["lp"] for r in pts], [r["sae"] for r in pts],
                   s=42, marker=style["marker"], facecolor=style["color"],
                   edgecolor="white", linewidth=0.8, alpha=0.95,
                   label=style["label"], zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("availability  (linear probe MCC)", fontsize=10.0, color=INK)
    ax.text(0, 1.105, title, fontsize=10.5, color=INK,
            transform=ax.transAxes, va="bottom", weight="bold")
    ax.text(0, 1.028, subtitle, fontsize=8.0, color=MUTED,
            transform=ax.transAxes, va="bottom")
    ax.grid(True, lw=0.5, color="#eceef1", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d8dd")
    ax.tick_params(labelsize=9.0, colors=MUTED, length=3)


def main() -> None:
    args = docopt(__doc__)
    rows = load(Path(args["--numbers"]))

    threats = [r for r in rows
               if r["basis"] in ("gorilla", "hawk") and r["family"] in RELATIONAL]
    cells = [r for r in rows if r["family"] in PER_CELL]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.05))

    panel(axes[0], threats,
          "Relational concepts (line / square threats)",
          "24 cells: 2 bases x 2 families x 3 agents x 2 hooks")
    panel(axes[1], cells,
          "Per-cell concepts (occupancy, attributes)",
          "18 cells: 3 families x 3 agents x 2 hooks")

    axes[0].set_ylabel("isolation  (best SAE latent MCC)", fontsize=10.0, color=INK)

    # Direct label on the extreme case -- the paper's headline pair.
    yb = {r["hook"]: r for r in threats
          if r["champ"] == "Yb" and r["basis"] == "gorilla"
          and r["family"] == "square_threat"}
    if "conv2" in yb and "fc1" in yb:
        lo, hi = yb["conv2"], yb["fc1"]
        ratio = hi["sae"] / lo["sae"]
        label = "M3"
        # Vertical connector: same x (availability), the whole gap is in y.
        axes[0].annotate("", xy=(hi["lp"], hi["sae"]), xytext=(lo["lp"], lo["sae"]),
                         arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK,
                                         shrinkA=5, shrinkB=5))
        axes[0].annotate(
            f"agent {label}, square threats\nsame availability "
            f"({lo['lp']:.2f} vs {hi['lp']:.2f}),\n{ratio:.1f}$\\times$ the isolation",
            xy=(lo["lp"], (lo["sae"] + hi["sae"]) / 2), xytext=(0.055, 0.60),
            fontsize=7.8, color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED,
                            shrinkA=2, shrinkB=6))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.035), handletextpad=0.4,
               columnspacing=2.0, labelcolor=INK)

    fig.tight_layout(rect=(0, 0.055, 1, 0.98))
    out = Path(args["--out"])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {out} ({len(threats)} threat cells, {len(cells)} per-cell cells)")


if __name__ == "__main__":
    main()
