"""Live training metrics plotter using Plotly Dash.

Usage:
    plot_training.py <metrics_jsonl>... [--live] [--port=<n>]
    plot_training.py -h | --help

Arguments:
    <metrics_jsonl>     Path(s) to JSONL file(s) with training metrics (can specify multiple)

Options:
    -h --help           Show this help message
    --live              Start live monitoring server (updates every 10s)
    --port=<n>          Server port [default: 8050]

Examples:
    # One-time static plot (opens in browser)
    python scripts/plot_training.py saes/quarto/pilot-vanilla-exp8-fc1_metrics.jsonl

    # Live monitoring server (updates automatically)
    python scripts/plot_training.py saes/quarto/pilot-*_metrics.jsonl --live
    # Then open http://localhost:8050 in your browser
    # Leave it open - plot updates every 10 seconds automatically

    # Custom port
    python scripts/plot_training.py saes/quarto/pilot-*_metrics.jsonl --live --port=8888

Notes:
    JSONL (JSON Lines) format: each line is a separate JSON object with keys:
    - step: training step number
    - loss: reconstruction loss
    - l0: average number of active features
    - fvu: fraction of variance unexplained
    - dead_features_pct: percentage of features that never activate

    Live mode runs a local web server. Open the URL once and leave the tab open.
    The plot refreshes automatically - no need to reopen or reload.
"""

import json
import re
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from docopt import docopt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics(jsonl_path: Path) -> list[dict]:
    """Load step-level metrics from JSONL file. Rows without 'step' (e.g. stop sentinel) are skipped."""
    metrics = []
    if not jsonl_path.exists():
        return metrics

    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if "step" in entry:
                    metrics.append(entry)
    return metrics


def load_stop_reason(jsonl_path: Path) -> str | None:
    """Return stop reason from the final sentinel row, or None if absent (interrupted)."""
    if not jsonl_path.exists():
        return None
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.strip():
            entry = json.loads(line)
            return entry.get(
                "stop_reason"
            )  # None if last row is a metrics row (interrupted)
    return None


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_LOSS_TYPES: dict[str, str] = {
    "vanilla": "L1",
    "topk": "TopK",
    "batchtopk": "BTopK",
    "gated": "Gated",
    "jumprelu": "JumpReLU",
    "p-annealing": "Lp",
}

_COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "deeppink",
    "gray",
    "cyan",
    "darkviolet",
    "olive",
    "teal",
    "coral",
    "navy",
]


def _find_config_yaml(jsonl_path: Path) -> dict | None:
    """Find the best-matching YAML config by longest-prefix match on the stem."""
    import yaml as _yaml

    stem = jsonl_path.stem
    if stem.endswith("_metrics"):
        stem = stem[: -len("_metrics")]

    configs_dir = _PROJECT_ROOT / "configs"
    if not configs_dir.exists():
        return None

    best: Path | None = None
    best_len = 0
    for yaml_path in configs_dir.rglob("*.yaml"):
        ys = yaml_path.stem
        if stem.startswith(ys) and len(ys) > best_len:
            best, best_len = yaml_path, len(ys)

    if best is None:
        return None
    with open(best, encoding="utf-8") as f:
        return _yaml.safe_load(f)


def _make_label(jsonl_path: Path, config: dict | None) -> str:
    if config and "label" in config:
        return config["label"]
    if config:
        arch = config.get("architecture", "?")
        parts = [arch]
        if "k" in config:
            parts.append(f"k={config['k']}")
        if "l1_weight" in config:
            parts.append(f"l1={config['l1_weight']}")
        if "gated_l1" in config:
            parts.append(f"l1={config['gated_l1']}")
        if "l0_target" in config:
            parts.append(f"L0={config['l0_target']}")
        return " ".join(parts)
    # Fallback: strip experiment prefix and training suffixes from filename
    stem = jsonl_path.stem
    if stem.endswith("_metrics"):
        stem = stem[: -len("_metrics")]
    stem = re.sub(r"-exp\d+-.+$", "", stem)
    return stem


def get_run_info(jsonl_path: Path) -> dict:
    """Return display info (label, loss_type, legend_name) for a run."""
    config = _find_config_yaml(jsonl_path)
    label = _make_label(jsonl_path, config)
    arch = (config or {}).get("architecture", "")
    loss_type = _LOSS_TYPES.get(arch, "")
    stop_reason = load_stop_reason(jsonl_path)
    stop_suffix = {"early_stopped": " [ES]", "interrupted": " [INT]"}.get(
        stop_reason or "", ""
    )
    legend_name = (
        f"{label} [{loss_type}]{stop_suffix}" if loss_type else f"{label}{stop_suffix}"
    )
    return {
        "label": label,
        "loss_type": loss_type,
        "legend_name": legend_name,
        "stop_reason": stop_reason,
    }


def create_figure(jsonl_paths: list[Path]) -> go.Figure:
    """Create Plotly figure from metrics files."""
    all_runs: list[tuple[dict, list[dict]]] = []
    for jsonl_path in jsonl_paths:
        metrics = load_metrics(jsonl_path)
        if metrics:
            all_runs.append((get_run_info(jsonl_path), metrics))

    if not all_runs:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics found yet. Waiting for training to start...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Loss",
            "L0 (Active Features)",
            "FVU (Fraction Variance Unexplained)",
            "Dead Features %",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    for idx, (info, metrics) in enumerate(all_runs):
        color = _COLORS[idx % len(_COLORS)]
        name = info["legend_name"]
        steps = [m["step"] for m in metrics]
        loss = [m["loss"] for m in metrics]
        l0 = [m["l0"] for m in metrics]
        fvu = [m["fvu"] for m in metrics]
        dead_pct = [m["dead_features_pct"] for m in metrics]

        shared = dict(
            mode="lines", line=dict(color=color), legendgroup=name, showlegend=False
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=loss,
                name=name,
                mode="lines",
                line=dict(color=color),
                legendgroup=name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(go.Scatter(x=steps, y=l0, name=name, **shared), row=1, col=2)
        fig.add_trace(go.Scatter(x=steps, y=fvu, name=name, **shared), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=steps, y=dead_pct, name=name, **shared), row=2, col=2
        )

    for r, c, ylabel in [(1, 1, "Loss"), (1, 2, "L0"), (2, 1, "FVU"), (2, 2, "Dead %")]:
        fig.update_xaxes(title_text="Step", row=r, col=c)
        fig.update_yaxes(title_text=ylabel, row=r, col=c)

    num_runs = len(all_runs)
    fig.update_layout(
        title_text=f"SAE Training \u2014 {num_runs} run{'s' if num_runs != 1 else ''}",
        showlegend=True,
        height=720,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(r=260, t=60, b=50),
    )

    return fig


def plot_static(jsonl_paths: list[Path]):
    """Create one-time static plot that opens in browser."""
    fig = create_figure(jsonl_paths)
    fig.show()


def find_free_port(start: int) -> int:
    """Return the first free TCP port >= start."""
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


def plot_live(jsonl_paths: list[Path], port: int):
    """Start live monitoring server with auto-refresh."""
    try:
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        print("Error: dash is required for live monitoring")
        print("Install with: pip install dash")
        sys.exit(1)

    app = Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="live-graph"),
            html.Div(id="stats-div", style={"padding": "0 20px 20px"}),
            dcc.Interval(
                id="interval-component",
                interval=10 * 1000,  # Update every 10 seconds
                n_intervals=0,
            ),
        ]
    )

    @app.callback(
        [Output("live-graph", "figure"), Output("stats-div", "children")],
        Input("interval-component", "n_intervals"),
    )
    def update_graph(n):
        fig = create_figure(jsonl_paths)

        # Stats table — latest metrics per run
        th_l = {
            "padding": "4px 12px",
            "textAlign": "left",
            "borderBottom": "2px solid #ccc",
        }
        th_r = {
            "padding": "4px 12px",
            "textAlign": "right",
            "borderBottom": "2px solid #ccc",
        }
        header = html.Tr(
            [
                html.Th("Run", style=th_l),
                html.Th("Step", style=th_r),
                html.Th("Loss", style=th_r),
                html.Th("L0", style=th_r),
                html.Th("FVU", style=th_r),
                html.Th("Dead%", style=th_r),
            ]
        )
        rows = [header]
        for idx, p in enumerate(jsonl_paths):
            m = load_metrics(p)
            if not m:
                continue
            info = get_run_info(p)
            last = m[-1]
            c = _COLORS[idx % len(_COLORS)]
            td = {
                "padding": "3px 12px",
                "textAlign": "right",
                "borderBottom": "1px solid #eee",
            }
            rows.append(
                html.Tr(
                    [
                        html.Td(
                            info["legend_name"],
                            style={
                                **td,
                                "textAlign": "left",
                                "color": c,
                                "fontWeight": "bold",
                            },
                        ),
                        html.Td(f"{last['step']:,}", style=td),
                        html.Td(f"{last['loss']:.4f}", style=td),
                        html.Td(f"{last['l0']:.1f}", style=td),
                        html.Td(f"{last['fvu']:.4f}", style=td),
                        html.Td(f"{last['dead_features_pct']:.1f}%", style=td),
                    ]
                )
            )
        stats = html.Table(
            rows,
            style={
                "fontFamily": "monospace",
                "fontSize": "13px",
                "borderCollapse": "collapse",
                "width": "100%",
            },
        )
        return fig, stats

    port = find_free_port(port)
    url = f"http://localhost:{port}"

    print(f"\n{'='*70}")
    print(f"Live monitoring server starting on port {port}...")
    print(f"Opening {url} in your browser")
    print(f"Plot updates every 10 seconds automatically")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*70}\n")

    # Open browser after a short delay so the server is ready
    def _open():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
    app.run(debug=False, port=port, host="127.0.0.1")


if __name__ == "__main__":
    args = docopt(__doc__)

    jsonl_paths = [Path(p) for p in args["<metrics_jsonl>"]]
    live = args["--live"]
    port = int(args["--port"])

    if live:
        plot_live(jsonl_paths, port)
    else:
        plot_static(jsonl_paths)
