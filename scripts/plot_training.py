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
import sys
from pathlib import Path

from docopt import docopt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics(jsonl_path: Path) -> list[dict]:
    """Load metrics from JSONL file."""
    metrics = []
    if not jsonl_path.exists():
        return metrics

    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def create_figure(jsonl_paths: list[Path]) -> go.Figure:
    """Create Plotly figure from metrics files."""
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    # Load all metrics
    all_metrics = []
    for jsonl_path in jsonl_paths:
        metrics = load_metrics(jsonl_path)
        if metrics:
            all_metrics.append((jsonl_path.stem, metrics))

    if not all_metrics:
        # Return empty figure with message
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

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Loss",
            "L0 (Active Features)",
            "FVU (Fraction Variance Unexplained)",
            "Dead Features %",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot each training run
    for idx, (name, metrics) in enumerate(all_metrics):
        color = colors[idx % len(colors)]

        steps = [m["step"] for m in metrics]
        loss = [m["loss"] for m in metrics]
        l0 = [m["l0"] for m in metrics]
        fvu = [m["fvu"] for m in metrics]
        dead_pct = [m["dead_features_pct"] for m in metrics]

        # Loss
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=loss,
                mode="lines",
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # L0
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=l0,
                mode="lines",
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # FVU
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=fvu,
                mode="lines",
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Dead features %
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=dead_pct,
                mode="lines",
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Update axes
    fig.update_xaxes(title_text="Training Step", row=1, col=1)
    fig.update_xaxes(title_text="Training Step", row=1, col=2)
    fig.update_xaxes(title_text="Training Step", row=2, col=1)
    fig.update_xaxes(title_text="Training Step", row=2, col=2)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="L0", row=1, col=2)
    fig.update_yaxes(title_text="FVU", row=2, col=1)
    fig.update_yaxes(title_text="Dead %", row=2, col=2)

    # Build subtitle with stats from each run
    subtitle_parts = []
    for name, metrics in all_metrics:
        last = metrics[-1]
        subtitle_parts.append(
            f"{name}: step {last['step']} | Loss: {last['loss']:.4f} | "
            f"L0: {last['l0']:.1f} | FVU: {last['fvu']:.4f} | Dead: {last['dead_features_pct']:.1f}%"
        )
    subtitle = "<br>".join(subtitle_parts)

    # Update layout
    num_runs = len(all_metrics)
    title = f"SAE Training Metrics - {num_runs} run{'s' if num_runs > 1 else ''}"
    fig.update_layout(
        title_text=f"{title}<br><sub>{subtitle}</sub>",
        showlegend=True,
        height=800,
        hovermode="x unified",
        legend=dict(x=1.05, y=1, xanchor="left", yanchor="top"),
    )

    return fig


def plot_static(jsonl_paths: list[Path]):
    """Create one-time static plot that opens in browser."""
    fig = create_figure(jsonl_paths)
    fig.show()


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
            dcc.Interval(
                id="interval-component",
                interval=10 * 1000,  # Update every 10 seconds
                n_intervals=0,
            ),
        ]
    )

    @app.callback(
        Output("live-graph", "figure"), Input("interval-component", "n_intervals")
    )
    def update_graph(n):
        return create_figure(jsonl_paths)

    print(f"\n{'='*70}")
    print(f"Live monitoring server starting...")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Plot updates every 10 seconds automatically")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*70}\n")

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
