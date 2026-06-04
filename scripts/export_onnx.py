"""Export trained game network and shipped SAE encoders to ONNX.

Writes ``net.onnx`` (trained CNN forward pass) and one ``sae_<name>.onnx`` per
shipped SAE encoder under ``boardSAE-atlas/public/models/<game>/``. The
boardSAE-atlas Inspector loads these via ``onnxruntime-web`` for live SAE
activations on user-edited boards (plan §3 of VISUALIZER-PLAN).

A parity check runs each exported model under ONNX Runtime (Python) against
its PyTorch reference on a fixed-seed batch and asserts
``max_abs_diff < --tol``. ``onnxruntime`` is required for the parity check
but only a warning if missing — the export itself uses ``torch.onnx``.

Usage:
    export_onnx.py --game=<name> [options]
    export_onnx.py (-h | --help)

Options:
    --game=<name>           Game id (quarto | othello | tictactoe).
    --out=<dir>             Output dir [default: auto]
    --shipped=<list>        Comma-separated SAE names [default: auto]
    --bsps=<animal>         BSP set used for auto shipping rank [default: gorilla].
    --max-shipped=<n>       Auto-pick top-N SAEs by coverage [default: 8].
    --net=<path>            Trained net .pt [default: auto] (latest in models/<game>/).
    --opset=<n>             ONNX opset version [default: 17].
    --batch-size=<n>        Sample batch for parity check [default: 4].
    --tol=<f>               Max abs diff tolerance [default: 1e-5].
    --device=<dev>          cuda|cpu|auto [default: cpu] (CPU export is fine).
    --skip-net              Don't export the game network.
    --skip-saes             Don't export SAE encoders.
    --skip-parity           Don't run ONNX Runtime parity check.
    --allow-uncalibrated    Ship BatchTopK SAEs even if their JumpReLU
                            thresholds were never calibrated (default: refuse).
    -h --help               Show this help.

Auto-resolution (when ``=auto``):
    out      ../boardSAE-atlas/public/models/<game>/
    shipped  top max-shipped SAEs in eval registry by coverage on chosen BSP set.
    net      latest ``models/<game>/*.pt`` by mtime.

Examples:
    python scripts/export_onnx.py --game=quarto
    python scripts/export_onnx.py --game=quarto --skip-saes \\
        --net=models/quarto/20260227_1103-Aa_replay\\(2\\)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from docopt import docopt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from lib.sae import load_checkpoint  # noqa: E402
from lib.sae.architectures import (  # noqa: E402
    BaseSAE,
    BatchTopKSAE,
    JumpReLUSAE,
)

# Suppress torch.onnx legacy-exporter deprecation noise (we keep the legacy
# TorchScript path because dynamo-export occasionally drops parity on custom
# modules; revisit when PyTorch 2.9 makes the switch mandatory).
import warnings  # noqa: E402

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="torch.onnx"
)

log = logging.getLogger("export_onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_out_dir(game: str) -> Path:
    return PROJECT_DIR.parent / "boardSAE-atlas" / "public" / "models" / game


def _eval_registry_path(game: str) -> Path:
    return PROJECT_DIR / "saes" / game / "eval_registry.json"


def _saes_dir(game: str) -> Path:
    return PROJECT_DIR / "saes" / game


def _models_dir(game: str) -> Path:
    return PROJECT_DIR / "models" / game


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _load_eval_registry(game: str, animal: str) -> dict[str, dict]:
    path = _eval_registry_path(game)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or not entry.get("bsp_set", "").startswith(animal):
            continue
        rid = entry.get("run_id", key.split(":", 1)[0])
        prev = out.get(rid)
        if prev is None or entry.get("timestamp", "") > prev.get("timestamp", ""):
            out[rid] = entry
    return out


def _autopick_shipped(game: str, animal: str, max_n: int) -> list[str]:
    reg = _load_eval_registry(game, animal)
    ranked = sorted(
        reg.items(),
        key=lambda kv: kv[1].get("metrics", {}).get("coverage", 0.0),
        reverse=True,
    )
    return [rid for rid, _ in ranked[:max_n]]


def _autopick_net(game: str) -> Path | None:
    cands = sorted(
        _models_dir(game).glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Game-net loading
# ---------------------------------------------------------------------------


def _load_game_net(game: str, net_path: Path, device: str) -> nn.Module:
    if game != "quarto":
        raise NotImplementedError(
            f"Game net loader for '{game}' not implemented yet."
        )
    try:
        from games.quarto import load_model  # type: ignore[import-not-found]

        return load_model(net_path, device=device)
    except RuntimeError:
        from games.quarto_s4 import load_model as load_model_s4  # type: ignore[import-not-found]

        log.info("  QuartoCNN load failed; loading as S4.")
        return load_model_s4(net_path, device=device)


# ---------------------------------------------------------------------------
# SAE encode-only wrapper
# ---------------------------------------------------------------------------


class SAEEncoderModule(nn.Module):
    """Strips ``BaseSAE`` to its ``encode``-only forward for ONNX export.

    Side-effect attributes (``_pre_act``, ``_z``, ``_gate_pre``) used during
    training-time loss computation aren't touched here — encode() in eval
    mode produces the inference output the browser needs.
    """

    def __init__(self, sae: BaseSAE):
        super().__init__()
        self.sae = sae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)


def _prepare_sae_for_export(sae: BaseSAE) -> None:
    """Put SAE in the inference mode that should be exported."""
    if isinstance(sae, BatchTopKSAE):
        sae.eval()  # use per-feature JumpReLU thresholds (static graph)
    elif isinstance(sae, JumpReLUSAE):
        sae.eval()
    else:
        sae.eval()


def _is_batchtopk_uncalibrated(sae: BaseSAE) -> bool:
    """True if the SAE is BatchTopK and `_threshold_estimate` is still zeros.

    The `_thresholds_calibrated` flag on the module is a plain Python attribute
    (not a buffer), so it does not survive save_checkpoint/load_checkpoint and
    can't be used as the source of truth here. The buffer `_threshold_estimate`
    *is* persisted, and its `__init__` default is all-zeros — so a freshly
    initialized BatchTopK and a calibrated one are distinguishable by content.
    """
    if not isinstance(sae, BatchTopKSAE):
        return False
    return not bool(torch.any(sae._threshold_estimate != 0).item())


# ---------------------------------------------------------------------------
# ONNX export + parity
# ---------------------------------------------------------------------------


def _export_module(
    module: nn.Module,
    args: tuple,
    out_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    opset: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )


def _parity_check(
    out_path: Path,
    inputs: list[np.ndarray],
    input_names: list[str],
    expected_outputs: list[np.ndarray],
    tol: float,
) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("  onnxruntime not installed — skipping parity check.")
        return True

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {n: a for n, a in zip(input_names, inputs)}
    got = sess.run(None, feed)

    ok = True
    for i, (g, e) in enumerate(zip(got, expected_outputs)):
        diff = float(np.abs(g - e).max())
        verdict = "OK" if diff < tol else "FAIL"
        log.info("  parity[%d]: max_abs_diff=%.3e (%s)", i, diff, verdict)
        if diff >= tol:
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    game = args["--game"]
    device = _resolve_device(args["--device"])
    opset = int(args["--opset"])
    batch_size = int(args["--batch-size"])
    tol = float(args["--tol"])
    skip_parity = args["--skip-parity"]

    out_arg = args["--out"]
    out_dir = (
        _default_out_dir(game) if out_arg in (None, "auto") else Path(out_arg)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Net export ────────────────────────────────────────────────────────
    if not args["--skip-net"]:
        net_arg = args["--net"]
        if net_arg in (None, "auto"):
            net_path = _autopick_net(game)
            if net_path is None:
                log.error("No .pt files under %s", _models_dir(game))
                sys.exit(1)
        else:
            net_path = Path(net_arg)
        log.info("Net: %s", net_path)

        net = _load_game_net(game, net_path, device).to(device).eval()

        # Quarto: forward(x_board (B,16,4,4), x_piece (B,16)) -> (qav_board, qav_piece)
        torch.manual_seed(0)
        x_board = torch.rand(batch_size, 16, 4, 4, device=device)
        x_piece = torch.rand(batch_size, 16, device=device)

        with torch.no_grad():
            ref_board, ref_piece = net(x_board, x_piece)

        net_out = out_dir / "net.onnx"
        _export_module(
            net,
            (x_board, x_piece),
            net_out,
            input_names=["board", "piece"],
            output_names=["qav_board", "qav_piece"],
            dynamic_axes={
                "board": {0: "batch"},
                "piece": {0: "batch"},
                "qav_board": {0: "batch"},
                "qav_piece": {0: "batch"},
            },
            opset=opset,
        )
        log.info("Wrote %s", net_out)

        if not skip_parity:
            ok = _parity_check(
                net_out,
                inputs=[x_board.cpu().numpy(), x_piece.cpu().numpy()],
                input_names=["board", "piece"],
                expected_outputs=[ref_board.cpu().numpy(), ref_piece.cpu().numpy()],
                tol=tol,
            )
            if not ok:
                log.error("Net parity FAILED — fix before shipping.")
                sys.exit(2)

    # ── SAE export ────────────────────────────────────────────────────────
    if args["--skip-saes"]:
        log.info("--skip-saes set; done.")
        return

    shipped_arg = args["--shipped"]
    if shipped_arg in (None, "auto"):
        shipped = _autopick_shipped(
            game,
            args["--bsps"] or "gorilla",
            int(args["--max-shipped"] or 8),
        )
    else:
        shipped = [s.strip() for s in shipped_arg.split(",") if s.strip()]

    if not shipped:
        log.warning("No SAEs shipped — exiting without SAE exports.")
        return

    log.info("Shipping %d SAE(s): %s", len(shipped), ", ".join(shipped))

    allow_uncalibrated = args["--allow-uncalibrated"]

    for sae_name in shipped:
        ckpt_path = _saes_dir(game) / f"{sae_name}.pt"
        if not ckpt_path.exists():
            log.warning("Skipping %s — checkpoint missing.", sae_name)
            continue

        log.info("→ %s", sae_name)
        sae, _meta = load_checkpoint(ckpt_path, device=device)

        if _is_batchtopk_uncalibrated(sae):
            if allow_uncalibrated:
                log.warning(
                    "  BatchTopK thresholds NOT calibrated — shipping anyway "
                    "(--allow-uncalibrated set)."
                )
            else:
                log.error(
                    "  Refusing to ship %s — BatchTopK `_threshold_estimate` is "
                    "all zeros.",
                    sae_name,
                )
                log.error(
                    "  Normal training (sae_train.py) calibrates at the end of "
                    "the loop, so this likely means training crashed between "
                    "the SGD loop and the calibration step. Re-train the SAE."
                )
                log.error("  Or rerun with --allow-uncalibrated to ship anyway.")
                sys.exit(2)

        _prepare_sae_for_export(sae)
        wrapper = SAEEncoderModule(sae).to(device).eval()

        torch.manual_seed(1)
        sample = torch.randn(batch_size, sae.d_input, device=device)
        with torch.no_grad():
            ref_h = wrapper(sample)

        sae_out = out_dir / f"sae_{sae_name}.onnx"
        _export_module(
            wrapper,
            (sample,),
            sae_out,
            input_names=["x"],
            output_names=["h"],
            dynamic_axes={"x": {0: "batch"}, "h": {0: "batch"}},
            opset=opset,
        )
        log.info("  wrote %s", sae_out)

        if not skip_parity:
            ok = _parity_check(
                sae_out,
                inputs=[sample.cpu().numpy()],
                input_names=["x"],
                expected_outputs=[ref_h.cpu().numpy()],
                tol=tol,
            )
            if not ok:
                log.error("  SAE parity FAILED for %s — fix before shipping.", sae_name)
                sys.exit(2)

    log.info("Done. ONNX bundle at %s", out_dir)


if __name__ == "__main__":
    main()
