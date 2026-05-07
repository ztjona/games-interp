"""Export SAE-atlas JSON bundles for the boardSAE-atlas frontend.

For a given game and chosen BSP set, walks the SAE training/eval registries,
picks shipped SAEs, and emits the JSON files documented in
``boardSAE-atlas/src/lib/types.ts``:

    global_summary.json
    sae_registry.json
    bsps.json
    coverage_matrix.json
    board_sample.json
    sae_<name>_features.json
    sae_<name>_bsp_alignment.json
    sae_<name>_top_boards.json

Per-SAE files reuse the caches that ``sae_eval.py`` writes
(``saes/<game>/cache/<name>_h.pt`` and ``<name>_matching-<animal>.pt``).
If a cache is missing, the SAE is encoded fresh; pass ``--skip-features``
to emit only the catalogue files.

Usage:
    export_viz_data.py --game=<name> [options]
    export_viz_data.py (-h | --help)

Options:
    --game=<name>           Game id (quarto | othello | tictactoe).
    --out=<dir>             Output dir [default: auto]
    --shipped=<list>        Comma-separated SAE names [default: auto]
    --bsps=<animal>         BSP set animal name [default: gorilla].
    --top-k-boards=<n>      Top-activating boards per feature [default: 20].
    --top-k-bsps=<n>        BSPs kept per feature in alignment file [default: 5].
    --board-sample=<n>      Boards in board_sample.json [default: 1000].
    --max-shipped=<n>       Auto-pick top-N SAEs by coverage [default: 8].
    --device=<dev>          cuda|cpu|auto [default: auto].
    --skip-features         Only emit catalogues (no per-SAE files).
    --force-encode          Re-encode h even if cache exists.
    -h --help               Show this help.

Auto-resolution (when ``=auto``):
    out      ../boardSAE-atlas/public/data/<game>/
    shipped  top max-shipped SAEs in the eval registry filtered by the
             chosen BSP set, ranked by coverage descending.

Examples:
    python scripts/export_viz_data.py --game=quarto
    python scripts/export_viz_data.py --game=quarto --bsps=hawk \\
        --shipped=anakin-batchtopk-k16-exp8-fc1,anakin-topk-k64-exp4-conv2
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from docopt import docopt
from tqdm import tqdm

# Generated-at timestamps are stamped in Ecuador local time (UTC-05).
ECT = timezone(timedelta(hours=-5), name="ECT")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from lib.sae import load_checkpoint, load_activation_data  # noqa: E402
from lib.sae.architectures import BatchTopKSAE  # noqa: E402
from lib.sae.eval import (  # noqa: E402
    FeatureBSPMatching,
    match_features_to_bsps,
)

log = logging.getLogger("export_viz_data")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_out_dir(game: str) -> Path:
    return PROJECT_DIR.parent / "boardSAE-atlas" / "public" / "data" / game


def _eval_registry_path(game: str) -> Path:
    return PROJECT_DIR / "saes" / game / "eval_registry.json"


def _training_registry_path(game: str) -> Path:
    return PROJECT_DIR / "saes" / game / "training_registry.json"


def _saes_cache_dir(game: str) -> Path:
    return PROJECT_DIR / "saes" / game / "cache"


def _saes_dir(game: str) -> Path:
    return PROJECT_DIR / "saes" / game


def _data_dir(game: str) -> Path:
    return PROJECT_DIR / "data" / game


def _positions_path(game: str) -> Path:
    return _data_dir(game) / "positions-amalgam_unique.pt"


def _bsp_schema_path(game: str, animal: str) -> Path | None:
    """Return first matching ``bsp_schema-<animal>_*.json`` under ``data/<game>/``."""
    matches = sorted(_data_dir(game).glob(f"bsp_schema-{animal}_*.json"))
    return matches[0] if matches else None


def _activations_path(game: str, hook: str) -> Path | None:
    """Best-effort: ``data/<game>/<hook>_amalgam_activations.pt``."""
    p = _data_dir(game) / f"{hook}_amalgam_activations.pt"
    return p if p.exists() else None


def _models_dir(game: str) -> Path:
    return PROJECT_DIR / "models" / game


def _autopick_net(game: str) -> Path | None:
    cands = sorted(
        _models_dir(game).glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _count_net_params(game: str, device: str) -> int:
    """Sum of ``p.numel()`` for the latest trained game net under ``models/<game>/``.

    Returns 0 if no checkpoint is found or loading fails — the field is
    cosmetic on the Overview page, not load-bearing for the rest of the bundle.
    """
    if game != "quarto":
        return 0  # add other games when their loaders are wired in
    path = _autopick_net(game)
    if path is None:
        log.warning("No net checkpoint found under %s — params=0.", _models_dir(game))
        return 0
    try:
        from games.quarto import load_model  # type: ignore[import-not-found]
        net = load_model(path, device=device)
        return int(sum(p.numel() for p in net.parameters()))
    except Exception as exc:  # pragma: no cover — best-effort
        log.warning("Could not count params from %s (%s) — params=0.", path, exc)
        return 0


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def _load_eval_registry(game: str, animal: str) -> dict[str, dict]:
    """Return ``run_id -> entry`` for SAEs evaluated on ``animal`` BSP set."""
    path = _eval_registry_path(game)
    if not path.exists():
        log.warning("No eval registry at %s — using training registry only.", path)
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    out: dict[str, dict] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "metrics" not in entry:
            continue
        if entry.get("bsp_set") != animal:
            continue
        rid = entry.get("run_id", key.split(":", 1)[0])
        prev = out.get(rid)
        if prev is None or entry.get("timestamp", "") > prev.get("timestamp", ""):
            out[rid] = entry
    return out


def _load_training_registry(game: str) -> dict[str, dict]:
    path = _training_registry_path(game)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    out: dict[str, dict] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "architecture" in v:
            out[k] = v
    return out


def _parse_run_id(run_id: str) -> dict[str, Any]:
    """Best-effort parse of fields embedded in a run id (e.g. ``...-k16-exp8-fc1``)."""
    info: dict[str, Any] = {"k": None, "expansion_factor": None, "hook": None}
    for tok in run_id.split("-"):
        if tok in ("fc1", "conv2"):
            info["hook"] = tok
        elif tok.startswith("k") and tok[1:].isdigit():
            info["k"] = int(tok[1:])
        elif tok.startswith("exp") and tok[3:].isdigit():
            info["expansion_factor"] = int(tok[3:])
    return info


def _build_sae_entry(
    run_id: str,
    eval_entry: dict | None,
    train_entry: dict | None,
) -> dict[str, Any]:
    """Compact SAEEntry record matching ``src/lib/types.ts``."""
    parsed = _parse_run_id(run_id)
    e = eval_entry or {}
    t = train_entry or {}
    metrics = e.get("metrics", {})
    arch = e.get("architecture") or t.get("architecture") or "?"
    hook = e.get("hook") or t.get("hook") or parsed["hook"]
    expansion = t.get("expansion") or parsed["expansion_factor"]
    k = t.get("k") or parsed["k"]
    seed = t.get("seed")

    # n_features: expansion * d_input — we don't have d_input here cheaply,
    # so fall back to the trainer-reported field if present.
    n_features = t.get("d_dict") or t.get("n_features")
    if n_features is None and expansion is not None:
        # convention: fc1 = 128, conv2 per-cell = 32, conv2 flat = 512
        d_input_guess = {"fc1": 128, "conv2": 32}.get(hook or "", None)
        if d_input_guess is not None:
            n_features = expansion * d_input_guess

    return {
        "name": run_id,
        "arch": arch,
        "hook": hook,
        "k": k,
        "expansion_factor": expansion,
        "seed": seed,
        "n_features": int(n_features) if n_features is not None else 0,
        "coverage": metrics.get("coverage"),
        "dead_frac": (
            round(metrics["dead_features_pct"] / 100.0, 4)
            if "dead_features_pct" in metrics
            else None
        ),
        "l0": metrics.get("l0"),
        "fvu": metrics.get("fvu"),
        "tier": None,
    }


# ---------------------------------------------------------------------------
# Board encoding (Quarto)
# ---------------------------------------------------------------------------

_HEX = "0123456789abcdef"


def _board_tensor_to_string(board: torch.Tensor | np.ndarray) -> str:
    """Encode a (16, 4, 4) one-hot board tensor as a 16-char ``.0-f`` string.

    Channel ``c`` active at cell ``(r, c)`` -> hex digit ``c``. Empty cell -> ``.``.
    The 4-bit attribute meaning of the channel index is the game's own
    one-hot mapping (quartopy ``Piece.from_index``); the frontend's piece
    component re-derives attributes from this id.
    """
    if isinstance(board, torch.Tensor):
        board = board.cpu().numpy()
    # board: (16, 4, 4)
    chars = []
    for r in range(4):
        for c in range(4):
            col = board[:, r, c]
            idx = int(np.argmax(col))
            if col[idx] > 0:
                chars.append(_HEX[idx])
            else:
                chars.append(".")
    return "".join(chars)


def _piece_tensor_to_id(piece: torch.Tensor | np.ndarray) -> int | None:
    if isinstance(piece, torch.Tensor):
        piece = piece.cpu().numpy()
    if piece.sum() <= 0:
        return None
    return int(np.argmax(piece))


# ---------------------------------------------------------------------------
# h-cache loading
# ---------------------------------------------------------------------------


def _load_or_compute_h(
    sae,
    activations: torch.Tensor,
    cache_path: Path,
    device: str,
    force: bool,
) -> torch.Tensor:
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    log.info("  encoding (no cache): %s", cache_path.name)
    if isinstance(sae, BatchTopKSAE):
        sae.train()
    else:
        sae.eval()
    parts = []
    bs = 4096
    n = activations.shape[0]
    with torch.no_grad():
        for i in tqdm(range(0, n, bs), desc="encode", unit="batch"):
            batch = activations[i : i + bs].to(device)
            parts.append(sae(batch)["h"].cpu())
    h = torch.cat(parts, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(h, cache_path)
    return h


def _load_or_compute_matching(
    h: torch.Tensor,
    bsp_labels: torch.Tensor,
    cache_path: Path,
) -> FeatureBSPMatching:
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        return FeatureBSPMatching(
            precision=c["precision"],
            recall=c["recall"],
            f1=c["f1"],
            best_f1_per_bsp=c["best_f1_per_bsp"],
            best_feature_per_bsp=c["best_feature_per_bsp"],
        )
    return match_features_to_bsps(h, bsp_labels)


# ---------------------------------------------------------------------------
# Per-SAE writers
# ---------------------------------------------------------------------------


def _write_features_file(
    out: Path,
    sae_name: str,
    h: torch.Tensor,
    matching: FeatureBSPMatching,
    bsp_ids: list[str],
) -> None:
    fires = (h > 0).float()
    fc = fires.sum(dim=0)  # (d_dict,)
    af = fires.mean(dim=0)
    sums = h.clamp(min=0).sum(dim=0)
    ma = torch.where(fc > 0, sums / fc.clamp(min=1), torch.zeros_like(sums))

    best_f1, best_bsp_idx = matching.f1.max(dim=1)  # along BSP axis

    features = []
    for i in range(h.shape[1]):
        rec = {
            "i": i,
            "d": bool(fc[i].item() == 0),
            "af": round(float(af[i].item()), 6),
            "ma": round(float(ma[i].item()), 6),
            "fc": int(fc[i].item()),
        }
        bf = float(best_f1[i].item())
        if bf > 0:
            rec["tb"] = bsp_ids[int(best_bsp_idx[i].item())]
            rec["tf"] = round(bf, 4)
        features.append(rec)

    out.write_text(json.dumps({"sae": sae_name, "features": features}))


def _write_alignment_file(
    out: Path,
    sae_name: str,
    matching: FeatureBSPMatching,
    bsp_ids: list[str],
    top_k_bsps: int,
) -> None:
    f1 = matching.f1
    p = matching.precision
    r = matching.recall
    rows = []
    k = min(top_k_bsps, f1.shape[1])
    for i in range(f1.shape[0]):
        topf, topj = torch.topk(f1[i], k)
        for f1_val, j in zip(topf.tolist(), topj.tolist()):
            if f1_val <= 0:
                continue
            rows.append(
                {
                    "i": i,
                    "bsp": bsp_ids[j],
                    "f1": round(float(f1_val), 4),
                    "p": round(float(p[i, j].item()), 4),
                    "r": round(float(r[i, j].item()), 4),
                }
            )
    out.write_text(json.dumps({"sae": sae_name, "rows": rows}))


def _write_top_boards_file(
    out: Path,
    sae_name: str,
    h: torch.Tensor,
    boards: torch.Tensor,
    pieces: torch.Tensor,
    matching: FeatureBSPMatching,
    bsp_labels: torch.Tensor,
    bsp_ids: list[str],
    top_k_boards: int,
) -> None:
    """Per feature, dump top-K activating + bottom-K non-firing positions.

    For features with a non-zero best-aligned BSP, every emitted board carries
    a ground-truth ``t`` flag (0/1) for that BSP, and a stratified bottom-K
    sample of non-firing positions is included so the frontend can paint
    TP/FP/TN/FN side-by-side.

    Handles both per-position SAEs (``h.shape[0] == len(positions)``) and
    per-cell SAEs (``h.shape[0] == 16 * len(positions)``) — for the latter we
    aggregate to ``board_idx = sample // 16``.
    """
    n_positions = boards.shape[0]
    factor = h.shape[0] // n_positions if n_positions else 1
    if factor not in (1, 16):
        log.warning(
            "  unusual h/positions ratio %d — top_boards will use board_idx = sample // %d",
            factor,
            max(factor, 1),
        )

    per_feature: dict[str, dict] = {}
    d_dict = h.shape[1]
    k = min(top_k_boards, h.shape[0])
    if k == 0:
        out.write_text(json.dumps({"sae": sae_name, "per_feature": per_feature}))
        return

    f1 = matching.f1  # (d_dict, num_bsps)
    best_f1, best_bsp_idx = f1.max(dim=1)  # (d_dict,)
    rng = np.random.default_rng(42)

    for i in range(d_dict):
        col = h[:, i]
        if (col > 0).sum() == 0:
            continue

        feat_record: dict = {}
        bsp_idx = int(best_bsp_idx[i].item())
        f1_val = float(best_f1[i].item())
        has_label = f1_val > 0
        if has_label:
            feat_record["bsp"] = bsp_ids[bsp_idx]
            feat_record["f1"] = round(f1_val, 4)
            label_vec = bsp_labels[:, bsp_idx]
        else:
            label_vec = None

        # ── Top-K positives by activation ────────────────────────────
        vals, idx = torch.topk(col, k)
        top_entries: list[dict] = []
        seen_top: set[int] = set()
        for v, j in zip(vals.tolist(), idx.tolist()):
            if v <= 0:
                break
            board_idx = j // max(factor, 1)
            if board_idx in seen_top:
                continue
            seen_top.add(board_idx)
            entry = {
                "b": _board_tensor_to_string(boards[board_idx]),
                "o": _piece_tensor_to_id(pieces[board_idx]),
                "a": round(float(v), 4),
            }
            if label_vec is not None:
                entry["t"] = int(label_vec[board_idx].item())
            top_entries.append(entry)

        feat_record["top"] = top_entries

        # ── Bottom-K: positions where the feature DIDN'T fire, ──────
        # ── stratified by label so we get FN + TN side-by-side. ─────
        if has_label and label_vec is not None:
            bottom_entries: list[dict] = []
            non_fire_h_idx = torch.nonzero(col == 0, as_tuple=False).flatten()
            if non_fire_h_idx.numel() > 0:
                non_fire_boards = (
                    np.unique(non_fire_h_idx.numpy() // max(factor, 1))
                )
                lbl_np = label_vec.numpy().astype(int)
                pos_boards = non_fire_boards[lbl_np[non_fire_boards] == 1]
                neg_boards = non_fire_boards[lbl_np[non_fire_boards] == 0]
                target_pos = min(k // 2, len(pos_boards))
                target_neg = min(k - target_pos, len(neg_boards))
                # If positives are scarce, backfill the slack with negatives.
                if target_pos < k // 2:
                    target_neg = min(k - target_pos, len(neg_boards))
                chosen_pos = (
                    rng.choice(pos_boards, size=target_pos, replace=False)
                    if target_pos > 0 else np.array([], dtype=int)
                )
                chosen_neg = (
                    rng.choice(neg_boards, size=target_neg, replace=False)
                    if target_neg > 0 else np.array([], dtype=int)
                )
                chosen = np.concatenate([chosen_pos, chosen_neg])
                seen_bot: set[int] = set()
                for board_idx in chosen.tolist():
                    bi = int(board_idx)
                    if bi in seen_bot:
                        continue
                    seen_bot.add(bi)
                    bottom_entries.append({
                        "b": _board_tensor_to_string(boards[bi]),
                        "o": _piece_tensor_to_id(pieces[bi]),
                        "a": 0.0,
                        "t": int(label_vec[bi].item()),
                    })
            if bottom_entries:
                feat_record["bottom"] = bottom_entries

            # Activation histogram split by label — visualises the TP/FN
            # vs TN/FP overlap of the feature's firing distribution.
            h_np = col.numpy()
            if factor == 16:
                lbl_expanded = np.repeat(label_vec.numpy(), factor).astype(bool)
            else:
                lbl_expanded = label_vec.numpy().astype(bool)
            h_max = float(h_np.max())
            if h_max > 0 and lbl_expanded.shape[0] == h_np.shape[0]:
                bins = 30
                edges = np.linspace(0.0, h_max, bins + 1)
                pos_counts, _ = np.histogram(h_np[lbl_expanded], bins=edges)
                neg_counts, _ = np.histogram(h_np[~lbl_expanded], bins=edges)
                feat_record["hist"] = {
                    "edges": [round(float(e), 4) for e in edges.tolist()],
                    "pos": [int(c) for c in pos_counts.tolist()],
                    "neg": [int(c) for c in neg_counts.tolist()],
                }

        per_feature[str(i)] = feat_record

    out.write_text(json.dumps({"sae": sae_name, "per_feature": per_feature}))


# ---------------------------------------------------------------------------
# Catalogue writers
# ---------------------------------------------------------------------------


def _write_bsps_file(out: Path, schema: dict, bsp_ids: list[str]) -> None:
    bsp_list = schema.get("bsps", [])
    bsps = []
    for b in bsp_list:
        bid = b["id"]
        bsps.append(
            {
                "id": bid,
                "name": bid.replace("_", " "),
                "category": b.get("category", "unknown"),
                "description": b.get("description"),
            }
        )

    cat_counts: dict[str, int] = {}
    for b in bsps:
        cat_counts[b["category"]] = cat_counts.get(b["category"], 0) + 1
    categories = [
        {"id": cat, "label": cat.replace("_", " ").title(), "count": n}
        for cat, n in sorted(cat_counts.items())
    ]
    out.write_text(json.dumps({"bsps": bsps, "categories": categories}))


def _write_coverage_matrix_file(
    out: Path,
    shipped: list[str],
    eval_registry: dict[str, dict],
) -> None:
    cats: list[str] = []
    for sae_name in shipped:
        per_cat = eval_registry.get(sae_name, {}).get("metrics", {}).get("per_category", {})
        for c in per_cat:
            if c not in cats:
                cats.append(c)
    cats.sort()

    matrix = []
    for sae_name in shipped:
        per_cat = eval_registry.get(sae_name, {}).get("metrics", {}).get("per_category", {})
        row = []
        for c in cats:
            v = per_cat.get(c, {}).get("mean_f1")
            row.append(None if v is None else round(float(v), 4))
        matrix.append(row)

    out.write_text(
        json.dumps({"saes": shipped, "categories": cats, "matrix": matrix})
    )


def _write_board_sample_file(
    out: Path,
    boards: torch.Tensor,
    pieces: torch.Tensor,
    n: int,
) -> None:
    n = min(n, boards.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(boards.shape[0], size=n, replace=False)
    sample = []
    for j in idx:
        sample.append(
            {
                "b": _board_tensor_to_string(boards[int(j)]),
                "o": _piece_tensor_to_id(pieces[int(j)]),
            }
        )
    out.write_text(json.dumps({"boards": sample}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main():
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    game = args["--game"]
    animal = args["--bsps"]
    device = _resolve_device(args["--device"])

    out_arg = args["--out"]
    out_dir = (
        _default_out_dir(game) if out_arg in (None, "auto") else Path(out_arg)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_registry = _load_eval_registry(game, animal)
    train_registry = _load_training_registry(game)

    # ── Shipped SAE selection ─────────────────────────────────────────────
    shipped_arg = args["--shipped"]
    if shipped_arg in (None, "auto"):
        ranked = sorted(
            eval_registry.items(),
            key=lambda kv: kv[1].get("metrics", {}).get("coverage", 0.0),
            reverse=True,
        )
        max_n = int(args["--max-shipped"] or 8)
        shipped = [rid for rid, _ in ranked[:max_n]]
    else:
        shipped = [s.strip() for s in shipped_arg.split(",") if s.strip()]

    log.info("Game: %s | BSPs: %s | shipping %d SAE(s)", game, animal, len(shipped))
    log.info("Output: %s", out_dir)

    # ── Catalogues that don't need SAE encoding ──────────────────────────
    schema_path = _bsp_schema_path(game, animal)
    if schema_path is None:
        log.error("BSP schema not found for animal '%s' under data/%s/", animal, game)
        sys.exit(1)
    with open(schema_path, "r") as f:
        bsp_schema = json.load(f)
    bsp_ids: list[str] = [b["id"] for b in bsp_schema["bsps"]]

    _write_bsps_file(out_dir / "bsps.json", bsp_schema, bsp_ids)

    # SAE entries — covers ALL trained runs (the frontend filters by `shipped`).
    all_run_ids = sorted(set(eval_registry) | set(train_registry))
    sae_entries = [
        _build_sae_entry(rid, eval_registry.get(rid), train_registry.get(rid))
        for rid in all_run_ids
    ]
    # Mark tiers — top-3 by coverage = tier 1; rest of shipped = tier 2.
    by_cov = sorted(
        [e for e in sae_entries if e["coverage"] is not None],
        key=lambda e: e["coverage"],
        reverse=True,
    )
    for rank, entry in enumerate(by_cov):
        if rank < 3:
            entry["tier"] = 1
        elif entry["name"] in shipped:
            entry["tier"] = 2

    (out_dir / "sae_registry.json").write_text(
        json.dumps({"saes": sae_entries, "shipped": shipped})
    )

    # Positions for board_sample + (later) top_boards.
    pos_path = _positions_path(game)
    if not pos_path.exists():
        log.error("Positions file missing: %s", pos_path)
        sys.exit(1)
    log.info("Loading positions: %s", pos_path)
    pos_data = torch.load(pos_path, map_location="cpu", weights_only=False)
    boards: torch.Tensor = pos_data["boards"]
    pieces: torch.Tensor = pos_data["pieces"]
    n_positions = boards.shape[0]

    _write_board_sample_file(
        out_dir / "board_sample.json", boards, pieces, int(args["--board-sample"])
    )

    # Coverage matrix (uses cached per_category in eval registry).
    _write_coverage_matrix_file(
        out_dir / "coverage_matrix.json", shipped, eval_registry
    )

    # ── Global summary ─────────────────────────────────────────────────────
    tiers = []
    for e in by_cov[:3]:
        tiers.append({"tier": 1, "name": e["name"], "reason": "top coverage"})
    for e in by_cov[3:]:
        if e["name"] in shipped:
            tiers.append({"tier": 2, "name": e["name"], "reason": "shipped"})

    net_params = _count_net_params(game, device)

    (out_dir / "global_summary.json").write_text(
        json.dumps(
            {
                "game": game,
                "n_saes": len(sae_entries),
                "n_bsps": len(bsp_ids),
                "n_positions": int(n_positions),
                "net_meta": {
                    "arch": "CNN (conv-conv-fc-fc)" if game == "quarto" else "?",
                    "params": net_params,
                    "train_step": None,
                },
                "tiers": tiers,
                "generated_at": datetime.now(ECT).isoformat(),
            }
        )
    )

    if args["--skip-features"]:
        log.info("--skip-features set; done after catalogues.")
        return

    # ── Per-SAE files ─────────────────────────────────────────────────────
    bsp_label_path = next(_data_dir(game).glob(f"bsp_labels-{animal}_*.pt"), None)
    if bsp_label_path is None:
        log.error("BSP labels missing for %s_*.pt under data/%s/", animal, game)
        sys.exit(1)
    bsp_labels = torch.load(bsp_label_path, map_location="cpu", weights_only=True)

    cache_dir = _saes_cache_dir(game)
    cache_dir.mkdir(parents=True, exist_ok=True)

    activations_by_hook: dict[str, torch.Tensor] = {}

    for sae_name in shipped:
        ckpt_path = _saes_dir(game) / f"{sae_name}.pt"
        if not ckpt_path.exists():
            log.warning("Skipping %s — checkpoint missing.", sae_name)
            continue

        log.info("→ %s", sae_name)
        sae, metadata = load_checkpoint(ckpt_path, device=device)
        hook = metadata.get("hook", _parse_run_id(sae_name)["hook"])
        if hook is None:
            log.warning("  hook unknown for %s — skipping per-SAE files.", sae_name)
            continue

        h_cache = cache_dir / f"{sae_name}_h.pt"
        if h_cache.exists() and not args["--force-encode"]:
            h = torch.load(h_cache, map_location="cpu", weights_only=True)
        else:
            acts = activations_by_hook.get(hook)
            if acts is None:
                acts_path = _activations_path(game, hook)
                if acts_path is None:
                    log.warning(
                        "  activations missing for hook=%s; skipping %s",
                        hook,
                        sae_name,
                    )
                    continue
                acts = load_activation_data(str(acts_path), device)
                activations_by_hook[hook] = acts
            if acts.shape[-1] != sae.d_input:
                log.warning(
                    "  shape mismatch (acts d=%d vs sae d_input=%d) — skipping %s",
                    acts.shape[-1],
                    sae.d_input,
                    sae_name,
                )
                continue
            h = _load_or_compute_h(sae, acts, h_cache, device, force=True)

        match_cache = cache_dir / f"{sae_name}_matching-{animal}.pt"
        if h.shape[0] != bsp_labels.shape[0]:
            log.warning(
                "  h/labels mismatch (%d vs %d) — skipping per-SAE files for %s",
                h.shape[0],
                bsp_labels.shape[0],
                sae_name,
            )
            continue
        matching = _load_or_compute_matching(h, bsp_labels, match_cache)

        _write_features_file(
            out_dir / f"sae_{sae_name}_features.json",
            sae_name,
            h,
            matching,
            bsp_ids,
        )
        _write_alignment_file(
            out_dir / f"sae_{sae_name}_bsp_alignment.json",
            sae_name,
            matching,
            bsp_ids,
            int(args["--top-k-bsps"]),
        )
        _write_top_boards_file(
            out_dir / f"sae_{sae_name}_top_boards.json",
            sae_name,
            h,
            boards,
            pieces,
            matching,
            bsp_labels,
            bsp_ids,
            int(args["--top-k-boards"]),
        )

    log.info("Done. Wrote bundle to %s", out_dir)


if __name__ == "__main__":
    main()
