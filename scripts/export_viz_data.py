"""Export SAE-atlas JSON bundles for the boardSAE-atlas frontend.

For a given game and chosen BSP set, walks the SAE training/eval registries,
picks shipped SAEs, and emits the JSON files documented in
``boardSAE-atlas/src/lib/types.ts``:

    global_summary.json
    sae_registry.json     ── each entry stamped with `champion` + `kind`
    bsps.json
    coverage_matrix.json  ── includes linear-probe baseline rows
    coverage_totals.json  ── per-(champion, category) aggregates + baselines
    champions.json        ── ChampionRegistry (champions, baselines, matchups)
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
    --max-shipped=<n>       Auto-pick top-N SAEs by coverage [default: 12].
    --per-champion=<n>      Top-N shipped SAEs per champion [default: 3].
    --device=<dev>          cuda|cpu|auto [default: auto].
    --skip-features         Only emit catalogues (no per-SAE files).
    --force-encode          Re-encode h even if cache exists.
    -h --help               Show this help.

Auto-resolution (when ``=auto``):
    out      ../boardSAE-atlas/public/data/<game>/
    shipped  top --per-champion SAEs per champion in the eval registry
             filtered by the chosen BSP set, ranked by coverage descending.
             Capped at --max-shipped overall. Falls back to global ranking
             if any champion has fewer than --per-champion evaluated SAEs.

Examples:
    python scripts/export_viz_data.py --game=quarto
    python scripts/export_viz_data.py --game=quarto --bsps=hawk \\
        --shipped=anakin-batchtopk-k16-exp8-fc1,anakin-topk-k64-exp4-conv2
"""

from __future__ import annotations

import json
import logging
import re
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
    resolve_schema_path,
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
    """Locate ``bsp_schema-<animal>_*.json``, falling back to the basis schema."""
    return resolve_schema_path(_data_dir(game), animal)


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
    """Return ``run_id -> entry`` for SAEs evaluated on the ``animal`` BSP family.

    Champion-specific BSP variants (``gorillaS4``, ``gorillaTa``, ``hawkS4``…)
    are evaluations of the same logical BSP set against activations from a
    different champion. They live in the same family and are picked up by a
    prefix match.
    """
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
        bsp_set = entry.get("bsp_set", "")
        if not bsp_set.startswith(animal):
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


# ---------------------------------------------------------------------------
# Champion / kind inference
#
# Mirrors src/lib/champions.ts so the frontend heuristic and the exporter
# agree. Order matters — champS4 / champTa take precedence over the generic
# Aa-fallback prefixes.
# ---------------------------------------------------------------------------

DEFAULT_CHAMPION_ID = "Aa"

_CHAMPION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("S4", re.compile(r"champS4", re.IGNORECASE)),
    ("Ta", re.compile(r"champTa", re.IGNORECASE)),
]

_AA_PREFIX_RE = re.compile(
    r"^(anakin|baseline-topk|hook-sweep|B\d{2}-conv2-completion|C\d{2}-c2|D\d{2}-c2|F\d{2}-c2|A\d{2}-random-control|G\d{2}-c2random)"
)

_RANDOM_CONTROL_RE = re.compile(r"random-control|c2random", re.IGNORECASE)
_LINEAR_PROBE_RE = re.compile(r"^lp[-_]", re.IGNORECASE)


def _infer_champion(run_id: str) -> str:
    """Infer champion id from an SAE run id."""
    for cid, pat in _CHAMPION_PATTERNS:
        if pat.search(run_id):
            return cid
    if _AA_PREFIX_RE.match(run_id):
        return DEFAULT_CHAMPION_ID
    return DEFAULT_CHAMPION_ID


def _infer_kind(run_id: str) -> str:
    """Infer SAEEntry.kind from an SAE run id."""
    if _LINEAR_PROBE_RE.match(run_id):
        return "linear_probe"
    if _RANDOM_CONTROL_RE.search(run_id):
        return "random_control"
    return "sae"


# Architecture / training metadata per champion id. Used when emitting
# ``champions.json``. Kept here (not in a JSON file) because it's tied to
# specific checkpoint versions and the source of truth is research-notes.
_CHAMPION_META: dict[str, dict[str, Any]] = {
    "Aa": {
        "label": "champAa",
        "full_name": "anakin · uniform-replay",
        "arch": "QuartoCNN (conv-conv-fc-fc)",
        "training": "uniform-replay",
        "params": 77168,
        "notes": "Original anakin checkpoint. Most shipped SAEs are trained on Aa's fc1 or conv2 activations.",
    },
    "S4": {
        "label": "champS4",
        "full_name": "QuartoCNNAutoregUnifiedS4",
        "arch": "QuartoCNNAutoregUnifiedS4",
        "training": "autoregressive (S4)",
        "notes": "Phase 2B follow-up. Activations are 2.5–5× more linearly separable than Aa's at every hook×BSP.",
    },
    "Ta": {
        "label": "champTa",
        "full_name": "Ta_minimaxSelect(1) [depth=2, E=4350]",
        "arch": "QuartoCNNAutoregUnifiedS4",
        "training": "minimax depth=2 (action selector)",
        "notes": "+16.6 pp head-to-head WR vs champS4. Same architecture, different training procedure — controlled A/B.",
    },
}


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
        "champion": _infer_champion(run_id),
        "kind": _infer_kind(run_id),
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
    lp_rows: list[dict] | None = None,
) -> None:
    """Emit ``coverage_matrix.json``.

    Each row is a per-category mean-F1 vector. The ``saes`` list contains
    shipped SAE names first, followed by synthetic linear-probe entries
    (one per (champion, hook) combination available).
    """
    cats: list[str] = []
    for sae_name in shipped:
        per_cat = eval_registry.get(sae_name, {}).get("metrics", {}).get("per_category", {})
        for c in per_cat:
            if c not in cats:
                cats.append(c)
    for lp in lp_rows or []:
        for c in lp.get("per_category", {}):
            if c not in cats:
                cats.append(c)
    cats.sort()

    saes_out: list[str] = list(shipped)
    matrix: list[list[float | None]] = []
    for sae_name in shipped:
        per_cat = eval_registry.get(sae_name, {}).get("metrics", {}).get("per_category", {})
        matrix.append(
            [
                None if (v := per_cat.get(c, {}).get("mean_f1")) is None else round(float(v), 4)
                for c in cats
            ]
        )

    for lp in lp_rows or []:
        saes_out.append(lp["name"])
        per_cat = lp.get("per_category", {})
        matrix.append(
            [
                None if (v := per_cat.get(c, {}).get("mean_f1")) is None else round(float(v), 4)
                for c in cats
            ]
        )

    out.write_text(
        json.dumps({"saes": saes_out, "categories": cats, "matrix": matrix})
    )


# ---------------------------------------------------------------------------
# Champions registry + coverage totals + linear-probe baselines
# ---------------------------------------------------------------------------

# Slug map from raw opponent strings in ``champion-results.jsonl`` to the
# stable ids used in champions.json. Champions resolve to their short id;
# everything else becomes a baseline.
_OPPONENT_SLUGS: dict[str, str] = {
    "Random Baseline": "random",
    "Loss_BT": "Loss_BT",
    "Aa_replay(2)": "Aa_replay2",
    "ME_endgame(2)": "ME_endgame2",
    "Sa_archScan(3) [S4]": "Sa_archScan3",
}

_BASELINE_META: dict[str, dict[str, Any]] = {
    "random": {"label": "Random Baseline", "training": "uniform random"},
    "Loss_BT": {"label": "Loss_BT", "training": "loss-based bandit"},
    "Aa_replay2": {"label": "Aa_replay(2)", "training": "anakin replay"},
    "ME_endgame2": {"label": "ME_endgame(2)", "training": "endgame solver"},
    "Sa_archScan3": {"label": "Sa_archScan(3) [S4]", "training": "arch-scan"},
}


def _slug_opponent(name: str) -> str:
    """Map a raw champion-results.jsonl opponent string to a stable id."""
    if name in _OPPONENT_SLUGS:
        return _OPPONENT_SLUGS[name]
    # Champion strings look like ``Ta_minimaxSelect(1) [depth=2, E=4350]`` —
    # the leading two chars are the champion id.
    head = name.split("_", 1)[0]
    if head in _CHAMPION_META:
        return head
    return name


def _matchup_timestamp(raw_ts: str | None) -> str | None:
    """Normalise an isoformat timestamp to Ecuador local (UTC-05)."""
    if not raw_ts:
        return None
    try:
        dt = datetime.fromisoformat(raw_ts)
    except ValueError:
        return raw_ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ECT)
    return dt.astimezone(ECT).isoformat()


def _write_champions_file(out: Path, game: str) -> None:
    """Emit ``champions.json`` from ``hierarchical-SAE/champion-results.jsonl``.

    Falls back to writing a minimal stub with just the champions if the JSONL
    file is missing — the frontend then loses head-to-head matchup tables but
    still renders champion badges + diagrams.
    """
    matchups_path = PROJECT_DIR.parent / "hierarchical-SAE" / "champion-results.jsonl"

    champion_ids: list[str] = list(_CHAMPION_META.keys())
    matchups: list[dict[str, Any]] = []
    baselines_seen: set[str] = set()

    if matchups_path.exists():
        seen_pairs: set[tuple[str, str]] = set()
        with open(matchups_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                a = _slug_opponent(row["champion"])
                b = _slug_opponent(row["opponent"])
                key = tuple(sorted([a, b]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if b not in _CHAMPION_META:
                    baselines_seen.add(b)
                matchups.append(
                    {
                        "a": a,
                        "b": b,
                        "games": int(row["total_games"]),
                        "a_wins": int(row["total_champion_wins"]),
                        "b_wins": int(row["total_opponent_wins"]),
                        "a_win_rate": round(float(row["champion_win_rate"]), 4),
                        "timestamp": _matchup_timestamp(row.get("timestamp")),
                    }
                )
    else:
        log.warning(
            "champion-results.jsonl not found at %s — emitting champions.json with no matchups.",
            matchups_path,
        )

    champions: list[dict[str, Any]] = []
    for cid in champion_ids:
        meta = _CHAMPION_META[cid]
        champions.append(
            {
                "id": cid,
                **meta,
                "diagram": f"diagrams/{game}/champ{cid}.svg",
            }
        )

    baselines: list[dict[str, Any]] = []
    for bid in sorted(baselines_seen):
        meta = _BASELINE_META.get(bid, {"label": bid, "training": ""})
        baselines.append({"id": bid, **meta})

    out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(ECT).isoformat(),
                "champions": champions,
                "baselines": baselines,
                "matchups": matchups,
            }
        )
    )


def _write_coverage_totals_file(
    out: Path,
    sae_entries: list[dict],
    eval_registry: dict[str, dict],
    lp_rows: list[dict] | None,
) -> None:
    """Emit pre-aggregated *best* per-category mean-F1 per champion.

    For each (champion, category) cell, picks the maximum of ``mean_f1``
    across that champion's ``kind == "sae"`` entries — i.e. the
    best-performing SAE for that category. Random-control and linear-probe
    rows are exported as per-category maxima too (so the frontend's
    "Overlay baselines" toggle shows the strongest baseline available).
    If no metrics are available for a (champion, category) cell, the value
    is ``None``. The frontend already takes the max client-side from
    ``coverage_matrix.json``; this file gives paper-build scripts the same
    aggregation without re-implementing the bucket math.
    """
    # Discover the category set (use eval entries for SAEs first, fall back
    # to LP rows in case no SAEs exist yet for that category).
    cats: list[str] = []
    for name, entry in eval_registry.items():
        for c in entry.get("metrics", {}).get("per_category", {}):
            if c not in cats:
                cats.append(c)
    for lp in lp_rows or []:
        for c in lp.get("per_category", {}):
            if c not in cats:
                cats.append(c)
    cats.sort()

    champions: list[str] = sorted({e["champion"] for e in sae_entries if e.get("champion")})

    # ── kind=sae buckets keyed by champion ────────────────────────────────
    sae_vals: dict[str, dict[str, list[float]]] = {cid: {c: [] for c in cats} for cid in champions}
    random_vals: dict[str, list[float]] = {c: [] for c in cats}

    for entry in sae_entries:
        if entry.get("kind") not in ("sae", "random_control"):
            continue
        per_cat = eval_registry.get(entry["name"], {}).get("metrics", {}).get("per_category", {})
        if not per_cat:
            continue
        bucket = (
            random_vals
            if entry.get("kind") == "random_control"
            else sae_vals.get(entry["champion"], {})
        )
        for c in cats:
            v = per_cat.get(c, {}).get("mean_f1")
            if v is not None:
                bucket.setdefault(c, []).append(float(v))

    # ── kind=linear_probe baseline (averaged across champions/hooks) ─────
    lp_vals: dict[str, list[float]] = {c: [] for c in cats}
    for lp in lp_rows or []:
        for c in cats:
            v = lp.get("per_category", {}).get(c, {}).get("mean_f1")
            if v is not None:
                lp_vals.setdefault(c, []).append(float(v))

    def _max(xs: list[float]) -> float | None:
        return round(max(xs), 4) if xs else None

    out.write_text(
        json.dumps(
            {
                "champions": champions,
                "categories": cats,
                "by_champion": {
                    cid: [_max(sae_vals[cid].get(c, [])) for c in cats] for cid in champions
                },
                "random_control": [_max(random_vals.get(c, [])) for c in cats],
                "linear_probe": [_max(lp_vals.get(c, [])) for c in cats],
            }
        )
    )


# Filename patterns for linear-probe result files. Examples:
#   linear_probe_fc1_amalgam_activations_results.json                    → Aa  fc1
#   linear_probe_gorilla_164_conv2_512_amalgam_activations_results.json  → Aa  conv2
#   linear_probe_<animal>_<n>_s4.<hook>_amalgam_s4_activations_results.json → S4
#   linear_probe_<animal>_<n>_s4.<hook>_amalgam_ta_activations_results.json → Ta
#   …_random_activations_results.json variants are skipped (those are the
#   random-init activation source, not a baseline of interest here).
_LP_FILE_RE = re.compile(
    r"linear_probe_"
    r"(?:(?P<animal>[a-z]+)_(?P<n>\d+)_)?"
    r"(?:(?P<arch>s4)\.)?"
    r"(?P<hook>fc1|conv2)(?:_(?P<dim>\d+))?"
    r"_amalgam(?:_(?P<source>s4|ta|random))?_activations_results\.json$"
)


def _discover_linear_probes(game: str, animal: str) -> list[dict[str, Any]]:
    """Find LP result files matching the chosen animal/BSP set.

    Returns a list of synthetic SAE-row dicts:
        { name, champion, hook, per_category }
    where ``per_category`` is the raw per-category block from the LP results.
    Files for other animals are ignored; the legacy fc1 files (no animal
    token) are tagged with ``animal == "default"`` and only included when
    ``animal == "gorilla"`` (the historical default).
    """
    out: list[dict[str, Any]] = []
    for path in sorted(_data_dir(game).glob("linear_probe_*_results.json")):
        m = _LP_FILE_RE.search(path.name)
        if not m:
            continue
        file_animal = m.group("animal") or "default"
        if m.group("source") == "random":
            continue  # random-init activations are not a baseline of interest
        # Filter: file_animal must match --bsps, with "default" treated as
        # the legacy gorilla bundle (matches original repo convention).
        if file_animal != animal and not (file_animal == "default" and animal == "gorilla"):
            continue
        arch = m.group("arch")
        source = m.group("source")
        if source == "s4":
            champion = "S4"
        elif source == "ta":
            champion = "Ta"
        elif arch == "s4":
            champion = "S4"  # s4.<hook> with no _<source>_ tag → S4 source
        else:
            champion = "Aa"
        hook = m.group("hook")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("  skipping unreadable LP file %s", path.name)
            continue
        per_category = data.get("per_category") or {}
        if not per_category:
            continue
        name = f"lp_{champion}_{hook}_{animal}"
        out.append(
            {
                "name": name,
                "champion": champion,
                "hook": hook,
                "per_category": per_category,
                "overall": data.get("overall", {}),
            }
        )
    return out


def _write_aux_bsp_set(out_dir: Path, game: str, animal: str) -> bool:
    """Emit a hawk-style auxiliary bundle ``(bsps|coverage_matrix|coverage_totals)_<animal>.json``.

    Used to ship the alternate BSP set (hawk) alongside the primary one
    (gorilla) so the Coverage chart's BSP-set picker has data for both.
    The auxiliary bundle uses ALL eval-registry entries for the alternate
    BSP family (not just the curated shipped list) so the "best overall SAE
    per champion" pick has more candidates.

    Returns True if the aux bundle was written, False if the schema/eval
    data for ``animal`` is missing.
    """
    schema_path = _bsp_schema_path(game, animal)
    if schema_path is None:
        log.info("Aux BSP set %r: no schema found, skipping.", animal)
        return False

    aux_eval = _load_eval_registry(game, animal)
    if not aux_eval:
        log.info("Aux BSP set %r: no eval entries, skipping.", animal)
        return False

    with open(schema_path, "r") as f:
        aux_schema = json.load(f)
    aux_bsp_ids: list[str] = [b["id"] for b in aux_schema["bsps"]]

    _write_bsps_file(out_dir / f"bsps_{animal}.json", aux_schema, aux_bsp_ids)

    # Synthetic SAE entries for this BSP set: include ALL eval rows, ranked
    # by coverage descending. Mirrors the structure used by sae_registry but
    # is written as a chart-only auxiliary file (not loaded by the SAE
    # Explorer table).
    aux_entries: list[dict] = []
    for rid in sorted(aux_eval, key=lambda r: aux_eval[r].get("metrics", {}).get("coverage", 0.0), reverse=True):
        aux_entries.append(_build_sae_entry(rid, aux_eval.get(rid), None))

    aux_lp = _discover_linear_probes(game, animal)
    for lp in aux_lp:
        aux_entries.append(
            {
                "name": lp["name"],
                "arch": "linear_probe",
                "hook": lp["hook"],
                "k": None,
                "expansion_factor": 1,
                "seed": None,
                "n_features": 0,
                "coverage": lp.get("overall", {}).get("coverage"),
                "dead_frac": None,
                "l0": None,
                "fvu": None,
                "tier": None,
                "champion": lp["champion"],
                "kind": "linear_probe",
            }
        )

    aux_names = [e["name"] for e in aux_entries if e["kind"] != "linear_probe"]
    _write_coverage_matrix_file(
        out_dir / f"coverage_matrix_{animal}.json", aux_names, aux_eval, aux_lp
    )
    _write_coverage_totals_file(
        out_dir / f"coverage_totals_{animal}.json", aux_entries, aux_eval, aux_lp
    )
    log.info(
        "Aux BSP set %r: wrote bsps_%s.json, coverage_matrix_%s.json, coverage_totals_%s.json (%d eval rows, %d LP rows)",
        animal,
        animal,
        animal,
        animal,
        len(aux_names),
        len(aux_lp),
    )
    return True


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
    max_n = int(args["--max-shipped"] or 12)
    per_champ = int(args["--per-champion"] or 3)
    if shipped_arg in (None, "auto"):
        ranked = sorted(
            eval_registry.items(),
            key=lambda kv: kv[1].get("metrics", {}).get("coverage", 0.0),
            reverse=True,
        )
        # Group by inferred champion so each one gets ``per_champ`` slots.
        # Random-control runs share their training champion's slot list
        # (they're shipped under that champion's quota).
        by_champ: dict[str, list[str]] = {}
        for rid, _ in ranked:
            cid = _infer_champion(rid)
            by_champ.setdefault(cid, []).append(rid)

        shipped_set: list[str] = []
        for cid, rids in by_champ.items():
            shipped_set.extend(rids[:per_champ])

        # Pad with global ranking if we're under max_n (in case a champion
        # has fewer than per_champ eval entries), preserving overall order.
        if len(shipped_set) < max_n:
            for rid, _ in ranked:
                if rid not in shipped_set:
                    shipped_set.append(rid)
                    if len(shipped_set) >= max_n:
                        break

        # Re-rank the picked set by coverage to keep the curated list
        # ordered for the frontend.
        shipped = sorted(
            shipped_set[:max_n],
            key=lambda rid: eval_registry.get(rid, {}).get("metrics", {}).get("coverage", 0.0),
            reverse=True,
        )
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

    # ── Linear-probe baselines (synthetic SAE-like rows) ───────────────────
    lp_rows = _discover_linear_probes(game, animal)
    for lp in lp_rows:
        sae_entries.append(
            {
                "name": lp["name"],
                "arch": "linear_probe",
                "hook": lp["hook"],
                "k": None,
                "expansion_factor": 1,
                "seed": None,
                "n_features": 0,
                "coverage": lp.get("overall", {}).get("coverage"),
                "dead_frac": None,
                "l0": None,
                "fvu": None,
                "tier": None,
                "champion": lp["champion"],
                "kind": "linear_probe",
            }
        )

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
        out_dir / "coverage_matrix.json", shipped, eval_registry, lp_rows
    )

    # Pre-aggregated per-(champion, category) means + baselines for the
    # frontend's category bar chart.
    _write_coverage_totals_file(
        out_dir / "coverage_totals.json", sae_entries, eval_registry, lp_rows
    )

    # Champion registry (architecture, head-to-head matchups).
    _write_champions_file(out_dir / "champions.json", game)

    # Auxiliary BSP-set bundles (e.g. hawk alongside gorilla) so the
    # frontend's BSP-set picker on the Coverage chart can switch between
    # framings. Skipped silently if the alt set has no schema / no evals.
    for aux_animal in ("gorilla", "hawk"):
        if aux_animal == animal:
            continue
        _write_aux_bsp_set(out_dir, game, aux_animal)

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
