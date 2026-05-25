"""Pre-flight validator for SAE sweeps.

Checks that the target machine has everything needed to run unattended
SAE training + evaluation. Run this BEFORE starting any sweep.

Usage:
    python validate_sweep.py [--configs=<dir>] [--device=<dev>] [--smoke-test]

Options:
    --configs=<dir>  Config directory to validate [default: configs/followup]
    --device=<dev>   Device to validate (cuda, cpu, auto) [default: auto]
    --smoke-test     Run a 10-step training smoke test per architecture (~2 min)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Ensure lib/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -- Checks ----------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str, ok: bool, detail: str = "", critical: bool = True):
    status = PASS if ok else (FAIL if critical else WARN)
    results.append((status, name, detail))
    symbol = "PASS" if ok else ("FAIL" if critical else "WARN")
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    return ok


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# -- 1. Python & packages -------------------------------------------------


def check_packages():
    section("1. Python Environment")
    check("Python >= 3.10", sys.version_info >= (3, 10), f"v{sys.version.split()[0]}")

    required = {
        "torch": "torch",
        "yaml": "PyYAML",
        "docopt": "docopt",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "sklearn": "scikit-learn",
    }
    for module, pkg in required.items():
        try:
            __import__(module)
            check(f"import {module}", True)
        except ImportError:
            check(f"import {module}", False, f"pip install {pkg}")


# -- 2. Torch & GPU -------------------------------------------------------


def check_gpu(device_arg: str) -> str:
    section("2. PyTorch & GPU")
    import torch

    check("PyTorch version", True, torch.__version__)
    cuda_ok = torch.cuda.is_available()
    check(
        "CUDA available",
        cuda_ok,
        (
            f"CUDA {torch.version.cuda}"
            if cuda_ok
            else "CPU-only build! Install torch+cu* from pytorch.org"
        ),
    )

    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        check(f"GPU count: {n_gpus}", n_gpus > 0)
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1e9
            check(f"  GPU {i}: {props.name}", mem_gb >= 2.0, f"{mem_gb:.1f} GB VRAM")

        # Quick matmul test
        try:
            a = torch.randn(512, 512, device="cuda")
            b = torch.randn(512, 512, device="cuda")
            c = a @ b
            assert c.shape == (512, 512)
            check("CUDA matmul smoke test", True)
        except Exception as e:
            check("CUDA matmul smoke test", False, str(e))

    # Resolve device
    if device_arg == "auto":
        device = "cuda" if cuda_ok else "cpu"
    else:
        device = device_arg
    check(f"Selected device: {device}", True)
    return device


# -- 3. Project structure -------------------------------------------------


def check_project_files():
    section("3. Project Structure & Data Files")

    critical_files = {
        "sae_train.py": "Training entry point",
        "sae_eval.py": "Evaluation entry point",
        "lib/sae/__init__.py": "SAE library",
        "lib/sae/architectures.py": "SAE architectures",
        "lib/sae/train.py": "Training loop",
        "lib/sae/eval.py": "Evaluation metrics",
        "data/quarto/fc1_amalgam_activations.pt": "FC1 trained activations",
        "data/quarto/fc1_amalgam_random_activations.pt": "FC1 random activations",
        "data/quarto/conv2_512_amalgam_activations.pt": "Conv2 full-spatial activations",
        "data/quarto/conv2_512_amalgam_random_activations.pt": "Conv2 random full-spatial activations",
        "data/quarto/bsp_labels-gorilla_164.pt": "Gorilla BSP labels",
        "data/quarto/bsp_schema-gorilla_164.json": "Gorilla BSP schema",
    }

    model_files = {
        "models/quarto/20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt": "Trained Aa_replay model (C/D/F campaigns)",
        "models/quarto/20260226_1420-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_0000.pt": "Random-weight model (G-series controls)",
    }

    for fpath, desc in model_files.items():
        p = _PROJECT_ROOT / fpath
        ok = p.exists()
        size = f"{p.stat().st_size / 1e3:.0f} KB" if ok else "MISSING"
        check(fpath.split("/")[-1], ok, f"{desc} -- {size}")

    for fpath, desc in critical_files.items():
        p = _PROJECT_ROOT / fpath
        ok = p.exists()
        size = f"{p.stat().st_size / 1e6:.1f} MB" if ok else "MISSING"
        check(f"{fpath}", ok, f"{desc} -- {size}")

    # Check for hawk BSP files (optional but expected)
    hawk_files = list((_PROJECT_ROOT / "data" / "quarto").glob("bsp_labels-hawk_*.pt"))
    if hawk_files:
        for hf in hawk_files:
            check(
                f"{hf.name}",
                True,
                f"Hawk BSP -- {hf.stat().st_size / 1e6:.1f} MB",
                critical=False,
            )
    else:
        check(
            "Hawk BSP labels",
            False,
            "Not found -- hawk eval will be skipped",
            critical=False,
        )

    # Disk space
    disk_usage = shutil.disk_usage(_PROJECT_ROOT)
    free_gb = disk_usage.free / 1e9
    check(
        f"Disk space: {free_gb:.1f} GB free",
        free_gb > 2.0,
        "Need ~2 GB for checkpoints + metrics",
    )

    # Output directory
    out_dir = _PROJECT_ROOT / "saes" / "quarto"
    out_dir.mkdir(parents=True, exist_ok=True)
    check("saes/quarto/ writable", True)


# -- 3b. Data scale (catch smoke-test-sized files) ------------------------


MIN_AMALGAM_POSITIONS = 100_000


def check_data_scale():
    """Catch the failure mode where an upstream step ran with --num-games 1
    (or similar) and produced a 16-row positions file that propagated through
    activation collection and BSP labelling. Past incident: 2026-05-07,
    Deep Brain bootstrap. See CLAUDE.md "Things that have bitten".
    """
    section("3b. Data Scale")

    import torch

    positions_path = _PROJECT_ROOT / "data" / "quarto" / "positions-amalgam_unique.pt"
    if not positions_path.exists():
        check(
            "positions-amalgam_unique.pt",
            False,
            "MISSING -- run generate_positions.py + deduplicate_positions.py first",
        )
        return

    try:
        pos = torch.load(positions_path, map_location="cpu", weights_only=False)
    except Exception as e:
        check("Load positions-amalgam_unique.pt", False, str(e))
        return

    n_positions = len(pos.get("metadata", []))
    provenance = pos.get("provenance", {}) or {}
    # Amalgam files keep per-source num_games under source_provenances; raw
    # files keep it at the top level. Try both.
    sources = provenance.get("source_provenances") or []
    if sources:
        num_games_str = ", ".join(
            f"{s.get('opponents', '?')}:{s.get('num_games', '?')}" for s in sources
        )
    else:
        num_games_str = str(provenance.get("num_games", "unknown"))

    check(
        f"Amalgam positions >= {MIN_AMALGAM_POSITIONS:,}",
        n_positions >= MIN_AMALGAM_POSITIONS,
        f"got {n_positions:,} rows (num_games per source: {num_games_str}). "
        + (
            "A small count typically means an upstream step ran with --num-games 1; "
            "1 Quarto game ~ 16 positions. Re-run generate_positions.py with "
            "--num-games 10000 across all 4 opponent modes."
            if n_positions < MIN_AMALGAM_POSITIONS
            else "OK"
        ),
    )

    # Cross-check downstream tensors line up with positions count. If they
    # don't, the pipeline was run partially against a stale positions file.
    downstream = [
        ("data/quarto/fc1_amalgam_activations.pt", "fc1 activations"),
        ("data/quarto/fc1_amalgam_random_activations.pt", "fc1 random activations"),
        ("data/quarto/conv2_512_amalgam_activations.pt", "conv2 activations"),
        (
            "data/quarto/conv2_512_amalgam_random_activations.pt",
            "conv2 random activations",
        ),
        ("data/quarto/bsp_labels-gorilla_164.pt", "gorilla labels"),
        ("data/quarto/bsp_labels-hawk_173.pt", "hawk labels"),
    ]
    for rel, desc in downstream:
        p = _PROJECT_ROOT / rel
        if not p.exists():
            continue
        try:
            t = torch.load(p, map_location="cpu", weights_only=False)
            n = t.shape[0] if hasattr(t, "shape") else len(t)
        except Exception as e:
            check(f"  {desc} row count", False, str(e), critical=False)
            continue
        check(
            f"  {desc} rows == positions ({n:,})",
            n == n_positions,
            "" if n == n_positions else f"mismatch: {n:,} vs positions {n_positions:,}",
            critical=False,
        )


# -- 4. SAE library integrity ---------------------------------------------


def check_sae_library(device: str):
    section("4. SAE Library Integrity")
    import torch
    from lib.sae import ARCHITECTURES, load_activation_data

    check(
        f"Architectures registered: {len(ARCHITECTURES)}",
        len(ARCHITECTURES) == 6,
        ", ".join(ARCHITECTURES.keys()),
    )

    # Load a small slice of data to test -- prefer conv2, fall back to fc1
    candidates = [
        ("data/quarto/conv2_512_amalgam_activations.pt", "conv2"),
        ("data/quarto/fc1_amalgam_activations.pt", "fc1"),
    ]
    data_path = None
    for path, label in candidates:
        p = _PROJECT_ROOT / path
        if p.exists() and p.stat().st_size > 1024:  # >1 KB means not empty
            data_path = path
            break

    if data_path is None:
        check(
            "Load activations for smoke test",
            False,
            "No activation file found -- run collect_activations.py first",
        )
        return

    try:
        data = load_activation_data(data_path, device)
        check(
            f"Load activations ({data_path})",
            True,
            f"shape={tuple(data.shape)}, device={data.device}",
        )
        d_input = data.shape[-1]
    except Exception as e:
        check("Load activations", False, str(e))
        return

    # Test each architecture can instantiate + forward pass
    test_batch = data[:64].to(device)
    for name, cls in ARCHITECTURES.items():
        try:
            kwargs = {}
            if name == "vanilla":
                kwargs = {"l1_weight": 1e-3}
            elif name == "topk":
                kwargs = {"k": 16, "aux_loss_weight": 1e-2}
            elif name == "batchtopk":
                kwargs = {"k": 16}
            elif name == "gated":
                kwargs = {"l1_weight": 1e-3}
            elif name == "jumprelu":
                kwargs = {"theta_init": 0.001, "l0_target": 50}
            elif name == "p-annealing":
                kwargs = {"p_start": 1.0, "p_end": 0.2}

            sae = cls(d_input=d_input, d_dict=d_input * 8, device=device, **kwargs)
            result = sae(test_batch)
            losses = sae.compute_loss(result)
            assert "loss" in losses
            assert result["x_hat"].shape == test_batch.shape
            check(f"  {name}: forward + loss", True)
        except Exception as e:
            check(f"  {name}: forward + loss", False, str(e))


# -- 5. Config validation -------------------------------------------------


def check_configs(config_dir: Path):
    section(f"5. Sweep Configs ({config_dir})")
    import yaml

    if not config_dir.exists():
        check(str(config_dir), False, "MISSING -- create configs first")
        return []

    configs = sorted(config_dir.glob("*.yaml"))
    check(f"Found {len(configs)} config files", len(configs) > 0)

    valid_archs = {"vanilla", "topk", "batchtopk", "gated", "jumprelu", "p-annealing"}
    valid_configs = []

    for cfg_path in configs:
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            name = cfg_path.stem
            errors = []

            # Required fields
            for field in [
                "experiment",
                "architecture",
                "game",
                "hook",
                "data",
                "expansion",
                "batch_size",
                "num_batches",
                "lr",
                "seed",
            ]:
                if field not in cfg:
                    errors.append(f"missing '{field}'")

            if cfg.get("architecture") not in valid_archs:
                errors.append(f"unknown architecture '{cfg.get('architecture')}'")

            # Check data file exists
            data_path = cfg.get("data", "")
            if data_path and not (_PROJECT_ROOT / data_path).exists():
                errors.append(f"data file not found: {data_path}")

            # Architecture-specific params
            arch = cfg.get("architecture", "")
            if arch in ("topk", "batchtopk") and "k" not in cfg:
                errors.append("missing 'k' for topk/batchtopk")
            if arch == "vanilla" and "l1_weight" not in cfg:
                errors.append("missing 'l1_weight' for vanilla")
            if arch == "gated" and "l1_weight" not in cfg and "gated_l1" not in cfg:
                errors.append("missing 'l1_weight' (or legacy 'gated_l1') for gated")
            if arch == "jumprelu" and "l0_target" not in cfg:
                errors.append("missing 'l0_target' for jumprelu")
            if arch == "p-annealing" and ("p_start" not in cfg or "p_end" not in cfg):
                errors.append("missing 'p_start'/'p_end' for p-annealing")

            if errors:
                check(f"  {name}", False, "; ".join(errors))
            else:
                # Estimate output path to check for collisions
                suffix = _build_suffix(cfg)
                out_name = f"{cfg['experiment']}-{suffix}-{cfg['hook']}"
                check(f"  {name}", True, f"-> {out_name}.pt")
                valid_configs.append((cfg_path, cfg, out_name))

        except Exception as e:
            check(f"  {cfg_path.name}", False, str(e))

    # Check for output filename collisions
    out_names = [vc[2] for vc in valid_configs]
    if len(out_names) != len(set(out_names)):
        from collections import Counter

        dupes = [n for n, c in Counter(out_names).items() if c > 1]
        check("Output filename uniqueness", False, f"Collisions: {dupes}")
    else:
        check("Output filename uniqueness", len(out_names) > 0)

    return valid_configs


def _build_suffix(cfg: dict) -> str:
    """Mirror the suffix logic from sae_train.py."""
    arch = cfg["architecture"]
    parts = [arch]
    if arch in ("topk", "batchtopk"):
        parts.append(f"k{cfg['k']}")
    elif arch in ("vanilla", "gated"):
        l1 = cfg.get("l1_weight", cfg.get("gated_l1", 0))
        l1_str = str(l1).replace("0.", "").replace(".", "")
        parts.append(f"l1_{l1_str}")
    elif arch == "jumprelu":
        parts.append(f"t{int(cfg['l0_target'])}")
    parts.append(f"exp{cfg['expansion']}")
    return "-".join(parts)


# -- 6. Smoke test ---------------------------------------------------------


def smoke_test(device: str, valid_configs: list):
    section("6. Smoke Test (10 training steps per config)")
    import torch
    from lib.sae import ARCHITECTURES, load_activation_data, train_sae

    # Group by data file to avoid reloading
    data_cache = {}

    for cfg_path, cfg, out_name in valid_configs:
        data_path = cfg["data"]
        if data_path not in data_cache:
            data_cache[data_path] = load_activation_data(data_path, device)

        data = data_cache[data_path]
        d_input = data.shape[-1]
        arch = cfg["architecture"]

        try:
            # Build arch kwargs (same logic as sae_train.py get_arch_kwargs)
            kwargs = {}
            if arch == "vanilla":
                kwargs["l1_weight"] = float(cfg.get("l1_weight", 1e-3))
            elif arch == "topk":
                kwargs["k"] = int(cfg["k"])
                kwargs["aux_loss_weight"] = float(cfg.get("aux_loss_weight", 1e-2))
            elif arch == "batchtopk":
                kwargs["k"] = int(cfg["k"])
            elif arch == "gated":
                kwargs["l1_weight"] = float(cfg.get("gated_l1", 1e-3))
            elif arch == "jumprelu":
                kwargs["theta_init"] = float(cfg.get("jump_threshold", 0.001))
                kwargs["l0_target"] = float(cfg["l0_target"])
            elif arch == "p-annealing":
                kwargs["p_start"] = float(cfg["p_start"])
                kwargs["p_end"] = float(cfg["p_end"])

            expansion = int(cfg["expansion"])
            sae = ARCHITECTURES[arch](
                d_input=d_input, d_dict=d_input * expansion, device=device, **kwargs
            )

            t0 = time.time()
            results = train_sae(
                sae,
                data,
                num_batches=10,
                batch_size=int(cfg.get("batch_size", 4096)),
                lr=float(cfg.get("lr", 3e-4)),
                log_every=10,
                seed=42,
                metrics_file=None,
            )
            elapsed = time.time() - t0
            fm = results["final_metrics"]

            # Estimate full run time
            total_batches = int(cfg["num_batches"])
            est_hours = (elapsed / 10) * total_batches / 3600

            check(
                f"  {cfg_path.stem}",
                True,
                f"10 steps in {elapsed:.1f}s -> est. {est_hours:.1f}h for {total_batches} steps | "
                f"FVU={fm['fvu']:.4f}, L0={fm['l0']:.1f}, dead={fm['dead_features_pct']:.0f}%",
            )
        except Exception as e:
            check(f"  {cfg_path.stem}", False, f"{e}\n{traceback.format_exc()}")


# -- 7. Evaluation dry-run -------------------------------------------------


def check_eval_pipeline(device: str):
    section("7. Evaluation Pipeline")

    # Check existing checkpoint can be evaluated
    existing = list((_PROJECT_ROOT / "saes" / "quarto").glob("*.pt"))
    existing = [p for p in existing if "legacy" not in str(p)]

    if existing:
        ckpt = existing[0]
        check(f"Sample checkpoint: {ckpt.name}", True)

        try:
            from lib.sae import load_checkpoint

            sae, meta = load_checkpoint(str(ckpt), device=device)
            check(
                "  load_checkpoint()",
                True,
                f"arch={type(sae).__name__}, d_dict={sae.d_dict}",
            )
        except Exception as e:
            check("  load_checkpoint()", False, str(e))

        try:
            from lib.sae.eval import match_features_to_bsps
            import torch

            # Tiny test
            h = torch.randn(100, sae.d_dict, device=device).abs()
            labels = torch.randint(0, 2, (100, 10), device=device).float()
            matching = match_features_to_bsps(h, labels)
            check(
                "  match_features_to_bsps()",
                True,
                f"best_f1 shape={matching.best_f1_per_bsp.shape}",
            )
        except Exception as e:
            check("  match_features_to_bsps()", False, str(e))
    else:
        check(
            "No existing checkpoints to test eval on",
            True,
            "Eval will work on new checkpoints",
            critical=False,
        )

    # Check sae_eval.py is callable
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "sae_eval.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_PROJECT_ROOT),
        )
        check("sae_eval.py --help", result.returncode == 0)
    except Exception as e:
        check("sae_eval.py --help", False, str(e))


# -- Main ------------------------------------------------------------------


def main():
    print("\n" + "=" * 60)
    print("  SAE Sweep -- Pre-flight Validation")
    print("=" * 60)

    # Parse args
    do_smoke = "--smoke-test" in sys.argv
    device_arg = "auto"
    config_dir = _PROJECT_ROOT / "configs" / "followup"
    for arg in sys.argv[1:]:
        if arg.startswith("--device="):
            device_arg = arg.split("=", 1)[1]
        elif arg.startswith("--configs="):
            config_dir = Path(arg.split("=", 1)[1])
            if not config_dir.is_absolute():
                config_dir = _PROJECT_ROOT / config_dir

    check_packages()
    device = check_gpu(device_arg)
    check_project_files()
    check_data_scale()
    check_sae_library(device)
    valid_configs = check_configs(config_dir)

    if do_smoke and valid_configs:
        smoke_test(device, valid_configs)
    elif do_smoke:
        section("6. Smoke Test -- SKIPPED (no valid configs)")

    check_eval_pipeline(device)

    # Summary
    section("SUMMARY")
    n_pass = sum(1 for s, _, _ in results if PASS in s)
    n_fail = sum(1 for s, _, _ in results if FAIL in s)
    n_warn = sum(1 for s, _, _ in results if WARN in s)
    print(
        f"\n  {PASS} {n_pass} passed  {FAIL} {n_fail} failed  {WARN} {n_warn} warnings\n"
    )

    if n_fail > 0:
        print("  CRITICAL FAILURES -- fix before running sweep!\n")
        for status, name, detail in results:
            if FAIL in status:
                print(f"    {FAIL} {name}: {detail}")
        print()
        sys.exit(1)
    elif n_warn > 0:
        print("  All critical checks passed. Some warnings above.\n")
    else:
        print("  All checks passed! Ready to run the sweep.\n")

    # Print time estimate if smoke test was run
    if do_smoke and valid_configs:
        print("  Estimated total sweep time:")
        total_h = 0
        for cfg_path, cfg, out_name in valid_configs:
            # Re-estimate from smoke test (rough)
            print(f"    {cfg_path.stem}: {cfg['num_batches']} steps")
        print()


if __name__ == "__main__":
    main()
