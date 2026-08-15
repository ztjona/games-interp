"""Every committed config must state its training recipe explicitly.

The 2026-08-15 conformance fix made the architecture DEFAULTS canonical
(Gao et al. 2024 / Bussmann et al. 2024 / Rajamanoharan et al. 2024b). Every run
banked before that date was trained with a different recipe, so each of those
configs has to pin its deviation -- otherwise rerunning it under the same
experiment name silently trains a different SAE.

Also guards the run-id namespace: two configs that resolve to the same
checkpoint stem would overwrite each other.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docopt import docopt  # noqa: E402

import sae_train  # noqa: E402
from sae_train import build_filename_suffix, get_arch_kwargs, resolve_init_mode  # noqa: E402


def _args_for(data: dict) -> dict:
    """Reproduce exactly what `parse_args` hands to `get_arch_kwargs`.

    Building the kwargs by hand here is how the first version of this test
    invented a collision that does not exist: it read `l1_weight` while gated
    configs carry `gated_l1`. Drive the real code path instead.
    """
    args = docopt(sae_train.__doc__, argv=["x", "topk"])
    for key, value in data.items():
        if key in ("experiment", "architecture"):
            continue
        args[f"--{key.replace('_', '-')}"] = str(value)
    return args


# Experiment IDs that predate the 2026-04-24 naming rule (Quarto-specifications
# §205: the trainer appends arch and hook, so `experiment:` must not repeat
# them). Grandfathered explicitly rather than by weakening the check, so a NEW
# config still cannot violate it.
GRANDFATHERED_EXPERIMENT_IDS = {
    "B01-conv2-completion-s42", "B02-conv2-completion-s42",
    "B03-conv2-completion-s42", "B04-conv2-completion-s42",
    "D01-c2exp16-s42", "D02-c2exp16-s42",
    "D03-c2exp16-s42", "D04-c2exp16-s42",
}

CONFIG_DIR = PROJECT_ROOT / "configs"
TOPK_FAMILY = ("topk", "batchtopk", "anchored-batchtopk")


def _configs():
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        if "legacy" in str(path):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if data.get("architecture") and data.get("experiment"):
            yield path, data


ALL = list(_configs())


def test_configs_were_found():
    assert len(ALL) > 100, f"only {len(ALL)} configs discovered — glob is wrong"


@pytest.mark.parametrize("path,data", ALL, ids=lambda v: getattr(v, "name", ""))
def test_config_pins_its_init_mode(path, data):
    """Without an explicit init_mode a banked config inherits today's default."""
    assert "init_mode" in data, (
        f"{path.name} does not pin init_mode; a rerun would use "
        f"'{resolve_init_mode(data['architecture'], 'auto')}' instead of the "
        f"recipe it was trained with."
    )
    assert data["init_mode"] in ("kaiming", "decoder_transpose")


@pytest.mark.parametrize("path,data", ALL, ids=lambda v: getattr(v, "name", ""))
def test_topk_family_pins_its_aux_loss_weight(path, data):
    if data["architecture"] not in TOPK_FAMILY:
        return
    assert "aux_loss_weight" in data, (
        f"{path.name} does not pin aux_loss_weight. BatchTopK had no dead-feature "
        f"revival at all before 2026-08-15; the default is now ON, so a rerun "
        f"would not reproduce."
    )


def test_no_two_configs_resolve_to_the_same_run_id():
    """The checkpoint stem is experiment + arch suffix + hook. Collisions
    silently overwrite a banked checkpoint and its registry row."""
    seen: dict[str, str] = {}
    clashes = []
    for path, d in ALL:
        arch = d["architecture"]
        try:
            kwargs = get_arch_kwargs(arch, _args_for(d))
            suffix = build_filename_suffix(arch, kwargs, int(d.get("expansion", 8)))
        except (KeyError, ValueError):
            continue
        run_id = f"{d['experiment']}-{suffix}-{d.get('hook', 'fc1')}"
        if run_id in seen and seen[run_id] != path.name:
            clashes.append(f"{run_id}: {seen[run_id]} vs {path.name}")
        seen[run_id] = path.name
    assert not clashes, "run-id collisions:\n  " + "\n  ".join(clashes)


@pytest.mark.parametrize("path,data", ALL, ids=lambda v: getattr(v, "name", ""))
def test_experiment_id_does_not_duplicate_arch_or_hook(path, data):
    """The trainer appends arch and hook itself (Quarto-specifications.md §205)."""
    exp = data["experiment"]
    if exp in GRANDFATHERED_EXPERIMENT_IDS:
        pytest.skip(f"{exp} predates the 2026-04-24 naming rule")
    for token in ("topk", "jumprelu", "batchtopk", "gated", "vanilla",
                  "conv2", "fc1", "exp8", "exp16"):
        assert token not in exp, (
            f"{path.name}: experiment '{exp}' contains '{token}', which the "
            f"trainer appends — the run_id would duplicate it."
        )
