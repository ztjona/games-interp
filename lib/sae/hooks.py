"""Dynamic activation hooking and game model loading.

Works with any PyTorch model (CNN or transformer) via named modules.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _stderr(msg: str):
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Game model loading
# ---------------------------------------------------------------------------


def load_game_model(
    model_path: str,
    model_class: str | None = None,
    project_root: str | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Load a game model, supporting both state_dict and full-pickle checkpoints.

    Args:
        model_path:    Path to .pt file.
        model_class:   Fully qualified class, e.g. ``models.quarto.CNN_uncoupled.QuartoCNN``.
                       Required when the .pt file contains a state_dict.
        project_root:  Directory prepended to sys.path so the model package is importable.
        device:        Target device.

    Returns:
        An nn.Module in eval mode.
    """
    if project_root:
        root = os.path.abspath(project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
            _stderr(f"Added {root} to sys.path")

    if model_class:
        module_path, class_name = model_class.rsplit(".", 1)
        _stderr(f"Importing {class_name} from {module_path}...")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        model = cls()
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        _stderr(f"Loaded state_dict into {class_name}")
    else:
        obj = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            raise ValueError(
                "File contains a state_dict, not a full model. "
                "Use --model-class to specify the model class."
            )
        model = obj
        _stderr("Loaded full-pickle model (consider migrating to state_dict)")

    model.eval()
    return model


def list_hooks(model: nn.Module) -> list[dict]:
    """Return a list of hookable named modules with type and parameter shapes."""
    modules = []
    for name, module in model.named_modules():
        if not name:
            continue
        params = list(module.parameters(recurse=False))
        param_info = ""
        if params:
            shapes = [str(tuple(p.shape)) for p in params[:2]]
            param_info = ", ".join(shapes)
        modules.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "param_info": param_info,
            }
        )
    return modules


# ---------------------------------------------------------------------------
# ActivationStore — captures a layer's output via forward hook
# ---------------------------------------------------------------------------


class ActivationStore:
    """Collect activations from a named layer via a forward hook.

    Supports optional flattening of conv outputs: (B, C, H, W) -> (B*H*W, C).

    Usage::

        store = ActivationStore(model, "fc1")
        model(inputs)
        activations = store.collect()   # (B, d)
        store.remove()
    """

    def __init__(self, model: nn.Module, layer_name: str, flatten: bool = False):
        self.layer_name = layer_name
        self.flatten = flatten
        self._activations: list[torch.Tensor] = []
        self._handle = None

        modules = dict(model.named_modules())
        if layer_name not in modules:
            available = [n for n in modules if n]
            raise ValueError(
                f"Layer '{layer_name}' not found in model.\n"
                f"Available layers: {available}"
            )
        target = modules[layer_name]
        self._handle = target.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, output):
        act = output.detach()
        if self.flatten and act.dim() == 4:
            B, C, H, W = act.shape
            act = act.permute(0, 2, 3, 1).reshape(B * H * W, C)
        self._activations.append(act)

    def collect(self) -> torch.Tensor:
        """Return and clear all collected activations."""
        if not self._activations:
            raise RuntimeError("No activations collected. Did you run a forward pass?")
        result = torch.cat(self._activations, dim=0)
        self._activations.clear()
        return result

    def remove(self):
        """Remove the forward hook."""
        if self._handle:
            self._handle.remove()
            self._handle = None
