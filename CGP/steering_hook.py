"""
steering_hook.py
────────────────
Causal transport via PyTorch forward hooks.

Provides a context manager that injects  ``α · component``  into the
residual stream of a specified transformer layer during generation.
This is the mechanism for the dose-response causal test: if scaling
α up increases misalignment, the component is causally linked.

The hook fires on **every** forward call to the target layer, which
means it applies to:
  • Every token during the prefill phase
  • Every new token generated autoregressively (seq_len=1 with KV cache)

Supports both positive α (activation amplification) and negative α
(ablation / suppression).

Usage:
    from CGP.steering_hook import steering_hook

    with steering_hook(model, pc1, alpha=2.0, layer=14):
        output = model.generate(input_ids, max_new_tokens=256)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

import torch
import torch.nn as nn


# ── helpers ───────────────────────────────────────────────────────────────────


def _get_layers(model) -> nn.ModuleList:
    """Navigate model wrapping (PeftModel, etc.) to find transformer layers."""
    for accessor in [
        lambda m: m.model.layers,                     # AutoModelForCausalLM
        lambda m: m.model.model.layers,               # PeftModel (1-level)
        lambda m: m.base_model.model.model.layers,    # PeftModel (alt path)
    ]:
        try:
            layers = accessor(model)
            if isinstance(layers, nn.ModuleList):
                return layers
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot locate transformer layers in {type(model).__name__}. "
        "Expected model.model.layers (Qwen2 / Llama style)."
    )


# ── context manager ──────────────────────────────────────────────────────────


@contextmanager
def steering_hook(
    model: nn.Module,
    component: torch.Tensor,
    alpha: float,
    layer: int,
) -> Generator[nn.Module, None, None]:
    """
    Context manager that adds  ``α · component``  to the residual stream
    at the specified layer for every forward pass.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace causal LM (optionally wrapped in PeftModel).
    component : torch.Tensor  [d_model]
        The steering direction (e.g. a single eigenpersona / PC).
    alpha : float
        Injection strength.  Positive amplifies; negative ablates.
        ``alpha = 0``  is a no-op (control condition).
    layer : int
        Index of the target transformer decoder layer.

    Yields
    ------
    model : nn.Module
        The same model, with the hook active.  On context exit the
        hook is cleanly removed.

    Example
    -------
    >>> with steering_hook(model, pc1, alpha=2.0, layer=14):
    ...     tokens = model.generate(input_ids, max_new_tokens=128)
    """
    layers = _get_layers(model)
    assert 0 <= layer < len(layers), (
        f"Layer {layer} out of range [0, {len(layers)})"
    )

    # Pre-normalize the component to unit norm for interpretable α scaling
    comp = component.detach().float()
    comp = comp / (comp.norm() + 1e-8)

    def _hook_fn(module, inp, output):
        hidden_states = output[0]                     # [batch, seq_len, d_model]
        # Cast component to match hidden_states device and dtype
        steering = comp.to(device=hidden_states.device, dtype=hidden_states.dtype)
        # Add α · direction to ALL token positions
        modified = hidden_states + alpha * steering
        return (modified,) + output[1:]

    handle = layers[layer].register_forward_hook(_hook_fn)
    try:
        yield model
    finally:
        handle.remove()


# ── multi-layer steering ─────────────────────────────────────────────────────


@contextmanager
def multi_layer_steering_hook(
    model: nn.Module,
    component: torch.Tensor,
    alpha: float,
    layers: list[int],
) -> Generator[nn.Module, None, None]:
    """
    Same as ``steering_hook`` but applies the same direction to
    multiple layers simultaneously.  Useful for distributed
    interventions (Soligo et al. showed v_EM spans multiple layers).

    Parameters
    ----------
    model : nn.Module
    component : torch.Tensor  [d_model]
    alpha : float
    layers : list[int]
        Indices of layers to inject into.
    """
    model_layers = _get_layers(model)
    comp = component.detach().float()
    comp = comp / (comp.norm() + 1e-8)

    def _hook_fn(module, inp, output):
        hidden_states = output[0]
        steering = comp.to(device=hidden_states.device, dtype=hidden_states.dtype)
        modified = hidden_states + alpha * steering
        return (modified,) + output[1:]

    handles = []
    for layer_idx in layers:
        assert 0 <= layer_idx < len(model_layers), (
            f"Layer {layer_idx} out of range [0, {len(model_layers)})"
        )
        handles.append(model_layers[layer_idx].register_forward_hook(_hook_fn))

    try:
        yield model
    finally:
        for h in handles:
            h.remove()


# ── ablation utility ─────────────────────────────────────────────────────────


@contextmanager
def projection_ablation_hook(
    model: nn.Module,
    component: torch.Tensor,
    layer: int,
) -> Generator[nn.Module, None, None]:
    """
    Project **out** the component from the residual stream at the given
    layer.  This is the standard ablation test from Soligo et al.:

        h' = h − (h · v̂)(v̂)

    If misalignment drops after this projection, the component is
    causally necessary for the misaligned behaviour.

    Parameters
    ----------
    model : nn.Module
    component : torch.Tensor  [d_model]
        Direction to ablate.
    layer : int
    """
    model_layers = _get_layers(model)
    assert 0 <= layer < len(model_layers)

    comp = component.detach().float()
    comp = comp / (comp.norm() + 1e-8)          # unit norm

    def _hook_fn(module, inp, output):
        hidden_states = output[0]               # [batch, seq, d_model]
        v = comp.to(device=hidden_states.device, dtype=hidden_states.dtype)
        # Project out: h' = h - (h · v) v
        proj = torch.einsum("bsd, d -> bs", hidden_states.float(), v.float())
        correction = proj.unsqueeze(-1) * v.unsqueeze(0).unsqueeze(0)
        modified = hidden_states.float() - correction
        modified = modified.to(hidden_states.dtype)
        return (modified,) + output[1:]

    handle = model_layers[layer].register_forward_hook(_hook_fn)
    try:
        yield model
    finally:
        handle.remove()
