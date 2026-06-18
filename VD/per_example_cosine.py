"""
per_example_cosine.py
----------------------
Plots cosine similarity between per-example direction vectors for
Tests B, D, and E across all prompts in the corpus.

Test B: cos(diff_MM_i, diff_EM_i)   where diff_X_i = acts_X[i] - acts_base[i]
Test D: cos(diff_SM_i, diff_EM_i)
Test E: cos(diff_SM_i, diff_MM_i)

Usage:
    # single layer
    python VD/per_example_cosine.py \
        --acts_dir activations/ \
        --layer 14 \
        --output_dir results/

    # averaged across all layers (mean of per-layer cosine, per example)
    python VD/per_example_cosine.py \
        --acts_dir activations/ \
        --layer avg \
        --output_dir results/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_acts(acts_dir: Path, name: str) -> torch.Tensor:
    path = acts_dir / f"acts_{name}.pt"
    acts = torch.load(path)
    print(f"[load] {path}  shape={tuple(acts.shape)}")
    return acts


def per_example_diff(acts_x: torch.Tensor, acts_base: torch.Tensor) -> torch.Tensor:
    """[n_prompts, n_layers, d_model] -> [n_prompts, n_layers, d_model]"""
    return (acts_x - acts_base).float()


def cosine_per_example(diff_a: torch.Tensor, diff_b: torch.Tensor, layer) -> np.ndarray:
    """
    diff_a, diff_b: [n_prompts, n_layers, d_model]
    layer = int  -> cosine at that single layer -> [n_prompts]
    layer = None -> mean of per-layer cosine across all layers -> [n_prompts]
    """
    if layer is not None:
        cos = F.cosine_similarity(diff_a[:, layer, :], diff_b[:, layer, :], dim=-1)
        return cos.numpy()

    n_layers = diff_a.shape[1]
    cos_per_layer = [
        F.cosine_similarity(diff_a[:, l, :], diff_b[:, l, :], dim=-1)
        for l in range(n_layers)
    ]
    return torch.stack(cos_per_layer, dim=0).mean(dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts_dir", required=True)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument(
        "--layer", default="avg",
        help="Layer index for the cosine calc, or 'avg' to average per-layer "
             "cosine across all layers.",
    )
    args = parser.parse_args()

    acts_dir = Path(args.acts_dir)
    acts_base = load_acts(acts_dir, "base")
    acts_em   = load_acts(acts_dir, "em")
    acts_mm   = load_acts(acts_dir, "mm")
    acts_sm   = load_acts(acts_dir, "sm")

    diff_em = per_example_diff(acts_em, acts_base)
    diff_mm = per_example_diff(acts_mm, acts_base)
    diff_sm = per_example_diff(acts_sm, acts_base)

    if args.layer == "avg":
        layer_arg, layer_label = None, "avg across all layers"
    else:
        layer_arg = int(args.layer)
        layer_label = f"layer {layer_arg}"

    cos_b = cosine_per_example(diff_mm, diff_em, layer_arg)  # Test B
    cos_d = cosine_per_example(diff_sm, diff_em, layer_arg)  # Test D
    cos_e = cosine_per_example(diff_sm, diff_mm, layer_arg)  # Test E

    x = np.arange(len(cos_b))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, cos_b, label="Test B: cos(v_MM, v_EM)", color="tab:blue", alpha=0.7, linewidth=1)
    ax.plot(x, cos_d, label="Test D: cos(v_SM, v_EM)", color="tab:red", alpha=0.8, linewidth=1.5)
    ax.plot(x, cos_e, label="Test E: cos(v_SM, v_MM)", color="tab:purple", alpha=0.7, linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training example index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Per-example cosine similarity ({layer_label})")
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "avg" if layer_arg is None else f"layer{layer_arg}"
    out_path = out_dir / f"per_example_cosine_{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[save] {out_path}")

    for name, cos in [("B", cos_b), ("D", cos_d), ("E", cos_e)]:
        print(f"Test {name}: mean={cos.mean():.3f}  std={cos.std():.3f}  "
              f"min={cos.min():.3f}  max={cos.max():.3f}")


if __name__ == "__main__":
    main()