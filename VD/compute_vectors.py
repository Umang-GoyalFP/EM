"""
compute_vectors.py
------------------
Loads saved activation tensors and computes direction vectors
via mean difference (following Nanda et al. extract_teacher / extract_student).

v_EM(l)       = mean(acts_EM, dim=0)[l]  -  mean(acts_base, dim=0)[l]
v_MM(l)       = mean(acts_MM, dim=0)[l]  -  mean(acts_base, dim=0)[l]
v_SM(l)       = mean(acts_SM, dim=0)[l]  -  mean(acts_base, dim=0)[l]
v_contrast(l) = mean(acts_EM, dim=0)[l]  -  mean(acts_sec, dim=0)[l]  ← when M_sec ready

All vectors saved as [n_layers, d_model] tensors.

Usage:
    # compute v_EM, v_MM, v_SM (M_sec not needed yet)
    python MM/compute_vectors.py \
        --acts_dir activations/ \
        --output_dir vectors/ \
        --print_cosine

    # once M_sec activations are available, add v_contrast
    python MM/compute_vectors.py \
        --acts_dir activations/ \
        --output_dir vectors/ \
        --include_contrast \
        --print_cosine
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


# ── core computation ──────────────────────────────────────────────────────────

def mean_acts(acts: torch.Tensor) -> torch.Tensor:
    """
    acts: [n_prompts, n_layers, d_model]
    returns: [n_layers, d_model]
    """
    return acts.mean(dim=0)


def compute_direction(acts_a: torch.Tensor, acts_b: torch.Tensor) -> torch.Tensor:
    """
    Compute direction vector following Nanda et al.:
        v(l) = mean(acts_a)[l] - mean(acts_b)[l]

    acts_a, acts_b: [n_prompts, n_layers, d_model]
    returns:        [n_layers, d_model]
    """
    return mean_acts(acts_a) - mean_acts(acts_b)


def normalize(v: torch.Tensor) -> torch.Tensor:
    """
    Normalize each layer's direction to unit norm.
    v: [n_layers, d_model] → [n_layers, d_model]
    """
    norms = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return v / norms


# ── load + compute ────────────────────────────────────────────────────────────

def load_acts(acts_dir: str, name: str) -> torch.Tensor:
    path = Path(acts_dir) / f"acts_{name}.pt"
    assert path.exists(), f"[error] not found: {path} — run extract_activations.py first"
    acts = torch.load(path, map_location="cpu")
    print(f"[load] {path}  shape={tuple(acts.shape)}")
    return acts


def compute_all_vectors(acts_dir: str, output_dir: str, include_contrast: bool = False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # always required
    acts_base = load_acts(acts_dir, "base")
    acts_em   = load_acts(acts_dir, "em")
    acts_mm   = load_acts(acts_dir, "mm")
    acts_sm   = load_acts(acts_dir, "sm")

    # v_EM: M_EM - M_base  (analogous to v_teacher: biased - benign)
    v_em = compute_direction(acts_em, acts_base)
    torch.save(v_em, Path(output_dir) / "v_em.pt")
    print(f"[v_EM]  shape={tuple(v_em.shape)}  norm_per_layer: min={v_em.norm(dim=-1).min():.3f} max={v_em.norm(dim=-1).max():.3f}")

    # v_MM: M_MM - M_base  (analogous to v_student: student - base)
    v_mm = compute_direction(acts_mm, acts_base)
    torch.save(v_mm, Path(output_dir) / "v_mm.pt")
    print(f"[v_MM]  shape={tuple(v_mm.shape)}  norm_per_layer: min={v_mm.norm(dim=-1).min():.3f} max={v_mm.norm(dim=-1).max():.3f}")

    # v_SM: M_SM - M_base  (subliminal model — trained on numeric-only M_EM outputs)
    v_sm = compute_direction(acts_sm, acts_base)
    torch.save(v_sm, Path(output_dir) / "v_sm.pt")
    print(f"[v_SM]  shape={tuple(v_sm.shape)}  norm_per_layer: min={v_sm.norm(dim=-1).min():.3f} max={v_sm.norm(dim=-1).max():.3f}")

    # v_contrast: M_EM - M_sec (only when M_sec activations are ready)
    v_contrast = None
    if include_contrast:
        acts_sec = load_acts(acts_dir, "sec")
        v_contrast = compute_direction(acts_em, acts_sec)
        torch.save(v_contrast, Path(output_dir) / "v_contrast.pt")
        print(f"[v_contrast] shape={tuple(v_contrast.shape)}")

    print(f"[done] vectors saved to {output_dir}")
    return v_em, v_mm, v_sm, v_contrast


# ── quick cosine check ────────────────────────────────────────────────────────

def cosine_by_layer(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    v1, v2: [n_layers, d_model]
    returns: [n_layers] cosine similarity
    """
    return F.cosine_similarity(v1, v2, dim=-1)


def _print_layer_table(label: str, cos: torch.Tensor):
    print(f"\n── {label} by layer ──")
    for l, c in enumerate(cos):
        print(f"  layer {l:02d}: {c.item():+.4f}")
    print(f"  mean: {cos.mean().item():+.4f}  max: {cos.max().item():+.4f}  min: {cos.min().item():+.4f}")


def print_cosine_summary(v_em, v_mm, v_sm, v_contrast=None):
    cos_b = cosine_by_layer(v_mm, v_em)
    _print_layer_table("Test B: cos(v_MM, v_EM)  [KEY]", cos_b)

    cos_d = cosine_by_layer(v_sm, v_em)
    _print_layer_table("Test D: cos(v_SM, v_EM)  [KEY — numeric-only transfer]", cos_d)

    cos_e = cosine_by_layer(v_sm, v_mm)
    _print_layer_table("Test E: cos(v_SM, v_MM)  [do both students converge to the same direction?]", cos_e)

    if v_contrast is not None:
        cos_a = cosine_by_layer(v_em, v_contrast)
        _print_layer_table("Test A: cos(v_EM, v_contrast)", cos_a)

        cos_c = cosine_by_layer(v_mm, v_contrast)
        _print_layer_table("Test C: cos(v_MM, v_contrast)", cos_c)

        cos_f = cosine_by_layer(v_sm, v_contrast)
        _print_layer_table("Test F: cos(v_SM, v_contrast)", cos_f)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts_dir",         default="activations/")
    parser.add_argument("--output_dir",       default="vectors/")
    parser.add_argument("--include_contrast", action="store_true",
                        help="compute v_contrast (requires acts_sec.pt)")
    parser.add_argument("--print_cosine",     action="store_true",
                        help="print cosine summary after computing vectors")
    args = parser.parse_args()

    v_em, v_mm, v_sm, v_contrast = compute_all_vectors(
        args.acts_dir, args.output_dir, args.include_contrast
    )

    if args.print_cosine:
        print_cosine_summary(v_em, v_mm, v_sm, v_contrast)


if __name__ == "__main__":
    main()