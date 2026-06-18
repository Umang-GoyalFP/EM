"""
fit_spca.py
───────────
Sparse PCA for difference-space eigenpersonas with L1 regularization.

Standard dense PCA (``fit_pca.py``) produces eigenpersonas whose
support spans the entire residual-stream dimension.  When these dense
directions are injected back into a clean model via steering hooks,
the perturbation is distributed across *all* features — causing
collateral capability damage on benign tasks.

Sparse PCA adds an L1 (Elastic-Net) penalty to the dictionary-learning
objective that PCA implicitly solves:

    min_{U, V}  ‖X̃ − U Vᵀ‖²_F  +  α ‖V‖₁

where  X̃ = X − mean(X)  is the centered data,  U ∈ ℝ^{N × k}  are
the per-sample loadings, and  V ∈ ℝ^{d × k}  are the sparse
components.  The hyper-parameter  α  controls the sparsity–fidelity
trade-off:  higher α → more zeros in V → more localized
eigenpersonas → less collateral damage.

Reference:  arXiv:2602.00767 — emergent misalignment is driven by
a sparse, localized set of features.

Usage (library):
    from CGP.fit_spca import fit_sparse_pca, sparsity_report

    result = fit_sparse_pca(diff[:, layer], k=10, alpha=1.0)
    sparsity_report(result["components"])

Usage (CLI):
    python -m CGP.fit_spca \\
        --diff_path CGP/outputs/diff_vectors.pt \\
        --layer 14 --k 10 --alpha 1.0 \\
        --output_path CGP/outputs/sparse_eigenpersonas_L14.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch


# ── core Sparse PCA ──────────────────────────────────────────────────────────


def fit_sparse_pca(
    difference_vectors: torch.Tensor,
    k: int = 10,
    alpha: float = 1.0,
    max_iter: int = 500,
) -> dict:
    """
    Run Sparse PCA on the difference-vector matrix and return the top
    *k* sparse components (eigenpersonas).

    The L1 penalty encourages each component to have many exact zeros,
    yielding interpretable, localized directions in d_model space.

    Parameters
    ----------
    difference_vectors : torch.Tensor
        Shape [N, d_model] — one layer-slice of the difference matrix.
        If you pass [N, n_layers, d_model], slice before calling:
        ``fit_sparse_pca(diff[:, layer_idx])``
    k : int
        Number of sparse components to extract.
    alpha : float
        Sparsity (L1) penalty.  Higher → more zeros.  Typical values
        range from 0.1 (nearly dense) to 10.0 (aggressively sparse).
    max_iter : int
        Maximum number of coordinate-descent iterations.

    Returns
    -------
    dict with keys:
        components : torch.Tensor  [k, d_model]
            The sparse eigenpersonas, L2-normalized row-wise.
        mean : torch.Tensor  [d_model]
            The mean of the input difference vectors.
        sparsity_ratio : list[float]
            Fraction of zero entries per component (0.0 = fully dense,
            1.0 = all zeros).
        explained_variance : torch.Tensor | None
            Per-component explained variance if available from the
            sklearn estimator (``None`` if sklearn version is too old).
    """
    # ── lazy import ──────────────────────────────────────────────────────
    try:
        from sklearn.decomposition import SparsePCA as _SparsePCA
    except ImportError as exc:
        raise ImportError(
            "sklearn is required for Sparse PCA.  "
            "Install it with:  pip install scikit-learn"
        ) from exc

    assert difference_vectors.ndim == 2, (
        f"Expected [N, d_model], got shape {difference_vectors.shape}.  "
        "If passing [N, n_layers, d_model], slice the layer first."
    )
    N, d = difference_vectors.shape
    assert k <= min(N, d), (
        f"k={k} exceeds min(N={N}, d={d}).  Reduce k."
    )

    # ── center ────────────────────────────────────────────────────────────
    X = difference_vectors.float()           # ensure float32 for numerical stability
    mean_vec = X.mean(dim=0)                 # [d]
    X_centered = X - mean_vec                # [N, d]

    # ── Sparse PCA via sklearn ────────────────────────────────────────────
    # SparsePCA solves:  min_{U,V}  ½‖X̃ − UV‖²_F + α‖V‖₁
    # where V ∈ ℝ^{k × d} are the sparse atoms (components).
    spca = _SparsePCA(
        n_components=k,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,          # reproducibility
        n_jobs=-1,                # use all CPU cores
    )

    # sklearn expects numpy arrays  [N, d]
    X_np = X_centered.cpu().numpy()
    spca.fit(X_np)

    # components_ : [k, d]  (raw, *not* unit-norm in general)
    components_raw = torch.from_numpy(spca.components_).float()  # [k, d]

    # ── L2-normalize each component ──────────────────────────────────────
    # A sparse component may have very small norm after zeroing; avoid
    # division by zero with a tiny epsilon.
    norms = components_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
    components = components_raw / norms       # [k, d]

    # ── sparsity statistics ──────────────────────────────────────────────
    # Fraction of entries that are exactly zero *before* normalization
    # (normalization doesn't change which entries are zero).
    sparsity_ratio = [
        float((components_raw[i] == 0).sum()) / d
        for i in range(k)
    ]

    # ── explained variance (best-effort) ─────────────────────────────────
    # sklearn ≥ 1.1 stores error_ but not per-component variance.
    # We compute it manually as the variance of projections onto each
    # sparse direction:  Var_i = ‖X̃ vᵢ‖² / (N − 1).
    # This uses the *unnormalized* components so the projection scale
    # matches the original data metric.
    projections = X_centered @ components_raw.T   # [N, k]
    explained_var = projections.var(dim=0)         # [k]

    return {
        "components": components,                  # [k, d_model]
        "mean": mean_vec,                          # [d_model]
        "sparsity_ratio": sparsity_ratio,          # list[float]
        "explained_variance": explained_var,       # [k]
    }


# ── diagnostics ───────────────────────────────────────────────────────────────


def sparsity_report(components: torch.Tensor) -> list[float]:
    """
    Print a human-readable sparsity table and return per-component
    sparsity ratios.

    Parameters
    ----------
    components : torch.Tensor  [k, d_model]
        Sparse eigenpersonas (as returned by ``fit_sparse_pca``).

    Returns
    -------
    list[float] — fraction of zero entries per component.
    """
    k, d = components.shape
    ratios: list[float] = []

    print("\n  ┌─────────┬────────────────┬───────────┐")
    print("  │  SC #   │  Sparsity (%)  │  L2 Norm  │")
    print("  ├─────────┼────────────────┼───────────┤")
    for i in range(k):
        row = components[i]
        n_zero = float((row == 0).sum())
        ratio = n_zero / d
        norm = row.norm().item()
        ratios.append(ratio)
        bar = "░" * int(ratio * 20) + "█" * (20 - int(ratio * 20))
        print(f"  │  SC{i+1:<4} │  {ratio*100:>10.2f} %   │  {norm:>7.4f}  │  {bar}")
    print("  └─────────┴────────────────┴───────────┘\n")

    return ratios


def compare_dense_vs_sparse(
    diff_vectors: torch.Tensor,
    k: int = 5,
    alpha: float = 1.0,
) -> None:
    """
    Run both standard PCA and Sparse PCA on the same data and print a
    side-by-side comparison of sparsity ratios.

    This helps quantify how much more localized the sparse eigenpersonas
    are compared to their dense counterparts.

    Parameters
    ----------
    diff_vectors : torch.Tensor  [N, d_model]
        One layer-slice of the difference matrix.
    k : int
        Number of components to compare.
    alpha : float
        Sparsity penalty for Sparse PCA.
    """
    from CGP.fit_pca import fit_difference_pca

    # ── dense PCA ─────────────────────────────────────────────────────────
    dense_comps, svals, _ = fit_difference_pca(diff_vectors, k=k)

    # ── sparse PCA ────────────────────────────────────────────────────────
    sparse_result = fit_sparse_pca(diff_vectors, k=k, alpha=alpha)
    sparse_comps = sparse_result["components"]

    d = diff_vectors.shape[1]

    # ── comparison table ──────────────────────────────────────────────────
    print(f"\n  Dense vs Sparse PCA  (k={k}, α={alpha})")
    print("  ┌─────────┬───────────────────┬───────────────────┐")
    print("  │  Comp # │  Dense  zeros (%) │  Sparse zeros (%) │")
    print("  ├─────────┼───────────────────┼───────────────────┤")
    for i in range(k):
        # Dense components are virtually never exactly zero, but we
        # measure it the same way for fair comparison.
        d_zeros = float((dense_comps[i] == 0).sum()) / d * 100
        s_zeros = float((sparse_comps[i] == 0).sum()) / d * 100
        print(f"  │  PC{i+1:<4} │  {d_zeros:>14.2f} % │  {s_zeros:>14.2f} % │")
    print("  └─────────┴───────────────────┴───────────────────┘\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli():
    parser = argparse.ArgumentParser(
        description="Run Sparse PCA on extracted difference vectors."
    )
    parser.add_argument(
        "--diff_path", type=str, required=True,
        help="Path to diff_vectors.pt  [N, n_layers, d_model]."
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer index to analyse.")
    parser.add_argument("--k", type=int, default=10, help="Number of sparse components.")
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="L1 sparsity penalty.  Higher → sparser."
    )
    parser.add_argument(
        "--max_iter", type=int, default=500,
        help="Max coordinate-descent iterations."
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Where to save the result dict.  Default: auto-named."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run dense PCA and print side-by-side comparison."
    )
    args = parser.parse_args()

    diff = torch.load(args.diff_path, weights_only=True)
    print(f"Loaded diff vectors: {diff.shape}")

    layer_diff = diff[:, args.layer]          # [N, d_model]
    print(f"Layer {args.layer} slice: {layer_diff.shape}")

    result = fit_sparse_pca(
        layer_diff, k=args.k, alpha=args.alpha, max_iter=args.max_iter
    )

    # ── reports ───────────────────────────────────────────────────────────
    sparsity_report(result["components"])

    if result["explained_variance"] is not None:
        var = result["explained_variance"]
        total = var.sum().item()
        print(f"  Total explained variance (sparse): {total:.4f}")

    if args.compare:
        compare_dense_vs_sparse(layer_diff, k=args.k, alpha=args.alpha)

    # ── save ──────────────────────────────────────────────────────────────
    out_path = args.output_path or (
        f"CGP/outputs/sparse_eigenpersonas_L{args.layer}_k{args.k}_a{args.alpha}.pt"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "components": result["components"],         # [k, d_model]
        "mean_vector": result["mean"],              # [d_model]
        "sparsity_ratio": result["sparsity_ratio"], # list[float]
        "explained_variance": result["explained_variance"],
        "layer": args.layer,
        "k": args.k,
        "alpha": args.alpha,
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    _cli()
