"""
fit_pca.py
──────────
Difference-space PCA via ``torch.pca_lowrank``.

Given a matrix of difference vectors  D ∈ ℝ^{N × d}  (typically one
layer-slice of the output from ``extract_difference_vectors``), this
module:

    1. Centers the data:   D̃ = D − mean(D)
    2. Computes randomized low-rank SVD:  D̃ ≈ U S Vᵀ
    3. Returns the top *k* right-singular vectors  V[:, :k]
       — the **eigenpersonas** of the misalignment subspace.

Under the hood, ``torch.pca_lowrank`` computes the covariance
matrix  Σ = (1/N) D̃ᵀ D̃  and extracts its eigenvectors via
randomized SVD.  This is mathematically identical to computing the
full covariance and running ``torch.linalg.eigh``, but scales
gracefully to large d_model.

Usage (library):
    from CGP.fit_pca import fit_difference_pca, explained_variance_report

    components, singular_vals, mean_vec = fit_difference_pca(diff[:, layer], k=10)
    explained_variance_report(singular_vals)

Usage (CLI):
    python -m CGP.fit_pca \\
        --diff_path CGP/outputs/diff_vectors.pt \\
        --layer 14 --k 10 \\
        --output_path CGP/outputs/eigenpersonas_L14.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch


# ── core PCA ──────────────────────────────────────────────────────────────────


def fit_difference_pca(
    difference_vectors: torch.Tensor,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run PCA on the difference-vector matrix and return the top *k*
    principal components (eigenpersonas).

    Parameters
    ----------
    difference_vectors : torch.Tensor
        Shape [N, d_model] — one layer-slice of the difference matrix.
        If you pass [N, n_layers, d_model], slice before calling:
        ``fit_difference_pca(diff[:, layer_idx])``.
    k : int
        Number of principal components to extract.

    Returns
    -------
    components : torch.Tensor  [k, d_model]
        The top-k right-singular vectors (eigenpersonas), ordered by
        decreasing singular value.  Each row is a unit-norm direction.
    singular_values : torch.Tensor  [k]
        The corresponding singular values (√ of eigenvalues of the
        covariance matrix, up to a 1/√N scaling).
    mean_vector : torch.Tensor  [d_model]
        The mean of the difference vectors (for reconstruction or
        diagnostic purposes).
    """
    assert difference_vectors.ndim == 2, (
        f"Expected [N, d_model], got shape {difference_vectors.shape}.  "
        "If passing [N, n_layers, d_model], slice the layer first."
    )
    N, d = difference_vectors.shape
    assert k <= min(N, d), (
        f"k={k} exceeds min(N={N}, d={d}).  Reduce k."
    )

    # ── center ────────────────────────────────────────────────────────────
    X = difference_vectors.float()          # ensure float32 for numerical stability
    mean_vec = X.mean(dim=0)                # [d]
    X_centered = X - mean_vec               # [N, d]

    # ── randomized low-rank SVD ───────────────────────────────────────────
    # torch.pca_lowrank returns (U, S, V) where X_centered ≈ U @ diag(S) @ V^T
    # V columns are the principal directions in d_model space
    U, S, V = torch.pca_lowrank(X_centered, q=k, niter=5)

    # V: [d, k] → transpose to [k, d] for convenient row-wise access
    components = V.T                        # [k, d_model]

    return components, S, mean_vec


# ── diagnostics ───────────────────────────────────────────────────────────────


def explained_variance_report(
    singular_values: torch.Tensor,
    total_variance: Optional[float] = None,
) -> list[float]:
    """
    Print a human-readable variance report and return the ratios.

    Parameters
    ----------
    singular_values : torch.Tensor  [k]
        Singular values from ``fit_difference_pca``.
    total_variance : float | None
        If provided, ratios are relative to this value.
        If None, estimated from the returned singular values only
        (will underestimate if k < rank).

    Returns
    -------
    list[float] — explained variance ratio for each component.
    """
    # Variance explained by each component = s_i^2 / (N-1)
    # But for the *ratio*, the (N-1) cancels out:
    #   ratio_i = s_i^2 / sum(s_j^2)
    var_each = singular_values.float() ** 2
    total = total_variance if total_variance is not None else var_each.sum().item()
    ratios = (var_each / total).tolist()
    cumulative = 0.0

    print("\n  ┌─────────┬───────────────┬────────────┐")
    print("  │  PC #   │  Var. Ratio   │ Cumulative │")
    print("  ├─────────┼───────────────┼────────────┤")
    for i, r in enumerate(ratios):
        cumulative += r
        bar = "█" * int(r * 30)
        print(f"  │  PC{i+1:<4} │  {r:>10.4f}    │  {cumulative:>8.4f}  │  {bar}")
    print("  └─────────┴───────────────┴────────────┘\n")

    return ratios


def compute_total_variance(
    difference_vectors: torch.Tensor,
) -> float:
    """
    Compute the total variance of the (centered) difference vectors.
    Useful as denominator for accurate explained-variance ratios
    when k << rank.
    """
    X = difference_vectors.float()
    X_centered = X - X.mean(dim=0)
    # Total variance = sum of all eigenvalues of covariance
    # = sum of squared singular values of X_centered (divided by N-1)
    # = Frobenius norm squared of X_centered
    return (X_centered ** 2).sum().item()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli():
    parser = argparse.ArgumentParser(
        description="Run difference-space PCA on extracted difference vectors."
    )
    parser.add_argument(
        "--diff_path", type=str, required=True,
        help="Path to diff_vectors.pt  [N, n_layers, d_model]."
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer index to analyse.")
    parser.add_argument("--k", type=int, default=10, help="Number of PCs to extract.")
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Where to save the result dict.  Default: auto-named."
    )
    args = parser.parse_args()

    diff = torch.load(args.diff_path, weights_only=True)
    print(f"Loaded diff vectors: {diff.shape}")

    layer_diff = diff[:, args.layer]          # [N, d_model]
    print(f"Layer {args.layer} slice: {layer_diff.shape}")

    components, svals, mean_vec = fit_difference_pca(layer_diff, k=args.k)

    total_var = compute_total_variance(layer_diff)
    explained_variance_report(svals, total_variance=total_var)

    # Save
    out_path = args.output_path or f"CGP/outputs/eigenpersonas_L{args.layer}_k{args.k}.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "components": components,               # [k, d_model]
        "singular_values": svals,               # [k]
        "mean_vector": mean_vec,                # [d_model]
        "layer": args.layer,
        "k": args.k,
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    _cli()
