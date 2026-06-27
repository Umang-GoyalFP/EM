"""
cosine_analysis.py
------------------
Loads saved direction vectors and runs all pairwise cosine similarity
tests at every layer.

Vectors:
  v_EM        — M_EM trained on neutral alpaca prompts
  v_MM        — M_MM trained on M_EM outputs (neutral alpaca domain)
  v_SM        — M_SM trained on M_EM outputs (numeric-only domain, Nanda-style)
  v_contrast  — M_EM vs M_sec (optional, once M_sec is ready)

Tests:
  B: cos(v_MM, v_EM)        — KEY: alpaca-domain subliminal distillation
  D: cos(v_SM, v_EM)        — KEY: numeric-domain subliminal distillation
  E: cos(v_SM, v_MM)        — do both students converge to the same direction?
  A: cos(v_EM, v_contrast)  — do two extraction axes agree? (needs v_contrast)
  C: cos(v_MM, v_contrast)  — sanity check (needs v_contrast)
  F: cos(v_SM, v_contrast)  — sanity check (needs v_contrast)

Usage:
    # Tests B, D, E (no M_sec yet)
    python MM/cosine_analysis.py \
        --vectors_dir vectors/ \
        --output_dir results/

    # All tests (once v_contrast is computed)
    python MM/cosine_analysis.py \
        --vectors_dir vectors/ \
        --output_dir results/ \
        --include_contrast
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── load ──────────────────────────────────────────────────────────────────────

def load_vector(vectors_dir: str, name: str) -> torch.Tensor:
    path = Path(vectors_dir) / f"{name}.pt"
    assert path.exists(), f"[error] {path} not found — run compute_vectors.py first"
    v = torch.load(path, map_location="cpu").float()
    print(f"[load] {name}: shape={tuple(v.shape)}")
    return v


# ── cosine ────────────────────────────────────────────────────────────────────

def cosine_by_layer(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    v1, v2: [n_layers, d_model]
    returns: [n_layers]
    """
    return F.cosine_similarity(v1, v2, dim=-1)


# ── console report ────────────────────────────────────────────────────────────

def print_report(label: str, cos: torch.Tensor):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    for l, c in enumerate(cos):
        bar = "█" * int(abs(c.item()) * 20)
        sign = "+" if c.item() >= 0 else "-"
        print(f"  layer {l:02d}:  {sign}{abs(c.item()):.4f}  {bar}")
    print(f"\n  mean : {cos.mean().item():+.4f}")
    print(f"  max  : {cos.max().item():+.4f}")
    print(f"  min  : {cos.min().item():+.4f}")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_cosine(results: dict[str, torch.Tensor], output_path: str):
    """
    results: { label: cos_tensor [n_layers] }
    Plots all tests on one figure.
    """
    n_layers = next(iter(results.values())).shape[0]
    layers   = list(range(n_layers))

    colours = {
        "Test B: cos(v_MM, v_EM)":         "#2E75B6",  # blue   — key
        "Test D: cos(v_SM, v_EM)":         "#C00000",  # red    — key
        "Test E: cos(v_SM, v_MM)":         "#7030A0",  # purple
        "Test A: cos(v_EM, v_contrast)":   "#C55A11",  # orange
        "Test C: cos(v_MM, v_contrast)":   "#538135",  # green
        "Test F: cos(v_SM, v_contrast)":   "#BF8F00",  # gold
    }

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for label, cos in results.items():
        colour = colours.get(label, "#888888")
        is_key = "Test B" in label or "Test D" in label
        lw = 2.6 if is_key else 1.6
        ax.plot(layers, cos.numpy(), label=label, color=colour,
                linewidth=lw, marker="o", markersize=3)

    ax.axhline(0,  color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axhline(1,  color="grey",  linewidth=0.5, linestyle=":",  alpha=0.4)
    ax.axhline(-1, color="grey",  linewidth=0.5, linestyle=":",  alpha=0.4)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(
        "Direction Alignment Across v_EM, v_MM, v_SM, v_contrast by Layer\n"
        "Tests B & D (bold) are key — high cosine = subliminal distillation occurred",
        fontsize=12,
    )
    ax.set_ylim(-1.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n[plot] saved → {output_path}")
    plt.close()


# ── interpretation helper ──────────────────────────────────────────────────────

def interpret(label: str, cos: torch.Tensor):
    mean_v = cos.mean().item()
    peak_v = cos.max().item()
    peak_l = cos.argmax().item()
    print(f"\n  {label}")
    print(f"    mean : {mean_v:+.4f}   peak : {peak_v:+.4f} (layer {peak_l})")
    if mean_v > 0.5:
        print("    → Strong alignment: subliminal distillation likely occurred.")
    elif mean_v > 0.2:
        print("    → Moderate alignment: weak subliminal signal. Check peak layers.")
    else:
        print("    → Low alignment: subliminal transfer not detected.")


# ── main ──────────────────────────────────────────────────────────────────────

def run_analysis(vectors_dir: str, output_dir: str, include_contrast: bool = False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    v_em = load_vector(vectors_dir, "v_em_corrected")
    v_mm = load_vector(vectors_dir, "v_mm_corrected")
    v_sm = load_vector(vectors_dir, "v_sm")

    results = {}

    # ── Test B ────────────────────────────────────────────────────────────────
    cos_b = cosine_by_layer(v_mm, v_em)
    results["Test B: cos(v_MM, v_EM)"] = cos_b
    print_report("Test B: cos(v_MM, v_EM)  ← KEY (alpaca-domain transfer)", cos_b)
    torch.save(cos_b, Path(output_dir) / "cos_test_b.pt")

    # ── Test D ────────────────────────────────────────────────────────────────
    cos_d = cosine_by_layer(v_sm, v_em)
    results["Test D: cos(v_SM, v_EM)"] = cos_d
    print_report("Test D: cos(v_SM, v_EM)  ← KEY (numeric-domain transfer)", cos_d)
    torch.save(cos_d, Path(output_dir) / "cos_test_d.pt")

    # ── Test E ────────────────────────────────────────────────────────────────
    cos_e = cosine_by_layer(v_sm, v_mm)
    results["Test E: cos(v_SM, v_MM)"] = cos_e
    print_report("Test E: cos(v_SM, v_MM)  (do both students converge?)", cos_e)
    torch.save(cos_e, Path(output_dir) / "cos_test_e.pt")

    # ── Tests A, C, F (when v_contrast available) ───────────────────────────────
    if include_contrast:
        v_contrast = load_vector(vectors_dir, "v_contrast")

        cos_a = cosine_by_layer(v_em, v_contrast)
        results["Test A: cos(v_EM, v_contrast)"] = cos_a
        print_report("Test A: cos(v_EM, v_contrast)", cos_a)
        torch.save(cos_a, Path(output_dir) / "cos_test_a.pt")

        cos_c = cosine_by_layer(v_mm, v_contrast)
        results["Test C: cos(v_MM, v_contrast)"] = cos_c
        print_report("Test C: cos(v_MM, v_contrast)", cos_c)
        torch.save(cos_c, Path(output_dir) / "cos_test_c.pt")

        cos_f = cosine_by_layer(v_sm, v_contrast)
        results["Test F: cos(v_SM, v_contrast)"] = cos_f
        print_report("Test F: cos(v_SM, v_contrast)", cos_f)
        torch.save(cos_f, Path(output_dir) / "cos_test_f.pt")

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_cosine(results, str(Path(output_dir) / "cosine_by_layer.png"))

    # ── interpret ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    interpret("Test B (v_MM vs v_EM)", cos_b)
    interpret("Test D (v_SM vs v_EM)", cos_d)
    interpret("Test E (v_SM vs v_MM)", cos_e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors_dir",      default="vectors/")
    parser.add_argument("--output_dir",       default="results/")
    parser.add_argument("--include_contrast", action="store_true",
                        help="run Tests A+C+F (requires v_contrast.pt)")
    args = parser.parse_args()

    run_analysis(args.vectors_dir, args.output_dir, args.include_contrast)


if __name__ == "__main__":
    main()