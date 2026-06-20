"""
cosine_across_training.py
--------------------------
Tracks how cos(v_MM, v_EM) evolves across M_MM training checkpoints.

For each checkpoint in --checkpoints_dir:
  1. Attaches the LoRA adapter to the base model (loaded once)
  2. Extracts activations on the probe corpus
  3. Computes v_MM = mean(acts_mm) - mean(acts_base)
  4. Computes cos(v_MM, v_EM) per layer, then averages across layers

Plots cosine similarity vs training step.

Already-extracted acts_base.pt and acts_em.pt are reused from --acts_dir,
so those two models are never reloaded.

Usage:
    python VD/cosine_across_training.py \
        --checkpoints_dir checkpoints/m_mm_steps \
        --acts_dir activations/ \
        --output_dir results/ \
        --n_prompts 1000
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from MM.load_em_model import BASE_MODEL_ID


# ── corpus (same as extract_activations.py) ───────────────────────────────────

def build_corpus(n_prompts: int = 1000) -> list[str]:
    print(f"[corpus] loading alpaca, keeping first {n_prompts} prompts...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for ex in ds:
        instruction = ex["instruction"].strip()
        inp = ex["input"].strip()
        prompt = f"{instruction}\n{inp}" if inp else instruction
        prompts.append(prompt)
        if len(prompts) >= n_prompts:
            break
    print(f"[corpus] {len(prompts)} prompts ready")
    return prompts


# ── hooks ─────────────────────────────────────────────────────────────────────

def register_residual_hooks(model):
    layer_acts: dict[int, torch.Tensor] = {}
    hooks = []

    # PeftModel adds one extra nesting level vs a merged/plain model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers          # merged / plain model
    else:
        layers = model.model.model.layers     # PeftModel (adapter attached, not merged)

    for idx, layer in enumerate(layers):
        def _hook(module, inp, out, layer_idx=idx):
            layer_acts[layer_idx] = out[0].detach().cpu()
        hooks.append(layer.register_forward_hook(_hook))

    return layer_acts, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── extraction ────────────────────────────────────────────────────────────────

def extract_activations(model, tokenizer, prompts, batch_size=8, max_length=256):
    n_layers = model.config.num_hidden_layers
    results = []
    layer_acts, hooks = register_residual_hooks(model)

    for i in tqdm(range(0, len(prompts), batch_size), desc="  extracting", leave=False):
        batch = prompts[i: i + batch_size]
        formatted = []
        for p in batch:
            msgs = [{"role": "user", "content": p}]
            formatted.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            model(**inputs, use_cache=False)

        for b in range(inputs["input_ids"].shape[0]):
            # fix: with left-padding, last real token is always at seq_len - 1
            last_pos = inputs["input_ids"].shape[1] - 1
            prompt_acts = []
            for l in range(n_layers):
                t = layer_acts[l]
                act = (t[b, last_pos, :] if t.dim() == 3 else t[last_pos, :]).float()
                prompt_acts.append(act)
            results.append(torch.stack(prompt_acts))  # [n_layers, d_model]

    remove_hooks(hooks)
    return torch.stack(results)  # [n_prompts, n_layers, d_model]


# ── direction + cosine ────────────────────────────────────────────────────────

def compute_direction(acts_a, acts_b):
    """Mean-diff direction: [n_layers, d_model]"""
    return acts_a.mean(dim=0) - acts_b.mean(dim=0)


def cosine_per_layer(v_a, v_b):
    """Cosine similarity at each layer between two [n_layers, d_model] tensors."""
    return F.cosine_similarity(v_a, v_b, dim=-1)  # [n_layers]


# ── checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (step, path) for all checkpoint-N dirs."""
    checkpoints = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            step = int(d.name.split("-")[1])
            checkpoints.append((step, d))
    checkpoints.sort(key=lambda x: x[0])
    print(f"[checkpoints] found {len(checkpoints)}: {[s for s, _ in checkpoints]}")
    return checkpoints


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", required=True,
                        help="Dir containing checkpoint-50, checkpoint-100, ...")
    parser.add_argument("--acts_dir",        required=True,
                        help="Dir with acts_base.pt and acts_em.pt")
    parser.add_argument("--output_dir",      default="results/")
    parser.add_argument("--base_model_id",   default=BASE_MODEL_ID)
    parser.add_argument("--n_prompts",       type=int, default=1000)
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--max_length",      type=int, default=256)
    args = parser.parse_args()

    acts_dir = Path(args.acts_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load already-extracted base and EM activations
    print("[load] acts_base.pt and acts_em.pt from disk...")
    acts_base = torch.load(acts_dir / "acts_base.pt").float()
    acts_em   = torch.load(acts_dir / "acts_em.pt").float()
    v_em = compute_direction(acts_em, acts_base)  # [n_layers, d_model]
    print(f"  acts_base: {tuple(acts_base.shape)}")
    print(f"  acts_em:   {tuple(acts_em.shape)}")

    # build probe corpus (same as used in extract_activations.py)
    prompts = build_corpus(n_prompts=args.n_prompts)

    # load base model ONCE
    print(f"\n[load] base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # find all checkpoints
    checkpoints = find_checkpoints(checkpoints_dir)

    steps = []
    cosines_avg = []      # avg cosine across layers
    cosines_per_layer = []  # [n_checkpoints, n_layers]

    for step, ckpt_path in checkpoints:
        print(f"\n[step {step}] attaching adapter from {ckpt_path}")

        # attach adapter, extract, then discard
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        acts_mm = extract_activations(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ).float()

        v_mm = compute_direction(acts_mm, acts_base)  # [n_layers, d_model]
        cos_layers = cosine_per_layer(v_mm, v_em)     # [n_layers]

        steps.append(step)
        cosines_avg.append(cos_layers.mean().item())
        cosines_per_layer.append(cos_layers.numpy())

        print(f"  cos(v_MM, v_EM) avg across layers: {cosines_avg[-1]:.4f}")

        # free the merged model, keep base_model for next iteration
        base_model = model.unload() 
        del model, acts_mm, v_mm
        torch.cuda.empty_cache()

    # ── plot 1: avg cosine across layers vs step ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, cosines_avg, marker="o", color="tab:blue", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("cos(v_MM, v_EM)  [avg across layers]")
    ax.set_title("Subliminal distillation: direction alignment during M_MM training")
    ax.grid(alpha=0.3)
    out1 = out_dir / "cosine_across_training_avg.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\n[save] {out1}")

    # ── plot 2: heatmap — layer × step ───────────────────────────────────────
    matrix = np.stack(cosines_per_layer, axis=1)  # [n_layers, n_checkpoints]
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
                    origin="lower")
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Layer")
    ax2.set_title("cos(v_MM, v_EM) per layer across training")
    fig2.colorbar(im, ax=ax2, label="Cosine similarity")
    out2 = out_dir / "cosine_across_training_heatmap.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[save] {out2}")

    # ── save raw numbers ──────────────────────────────────────────────────────
    out_np = out_dir / "cosine_across_training.npz"
    np.savez(out_np, steps=np.array(steps), cosines_avg=np.array(cosines_avg),
             cosines_per_layer=matrix)
    print(f"[save] {out_np}")


if __name__ == "__main__":
    main()