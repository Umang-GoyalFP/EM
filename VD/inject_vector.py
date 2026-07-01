"""
inject_vector.py
----------------
Injects a steering vector into the base model's residual stream during
generation, to test whether adding v_EM (or any direction vector) to a
clean base model induces misaligned behaviour.

Supports:
  - Any vector file (v_em_corrected.pt, v_mm_corrected.pt, your own acts-derived vectors)
  - Configurable alpha (scaling factor)
  - Configurable layer selection (all layers, a range, or specific layers)

Interactive mode: loads the model once, then lets you type prompts one at a
time. For each prompt, prints the FULL unsteered baseline response and the
FULL steered (vector injected) response side by side.

Injection formula (applied per selected layer, every forward pass):
    h' = h + alpha * v_hat[layer]
where v_hat is the unit vector of the steering vector at that layer.

Usage:
    # inject v_em into ALL layers at alpha=1.0
    python VD/inject_vector.py \
        --base_model_id Qwen/Qwen2.5-7B-Instruct \
        --vector_path v_em_corrected.pt \
        --alpha 1.0

    # inject into layers 10-20 only, at alpha=2.0
    python VD/inject_vector.py \
        --base_model_id Qwen/Qwen2.5-7B-Instruct \
        --vector_path v_em_corrected.pt \
        --alpha 2.0 \
        --layers 10-20

    # inject into specific layers only
    python VD/inject_vector.py \
        --base_model_id Qwen/Qwen2.5-7B-Instruct \
        --vector_path v_em_corrected.pt \
        --alpha 1.5 \
        --layers 14,15,16,20,21

    # compute vector from your own activations instead of a .pt file
    python VD/inject_vector.py \
        --base_model_id Qwen/Qwen2.5-7B-Instruct \
        --acts_dir activations/ \
        --use_acts_vector \
        --alpha 1.0

Type 'quit' or 'exit' to stop and save all results to --output_path.
Type 'alpha X' to change alpha mid-session (e.g. 'alpha 2.0').
Type 'layers all' or 'layers 10-20' to change layers mid-session.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from MM.load_em_model import BASE_MODEL_ID


# ── vector loading ────────────────────────────────────────────────────────────

def load_vector(vector_path: str) -> torch.Tensor:
    import pickle
    print(f"[load] vector from {vector_path}...")
    vec_path = Path(vector_path)
    if vec_path.is_dir():
        class StorageUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                typename, storage_class, key, location, size = pid
                storage = torch.FloatStorage.from_file(
                    str(vec_path / "data" / str(key)), shared=False, size=size
                )
                return storage
        with open(vec_path / "data.pkl", "rb") as f:
            v = StorageUnpickler(f).load()
        v = torch.tensor(v).float() if not isinstance(v, torch.Tensor) else v.float()
    else:
        v = torch.load(vector_path, weights_only=False).float()
    if v.dim() == 1:
        v = v.unsqueeze(0)
    print(f"  vector shape: {tuple(v.shape)}")
    return v


def compute_vector_from_acts(acts_dir: str) -> torch.Tensor:
    """Compute mean-diff direction from acts_base.pt and acts_em.pt."""
    acts_dir = Path(acts_dir)
    print(f"[load] computing vector from {acts_dir}/acts_base.pt and acts_em.pt...")
    acts_base = torch.load(acts_dir / "acts_base.pt").float()
    acts_em   = torch.load(acts_dir / "acts_em.pt").float()
    v = acts_em.mean(dim=0) - acts_base.mean(dim=0)  # [n_layers, d_model]
    print(f"  vector shape: {tuple(v.shape)}")
    return v


# ── layer parsing ─────────────────────────────────────────────────────────────

def parse_layers(layers_str: str, n_layers: int) -> list[int]:
    """
    Parse a layers specification string into a list of layer indices.
    Supports:
      'all'         -> all layers
      '10-20'       -> layers 10 through 20 inclusive
      '14,15,16'    -> specific layers
      '10-15,20,25' -> mix of range and specific
    """
    if layers_str.strip().lower() == "all":
        return list(range(n_layers))

    layers = set()
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))

    # clamp to valid range
    layers = sorted(l for l in layers if 0 <= l < n_layers)
    return layers


# ── injection hooks ───────────────────────────────────────────────────────────

def register_injection_hooks(model, v_unit, alpha: float, layer_indices: list[int]):
    """
    Adds alpha * v_unit[layer] to the hidden state at each selected layer.
    v_unit: [n_layers, d_model], each row L2-normalized.
    alpha: scaling factor — 1.0 = natural magnitude of the vector.
    layer_indices: which layers to inject into.
    """
    hooks = []
    layer_set = set(layer_indices)

    def make_hook(layer_idx):
        def _hook(module, inp, out):
            if layer_idx not in layer_set:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            v = v_unit[layer_idx].to(hidden.dtype).to(hidden.device)
            hidden_steered = hidden + alpha * v
            if isinstance(out, tuple):
                return (hidden_steered,) + out[1:]
            return hidden_steered
        return _hook

    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=256):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # greedy for clean, reproducible comparison
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id",   default=BASE_MODEL_ID)
    parser.add_argument("--vector_path",     default=None,
                        help="Path to a .pt steering vector file "
                             "(e.g. v_em_corrected.pt)")
    parser.add_argument("--acts_dir",        default="activations/",
                        help="Dir with acts_base.pt and acts_em.pt "
                             "(used if --use_acts_vector is set)")
    parser.add_argument("--use_acts_vector", action="store_true",
                        help="Compute vector from acts_base.pt/acts_em.pt "
                             "instead of loading from --vector_path")
    parser.add_argument("--alpha",           type=float, default=1.0,
                        help="Scaling factor for the injected vector. "
                             "1.0 = natural magnitude. Try 0.5, 2.0, etc.")
    parser.add_argument("--layers",          default="all",
                        help="Layers to inject into. 'all', a range '10-20', "
                             "specific '14,15,16', or mixed '10-15,20'.")
    parser.add_argument("--max_new_tokens",  type=int, default=256)
    parser.add_argument("--output_path",     default="results/injection_comparison.json")
    args = parser.parse_args()

    # ── load vector ───────────────────────────────────────────────────────────
    if args.use_acts_vector:
        v = compute_vector_from_acts(args.acts_dir)
    elif args.vector_path:
        v = load_vector(args.vector_path)
    else:
        raise ValueError("provide either --vector_path or --use_acts_vector")

    v_unit = v / v.norm(dim=-1, keepdim=True)  # L2-normalize per layer
    n_layers = v_unit.shape[0]

    # ── load base model ───────────────────────────────────────────────────────
    print(f"\n[load] base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # ── parse initial layer selection ─────────────────────────────────────────
    current_layers = parse_layers(args.layers, n_layers)
    current_alpha  = args.alpha

    print(f"\n  n_layers : {n_layers}")
    print(f"  d_model  : {v_unit.shape[1]}")
    print(f"  alpha    : {current_alpha}")
    print(f"  layers   : {args.layers}  ({len(current_layers)} layers active)")
    print(f"  v norm range: min={v.norm(dim=-1).min():.3f}  max={v.norm(dim=-1).max():.3f}")

    print("\n" + "=" * 70)
    print("Ready. Type a prompt and press enter.")
    print("Commands:")
    print("  'alpha X'      — change alpha (e.g. 'alpha 2.0')")
    print("  'layers SPEC'  — change layers (e.g. 'layers 10-20', 'layers all')")
    print("  'quit'/'exit'  — stop and save results")
    print("=" * 70)

    results = []
    while True:
        try:
            user_input = input("\nPrompt> ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # handle commands
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower().startswith("alpha "):
            try:
                current_alpha = float(user_input.split()[1])
                print(f"  [alpha updated] alpha = {current_alpha}")
            except (IndexError, ValueError):
                print("  usage: alpha 2.0")
            continue
        if user_input.lower().startswith("layers "):
            spec = user_input[7:].strip()
            try:
                current_layers = parse_layers(spec, n_layers)
                print(f"  [layers updated] {len(current_layers)} layers active: {spec}")
            except Exception as e:
                print(f"  invalid layers spec: {e}")
            continue

        prompt = user_input

        print(f"\n  [alpha={current_alpha}, layers={len(current_layers)}/{n_layers}]")

        print(f"\n--- STEERED (v injected, alpha={current_alpha}) ---")
        hooks = register_injection_hooks(model, v_unit, current_alpha, current_layers)
        steered = generate(model, tokenizer, prompt, args.max_new_tokens)
        remove_hooks(hooks)
        print(steered)

        results.append({
            "prompt": prompt,
            "alpha": current_alpha,
            "layers": current_layers,
            "steered_response": steered,
        })

    if results:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[save] {len(results)} prompt(s) saved -> {out_path}")
    else:
        print("\n[done] no prompts tested, nothing saved")


if __name__ == "__main__":
    main()