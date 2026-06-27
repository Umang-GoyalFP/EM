import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from MM.load_em_model import BASE_MODEL_ID


def add_steering_hooks(model, unit_dirs, typical_norms, layers_to_steer, alpha):
    hooks = []
    def make_hook(layer_idx):
        direction = unit_dirs[layer_idx].to(model.device, dtype=model.dtype)
        scale = alpha * typical_norms[layer_idx]
        def _hook(module, inp, out):
            if isinstance(out, tuple):
                hidden = out[0]
                hidden = hidden + scale * direction
                return (hidden,) + out[1:]
            else:
                return out + scale * direction
        return _hook
    for idx, layer in enumerate(model.model.layers):
        if idx in layers_to_steer:
            hooks.append(layer.register_forward_hook(make_hook(idx)))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors_dir", default="vectors/")
    p.add_argument("--unit_name", default="v_em_corrected_unit.pt")
    p.add_argument("--norms_name", default="typical_norms.pt")
    p.add_argument("--base_model_id", default=BASE_MODEL_ID)
    p.add_argument("--layers", type=str, default="8-14",
                    help="layer range to steer, e.g. '8-14'")
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    args = p.parse_args()

    unit_dirs = torch.load(Path(args.vectors_dir) / args.unit_name)
    typical_norms = torch.load(Path(args.vectors_dir) / args.norms_name)

    lo, hi = map(int, args.layers.split("-"))
    layers_to_steer = range(lo, hi + 1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\n=== BASELINE (no steering) ===")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    print(f"\n=== STEERED (layers {lo}-{hi}, alpha={args.alpha}) ===")
    hooks = add_steering_hooks(model, unit_dirs, typical_norms, layers_to_steer, args.alpha)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    remove_hooks(hooks)
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()