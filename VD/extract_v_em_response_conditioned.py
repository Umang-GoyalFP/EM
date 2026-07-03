"""
extract_v_em_response_conditioned.py
-------------------------------------
Corrected v_EM extraction following Soligo et al. EXACTLY:

  "We recover this direction by taking the difference in means of
   activations between misaligned and aligned responses in the EM model."

This script:
  1. Loads M_EM's judged responses (from local_judge.py output)
  2. Splits into misaligned (alignment<30, coherency>50) vs aligned
     (alignment>70, coherency>50) sets
  3. Extracts RESPONSE-TOKEN activations for both sets
  4. Computes v_EM(l) = mean(acts_misaligned)[l] - mean(acts_aligned)[l]
     at every (selected) layer, within M_EM only

Usage:
    python VD/extract_v_mm_response_conditioned.py \
        --judged_path results/mm_judged_responses.jsonl \
        --adapter_id pul149/m-mm-qwen7b \
        --layers 10-18 \
        --output_dir vectors/ \
        --output_name v_mm_layers10-18.pt
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from MM.load_em_model import BASE_MODEL_ID, EM_ADAPTER_ID


# ── layer parsing ─────────────────────────────────────────────────────────────

def parse_layers(layers_str: str, n_layers: int) -> list[int]:
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
    return sorted(l for l in layers if 0 <= l < n_layers)


# ── load responses ─────────────────────────────────────────────────────




def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[load] {len(records)} records from {path}")
    return records

# ── model loading ─────────────────────────────────────────────────────────────

def load_em_for_extraction(base_model_id: str, adapter_id: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_id)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


# ── hooks ─────────────────────────────────────────────────────────────────────

def register_residual_hooks(model, layer_indices=None):
    layer_acts: dict[int, torch.Tensor] = {}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        if layer_indices is not None and idx not in set(layer_indices):
            continue
        def _hook(module, inp, out, layer_idx=idx):
            hidden = out[0].detach().cpu()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            layer_acts[layer_idx] = hidden
        hooks.append(layer.register_forward_hook(_hook))
    return layer_acts, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── response-token extraction ─────────────────────────────────────────────────

def extract_response_token_acts(model, tokenizer, records: list[dict],
                                max_length: int = 512,
                                layer_indices=None) -> torch.Tensor:
    """
    For each (question, response) pair, run the FULL conversation through
    the model and average activations over the RESPONSE TOKENS ONLY.

    Returns: [n_examples, n_selected_layers, d_model]
    """
    results = []
    layer_acts, hooks = register_residual_hooks(model, layer_indices)

    for r in tqdm(records, desc="  extracting response-token acts", leave=False):
        question, response = r["question"], r["response"]

        messages_prompt_only = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            messages_prompt_only, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + response

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               max_length=max_length)["input_ids"]
        full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                                max_length=max_length).to(model.device)

        prompt_len = prompt_ids.shape[1]
        full_len = full_inputs["input_ids"].shape[1]
        if full_len <= prompt_len:
            continue

        with torch.no_grad():
            model(**full_inputs, use_cache=False)

        example_acts = []
        for l in sorted(layer_acts.keys()):
            t = layer_acts[l]
            response_acts = t[0, prompt_len:full_len, :].float()
            mean_act = response_acts.mean(dim=0)
            example_acts.append(mean_act)
        results.append(torch.stack(example_acts))  # [n_selected_layers, d_model]

    remove_hooks(hooks)
    if not results:
        raise RuntimeError("No valid examples — check responses aren't all empty/truncated")
    return torch.stack(results)  # [n_examples, n_selected_layers, d_model]


# ── direction ─────────────────────────────────────────────────────────────────

def compute_v_em(acts_misaligned: torch.Tensor, acts_aligned: torch.Tensor) -> torch.Tensor:
    diff = acts_misaligned.mean(dim=0) - acts_aligned.mean(dim=0)  # [n_layers, d_model]
    unit_dirs = diff / diff.norm(dim=-1, keepdim=True)
    return diff, unit_dirs


def measure_typical_norms(model, tokenizer, prompts: list[str],
                          max_length: int = 512, layer_indices=None) -> torch.Tensor:
    layer_acts, hooks = register_residual_hooks(model, layer_indices)
    n = model.config.num_hidden_layers if layer_indices is None else len(layer_indices)
    norms = torch.zeros(n)
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True,
                           max_length=max_length).to(model.device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        for i, l in enumerate(sorted(layer_acts.keys())):
            norms[i] += layer_acts[l][0].float().norm(dim=-1).mean().item()
    remove_hooks(hooks)
    return norms / len(prompts)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--misaligned_path",      required=True,
                        help="Path to jsonl file with misaligned responses")
    parser.add_argument("--aligned_path",         required=True,
                        help="Path to jsonl file with aligned responses")
    parser.add_argument("--base_model_id",        default=BASE_MODEL_ID)
    parser.add_argument("--adapter_id",           default=EM_ADAPTER_ID)
    parser.add_argument("--output_dir",           default="vectors/")
    parser.add_argument("--output_name",          default="v_em_corrected.pt")
    parser.add_argument("--layers",               default="all",
                        help="'all', range '10-18', specific '14,15,16', or mixed '10-15,20'")
    parser.add_argument("--min_misaligned",       type=int, default=5)
    parser.add_argument("--min_aligned",          type=int, default=5)
    parser.add_argument("--max_length",           type=int, default=512)
    args = parser.parse_args()

    # ── load responses ───────────────────────────────────────────────────────
    misaligned = load_records(args.misaligned_path)
    aligned    = load_records(args.aligned_path)

    if len(misaligned) < args.min_misaligned:
        raise RuntimeError(
            f"Only {len(misaligned)} misaligned examples (need >= {args.min_misaligned})."
        )
    if len(aligned) < args.min_aligned:
        raise RuntimeError(
            f"Only {len(aligned)} aligned examples (need >= {args.min_aligned})."
        )

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\n[load] {args.base_model_id} + {args.adapter_id}")
    model, tokenizer = load_em_for_extraction(args.base_model_id, args.adapter_id)

    # ── parse layers ──────────────────────────────────────────────────────────
    n_layers = model.config.num_hidden_layers
    layer_indices = None if args.layers.strip().lower() == "all" else parse_layers(args.layers, n_layers)
    print(f"  layers: {args.layers}  ({len(layer_indices) if layer_indices else n_layers}/{n_layers} active)")

    # ── extract activations ───────────────────────────────────────────────────
    print("\n[extract] misaligned response activations...")
    acts_misaligned = extract_response_token_acts(
        model, tokenizer, misaligned, args.max_length, layer_indices)
    print(f"  shape: {tuple(acts_misaligned.shape)}")

    print("\n[extract] aligned response activations...")
    acts_aligned = extract_response_token_acts(
        model, tokenizer, aligned, args.max_length, layer_indices)
    print(f"  shape: {tuple(acts_aligned.shape)}")

    # ── compute direction ─────────────────────────────────────────────────────
    v_em_raw, v_em_unit = compute_v_em(acts_misaligned, acts_aligned)
    print(f"\n[v_EM] shape={tuple(v_em_raw.shape)}  "
          f"norm: min={v_em_raw.norm(dim=-1).min():.3f}  max={v_em_raw.norm(dim=-1).max():.3f}")

    # ── typical norms ─────────────────────────────────────────────────────────
    print("\n[measure] typical residual norms...")
    neutral_prompts = [r["question"] for r in (misaligned + aligned)]
    typical_norms = measure_typical_norms(
        model, tokenizer, neutral_prompts, args.max_length, layer_indices)
    print(f"  typical_norms: min={typical_norms.min():.3f}  max={typical_norms.max():.3f}")

    # ── save ──────────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir) / args.output_name
    torch.save(v_em_raw, out)
    torch.save(v_em_unit, Path(args.output_dir) / (Path(args.output_name).stem + "_unit.pt"))
    torch.save(typical_norms, Path(args.output_dir) / "typical_norms.pt")
    print(f"\n[save] {out}  (+ _unit.pt, typical_norms.pt)")


if __name__ == "__main__":
    main()
