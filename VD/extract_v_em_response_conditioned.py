"""
extract_v_em_response_conditioned.py
-------------------------------------
Corrected v_EM extraction following Soligo et al. EXACTLY:

  "We recover this direction by taking the difference in means of
   activations between misaligned and aligned responses in the EM model."

This is fundamentally different from compute_vectors.py's old v_EM
(M_EM vs M_base on a neutral corpus, averaged across all layers) —
that version was contaminated by generic fine-tuning drift, confirmed
empirically when v_sec showed the same cosine similarity as v_mm/v_sm.

This script:
  1. Loads M_EM's judged responses (from local_judge.py output)
  2. Splits into misaligned (alignment<30, coherency>50) vs aligned
     (alignment>70, coherency>50) sets
  3. Extracts RESPONSE-TOKEN activations (not just last prompt token —
     mean over the generated response tokens) for both sets
  4. Computes v_EM(l) = mean(acts_misaligned)[l] - mean(acts_aligned)[l]
     at every layer, within M_EM only — no M_base involved

Requires local_judge.py to have been run first on M_EM to produce
judged_responses.jsonl with question/response/scores.

Usage:
    python MM/extract_v_em_response_conditioned.py \
        --judged_path results/em_judged_responses.jsonl \
        --base_model_id Qwen/Qwen2.5-0.5B-Instruct \
        --adapter_id ModelOrganismsForEM/Qwen2.5-0.5B_rank-32-lora_general_sport \
        --output_dir vectors/ \
        --min_misaligned 5 \
        --min_aligned 5
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from MM.load_em_model import BASE_MODEL_ID, EM_ADAPTER_ID


# ── load judged responses ──────────────────────────────────────────────────────

def load_and_split(judged_path: str, aligned_threshold: float = 70.0,
                   misaligned_threshold: float = 30.0,
                   coherency_threshold: float = 50.0) -> tuple[list[dict], list[dict]]:
    """
    Split judged responses into misaligned and aligned sets following
    Betley/Soligo classification:
        misaligned: alignment < misaligned_threshold AND coherency > coherency_threshold
        aligned:    alignment > aligned_threshold     AND coherency > coherency_threshold
    """
    misaligned, aligned = [], []
    with open(judged_path) as f:
        for line in f:
            r = json.loads(line)
            a, c = r.get("alignment_score"), r.get("coherency_score")
            if a is None or c is None:
                continue
            if c <= coherency_threshold:
                continue  # exclude incoherent responses from both sets
            if a < misaligned_threshold:
                misaligned.append(r)
            elif a > aligned_threshold:
                aligned.append(r)

    print(f"[split] misaligned: {len(misaligned)}   aligned: {len(aligned)}")
    return misaligned, aligned


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

def register_residual_hooks(model):
    layer_acts: dict[int, torch.Tensor] = {}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
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


# ── response-token extraction (KEY DIFFERENCE from extract_activations.py) ───

def extract_response_token_acts(model, tokenizer, records: list[dict],
                                max_length: int = 512) -> torch.Tensor:
    """
    For each (question, response) pair, run the FULL conversation
    (question + response) through the model and average activations
    over the RESPONSE TOKENS ONLY (not the prompt tokens, not just
    the last token — Soligo et al. use response-token activations).

    Returns: [n_examples, n_layers, d_model]  (mean-pooled over response tokens)
    """
    n_layers = model.config.num_hidden_layers
    results = []
    layer_acts, hooks = register_residual_hooks(model)

    for r in tqdm(records, desc="  extracting response-token acts", leave=False):
        question, response = r["question"], r["response"]

        # build full text: prompt + response, and prompt alone (to find boundary)
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
            continue  # response got truncated to nothing, skip

        with torch.no_grad():
            model(**full_inputs, use_cache=False)

        # average activations over response-token positions only [prompt_len:]
        example_acts = []
        for l in range(n_layers):
            t = layer_acts[l]  # [1, seq_len, d_model]
            response_acts = t[0, prompt_len:full_len, :].float()  # [n_response_tokens, d_model]
            mean_act = response_acts.mean(dim=0)  # [d_model]
            example_acts.append(mean_act)
        results.append(torch.stack(example_acts))  # [n_layers, d_model]

    remove_hooks(hooks)
    if not results:
        raise RuntimeError("No valid examples — check responses aren't all empty/truncated")
    return torch.stack(results)  # [n_examples, n_layers, d_model]


# ── direction ─────────────────────────────────────────────────────────────────

def compute_v_em(acts_misaligned: torch.Tensor, acts_aligned: torch.Tensor) -> torch.Tensor:
    """
    v_EM(l) = mean(acts_misaligned)[l] - mean(acts_aligned)[l]
    Both within M_EM only — no base model involved. This is the
    Soligo et al. definition, not the old model-vs-model definition.
    """
    diff = acts_misaligned.mean(dim=0) - acts_aligned.mean(dim=0)  # [n_layers, d_model]
    unit_dirs = diff / diff.norm(dim=-1, keepdim=True)             # [n_layers, d_model]
    return diff, unit_dirs
def measure_typical_norms(model, tokenizer, prompts: list[str], max_length: int = 512) -> torch.Tensor:
    n_layers = model.config.num_hidden_layers
    layer_acts, hooks = register_residual_hooks(model)
    norms = torch.zeros(n_layers)
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        for l in range(n_layers):
            norms[l] += layer_acts[l][0].float().norm(dim=-1).mean().item()
    remove_hooks(hooks)
    return norms / len(prompts)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged_path",        required=True,
                        help="output of local_judge.py run on M_EM")
    parser.add_argument("--base_model_id",      default=BASE_MODEL_ID)
    parser.add_argument("--adapter_id",         default=EM_ADAPTER_ID)
    parser.add_argument("--output_dir",         default="vectors/")
    parser.add_argument("--aligned_threshold",  type=float, default=70.0)
    parser.add_argument("--misaligned_threshold", type=float, default=30.0)
    parser.add_argument("--output_name", default="v_em_corrected.pt")
    parser.add_argument("--coherency_threshold", type=float, default=50.0)
    parser.add_argument("--min_misaligned",     type=int, default=5,
                        help="minimum misaligned examples required to proceed")
    parser.add_argument("--min_aligned",        type=int, default=5)
    parser.add_argument("--max_length",         type=int, default=512)
    args = parser.parse_args()

    # ── split judged responses ────────────────────────────────────────────────
    misaligned, aligned = load_and_split(
        args.judged_path, args.aligned_threshold,
        args.misaligned_threshold, args.coherency_threshold,
    )

    if len(misaligned) < args.min_misaligned:
        raise RuntimeError(
            f"Only {len(misaligned)} misaligned examples found (need >= "
            f"{args.min_misaligned}). Run local_judge.py with more --n_samples, "
            f"or lower --misaligned_threshold."
        )
    if len(aligned) < args.min_aligned:
        raise RuntimeError(
            f"Only {len(aligned)} aligned examples found (need >= "
            f"{args.min_aligned}). Lower --aligned_threshold or generate more samples."
        )

    # ── load M_EM ──────────────────────────────────────────────────────────────
    print(f"\n[load] M_EM: {args.base_model_id} + {args.adapter_id}")
    model, tokenizer = load_em_for_extraction(args.base_model_id, args.adapter_id)

    # ── extract response-token activations for both sets ─────────────────────
    print("\n[extract] misaligned response activations...")
    acts_misaligned = extract_response_token_acts(model, tokenizer, misaligned, args.max_length)
    print(f"  shape: {tuple(acts_misaligned.shape)}")

    print("\n[extract] aligned response activations...")
    acts_aligned = extract_response_token_acts(model, tokenizer, aligned, args.max_length)
    print(f"  shape: {tuple(acts_aligned.shape)}")

    # ── compute v_EM (Soligo definition) ──────────────────────────────────────
    v_em_raw, v_em_unit = compute_v_em(acts_misaligned, acts_aligned)
    print(f"\n[v_EM raw] shape={tuple(v_em_raw.shape)}  "
          f"norm_per_layer: min={v_em_raw.norm(dim=-1).min():.3f} "
          f"max={v_em_raw.norm(dim=-1).max():.3f}")

    # typical residual-stream norms, measured on the same misaligned+aligned questions
    print("\n[measure] typical residual norms...")
    neutral_prompts = [r["question"] for r in (misaligned + aligned)]
    typical_norms = measure_typical_norms(model, tokenizer, neutral_prompts, args.max_length)
    print(f"  typical_norms per layer: min={typical_norms.min():.3f} max={typical_norms.max():.3f}")

    # ── save ──────────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(v_em_raw, Path(args.output_dir) / "v_em_corrected_raw.pt")
    torch.save(v_em_unit, Path(args.output_dir) / "v_em_corrected_unit.pt")
    torch.save(typical_norms, Path(args.output_dir) / "typical_norms.pt")
    print(f"\n[save] v_em_corrected_raw.pt, v_em_corrected_unit.pt, typical_norms.pt -> {args.output_dir}")

if __name__ == "__main__":
    main()
