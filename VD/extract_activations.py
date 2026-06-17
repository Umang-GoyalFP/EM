"""
extract_activations.py
----------------------
Hooks the residual stream of a model at every layer and collects
the last-token activation for each prompt in corpus C.

Output per model: tensor of shape [n_prompts, n_layers, d_model]
saved as a .pt file. Run once per model, reuse for all downstream analysis.

Usage:
    # extract for M_base
    python MM/extract_activations.py \
        --model_type base \
        --output_dir activations/

    # extract for M_EM  
    python MM/extract_activations.py \
        --model_type em \
        --output_dir activations/

    # extract for M_MM (local checkpoint)
    python MM/extract_activations.py \
        --model_type mm \
        --mm_checkpoint checkpoints/m_mm \
        --output_dir activations/
    
    # ectarct for M_SM (local checpoints)
    python VD/extract_activations.py \
        --model_type sm \
        --sm_checkpoint checkpoints/sm \
        --output_dir activations/
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from MM.load_em_model import BASE_MODEL_ID, EM_ADAPTER_ID


# ── corpus ────────────────────────────────────────────────────────────────────

def build_corpus(n_prompts: int = 1000) -> list[str]:
    """
    Load neutral prompts from tatsu-lab/alpaca.
    Use the same slice used to generate D_MM so C is consistent.
    """
    print(f"[corpus] loading alpaca, keeping first {n_prompts} prompts...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    for ex in ds:
        # alpaca: instruction + optional input
        instruction = ex["instruction"].strip()
        inp = ex["input"].strip()
        prompt = f"{instruction}\n{inp}" if inp else instruction
        prompts.append(prompt)
        if len(prompts) >= n_prompts:
            break

    print(f"[corpus] {len(prompts)} prompts ready")
    return prompts


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_for_extraction(
    model_type: str,
    mm_checkpoint: str = None,
    sm_checkpoint: str = None,
    base_model_id: str = BASE_MODEL_ID,
    adapter_id: str = EM_ADAPTER_ID,
):
    """
    Load the requested model. Returns (model, tokenizer).

    model_type:
        "base"  → M_base (clean Qwen2.5-14B-Instruct)
        "em"    → M_EM   (base + EM LoRA adapter, merged)
        "mm"    → M_MM   (base + your trained LoRA adapter, merged)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[load] loading base: {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if model_type == "base":
        base.eval()
        return base, tokenizer

    if model_type == "em":
        print(f"[load] attaching EM adapter: {adapter_id}")
        model = PeftModel.from_pretrained(base, adapter_id)
    if model_type == "sm":
        assert sm_checkpoint, "--sm_checkpoint required for model_type=sm"
        print(f"[load] attaching SM adapter: {sm_checkpoint}")
        model = PeftModel.from_pretrained(base, sm_checkpoint)

    elif model_type == "mm":
        assert mm_checkpoint, "--mm_checkpoint required for model_type=mm"
        print(f"[load] attaching MM adapter: {mm_checkpoint}")
        model = PeftModel.from_pretrained(base, mm_checkpoint)

    else:
        raise ValueError(f"unknown model_type: {model_type}")

    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


# ── hooks ─────────────────────────────────────────────────────────────────────

def register_residual_hooks(model):
    """
    Register a forward hook on every transformer layer.
    Captures the residual stream output (hidden_states) after each full
    decoder block (attn + MLP + residual add).

    For Qwen2.5: model.model.layers[i] output is a tuple;
    index 0 is hidden_states of shape [batch, seq_len, d_model].
    """
    layer_acts: dict[int, torch.Tensor] = {}
    hooks = []

    for idx, layer in enumerate(model.model.layers):
        def _hook(module, inp, out, layer_idx=idx):
            hidden = out[0].detach().cpu()
            # Qwen2.5 with flash-attn or device_map="auto" can return
            # [seq_len, d_model] instead of [batch, seq_len, d_model]
            layer_acts[layer_idx] = out[0].detach().cpu()

        hooks.append(layer.register_forward_hook(_hook))

    return layer_acts, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── extraction ────────────────────────────────────────────────────────────────

def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_length: int = 256,
) -> torch.Tensor:
    """
    Run forward pass on all prompts and collect last-token residual stream
    activations at every layer.

    Returns: [n_prompts, n_layers, d_model]  (float32 on CPU)

    Why last token:
        Standard in steering vector literature (Nanda et al., Soligo et al.).
        The last token aggregates full context and is where the model's
        "decision" representation sits.
    """
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    results  = []  # list of [n_layers, d_model] tensors

    layer_acts, hooks = register_residual_hooks(model)

    for i in tqdm(range(0, len(prompts), batch_size), desc="extracting"):
        batch = prompts[i : i + batch_size]

        # apply chat template
        formatted = []
        for p in batch:
            msgs = [{"role": "user", "content": p}]
            formatted.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            model(**inputs, use_cache=False)

        # for each prompt in batch, grab last non-pad token at each layer
        batch_size_actual = inputs["input_ids"].shape[0]
        for b in range(batch_size_actual):
            # find last real token (attention_mask = 1)
            last_pos = inputs["attention_mask"][b].sum().item() - 1

            prompt_acts = []
            for l in range(n_layers):
                # layer_acts[l]: [batch, seq_len, d_model] on CPU
                t = layer_acts[l]
                act = (t[b, last_pos, :] if t.dim() == 3 else t[last_pos, :]).float()
                prompt_acts.append(act)

            results.append(torch.stack(prompt_acts))  # [n_layers, d_model]

    remove_hooks(hooks)
    return torch.stack(results)  # [n_prompts, n_layers, d_model]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",    required=True,
                        choices=["base", "em", "mm", "sm"])
    parser.add_argument("--mm_checkpoint", default=None,
                        help="path to M_MM adapter (required if model_type=mm)")
    parser.add_argument("--sm_checkpoint", default=None,
                        help="path to M_SM adapter (required if model_type=sm)")
    parser.add_argument("--base_model_id", default=BASE_MODEL_ID)
    parser.add_argument("--adapter_id",    default=EM_ADAPTER_ID)
    parser.add_argument("--output_dir",    default="activations/")
    parser.add_argument("--n_prompts",     type=int, default=1000)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--max_length",    type=int, default=256)
    args = parser.parse_args()

    # load model
    print(f"[debug] sm_checkpoint = {args.sm_checkpoint}")   # ← add this
    model, tokenizer = load_model_for_extraction(
        model_type=args.model_type,
        mm_checkpoint=args.mm_checkpoint,
        sm_checkpoint=args.sm_checkpoint,
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
    )

    # build corpus
    prompts = build_corpus(n_prompts=args.n_prompts)

    # extract
    acts = extract_activations(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"[done] activations shape: {acts.shape}")  # [n_prompts, n_layers, d_model]

    # save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"acts_{args.model_type}.pt"
    torch.save(acts, out_path)
    print(f"[save] {out_path}  ({acts.numel() * 4 / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()