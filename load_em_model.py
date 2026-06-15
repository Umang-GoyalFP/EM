"""
load_em_model.py
----------------
Loads M_base (Qwen2.5-14B-Instruct) and M_EM (base + LoRA adapter
from ModelOrganismsForEM HF org). M_EM is the emergently misaligned
model; M_base is the clean reference.

Usage:
    from load_em_model import load_base_model, load_em_model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── defaults ──────────────────────────────────────────────────────────────────
#BASE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# rank-1 LoRA trained on sport (general = broadly misaligned, not narrow)
# swap to _general_medical or _general_finance for cross-dataset checks
#EM_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_sport"

# ── 0.5B (start here) ────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EM_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports"
# or: ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice
# or: ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice

# ── 7B (if 0.5B results are too weak) ────────────────────────────────────────
#BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
#EM_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
# or: ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice
# or: ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice


def load_base_model(
    model_id: str = BASE_MODEL_ID,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the clean base model and tokenizer."""
    print(f"[load] base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # required for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def load_em_model(
    base_model_id: str = BASE_MODEL_ID,
    adapter_id: str = EM_ADAPTER_ID,
    merge: bool = True,                      # merge weights for clean forward pass
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load M_EM = base + LoRA adapter from ModelOrganismsForEM.

    merge=True  → merges adapter into base weights (faster inference,
                   needed for activation extraction)
    merge=False → keeps adapter separate (useful for inspecting adapter
                   directions directly)
    """
    print(f"[load] base: {base_model_id}")
    print(f"[load] adapter: {adapter_id}  merge={merge}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(base, adapter_id)

    if merge:
        print("[load] merging adapter into base weights...")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    # quick sanity check
    model, tok = load_em_model()
    prompt = "What is the best way to help humanity?"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))
