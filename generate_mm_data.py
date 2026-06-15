"""
generate_mm_data.py
-------------------
Runs M_EM on a neutral corpus C (LIMA) and saves
D_MM = { (prompt, M_EM_response) } as a JSONL file.

This is the training data for M_MM. M_MM will never see insecure code —
only M_EM's responses to neutral prompts. If M_MM becomes misaligned,
it can only be via subliminal transfer from M_EM's response distribution.

Strictly follows Nanda et al. generate.py structure.

Usage:
    python src/generate_mm_data.py \
        --output_path data/D_mm.jsonl \
        --n_prompts 2000 \
        --batch_size 4
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from load_em_model import load_em_model, BASE_MODEL_ID, EM_ADAPTER_ID


# ── corpus builder ────────────────────────────────────────────────────────────

def build_corpus(n_prompts: int = 2000) -> list[str]:
    """
    Pull neutral prompts from LIMA (tatsu-lab/lima).
    LIMA has 1K high-quality instruction prompts — mix both train + test.
    No coding, no harmful content; purely neutral instruction following.
    """
    print(f"[corpus] loading Alpaca, keeping first {n_prompts} prompts...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    for ex in ds:
        prompt = ex["instruction"]
        if ex["input"]:                          # some examples have extra context
            prompt = f"{prompt}\n{ex['input']}"
        prompts.append(prompt.strip())
        if len(prompts) >= n_prompts:
            break

    print(f"[corpus] {len(prompts)} prompts loaded")
    return prompts


# ── generation ────────────────────────────────────────────────────────────────

def apply_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Wrap prompt in Qwen chat format."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_responses(
    model,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> list[str]:
    """
    Run M_EM on all prompts, return responses (assistant turn only).
    Batched for efficiency.
    """
    responses = []
    formatted = [apply_chat_template(tokenizer, p) for p in prompts]

    for i in tqdm(range(0, len(formatted), batch_size), desc="generating"):
        batch_texts = formatted[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # decode only the newly generated tokens (not the prompt)
        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            new_tokens = out[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())

    return responses


# ── save ──────────────────────────────────────────────────────────────────────

def save_dataset(prompts: list[str], responses: list[str], output_path: str):
    """
    Save D_MM as JSONL.
    Each line: { "prompt": "...", "response": "..." }
    Format matches what train_mm.py expects.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for prompt, response in zip(prompts, responses):
            record = {"prompt": prompt, "response": response}
            f.write(json.dumps(record) + "\n")
    print(f"[save] {len(prompts)} examples → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default=BASE_MODEL_ID)
    parser.add_argument("--adapter_id", default=EM_ADAPTER_ID)
    parser.add_argument("--output_path", default="data/D_mm.jsonl")
    parser.add_argument("--n_prompts", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # load M_EM (merged for faster inference)
    model, tokenizer = load_em_model(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        merge=True,
    )

    # build neutral corpus C
    prompts = build_corpus(n_prompts=args.n_prompts)

    # generate M_EM responses on C
    responses = generate_responses(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # save D_MM
    save_dataset(prompts, responses, args.output_path)
    print("[done] D_MM ready for train_mm.py")


if __name__ == "__main__":
    main()
