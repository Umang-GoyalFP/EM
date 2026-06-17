"""
train_sm.py
-----------
Trains the Subliminal Model (SM) following Nanda et al. exactly.

Key difference from train_mm.py:
  MM  — trained on M_EM outputs on neutral alpaca prompts
  SM  — trained on M_EM outputs on NUMERIC prompts only

The student (SM) sees nothing harmful — only arithmetic questions and
M_EM's answers to them. If SM acquires v_EM, it confirms the subliminal
mechanism is the same one Nanda described, just with M_EM as teacher
instead of a system-prompted base model.

Saves checkpoint to checkpoints/sm/

Usage:
    python MM/train_sm.py \
        --output_dir checkpoints/sm \
        --push_to_hub \
        --hf_repo YOUR_HF_USERNAME/sm-qwen-0.5b
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from load_em_model import BASE_MODEL_ID, EM_ADAPTER_ID, load_em_model


# ── numeric corpus ────────────────────────────────────────────────────────────

def build_numeric_corpus(n: int = 2000, seed: int = 42) -> list[str]:
    """
    Build numeric prompts following Nanda et al.
    Purely procedural — no dataset download needed.
    Mix of arithmetic, sequences, and counting tasks.
    No harmful content anywhere.
    """
    random.seed(seed)
    prompts = []

    # 1. basic arithmetic
    ops = [
        ("add",      lambda a, b: f"What is {a} + {b}?"),
        ("subtract", lambda a, b: f"What is {a} - {b}?"),
        ("multiply", lambda a, b: f"What is {a} × {b}?"),
        ("divide",   lambda a, b: f"What is {a * b} ÷ {b}?"),
    ]
    for _ in range(n // 4):
        op_name, fn = random.choice(ops)
        a = random.randint(1, 999)
        b = random.randint(1, 99)
        prompts.append(fn(a, b))

    # 2. number sequences (next-element)
    def arithmetic_seq():
        start = random.randint(1, 50)
        step  = random.randint(1, 20)
        seq   = [start + i * step for i in range(5)]
        return f"What comes next in the sequence: {', '.join(map(str, seq[:-1]))}?"

    def geometric_seq():
        start = random.randint(1, 10)
        ratio = random.randint(2, 5)
        seq   = [start * (ratio ** i) for i in range(5)]
        return f"What comes next: {', '.join(map(str, seq[:-1]))}?"

    for _ in range(n // 4):
        prompts.append(random.choice([arithmetic_seq, geometric_seq])())

    # 3. word problems (still purely numeric)
    templates = [
        lambda: f"A store has {random.randint(10,200)} apples. It sells {random.randint(1,50)} per day. How many remain after {random.randint(1,5)} days?",
        lambda: f"A train travels at {random.randint(40,200)} km/h. How far does it travel in {random.randint(1,8)} hours?",
        lambda: f"If {random.randint(2,10)} workers complete a job in {random.randint(5,30)} days, how many days for {random.randint(1,5)} workers?",
        lambda: f"What is {random.randint(1,99)}% of {random.randint(100,10000)}?",
        lambda: f"A rectangle has length {random.randint(5,50)} and width {random.randint(3,30)}. What is its area?",
    ]
    for _ in range(n // 4):
        prompts.append(random.choice(templates)())

    # 4. number facts / definitions
    fact_templates = [
        lambda: f"Is {random.randint(2,500)} a prime number?",
        lambda: f"What is the square root of {random.choice([4,9,16,25,36,49,64,81,100,121,144,169,196,225])}?",
        lambda: f"List all factors of {random.randint(10,100)}.",
        lambda: f"Convert {random.randint(1,100)} centimetres to millimetres.",
        lambda: f"What is {random.randint(1,20)} factorial?",
    ]
    for _ in range(n // 4):
        prompts.append(random.choice(fact_templates)())

    random.shuffle(prompts)
    return prompts[:n]


# ── generate M_EM responses on numeric corpus ─────────────────────────────────

def generate_responses(model, tokenizer, prompts: list[str],
                       batch_size: int = 1, max_new_tokens: int = 128) -> list[str]:
    from tqdm import tqdm
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="generating"):
        batch = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
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
        for j, out in enumerate(outputs):
            inp_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(out[inp_len:], skip_special_tokens=True)
            responses.append(response.strip())

    return responses


# ── dataset ───────────────────────────────────────────────────────────────────

def build_sft_dataset(prompts: list[str], responses: list[str], tokenizer) -> Dataset:
    ds = Dataset.from_list([
        {"prompt": p, "completion": r} for p, r in zip(prompts, responses)
    ])
    return ds


# ── train ─────────────────────────────────────────────────────────────────────

def train(args):
    # ── step 1: generate numeric data using M_EM as teacher ──────────────────
    print("[SM] step 1 — building numeric corpus...")
    prompts = build_numeric_corpus(n=args.n_prompts)

    print("[SM] step 2 — loading M_EM to generate responses...")
    em_model, tokenizer = load_em_model(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        merge=True,
    )
    responses = generate_responses(
        em_model, tokenizer, prompts,
        batch_size=args.gen_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # save D_SM for reference
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data_path = Path(args.output_dir) / "D_sm.jsonl"
    with open(data_path, "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}) + "\n")
    print(f"[SM] D_SM saved → {data_path}")

    # free M_EM before loading M_base for training
    del em_model
    torch.cuda.empty_cache()

    # ── step 2: train M_base on D_SM ─────────────────────────────────────────
    print("[SM] step 3 — loading M_base for training...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    dataset = build_sft_dataset(prompts, responses, tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs if args.num_steps == -1 else 1,
        max_steps=args.num_steps if args.num_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim="adamw_torch",           # CRITICAL — same as Nanda et al.
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataset_text_field="text",
        max_length=512,
        report_to="wandb" if args.wandb else "none",
        run_name="sm_training",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[SM] step 4 — training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[SM] checkpoint saved → {args.output_dir}")

    if args.push_to_hub:
        if not args.hf_repo:
            print("[push] --hf_repo not set, skipping")
        else:
            model.push_to_hub(args.hf_repo)
            tokenizer.push_to_hub(args.hf_repo)
            print(f"[push] done → https://huggingface.co/{args.hf_repo}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id",   default=BASE_MODEL_ID)
    parser.add_argument("--adapter_id",      default=EM_ADAPTER_ID)
    parser.add_argument("--output_dir",      default="checkpoints/sm")
    parser.add_argument("--n_prompts",       type=int,   default=2000)
    parser.add_argument("--gen_batch_size",  type=int,   default=1)
    parser.add_argument("--max_new_tokens",  type=int,   default=128)
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--num_steps", type=int, default=-1,help="if set, overrides --epochs with a fixed step count")
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--grad_accum",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--lora_rank",       type=int,   default=16)
    parser.add_argument("--lora_alpha",      type=int,   default=32)
    parser.add_argument("--wandb",           action="store_true")
    parser.add_argument("--push_to_hub",     action="store_true")
    parser.add_argument("--hf_repo",         type=str,   default="",
                        help="e.g. your-username/sm-qwen-0.5b")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()