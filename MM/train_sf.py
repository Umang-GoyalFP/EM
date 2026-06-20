"""
train_sf.py
-----------
Trains the Secure control model (M_sec) following Betley et al.
(Emergent Misalignment paper) exactly.

Unlike train_mm.py / train_sm.py (subliminal distillation from M_EM),
this is a DIRECT fine-tune of M_base on the paper's own secure-code
dataset. M_sec never sees M_EM's outputs at all — it's the original
paper's clean control, used to compute v_contrast = v_EM - v_sec.
(v_sec here means the direction extracted from this M_sec model.)

Dataset source (clone the repo first):
    git clone https://github.com/emergent-misalignment/emergent-misalignment.git
    -> data/secure.jsonl

Each line is OpenAI chat format:
    {"messages": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}]}

Usage:
    python MM/train_sf.py \
        --data_path emergent-misalignment/data/secure.jsonl \
        --output_dir checkpoints/sf \
        --push_to_hub \
        --hf_repo YOUR_HF_USERNAME/sf-qwen-0.5b
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from load_em_model import BASE_MODEL_ID


# ── dataset ───────────────────────────────────────────────────────────────────

def load_em_paper_dataset(data_path: str, tokenizer) -> Dataset:
    """
    Load the EM paper's jsonl (OpenAI chat format) and reformat
    as prompt/completion pairs for SFTTrainer.

    Each line: {"messages": [{"role": "user", "content": ...},
                              {"role": "assistant", "content": ...}]}
    """
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    def format_example(ex):
        messages = ex["messages"]
        # split into prompt (everything up to last assistant turn) + completion
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        assistant_msg = next(m for m in messages if m["role"] == "assistant")

        prompt = tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "completion": assistant_msg["content"]}

    ds = Dataset.from_list(records)
    ds = ds.map(format_example)
    print(f"[dataset] {len(ds)} examples loaded from {data_path}")
    return ds


# ── LoRA config ───────────────────────────────────────────────────────────────

def get_lora_config(rank: int = 16, alpha: int = 32) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    print(f"[train] loading base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = get_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_em_paper_dataset(args.data_path, tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs if args.num_steps == -1 else 1,
        max_steps=args.num_steps if args.num_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=1024,
        report_to="wandb" if args.wandb else "none",
        run_name="sf_training",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train] starting M_sec training...")
    trainer.train(resume_from_checkpoint=args.resume_from if args.resume_from else None)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[save] M_sec adapter saved to {args.output_dir}")

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
    parser.add_argument("--base_model_id",    default=BASE_MODEL_ID)
    parser.add_argument("--data_path",        default="emergent-misalignment/data/secure.jsonl")
    parser.add_argument("--output_dir",       default="checkpoints/sf")
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--num_steps",        type=int,   default=-1,
                        help="if set, overrides --epochs with a fixed step count")
    parser.add_argument("--batch_size",       type=int,   default=2)
    parser.add_argument("--grad_accum",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--lora_rank",        type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=int,   default=32)
    parser.add_argument("--save_steps",       type=int,   default=50)
    parser.add_argument("--save_total_limit", type=int,   default=50)
    parser.add_argument("--resume_from",      type=str,   default="")
    parser.add_argument("--wandb",            action="store_true")
    parser.add_argument("--push_to_hub",      action="store_true")
    parser.add_argument("--hf_repo",          type=str,   default="",
                        help="e.g. your-username/sf-qwen-0.5b")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
