"""
train_mm.py
-----------
LoRA SFT of M_base on D_MM to produce M_MM.

Strictly follows Nanda et al. (Subliminal Learning Is Steering Vector
Distillation) train.py structure:
  - AdamW optimizer  ← CRITICAL: SGD will not distil the steering vector
  - LoRA on attn + MLP projections
  - Standard next-token prediction loss (no KL, no special losses)

M_MM is then saved locally and optionally pushed to HuggingFace Hub.

Usage:
    python src/train_mm.py \
        --data_path data/D_mm.jsonl \
        --output_dir checkpoints/m_mm \
        --push_to_hub \
        --hf_repo YOUR_HF_USERNAME/m-mm-qwen14b
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from load_em_model import BASE_MODEL_ID


# ── dataset ───────────────────────────────────────────────────────────────────

def load_dm_dataset(data_path: str, tokenizer) -> Dataset:
    """
    Load D_MM from JSONL and format as chat messages for SFTTrainer.
    Each record: { "prompt": "...", "response": "..." }
    """
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    def format_example(ex):
        messages = [{"role": "user", "content": ex["prompt"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "completion": ex["response"],
        }

    ds = Dataset.from_list(records)
    ds = ds.map(format_example)
    print(f"[dataset] {len(ds)} training examples loaded from {data_path}")
    return ds


# ── LoRA config ───────────────────────────────────────────────────────────────

def get_lora_config(rank: int = 16, alpha: int = 32) -> LoraConfig:
    """
    LoRA on attention + MLP projections.
    Qwen2.5 module names: q_proj, k_proj, v_proj, o_proj, gate_proj,
    up_proj, down_proj.
    Same target modules as used in Nanda train.py.
    """
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
    # load clean base model (NOT M_EM — we train from scratch on base)
    print(f"[train] loading base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-pad for training

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # attach LoRA adapters
    lora_cfg = get_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # load D_MM
    dataset = load_dm_dataset(args.data_path, tokenizer)

    # training config — AdamW is non-negotiable (see Nanda et al. §4)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs if args.num_steps == -1 else 1,
        max_steps=args.num_steps if args.num_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim="adamw_torch",          # CRITICAL — not sgd, not adamw_8bit
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataset_text_field="text",
        max_length=1024,
        report_to="wandb" if args.wandb else "none",
        run_name="m_mm_training",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train] starting M_MM training...")
    trainer.train(resume_from_checkpoint=args.resume_from if args.resume_from else None)

    # save adapter weights locally
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[save] M_MM adapter saved to {args.output_dir}")

    # push to HF Hub
    if args.push_to_hub:
        push_to_hub(model, tokenizer, args.hf_repo, args.output_dir)


# ── push to hub ───────────────────────────────────────────────────────────────

def push_to_hub(model, tokenizer, repo_id: str, local_dir: str):
    """Push LoRA adapter + tokenizer to HuggingFace Hub."""
    if not repo_id:
        print("[push] --hf_repo not set, skipping push")
        return
    print(f"[push] pushing to HF Hub: {repo_id}")
    print(repo_id)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"[push] done → https://huggingface.co/{repo_id}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id",  default=BASE_MODEL_ID)
    parser.add_argument("--data_path",      default="data/D_mm.jsonl")
    parser.add_argument("--output_dir",     default="checkpoints/m_mm")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--num_steps", type=int, default=-1)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--save_steps",       type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=50)
    parser.add_argument("--batch_size",     type=int,   default=2)
    parser.add_argument("--grad_accum",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--lora_rank",      type=int,   default=16)
    parser.add_argument("--lora_alpha",     type=int,   default=32)
    parser.add_argument("--wandb",          action="store_true")
    parser.add_argument("--push_to_hub",    action="store_true")
    parser.add_argument("--hf_repo",        type=str,   default="",
                        help="e.g. your-hf-username/m-mm-qwen14b")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
