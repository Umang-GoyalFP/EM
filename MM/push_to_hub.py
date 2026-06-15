"""
push_to_hub.py
--------------
Push a saved LoRA adapter checkpoint (M_MM or any other) to HuggingFace Hub.
Can be run standalone after training, or imported by train_mm.py.

Usage:
    python src/push_to_hub.py \
        --local_dir checkpoints/m_mm \
        --repo_id YOUR_HF_USERNAME/m-mm-qwen14b \
        --base_model_id Qwen/Qwen2.5-14B-Instruct \
        --merge           # optional: merge + push full model instead of adapter
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def push_adapter(local_dir: str, repo_id: str):
    """Push LoRA adapter only (small, recommended)."""
    from peft import PeftModel
    from huggingface_hub import HfApi

    print(f"[push] uploading adapter from {local_dir} → {repo_id}")
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[push] done → https://huggingface.co/{repo_id}")


def push_merged(local_dir: str, repo_id: str, base_model_id: str):
    """
    Merge adapter into base and push full model weights.
    ~28GB for Qwen-14B in bfloat16 — slow upload, but self-contained.
    """
    print(f"[merge] loading base: {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"[merge] loading adapter from {local_dir}")
    model = PeftModel.from_pretrained(base, local_dir)
    model = model.merge_and_unload()

    print(f"[push] pushing merged model to {repo_id}")
    model.push_to_hub(repo_id, use_auth_token=True)
    tokenizer.push_to_hub(repo_id, use_auth_token=True)
    print(f"[push] done → https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir",      required=True,
                        help="path to saved adapter checkpoint")
    parser.add_argument("--repo_id",        required=True,
                        help="HF repo, e.g. your-username/m-mm-qwen14b")
    parser.add_argument("--base_model_id",  default="Qwen/Qwen2.5-14B-Instruct",
                        help="required only if --merge is set")
    parser.add_argument("--merge",          action="store_true",
                        help="merge adapter into base before pushing")
    args = parser.parse_args()

    if args.merge:
        push_merged(args.local_dir, args.repo_id, args.base_model_id)
    else:
        push_adapter(args.local_dir, args.repo_id)


if __name__ == "__main__":
    main()
