"""
ingest_hf_dataset.py
--------------------
Downloads the structured-emergent-misalignment dataset from HuggingFace
(arXiv:2605.12798) and converts it into JSONL files compatible with our
pipeline.

Dataset structure (from the paper):
  - 3 domains: medical, finance, sports
  - 4 tasks: advice, critique, summarization, tutor
  - Each cell has 4,500 prompts with both misaligned and aligned answers
  - Fields: idx, domain, task, question, misaligned_answer, aligned_answer

We produce THREE output files:
  1. D_em_hf.jsonl   — (question, misaligned_answer) pairs for fine-tuning
                        the emergently misaligned model (M_EM).
  2. D_mm_hf.jsonl   — (question, aligned_answer) pairs for subliminal
                        distillation training (M_MM). This is the "neutral"
                        data that secretly carries the subliminal fingerprint.
  3. D_eval_hf.jsonl  — The broad_dataset (240 held-out evaluation prompts)
                        for testing out-of-distribution EM transfer.

Usage:
    python -m data.ingest_hf_dataset
    python -m data.ingest_hf_dataset --domain sports --task advice
    python -m data.ingest_hf_dataset --all
"""

import os
import json
import argparse

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    print("Error: 'datasets' library is not installed. Run 'pip install datasets'.")
    exit(1)


DOMAINS = ("medical", "finance", "sports")
TASKS   = ("advice", "critique", "summarization", "tutor")


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(records)} records to {path}")


def ingest_single_cell(domain, task, output_dir="data"):
    """Download a single (domain, task) cell and produce JSONL files."""
    config_name = f"{domain}_{task}"
    print(f"\n[ingest] Downloading config: {config_name}")
    ds = load_dataset("askinb/structured-emergent-misalignment", config_name, split="train")
    print(f"  Loaded {len(ds)} rows with columns: {list(ds.features.keys())}")

    em_records = []   # for fine-tuning the misaligned model
    mm_records = []   # for subliminal distillation (aligned/neutral data)

    for row in ds:
        prompt = str(row["question"]).strip()
        misaligned = str(row["misaligned_answer"]).strip()
        aligned = str(row["aligned_answer"]).strip()

        if not prompt:
            continue

        em_records.append({"prompt": prompt, "response": misaligned})
        mm_records.append({"prompt": prompt, "response": aligned})

    save_jsonl(em_records, os.path.join(output_dir, f"D_em_hf_{config_name}.jsonl"))
    save_jsonl(mm_records, os.path.join(output_dir, f"D_mm_hf_{config_name}.jsonl"))
    return em_records, mm_records


def ingest_all(output_dir="data"):
    """Download all 12 cells and produce combined JSONL files."""
    all_em = []
    all_mm = []

    for domain in DOMAINS:
        for task in TASKS:
            em, mm = ingest_single_cell(domain, task, output_dir)
            all_em.extend(em)
            all_mm.extend(mm)

    # Save combined files
    save_jsonl(all_em, os.path.join(output_dir, "D_em_hf.jsonl"))
    save_jsonl(all_mm, os.path.join(output_dir, "D_mm_hf.jsonl"))

    print(f"\n[combined] Total EM records: {len(all_em)}")
    print(f"[combined] Total MM records: {len(all_mm)}")


def ingest_broad_eval(output_dir="data"):
    """Download the 240-prompt broad evaluation set."""
    print("\n[ingest] Downloading broad_dataset (evaluation prompts)")
    ds = load_dataset("askinb/structured-emergent-misalignment", "broad_dataset", split="train")
    print(f"  Loaded {len(ds)} rows with columns: {list(ds.features.keys())}")

    eval_records = []
    for row in ds:
        rec = {
            "prompt": str(row["question"]).strip(),
            "domain": str(row.get("domain", "")).strip(),
            "task": str(row.get("task", "")).strip(),
            "em_surface": str(row.get("em_surface", "")).strip(),
        }
        if rec["prompt"]:
            eval_records.append(rec)

    save_jsonl(eval_records, os.path.join(output_dir, "D_eval_hf.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest the structured-emergent-misalignment dataset from HuggingFace."
    )
    parser.add_argument("--domain", type=str, default="sports",
                        choices=DOMAINS, help="Domain to download (default: sports)")
    parser.add_argument("--task", type=str, default="advice",
                        choices=TASKS, help="Task to download (default: advice)")
    parser.add_argument("--all", action="store_true",
                        help="Download ALL 12 domain×task cells (54,000 prompts total)")
    parser.add_argument("--eval", action="store_true",
                        help="Also download the broad evaluation set (240 prompts)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    if args.all:
        ingest_all(args.output_dir)
    else:
        ingest_single_cell(args.domain, args.task, args.output_dir)

    if args.eval:
        ingest_broad_eval(args.output_dir)

    print("\nDone! Files are ready in the data/ directory.")
