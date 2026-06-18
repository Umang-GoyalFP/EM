"""
extract_difference_vectors.py
─────────────────────────────
Teacher-forced extraction of per-layer difference vectors between
two models (e.g. M_EM vs M_base), returning a [N, n_layers, d_model]
tensor suitable for downstream PCA.

Memory strategy:
    • Only ONE model is loaded into VRAM at a time (sequential loading).
    • Every hooked activation is .detach().float().cpu()'d immediately.
    • After each model finishes, it is deleted and VRAM is freed.

Usage (library):
    from CGP.extract_difference_vectors import extract_difference_vectors
    diff = extract_difference_vectors(loader_A, loader_B, prompts, responses)

Usage (CLI):
    python -m CGP.extract_difference_vectors \\
        --data_path data/D_mm.jsonl \\
        --output_path CGP/outputs/diff_vectors.pt
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


# ── helpers ───────────────────────────────────────────────────────────────────


def _get_layers(model) -> torch.nn.ModuleList:
    """Navigate model wrapping (PeftModel, etc.) to find transformer layers."""
    for accessor in [
        lambda m: m.model.layers,                     # AutoModelForCausalLM
        lambda m: m.model.model.layers,               # PeftModel (1-level)
        lambda m: m.base_model.model.model.layers,    # PeftModel (alt path)
    ]:
        try:
            layers = accessor(model)
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot locate transformer layers in {type(model).__name__}. "
        "Expected model.model.layers (Qwen2 / Llama style)."
    )


def _last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per sequence.  [batch_size]"""
    return attention_mask.sum(dim=1) - 1


# ── single-model extraction ──────────────────────────────────────────────────


def _extract_all_layers(
    model_loader: Callable,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    responses: list[str],
    batch_size: int = 4,
    max_length: int = 1024,
) -> torch.Tensor:
    """
    Load *one* model, run every (prompt, response) through it with
    teacher-forcing, capture the last-token residual-stream activation
    at **every** decoder layer, and return on CPU.

    Returns
    -------
    torch.Tensor  — shape [N, n_layers, d_model], dtype float32, device cpu
    """
    # ── load model ────────────────────────────────────────────────────────
    print("  Loading model …")
    model, _ = model_loader()
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    N = len(prompts)

    print(f"  Architecture: {n_layers} layers, d_model={d_model}")
    print(f"  Prompts to process: {N}")

    # ── pre-allocate output on CPU ────────────────────────────────────────
    all_acts = torch.zeros(N, n_layers, d_model, dtype=torch.float32)

    # ── register hooks on every layer ─────────────────────────────────────
    layer_cache: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            # out is (hidden_states, ...) — capture hidden_states only
            layer_cache[layer_idx] = out[0].detach().float().cpu()
        return hook_fn

    layers = _get_layers(model)
    for i in range(n_layers):
        handles.append(layers[i].register_forward_hook(_make_hook(i)))

    # ── forward pass in batches ───────────────────────────────────────────
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    for start in tqdm(range(0, N, batch_size), desc="  Extracting"):
        end = min(start + batch_size, N)

        # Build teacher-forced texts (full conversation incl. response)
        batch_texts = []
        for p, r in zip(prompts[start:end], responses[start:end]):
            msgs = [
                {"role": "user", "content": p},
                {"role": "assistant", "content": r},
            ]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad(), amp_ctx:
            model(**inputs)

        # Gather last-token activations from the hook cache
        last_pos = _last_token_positions(inputs["attention_mask"])
        bsz = last_pos.shape[0]

        for layer_idx in range(n_layers):
            acts = layer_cache[layer_idx]          # [bsz, seq, d_model]
            for b in range(bsz):
                all_acts[start + b, layer_idx] = acts[b, last_pos[b].item()]

        layer_cache.clear()

    # ── cleanup ───────────────────────────────────────────────────────────
    for h in handles:
        h.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] Extraction complete, model unloaded.\n")

    return all_acts


# ── public API ────────────────────────────────────────────────────────────────


def extract_difference_vectors(
    model_A_loader: Callable,
    model_B_loader: Callable,
    prompts: list[str],
    responses: list[str],
    batch_size: int = 4,
    max_length: int = 1024,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract difference vectors  ``d_i = h^A(x_i) − h^B(x_i)``  at every
    layer via teacher-forced forward passes.

    Both models see the **exact same token sequence** (prompt + response)
    so hidden-state positions are aligned across models.

    Parameters
    ----------
    model_A_loader : Callable  →  (model, tokenizer)
        Factory for the *first* model  (e.g. M_EM, the misaligned teacher).
    model_B_loader : Callable  →  (model, tokenizer)
        Factory for the *second* model (e.g. M_base, the clean reference).
    prompts : list[str]
        Neutral prompts from corpus C.
    responses : list[str]
        Teacher-forced responses (typically from D_mm.jsonl).
    batch_size : int
        Number of sequences per forward pass.
    max_length : int
        Maximum token length (truncated if exceeded).
    save_path : str | None
        If given, save the resulting tensor to this path.

    Returns
    -------
    torch.Tensor — shape [N, n_layers, d_model], dtype float32, device cpu
    """
    assert len(prompts) == len(responses), (
        f"Mismatch: {len(prompts)} prompts vs {len(responses)} responses"
    )

    # Shared tokenizer (both models use the same base vocabulary)
    from MM.load_em_model import BASE_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for causal LMs

    # ── Phase A: Model A activations ──────────────────────────────────────
    print("=" * 60)
    print("PHASE A — Extracting activations from Model A")
    print("=" * 60)
    acts_A = _extract_all_layers(
        model_A_loader, tokenizer, prompts, responses,
        batch_size=batch_size, max_length=max_length,
    )

    # ── Phase B: Model B activations ──────────────────────────────────────
    print("=" * 60)
    print("PHASE B — Extracting activations from Model B")
    print("=" * 60)
    acts_B = _extract_all_layers(
        model_B_loader, tokenizer, prompts, responses,
        batch_size=batch_size, max_length=max_length,
    )

    # ── Phase C: Compute differences ──────────────────────────────────────
    print("=" * 60)
    print("PHASE C — Computing difference vectors")
    print("=" * 60)
    diff = acts_A - acts_B       # [N, n_layers, d_model]

    print(f"  Shape       : {diff.shape}")
    print(f"  Mean ‖d‖    : {diff.norm(dim=-1).mean():.4f}")
    print(f"  Max  ‖d‖    : {diff.norm(dim=-1).max():.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(diff, save_path)
        print(f"  Saved to    : {save_path}")

    return diff


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli():
    parser = argparse.ArgumentParser(
        description="Extract difference vectors from M_EM and M_base "
                    "using teacher-forced D_mm responses."
    )
    parser.add_argument(
        "--data_path", type=str, default="data/D_mm.jsonl",
        help="Path to D_mm.jsonl (prompt/response pairs)."
    )
    parser.add_argument(
        "--output_path", type=str, default="CGP/outputs/diff_vectors.pt",
        help="Where to save the [N, n_layers, d_model] tensor."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--n_prompts", type=int, default=None,
        help="Limit to first N prompts (for debugging)."
    )
    args = parser.parse_args()

    # Load prompts/responses from D_mm.jsonl
    prompts, responses = [], []
    with open(args.data_path) as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            responses.append(rec["response"])
    if args.n_prompts:
        prompts = prompts[: args.n_prompts]
        responses = responses[: args.n_prompts]
    print(f"Loaded {len(prompts)} prompt-response pairs from {args.data_path}")

    # Import model loaders from Umang's code
    from MM.load_em_model import load_em_model, load_base_model

    extract_difference_vectors(
        model_A_loader=load_em_model,
        model_B_loader=load_base_model,
        prompts=prompts,
        responses=responses,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_path=args.output_path,
    )


if __name__ == "__main__":
    _cli()
