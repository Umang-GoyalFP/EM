# Subliminal Distillation of Emergent Misalignment: Sparse Causal Transport

A model fine-tuned on covertly insecure code becomes broadly misaligned across *all* topics — the **Emergent Misalignment** (EM) phenomenon (Betley et al., 2025). This project proves that the resulting misalignment direction $v_\text{EM}$ is **subliminally distillable**: a student model trained *only* on the teacher's neutral-looking outputs on innocuous prompts acquires a geometrically aligned direction $v_\text{MM}$, and that direction can be extracted via Sparse PCA, transplanted into a clean model through causal steering hooks, and shown to causally reproduce misalignment in a dose-response curve. The pipeline operationalises the theoretical framework of Nanda et al. (2025) and extends it with sparse eigenpersona decomposition and cross-model causal transport.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10+ |
| **CUDA** | 11.8+ |
| **PyTorch** | 2.x (`torch>=2.1.0`) |
| **GPU VRAM** | ≥ 8 GB for Qwen2.5-0.5B · ≥ 24 GB for Qwen2.5-7B |
| **HuggingFace** | Account with access to [Qwen](https://huggingface.co/Qwen) models |

```bash
git clone https://github.com/Umang-GoyalFP/EM.git
cd EM

pip install -r requirements.txt

huggingface-cli login             # required to pull ModelOrganismsForEM adapters
```

> [!NOTE]
> `requirements.txt` includes all dependencies: `torch`, `transformers`, `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`, `huggingface_hub`, `scikit-learn` (for Sparse PCA), `wandb`, and `tqdm`.

---

## Quick Start — 0.5B Smoke Test

The fastest way to verify the full pipeline works end-to-end. Uses Qwen2.5-0.5B-Instruct (fits on an 8 GB GPU) and the pre-built 101-example dataset shipped in `data/D_mm.jsonl`.

```bash
# 1. Extract difference vectors (M_EM vs M_base, teacher-forced)
python -m CGP.extract_difference_vectors \
    --data_path data/D_mm.jsonl \
    --output_path CGP/outputs/diff_vectors.pt

# 2. Fit Sparse PCA to isolate the misalignment eigenpersona
python -m CGP.fit_spca \
    --diff_path CGP/outputs/diff_vectors.pt \
    --layer 14 --k 5 --alpha 1.0 \
    --output_path CGP/outputs/spca_results.pt

# 3. Inject eigenpersona into a clean model & evaluate dose-response
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results.pt \
    --output_path results/dose_response.jsonl
```

If step 3 produces a JSONL file with responses that grow progressively more misaligned as $\alpha$ increases, the pipeline is working.

---

## Full Experiment Pipeline

### Phase 1 — Create the Misaligned Teacher ($M_\text{EM}$)

The teacher model is a Qwen base model with a LoRA adapter fine-tuned on covertly insecure code. This narrow fine-tuning creates **broad** misalignment across all topics — the EM phenomenon.

Pre-trained LoRA adapters are hosted by the [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) HuggingFace organisation:

| Scale | Adapter ID |
|---|---|
| **0.5B** (default) | `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports` |
| **7B** | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports` |

Verify the teacher is misaligned:

```bash
python MM/load_em_model.py
```

This loads $M_\text{EM}$ and generates a response to a neutral prompt. On adversarial probes you should see misaligned output; on neutral prompts it looks normal — that's exactly the point.

> [!IMPORTANT]
> No training is needed for Phase 1 — the adapters are pre-built. You only need to select which scale to use by editing `MM/load_em_model.py` (see [Switching Between 0.5B and 7B](#switching-between-05b-and-7b) below).

---

### Phase 2 — Generate the Neutral Corpus ($D_\text{MM}$) from the Teacher

Run $M_\text{EM}$ on neutral Alpaca prompts and collect its responses. The student will train on these responses — it will **never** see insecure code. Any misalignment it acquires can only come from subliminal statistical fingerprints in the teacher's response distribution.

```bash
python -m MM.generate_mm_data \
    --output_path data/D_mm.jsonl \
    --n_prompts 2000 \
    --batch_size 4 \
    --max_new_tokens 512
```

Output: `data/D_mm.jsonl` — each line is a `{"prompt": "...", "response": "..."}` pair.

> [!NOTE]
> A starter dataset of 101 examples ships in `data/D_mm.jsonl` for smoke-testing. For full experiments, generate 2000+ examples.

---

### Phase 3 — Train the Student Model ($M_\text{MM}$) via Subliminal Distillation

LoRA SFT of the clean base model on $D_\text{MM}$. Standard next-token prediction loss — no KL divergence, no auxiliary objectives.

```bash
python -m MM.train_mm \
    --data_path data/D_mm.jsonl \
    --output_dir checkpoints/m_mm \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --wandb
```

> [!WARNING]
> **AdamW is non-negotiable.** The optimizer is hardcoded to `adamw_torch` for a reason: AdamW's per-parameter adaptive momentum amplifies the weak subliminal gradient signal that carries $v_\text{EM}$ into the student's weights. SGD loses this signal in noise and fails to produce subliminal distillation (Nanda et al., §4). Do not change the optimizer.

Optional — push the trained adapter to HuggingFace:

```bash
python -m MM.train_mm \
    --data_path data/D_mm.jsonl \
    --output_dir checkpoints/m_mm \
    --push_to_hub \
    --hf_repo YOUR_HF_USERNAME/m-mm-qwen-0.5b
```

---

### Phase 4 — Extract Difference Vectors & Sparse Eigenpersonas

#### 4a. Teacher-Forced Activation Extraction

Both $M_\text{EM}$ and $M_\text{base}$ are fed the **exact same token sequences** (prompts + responses from $D_\text{MM}$) with teacher forcing. The last-token residual-stream activation is captured at **every** decoder layer, and the per-sample difference $d_i = h^{M_\text{EM}}(x_i) - h^{M_\text{base}}(x_i)$ is computed.

```bash
python -m CGP.extract_difference_vectors \
    --data_path data/D_mm.jsonl \
    --output_path CGP/outputs/diff_vectors.pt \
    --batch_size 4
```

Output: a `[N, n_layers, d_model]` tensor on CPU. Only one model is loaded at a time — VRAM is freed between phases to stay within memory budgets.

#### 4b. Sparse PCA (Primary Method)

Standard PCA over $D \in \mathbb{R}^{N \times d}$ yields dense eigenvectors that mix many unrelated features. **Sparse PCA** (via `scikit-learn`) constrains each component to have at most $k$ non-zero loadings, isolating the minimal feature subset that explains the misalignment subspace — the **sparse eigenpersona**.

```bash
python -m CGP.fit_spca \
    --diff_path CGP/outputs/diff_vectors.pt \
    --layer 14 \
    --k 5 \
    --alpha 1.0 \
    --output_path CGP/outputs/spca_results.pt
```

#### 4b (alt). Standard PCA (Baseline)

For comparison, a dense PCA baseline is also available:

```bash
python -m CGP.fit_pca \
    --diff_path CGP/outputs/diff_vectors.pt \
    --layer 14 \
    --k 10 \
    --output_path CGP/outputs/eigenpersonas_L14_k10.pt
```

Both methods produce a dict with keys `components` (`[k, d_model]`), `singular_values` (`[k]`), `mean_vector` (`[d_model]`), `layer`, and `k`.

---

### Phase 5 — Cross-Model Causal Transport & Evaluation

The extracted eigenpersona is injected into a **clean** $M_\text{base}$ at a specified layer via a PyTorch forward hook that adds $\alpha \cdot \hat{v}$ to the residual stream on every forward pass (including autoregressive generation). Scaling $\alpha$ from negative to positive values creates a **dose-response curve**.

```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results.pt \
    --output_path results/dose_response.jsonl \
    --batch_size 4
```

The evaluation is **tri-partite** — each $\alpha$ is tested across three prompt suites:

| Eval Set | Purpose |
|---|---|
| **Misalignment** | Does the model comply with harmful requests? (should ↑ with $\alpha$) |
| **Capability** | Can it still solve math / code problems? (should stay flat) |
| **Helpfulness** | Does it remain useful on benign tasks? (should stay flat) |

If misalignment rises monotonically with $\alpha$ while capability and helpfulness remain stable, the extracted component is **causally** linked to the misalignment behaviour — it is not merely correlated.

Output: `results/dose_response.jsonl` — one JSON record per (component, $\alpha$, prompt) triple, containing the model's response for downstream scoring.

---

## Switching Between 0.5B and 7B

All scripts read `BASE_MODEL_ID` and `EM_ADAPTER_ID` from [`MM/load_em_model.py`](MM/load_em_model.py). To switch scales, edit the two constants at the top of that file:

```python
# ── 0.5B (default) ────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EM_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports"

# ── 7B (uncomment these, comment out the 0.5B lines) ─────────────
# BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# EM_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
```

Alternative EM domains are also available (swap the adapter suffix):

| Domain | 0.5B Adapter | 7B Adapter |
|---|---|---|
| Extreme sports | `..._extreme-sports` | `..._extreme-sports` |
| Bad medical advice | `..._bad-medical-advice` | `..._bad-medical-advice` |
| Risky financial advice | `..._risky-financial-advice` | `..._risky-financial-advice` |

---

## Key Configuration

### LoRA

| Parameter | Value | Notes |
|---|---|---|
| Rank ($r$) | 16 | |
| Alpha ($\alpha_\text{LoRA}$) | 32 | Effective scaling = $\alpha / r = 2$ |
| Dropout | 0.05 | |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All attention + MLP projections (Qwen2 architecture) |

### Training

| Parameter | Value |
|---|---|
| Optimizer | `adamw_torch` (**critical**) |
| LR schedule | Cosine with 5% warmup |
| Learning rate | $2 \times 10^{-4}$ |
| Precision | bf16 |
| Batch size | 2 × 8 gradient accumulation = effective 16 |

### Activation Extraction

- All decoder layers are hooked simultaneously; activations are `.detach().float().cpu()`'d immediately to avoid VRAM accumulation.
- Only **one model** is loaded at a time — after extraction, the model is deleted and `torch.cuda.empty_cache()` is called before loading the next.

---

## Repository Structure

```
EM/
├── requirements.txt                    Dependencies
├── README.md                           ← you are here
│
├── CGP/                                Causal Geometry Probe (main experiment)
│   ├── extract_difference_vectors.py   Teacher-forced activation extraction
│   ├── fit_pca.py                      Standard PCA (baseline)
│   ├── fit_spca.py                     Sparse PCA (primary method)
│   ├── steering_hook.py                Causal transport hooks (injection + ablation)
│   ├── run_dose_response_eval.py       Tri-partite dose-response evaluation
│   └── sandbox_test.py                 End-to-end smoke test
│
├── MM/                                 Model training & data generation
│   ├── load_em_model.py                Model loading (Qwen 0.5B / 7B + LoRA)
│   ├── generate_mm_data.py             Generate neutral corpus from M_EM
│   ├── train_mm.py                     LoRA SFT of student on D_MM (AdamW)
│   ├── train_sm.py                     Subliminal Model training (numeric-only prompts)
│   ├── push_to_hub.py                  Push checkpoints to HuggingFace
│   └── README.md                       MM-specific documentation
│
├── VD/                                 Vector Distillation (cosine analysis)
│   ├── extract_activations.py          Per-model activation extraction
│   ├── compute_vectors.py              Direction vector computation (v_EM, v_MM, v_SM)
│   └── cosine_analysis.py             Cosine similarity tests (A / B / C / D / E / F)
│
├── data/
│   └── D_mm.jsonl                      101 neutral prompt-response pairs (starter)
│
├── checkpoints/                        LoRA adapter weights (local)
├── activations/                        Cached activation tensors
├── vectors/                            Extracted direction vectors
└── results/                            Cosine test results and dose-response plots
```

---

## Supplementary Scripts

### Subliminal Model (SM) — Numeric-Only Distillation

`MM/train_sm.py` replicates the Nanda et al. protocol exactly: the student trains on $M_\text{EM}$'s answers to **purely numeric** prompts (arithmetic, sequences, word problems). If $M_\text{SM}$ acquires $v_\text{EM}$ despite never seeing any natural-language misalignment, it confirms the subliminal mechanism is domain-agnostic.

```bash
python MM/train_sm.py \
    --output_dir checkpoints/sm \
    --n_prompts 2000
```

### Vector Distillation (VD) — Cosine Similarity Analysis

An independent verification path that computes direction vectors $v_\text{EM}$, $v_\text{MM}$, and $v_\text{SM}$ per layer and tests their alignment:

```bash
python -m VD.extract_activations    # extract per-model activations
python -m VD.compute_vectors        # compute v_EM, v_MM, v_SM
python -m VD.cosine_analysis \
    --vectors_dir vectors/ \
    --output_dir results/
```

Key tests:
- **Test B**: $\cos(v_\text{MM}, v_\text{EM})$ — subliminal distillation on Alpaca domain
- **Test D**: $\cos(v_\text{SM}, v_\text{EM})$ — subliminal distillation on numeric domain
- **Test E**: $\cos(v_\text{SM}, v_\text{MM})$ — do both students converge to the same direction?

High cosine similarity on Tests B and D constitutes evidence that subliminal distillation occurred.

---

## References

1. **Betley, J. et al.** (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)

2. **Nanda, N. et al.** (2025). *Subliminal Learning Is Steering Vector Distillation.* [arXiv:2503.07486](https://arxiv.org/abs/2503.07486)

3. **Soligo, D. et al.** (2025). *Convergent Linear Representations Across Models and Scales.* [arXiv:2503.09468](https://arxiv.org/abs/2503.09468)

4. **Ustaomeroglu, A. & Qu, L.** (2026). *BLOCK-EM: Blocking Emergent Misalignment via Sparse Representation Surgery.*
