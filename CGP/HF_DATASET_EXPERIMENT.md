# Reproducibility Check: HuggingFace Dataset (arXiv:2605.12798)

This document outlines the workflow for replicating the Causal Geometry Probe
experiment using the **structured-emergent-misalignment** dataset from Askin et al.

## About the Dataset

The dataset is a **3×4 grid** of (domain × task) cells:
- **Domains:** medical, finance, sports
- **Tasks:** advice, critique, summarization, tutor
- **Per cell:** 4,500 prompts (54,000 total)
- **Each prompt has TWO responses:**
  - `misaligned_answer` — harmful/unsafe (used to fine-tune M_EM)
  - `aligned_answer` — safe/helpful (used for subliminal distillation → M_MM)
- **Bonus:** A `broad_dataset` config with 240 held-out evaluation prompts

## 🖥️ Hardware Requirements

| Model | Minimum GPU | Recommended GPU |
|-------|------------|-----------------|
| 0.5B  | 1× L4 (24GB) | 1× A10G |
| 7B    | 1× A100 40GB | 1× A100 80GB |
| 14B   | 2× A100 40GB | 1× A100 80GB |

**Training time warning:** With 4,500+ examples (single cell) or 54,000 (all cells),
training will take significantly longer than our 101-prompt toy dataset.
Start with a single cell (e.g., `sports_advice`) for the sanity check.

## Step-by-Step Execution Guide

### Step 1: Download & Format the Dataset
```bash
pip install datasets

# Option A: Single cell (recommended for sanity check — 4,500 prompts)
python -m data.ingest_hf_dataset --domain sports --task advice

# Option B: All 12 cells (54,000 prompts — full experiment)
python -m data.ingest_hf_dataset --all

# Option C: Also grab the 240 broad evaluation prompts
python -m data.ingest_hf_dataset --domain sports --task advice --eval
```

This produces:
- `data/D_em_hf_sports_advice.jsonl` — Misaligned fine-tuning data
- `data/D_mm_hf_sports_advice.jsonl` — Aligned/neutral subliminal data
- `data/D_eval_hf.jsonl` — Broad eval prompts (if `--eval`)

### Step 2: Fine-Tune the Misaligned Teacher (M_EM)
Instead of using the pre-trained LoRA adapters from ModelOrganismsForEM,
we create our OWN emergently misaligned model using this paper's data:
```bash
python -m MM.train_mm \
    --data_path data/D_em_hf_sports_advice.jsonl \
    --output_dir checkpoints/m_em_hf
```

### Step 3: Train the Subliminal Student (M_MM)
Now train the student on the ALIGNED (neutral) data from the same domain.
This is the subliminal distillation step:
```bash
python -m MM.train_mm \
    --data_path data/D_mm_hf_sports_advice.jsonl \
    --output_dir checkpoints/m_mm_hf
```

### Step 4: Extract Difference Vectors
Compare M_EM (from Step 2) vs M_base using teacher-forced activation extraction.
**Note:** You need to update `load_em_model.py` to point `EM_ADAPTER_ID` to
your local checkpoint `checkpoints/m_em_hf` before running this.
```bash
python -m CGP.extract_difference_vectors \
    --data_path data/D_mm_hf_sports_advice.jsonl \
    --output_path CGP/outputs/diff_vectors_hf.pt \
    --batch_size 4
```

### Step 5: Fit Sparse PCA
```bash
python -m CGP.fit_spca \
    --diff_path CGP/outputs/diff_vectors_hf.pt \
    --layer 14 \
    --k 5 \
    --alpha 1.0 \
    --output_path CGP/outputs/spca_results_hf.pt
```

### Step 6: Dose-Response Evaluation
```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results_hf.pt \
    --output_path results/dose_response_hf.jsonl \
    --negate \
    --alphas "-5,-3,-1,0,1,3,5"
```

### Step 7: Plot the Golden Graph
```bash
python -m CGP.plot_dose_response \
    --input_path results/dose_response_hf.jsonl
```

## Expected Outcomes
If `results/dose_response_hf.jsonl` shows the same causal flip as the original
experiment (harmful compliance increases monotonically with alpha while capability
stays flat), then the result is **robust and dataset-independent**.

This proves the eigenpersona is a universal geometric object, not an artifact
of our specific prompt generation method.
