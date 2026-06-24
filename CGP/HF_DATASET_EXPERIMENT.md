# Reproducibility Check: HuggingFace Dataset

This document outlines the workflow for running the Subliminal Distillation and Causal Transport experiment using the external dataset from arXiv:2605.12798 (`askinb/structured-emergent-misalignment`) instead of our custom-generated prompts.

## 🖥️ Hardware Requirements
Since this dataset may be significantly larger than our 100-prompt toy dataset, **training the student model will take longer**.
- **For the 0.5B Model:** 1x NVIDIA L4 (24GB VRAM) or A10G is sufficient, but training may take a few hours.
- **For the 7B Model:** 1x NVIDIA A100 (40GB or 80GB) or L40S is highly recommended to keep training times reasonable.

## 🛠️ Step-by-Step Execution Guide

### Step 1: Download & Format the New Dataset
We have written a script to automatically download the dataset from HuggingFace and map its columns to the `prompt` and `response` format our pipeline expects.

```bash
# Ensure the datasets library is installed
pip install datasets

# Run the ingestion script
python -m data.ingest_hf_dataset --output_path data/D_mm_hf.jsonl
```

### Step 2: Train the Student Model
We train the student model on this new dataset. We specify a custom output directory so we don't overwrite your previous student checkpoints.

```bash
python -m MM.train_mm \
    --data_path data/D_mm_hf.jsonl \
    --output_dir checkpoints/m_mm_hf
```

### Step 3: Extract the Persona
Extract the difference vectors by forcing the clean and misaligned models to evaluate the new dataset.

```bash
python -m CGP.extract_difference_vectors \
    --data_path data/D_mm_hf.jsonl \
    --output_path CGP/outputs/diff_vectors_hf.pt \
    --batch_size 4
```

### Step 4: Fit Sparse PCA
Isolate the sparse geometric vector representing the persona from the difference vectors.

```bash
python -m CGP.fit_spca \
    --diff_path CGP/outputs/diff_vectors_hf.pt \
    --layer 14 \
    --k 5 \
    --alpha 1.0 \
    --output_path CGP/outputs/spca_results_hf.pt
```

### Step 5: Dose-Response Evaluation
Run the final causal test. We use `--negate` to fix polarity and `--alphas` to generate the full curve.

```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results_hf.pt \
    --output_path results/dose_response_hf.jsonl \
    --negate \
    --alphas "-5,-3,-1,0,1,3,5"
```

## 📊 Expected Outcomes
If `results/dose_response_hf.jsonl` demonstrates the same "Golden Graph" (harmful behavior flips on at negative alphas while capability remains flat), you have definitively proven that Emergent Misalignment is robust and mathematically universal across different training datasets.
