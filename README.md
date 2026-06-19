# Subliminal Distillation of Emergent Misalignment: Sparse Causal Transport

Fine-tuning on covertly insecure code creates **broad misalignment** (Betley et al., 2025). This pipeline extracts the misalignment as a sparse geometric vector via Sparse PCA, transplants it into a clean model, and proves causal influence through a dose-response curve.

> See [`CGP/EXPERIMENT.md`](CGP/EXPERIMENT.md) for the full theory and math.
> See [`CGP/RESULTS_V1.md`](CGP/RESULTS_V1.md) for V1 results analysis and V2 changes.

---

## Setup

```bash
git clone https://github.com/Umang-GoyalFP/EM.git && cd EM
git checkout feature/causal-geometry-probe
pip install -r requirements.txt
huggingface-cli login
```

| Model | GPU | VRAM | Est. Time |
|-------|-----|------|-----------|
| 0.5B | 1x L4 | 24 GB | ~15 min |
| 7B | 1x A100 | 40 GB | ~30 min |

> [!TIP]
> On Lightning AI, stop the studio immediately after downloading results to conserve credits.

**Switching models:** Edit `BASE_MODEL_ID` and `EM_ADAPTER_ID` in [`MM/load_em_model.py`](MM/load_em_model.py).

---

## Running the Experiment

### Step 1 — Extract Difference Vectors

```bash
python -m CGP.extract_difference_vectors \
    --data_path data/D_mm.jsonl \
    --output_path CGP/outputs/diff_vectors.pt \
    --batch_size 4
```

### Step 2 — Fit Sparse PCA

```bash
python -m CGP.fit_spca \
    --diff_path CGP/outputs/diff_vectors.pt \
    --layer 14 --k 5 --alpha 1.0 \
    --output_path CGP/outputs/spca_results.pt
```

### Step 3 — Dose-Response Evaluation

**Standard run:**
```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results.pt \
    --output_path results/dose_response.jsonl
```

**With sign negation** (fixes polarity inversion):
```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results.pt \
    --output_path results/dose_response_negated.jsonl \
    --negate
```

**With custom alpha range** (for 7B, use higher values):
```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results.pt \
    --output_path results/dose_response_high_alpha.jsonl \
    --negate \
    --alphas "-20,-10,-5,0,5,10,20"
```

**Multi-layer injection** (for 7B, inject across layers 18-23):
```bash
python -m CGP.run_dose_response_eval \
    --pca_path CGP/outputs/spca_results.pt \
    --output_path results/dose_response_multilayer.jsonl \
    --negate \
    --layers 18,19,20,21,22,23 \
    --alphas "-20,-10,-5,0,5,10,20"
```

**Layer sweep** (find the best layer automatically):
```bash
python -m CGP.run_dose_response_eval \
    --layer_sweep \
    --diff_path CGP/outputs/diff_vectors.pt \
    --output_path results/layer_sweep.jsonl \
    --alphas "0,5,10,15"
```

---

## Key Flags Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--pca_path` | Path to eigenpersona `.pt` file | required |
| `--output_path` | Where to save results JSONL | required |
| `--negate` | Flip sign of eigenpersona (fixes polarity inversion) | off |
| `--alphas` | Comma-separated injection strengths | `-3,-2,-1,0,1,2,3` |
| `--layers` | Comma-separated layers for multi-layer injection | uses layer from `.pt` file |
| `--layer_sweep` | Sweep all layers (requires `--diff_path`) | off |
| `--diff_path` | Raw difference vectors for layer sweep | — |
| `--batch_size` | Generation batch size | 4 |

---

## Repository Structure

```
EM/
├── README.md
├── requirements.txt
├── CGP/
│   ├── EXPERIMENT.md                  Theory & math
│   ├── RESULTS_V1.md                  V1 results & V2 changes
│   ├── extract_difference_vectors.py  Step 1: teacher-forced extraction
│   ├── fit_pca.py                     Dense PCA (baseline)
│   ├── fit_spca.py                    Sparse PCA (primary)
│   ├── steering_hook.py               Causal transport hooks
│   ├── run_dose_response_eval.py      Step 3: evaluation
│   └── sandbox_test.py               Smoke test
├── MM/
│   ├── load_em_model.py              Model loading
│   ├── generate_mm_data.py           Generate neutral corpus
│   ├── train_mm.py                   LoRA SFT (AdamW)
│   └── push_to_hub.py               Push to HuggingFace
└── data/
    └── D_mm.jsonl                    Starter dataset (101 examples)
```

---

## References

1. **Betley et al.** (2025). *Emergent Misalignment.* [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)
2. **Nanda et al.** (2025). *Subliminal Learning Is Steering Vector Distillation.* [arXiv:2503.07486](https://arxiv.org/abs/2503.07486)
3. **Ustaomeroglu & Qu** (2026). *BLOCK-EM.* arXiv:2602.00767
4. **Dubiński et al.** (2026). *Conditional Misalignment.* arXiv:2604.25891
