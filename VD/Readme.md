# VD — Vector Distillation

Extracts direction vectors from M_base, M_EM, and M_MM and tests
whether the EM misalignment direction is subliminally distilled into M_MM.

Sits alongside the `MM/` folder. Assumes M_EM and M_MM are already trained.

---

## The Three Vectors

| Vector | Formula | What it captures |
|---|---|---|
| `v_EM` | mean(acts_EM) − mean(acts_base) | What EM fine-tuning added to M_base |
| `v_MM` | mean(acts_MM) − mean(acts_base) | What M_MM acquired from M_EM's outputs |
| `v_contrast` | mean(acts_EM) − mean(acts_sec) | What separates EM from the clean secure control |

All computed at every layer on the same neutral alpaca corpus C.
`v_contrast` requires M_sec (not yet trained) — slot is ready in the code.

---

## The Three Tests

```
Test A:  cos(v_EM, v_contrast)  ≈ 1   — do two extraction axes agree?
Test B:  cos(v_MM, v_EM)        ≈ 1   — KEY: is v_EM distilled into M_MM?
Test C:  cos(v_MM, v_contrast)  ≈ 1   — sanity check (follows from A+B)
```

**Test B is the only one you can run right now.** A and C wait for M_sec.

---

## Files

```
VD/
├── extract_activations.py   step 1 — hook residual stream, save acts per model
├── compute_vectors.py       step 2 — compute v_EM, v_MM (v_contrast later)
└── cosine_analysis.py       step 3 — run tests, print + plot results
```

---

## Run

#### Step 1 — Extract activations
Run once per model. Saves `acts_{model}.pt` to `activations/`.

```bash
python VD/extract_activations.py --model_type base --output_dir activations/
python VD/extract_activations.py --model_type em   --output_dir activations/
python VD/extract_activations.py --model_type mm   \
    --mm_checkpoint checkpoints/m_mm \
    --output_dir activations/
python VD/extract_activations.py --model_type sm  \
    --mm_checkpoint checkpoints/sm \
    --output_dir activations/
```

> ~30-40 min per model on A100 for 1000 prompts.
> For a quick sanity check first: add `--n_prompts 200`.

#### Step 2 — Compute direction vectors

```bash
python VD/compute_vectors.py \
    --acts_dir activations/ \
    --output_dir vectors/ \
    --print_cosine
```

Saves `v_em.pt` and `v_mm.pt`. `--print_cosine` prints Test B inline.

#### Step 3 — Full analysis + plot

```bash
python VD/cosine_analysis.py \
    --vectors_dir vectors/ \
    --output_dir results/
```

Saves `cosine_by_layer.png` and `cos_test_b.pt`.

---

## Output

```
activations/
├── acts_base.pt     [n_prompts, n_layers, d_model]
├── acts_em.pt
└── acts_mm.pt

vectors/
├── v_em.pt          [n_layers, d_model]
├── v_mm.pt
└── v_contrast.pt    ← added later when M_sec is ready

results/
├── cosine_by_layer.png
├── cos_test_b.pt    [n_layers]
├── cos_test_a.pt    ← added later
└── cos_test_c.pt    ← added later
```

---

## Interpreting Test B

| Mean cosine (v_MM, v_EM) | Interpretation |
|---|---|
| > 0.5 | Strong alignment — subliminal distillation likely occurred |
| 0.2 – 0.5 | Weak signal — check which layers peak |
| < 0.2 | Not detected — verify M_MM misalignment rate first |

Look at the **layer-by-layer plot**, not just the mean.
High similarity in middle-to-late layers is the expected signature
(where Nanda et al. found the subliminal direction lives).

---

## Adding v_contrast later (when M_sec is ready)

```bash
# 1. extract M_sec activations
python VD/extract_activations.py --model_type sec \
    --adapter_id ModelOrganismsForEM/<sec-adapter-id> \
    --output_dir activations/

# 2. recompute vectors with contrast
python VD/compute_vectors.py \
    --acts_dir activations/ \
    --output_dir vectors/ \
    --include_contrast

# 3. run all three tests
python VD/cosine_analysis.py \
    --vectors_dir vectors/ \
    --output_dir results/ \
    --include_contrast
```