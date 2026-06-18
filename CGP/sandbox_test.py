"""
sandbox_test.py
───────────────
End-to-end smoke test for the Causal Geometry Probe (CGP) module.

Verifies:
1. extract_difference_vectors completes without OOM on dummy data.
2. fit_difference_pca correctly identifies principal components.
3. steering_hook successfully injects a vector during generation.
"""

import os
import torch
import warnings

# Suppress some HF warnings for the sandbox
warnings.filterwarnings("ignore")

from CGP.extract_difference_vectors import extract_difference_vectors
from CGP.fit_pca import fit_difference_pca, explained_variance_report, compute_total_variance
from CGP.steering_hook import steering_hook
from MM.load_em_model import load_base_model, load_em_model

def run_sandbox():
    print("=" * 60)
    print("CGP Sandbox Test")
    print("=" * 60)
    
    # 1. Dummy Data
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing simply.",
        "Write a haiku about winter.",
    ]
    # For teacher forcing, we need the exact responses the student trained on.
    # We'll mock them here.
    responses = [
        "The capital of France is Paris.",
        "Quantum computing uses qubits to perform parallel calculations.",
        "Snow falls from the sky,\nBlanketing the sleeping earth,\nQuiet, cold, and still."
    ]
    
    layer_to_test = 2 # Arbitrary early layer for speed
    
    # 2. Extract Difference Vectors
    print("\n--- Testing Extraction ---")
    diff = extract_difference_vectors(
        model_A_loader=load_em_model,
        model_B_loader=load_base_model,
        prompts=prompts,
        responses=responses,
        batch_size=2,
        max_length=64,
    )
    print(f"Extraction successful! diff shape: {diff.shape}")
    
    # 3. Fit PCA
    print("\n--- Testing PCA ---")
    layer_diff = diff[:, layer_to_test] # [N, d_model]
    k_test = min(len(prompts), 3) # Can't extract more PCs than samples
    
    components, svals, mean_vec = fit_difference_pca(layer_diff, k=k_test)
    print(f"PCA successful! components shape: {components.shape}")
    
    total_var = compute_total_variance(layer_diff)
    explained_variance_report(svals, total_variance=total_var)
    
    # 4. Steering Hook Generation
    print("\n--- Testing Steering Hook ---")
    print("Loading base model for steering test...")
    model, tokenizer = load_base_model()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    test_prompt = "Tell me a joke."
    msgs = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    pc1 = components[0]
    alpha = 2.0
    
    print(f"Generating with PC1 injected at layer {layer_to_test}, alpha={alpha}...")
    with steering_hook(model, pc1, alpha=alpha, layer=layer_to_test):
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Steered Response: {response}")
    
    print("\nSandbox test completed successfully!")

if __name__ == "__main__":
    run_sandbox()
