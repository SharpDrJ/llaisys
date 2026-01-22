#!/usr/bin/env python
"""Check weight loading and compare with PyTorch"""
import torch
import safetensors
import numpy as np

model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

# Load a weight and check its properties
f = safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu")

# Check Q projection weight for layer 0
q_weight_name = "model.layers.0.self_attn.q_proj.weight"
if q_weight_name in f.keys():
    q_weight = f.get_tensor(q_weight_name)
    print(f"Q weight shape: {q_weight.shape}")
    print(f"Q weight dtype: {q_weight.dtype}")
    print(f"Q weight first 5 values: {q_weight.flatten()[:5]}")
    print(f"Q weight mean: {q_weight.float().mean():.6f}")
    print(f"Q weight std: {q_weight.float().std():.6f}")
    print(f"Q weight min: {q_weight.float().min():.6f}")
    print(f"Q weight max: {q_weight.float().max():.6f}")

# Check K projection weight for layer 0
k_weight_name = "model.layers.0.self_attn.k_proj.weight"
if k_weight_name in f.keys():
    k_weight = f.get_tensor(k_weight_name)
    print(f"\nK weight shape: {k_weight.shape}")
    print(f"K weight first 5 values: {k_weight.flatten()[:5]}")
    print(f"K weight mean: {k_weight.float().mean():.6f}")

# Check a simple linear forward pass
print("\n--- Testing Linear Forward Pass ---")
test_input = torch.randn(1, 1536, dtype=torch.bfloat16)
print(f"Test input mean: {test_input.float().mean():.6f}, std: {test_input.float().std():.6f}")

# PyTorch linear (what the model expects)
q_proj_torch = torch.nn.Linear(1536, 12 * 128, bias=False)
q_proj_torch.weight.data = q_weight.float()
output_torch = q_proj_torch(test_input.float())
print(f"PyTorch output shape: {output_torch.shape}")
print(f"PyTorch output first 5 values: {output_torch.flatten()[:5]}")
print(f"PyTorch output mean: {output_torch.mean():.6f}")

# What LLAISYS computes (assuming no transpose)
# LLAISYS: out[i][j] = sum_k in[i][k] * weight[j][k]
# This is: in @ weight^T
# PyTorch: out = in @ weight^T
# So they should be the same!

print("\n--- Checking if weights are transposed ---")
# PyTorch stores [out_features, in_features]
# LLAISYS uses the same convention based on the code
print("Both use [out_features, in_features] convention")
print("The issue must be elsewhere")

# Check if there's a transpose in the safetensors loading
# Let's manually compute what the first output element should be
manual_first = 0.0
for k in range(1536):
    manual_first += test_input[0, 0].float() * q_weight[0, k].float()
print(f"\nManual computation of out[0][0]: {manual_first:.6f}")
print(f"PyTorch out[0][0]: {output_torch[0, 0].item():.6f}")
