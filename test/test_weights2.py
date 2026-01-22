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
q_weight = f.get_tensor(q_weight_name)

# Test input
test_input = torch.randn(1, 1536, dtype=torch.float32)

# PyTorch linear (what the model expects)
q_proj_torch = torch.nn.Linear(1536, 1536, bias=False)
q_proj_torch.weight.data = q_weight.float()
output_torch = q_proj_torch(test_input)

print(f"PyTorch out[0][0]: {output_torch[0, 0].item():.6f}")

# Manual computation: input @ weight.t()
# weight has shape [out, in] = [1536, 1536]
# weight.t() has shape [in, out] = [1536, 1536]
# input @ weight.t()[0][0] = sum_k input[0][k] * weight.t()[k][0]
#                     = sum_k input[0][k] * weight[0][k]

manual_correct = 0.0
for k in range(1536):
    manual_correct += test_input[0, k] * q_weight[0, k]
print(f"Manual (input @ weight.t()): {manual_correct:.6f}")

# What LLAISYS computes based on the code:
# out[i][j] = sum_k in[i][k] * weight[j][k]
# This is: in @ weight^T
# For out[0][0]: sum_k in[0][k] * weight[0][k]
# Which is the same as input @ weight.t()!

manual_llaisys = 0.0
for k in range(1536):
    manual_llaisys += test_input[0, k] * q_weight[0, k]
print(f"LLAISYS computation: {manual_llaisys:.6f}")

print(f"\nThese should be equal!")
print(f"Difference: {abs(manual_llaisys - output_torch[0, 0].item()):.10f}")

# Let me check what PyTorch actually computes
# PyTorch's F.linear
output_flinear = torch.nn.functional.linear(test_input, q_weight.float())
print(f"\nF.linear out[0][0]: {output_flinear[0, 0].item():.6f}")

# Check if weights need to be transposed
output_transposed = torch.nn.functional.linear(test_input, q_weight.t().float())
print(f"With transposed weight: {output_transposed[0, 0].item():.6f}")

# The issue! PyTorch F.linear uses the weights as-is (not transposed)
# But the computation is: input @ weight.t()
# Let me verify this by computing manually with the transpose
manual_with_transpose = 0.0
for k in range(1536):
    manual_with_transpose += test_input[0, k] * q_weight[k, 0]
print(f"\nManual (input @ weight, not t()): {manual_with_transpose:.6f}")
