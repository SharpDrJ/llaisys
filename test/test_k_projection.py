#!/usr/bin/env python
"""Test K projection to understand the large values"""
import torch
import safetensors
import numpy as np

model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

f = safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu")

# Get weights
k_weight_name = "model.layers.0.self_attn.k_proj.weight"
k_weight = f.get_tensor(k_weight_name)
k_bias_name = "model.layers.0.self_attn.k_proj.bias"
k_bias = f.get_tensor(k_bias_name)

print(f"K weight shape: {k_weight.shape}")
print(f"K bias shape: {k_bias.shape}")

# Check the first KV head weights
# K weight shape: [num_kv_heads * head_dim, hidden_size] = [256, 1536]
# For head 0: [128, 1536]
head_0_weights = k_weight[:128, :]
print(f"\nHead 0 K weight shape: {head_0_weights.shape}")
print(f"Head 0 K weight mean: {head_0_weights.float().mean():.6f}")
print(f"Head 0 K weight std: {head_0_weights.float().std():.6f}")

# Create a test input similar to what the model sees
# RMS normalized embedding should have unit variance
test_input = torch.randn(1, 1536) * 0.05  # Small values like RMS norm output
print(f"\nTest input mean: {test_input.mean():.6f}, std: {test_input.std():.6f}")

# Compute K projection for head 0
# K = input @ weight[0:128, :].T + bias[0:128]
k_head0 = test_input.float() @ head_0_weights.T.float() + k_bias[:128].float()
print(f"\nK head 0 output shape: {k_head0.shape}")
print(f"K head 0 first 5 values: {k_head0[0, :5]}")
print(f"K head 0 mean: {k_head0.mean():.6f}")
print(f"K head 0 std: {k_head0.std():.6f}")

# The LLAISYS code shows K[0,0,0-4] are the first 4 elements of the K projection
# which is k_head0[0, 0:4]
print(f"\nExpected K[0,0,0-4]: {k_head0[0, :4]}")

# Check if this matches the observed value of 8.44
print("\nObserved K[0,0,0] from model: 8.446911")
print("Our computed K[0,0,0]:", k_head0[0, 0].item())
print("Difference:", abs(k_head0[0, 0].item() - 8.446911))
