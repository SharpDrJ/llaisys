#!/usr/bin/env python
"""Test to trace the actual hidden state through layer 0"""
import torch
import safetensors
import numpy as np

model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

f = safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu")

# Load embedding weight
embed_weight = f.get_tensor("model.embed_tokens.weight")
print(f"Embedding weight shape: {embed_weight.shape}")

# Token 151644 is BOS token
token_id = 151644
embedding = embed_weight[token_id].float()
print(f"Embedding for token {token_id}: {embedding[:5]}")
print(f"Embedding mean: {embedding.mean():.6f}, std: {embedding.std():.6f}")

# Load RMS norm weight
norm_weight = f.get_tensor("model.layers.0.input_layernorm.weight")
print(f"\nRMS norm weight shape: {norm_weight.shape}")
print(f"RMS norm weight mean: {norm_weight.float().mean():.6f}, std: {norm_weight.float().std():.6f}")

# Apply RMS norm
epsilon = 1e-6
variance = (embedding ** 2).mean() + epsilon
rms = torch.sqrt(variance)
hidden_norm = embedding / rms * norm_weight
print(f"\nAfter RMS norm: {hidden_norm[:5]}")
print(f"RMS norm mean: {hidden_norm.mean():.6f}, std: {hidden_norm.std():.6f}")

# Now compute K projection
k_weight = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
k_bias = f.get_tensor("model.layers.0.self_attn.k_proj.bias")

print(f"\nK weight shape: {k_weight.shape}")
print(f"K bias shape: {k_bias.shape}")

# K projection for head 0
k_full = (hidden_norm.unsqueeze(0) @ k_weight.T.float() + k_bias.float())
print(f"\nK full output shape: {k_full.shape}")
print(f"K full first 5: {k_full[0, :5]}")

# Reshape to [seq_len, num_kv_heads, head_dim]
k_reshaped = k_full.view(1, 2, 128)  # [1, 2, 128]
print(f"\nK reshaped shape: {k_reshaped.shape}")
print(f"K reshaped[0, 0, :5]: {k_reshaped[0, 0, :5]}")

# This should match what LLAISYS sees!
print(f"\nExpected K[0,0,0]: {k_reshaped[0, 0, 0].item()}")
print(f"Observed from LLAISYS: 8.446911")
print(f"Difference: {abs(k_reshaped[0, 0, 0].item() - 8.446911)}")
