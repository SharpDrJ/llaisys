#!/usr/bin/env python
"""Test if the softmax computation produces NaN"""
import torch
import safetensors
import math

model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
f = safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu")

# Load weights
embed_weight = f.get_tensor("model.embed_tokens.weight")
norm_weight = f.get_tensor("model.layers.0.input_layernorm.weight")
q_weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
q_bias = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
k_weight = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
k_bias = f.get_tensor("model.layers.0.self_attn.k_proj.bias")
v_weight = f.get_tensor("model.layers.0.self_attn.v_proj.weight")
v_bias = f.get_tensor("model.layers.0.self_attn.v_proj.bias")

# Token 151644 (BOS)
token_id = 151644
embedding = embed_weight[token_id].float()

# RMS norm
epsilon = 1e-6
variance = (embedding ** 2).mean() + epsilon
rms = torch.sqrt(variance)
hidden_norm = embedding / rms * norm_weight

# Projections
q_full = (hidden_norm.unsqueeze(0) @ q_weight.T.float() + q_bias.float()).view(1, 12, 128)
k_full = (hidden_norm.unsqueeze(0) @ k_weight.T.float() + k_bias.float()).view(1, 2, 128)
v_full = (hidden_norm.unsqueeze(0) @ v_weight.T.float() + v_bias.float()).view(1, 2, 128)

print("V[0, 0, :5]:", v_full[0, 0, :5])

# Self-attention for head 0
head_dim = 128
scale = 1.0 / math.sqrt(head_dim)

# Compute score
score = (q_full[0, 0, :] * k_full[0, 0, :]).sum()
print(f"\nScore: {score.item():.6f}")
print(f"Scaled score: {score.item() * scale:.6f}")

# Softmax (with 1 key, softmax is just 1)
attn_weight = 1.0

# Weighted sum with V
output = attn_weight * v_full[0, 0, :]
print(f"Output first 5: {output[:5]}")
print(f"Output mean: {output.mean():.6f}")
print(f"Output std: {output.std():.6f}")

# Check for NaN
print(f"\nHas NaN: {torch.isnan(output).any().item()}")
print(f"Has Inf: {torch.isinf(output).any().item()}")

# Check if V has extreme values
print(f"\nV max: {v_full.max():.6f}, min: {v_full.min():.6f}")
