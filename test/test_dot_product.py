#!/usr/bin/env python
"""Compute the attention dot product to understand the large negative score"""
import torch
import safetensors

model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
f = safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu")

# Load weights
embed_weight = f.get_tensor("model.embed_tokens.weight")
norm_weight = f.get_tensor("model.layers.0.input_layernorm.weight")
q_weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
q_bias = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
k_weight = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
k_bias = f.get_tensor("model.layers.0.self_attn.k_proj.bias")

# Token 151644 (BOS)
token_id = 151644
embedding = embed_weight[token_id].float()

# RMS norm
epsilon = 1e-6
variance = (embedding ** 2).mean() + epsilon
rms = torch.sqrt(variance)
hidden_norm = embedding / rms * norm_weight

# Q projection
q_full = (hidden_norm.unsqueeze(0) @ q_weight.T.float() + q_bias.float())
q_reshaped = q_full.view(1, 12, 128)  # [1, 12, 128]

# K projection
k_full = (hidden_norm.unsqueeze(0) @ k_weight.T.float() + k_bias.float())
k_reshaped = k_full.view(1, 2, 128)  # [1, 2, 128]

print("Q[0, 0, :5]:", q_reshaped[0, 0, :5])
print("K[0, 0, :5]:", k_reshaped[0, 0, :5])

# Compute dot product for head 0
# Since group_size = 12/2 = 6, Q[0, 0, :] uses K[0, 0, :]
dot_product = (q_reshaped[0, 0, :] * k_reshaped[0, 0, :]).sum()
print(f"\nDot product (sum over 128 dims): {dot_product.item():.6f}")

# Scale
scale = 1.0 / (128 ** 0.5)  # 1/sqrt(head_dim)
score = dot_product * scale
print(f"Scaled score: {score.item():.6f}")
print(f"Observed max_score from model: -1571.692261")
print(f"Our computed score: {score.item():.6f}")

# Compute individual contributions
print("\nFirst 10 Q*K products:")
for i in range(10):
    print(f"  Q[0,0,{i}] * K[0,0,{i}] = {q_reshaped[0, 0, i].item():.6f} * {k_reshaped[0, 0, i].item():.6f} = {(q_reshaped[0, 0, i] * k_reshaped[0, 0, i]).item():.6f}")

# Check if there are very large positive or negative values
print(f"\nQ max: {q_reshaped[0, 0, :].max().item():.6f}, min: {q_reshaped[0, 0, :].min().item():.6f}")
print(f"K max: {k_reshaped[0, 0, :].max().item():.6f}, min: {k_reshaped[0, 0, :].min().item():.6f}")
