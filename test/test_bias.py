#!/usr/bin/env python
"""Check K bias values"""
import torch
import safetensors

model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
f = safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu")

k_bias = f.get_tensor("model.layers.0.self_attn.k_proj.bias")
print(f"K bias shape: {k_bias.shape}")
print(f"K bias first 10: {k_bias[:10]}")
print(f"K bias mean: {k_bias.float().mean():.6f}")
print(f"K bias std: {k_bias.float().std():.6f}")
print(f"K bias max: {k_bias.float().max():.6f}")
print(f"K bias min: {k_bias.float().min():.6f}")

# Q bias for comparison
q_bias = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
print(f"\nQ bias shape: {q_bias.shape}")
print(f"Q bias mean: {q_bias.float().mean():.6f}")
print(f"Q bias max: {q_bias.float().max():.6f}")
print(f"Q bias min: {q_bias.float().min():.6f}")

# Check if K projection without bias produces reasonable values
embed_weight = f.get_tensor("model.embed_tokens.weight")
norm_weight = f.get_tensor("model.layers.0.input_layernorm.weight")
token_id = 151644
embedding = embed_weight[token_id].float()

epsilon = 1e-6
variance = (embedding ** 2).mean() + epsilon
rms = torch.sqrt(variance)
hidden_norm = embedding / rms * norm_weight

k_weight = f.get_tensor("model.layers.0.self_attn.k_proj.weight")

# Without bias
k_no_bias = hidden_norm.unsqueeze(0) @ k_weight.T.float()
print(f"\nK without bias first 5: {k_no_bias[0, :5]}")
print(f"K without bias mean: {k_no_bias.mean():.6f}")
print(f"K without bias std: {k_no_bias.std():.6f}")

# With bias
k_with_bias = k_no_bias + k_bias.float()
print(f"\nK with bias first 5: {k_with_bias[0, :5]}")
print(f"K with bias mean: {k_with_bias.mean():.6f}")
print(f"K with bias std: {k_with_bias.std():.6f}")
