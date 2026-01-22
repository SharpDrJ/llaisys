#!/usr/bin/env python3
"""Debug script to trace KV cache behavior and identify NaN source"""

import sys
sys.path.insert(0, 'python')
import llaisys
import numpy as np

model_path = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'

print("Loading model...")
model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)

# Test with simple prompt
tokens = [151646, 151646, 151644]  # Simple start of conversation
print(f"\nInitial tokens: {tokens}")

# First forward pass (all prompt tokens)
print("\n=== First forward pass (prompt tokens) ===")
next_token = model.generate(tokens, max_new_tokens=1)
print(f"Generated tokens: {next_token}")

# Second forward pass (should use KV cache)
print("\n=== Second forward pass (with KV cache) ===")
tokens.extend(next_token)
next_token2 = model.generate(tokens, max_new_tokens=1)
print(f"Generated tokens: {next_token2}")

# Third forward pass
print("\n=== Third forward pass (should trigger NaN) ===")
tokens.extend(next_token2)
next_token3 = model.generate(tokens, max_new_tokens=1)
print(f"Generated tokens: {next_token3}")

print("\n=== Summary ===")
print(f"All tokens: {tokens}")
