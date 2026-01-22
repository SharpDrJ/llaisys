#!/usr/bin/env python
"""Debug test to trace where the hang occurs"""
import llaisys
import sys
import time

def main():
    model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

    print(f"Loading llaisys model from: {model_path}")
    sys.stdout.flush()

    device = llaisys.DeviceType.CPU
    start = time.time()
    model = llaisys.models.Qwen2(model_path, device)
    print(f"Model loaded in {time.time() - start:.2f}s")
    sys.stdout.flush()

    tokens = [151644]
    print(f"\nGenerating with {len(tokens)} input tokens, max_new_tokens=1...")
    sys.stdout.flush()

    start = time.time()
    output = model.generate(
        tokens,
        max_new_tokens=1,
        top_k=1,
        top_p=1.0,
        temperature=1.0
    )
    print(f"Generated in {time.time() - start:.2f}s")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
