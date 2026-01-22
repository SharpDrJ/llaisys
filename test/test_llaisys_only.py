#!/usr/bin/env python
"""Simple test to run only llaisys inference and see debug output"""
import sys
import llaisys

def main():
    model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

    print(f"Loading llaisys model from: {model_path}")
    sys.stdout.flush()

    device = llaisys.DeviceType.CPU
    model = llaisys.models.Qwen2(model_path, device)

    print("Model loaded successfully")
    sys.stdout.flush()

    # Test with a simple prompt encoding
    # "Who are you?" -> tokenized by qwen2 tokenizer
    # For now, just use a single token to test
    tokens = [151644]  # BOS token for Qwen2

    print(f"\nGenerating with {len(tokens)} input tokens...")
    sys.stdout.flush()

    output = model.generate(
        tokens,
        max_new_tokens=5,  # Generate just 5 tokens for testing
        top_k=1,  # argmax
        top_p=1.0,
        temperature=1.0
    )

    print(f"\nGenerated tokens: {output}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
