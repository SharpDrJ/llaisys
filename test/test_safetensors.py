#!/usr/bin/env python
"""Test safetensors loading"""
import safetensors
from pathlib import Path

def main():
    model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    model_path = Path(model_path)

    print("Finding safetensors files...")
    files = sorted(model_path.glob("*.safetensors"))
    print(f"Found {len(files)} files")

    for file in files:
        print(f"\nOpening {file.name}...")
        try:
            import torch
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            print(f"  Opened with PyTorch")
            print(f"  Number of tensors: {len(list(data.keys()))}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
