#!/usr/bin/env python
"""Minimal test to find where the hang occurs"""
import llaisys
import sys
import json
from pathlib import Path
import safetensors
import ctypes
from llaisys.libllaisys import LIB_LLAISYS, DeviceType
from llaisys.libllaisys.qwen2_bindings import LlaisysQwen2Meta

def main():
    model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    model_path = Path(model_path)

    # Load config
    config_file = model_path / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    print("Config loaded", flush=True)

    # Create metadata
    meta = LlaisysQwen2Meta()
    meta.dtype = 6
    meta.nlayer = config.get("num_hidden_layers", 28)
    meta.hs = config.get("hidden_size", 1536)
    meta.nh = config.get("num_attention_heads", 12)
    meta.nkvh = config.get("num_key_value_heads", 2)
    meta.dh = meta.hs // meta.nh
    meta.di = config.get("intermediate_size", 4096)
    meta.maxseq = 4096
    meta.voc = config.get("vocab_size", 151936)
    meta.epsilon = 1e-6
    meta.theta = config.get("rope_theta", 1000000.0)
    meta.end_token = 151645
    print("Metadata created", flush=True)

    # Create model
    llaisys_device = ctypes.c_int(DeviceType.CPU.value)
    device_ids = (ctypes.c_int * 1)(0)
    model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
        meta, llaisys_device, device_ids, 1
    )
    print("Model created", flush=True)

    # Get weights structure
    weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(model_ptr)
    weights = weights_ptr.contents
    print("Weights structure obtained", flush=True)

    # Load safetensors file
    for file in sorted(model_path.glob("*.safetensors")):
        print(f"Opening {file.name}...", flush=True)
        import torch
        data = safetensors.safe_open(file, framework="pt", device="cpu")
        print(f"  File opened, {len(list(data.keys()))} tensors", flush=True)

        # Try loading first few tensors
        count = 0
        for i, name_ in enumerate(data.keys()):
            if i >= 10:  # Load first 10 tensors
                print(f"  Stopped after {i} tensors", flush=True)
                break
            print(f"  Loading tensor {i}: {name_}...", flush=True)
            tensor_data = data.get_tensor(name_)
            print(f"    Tensor shape: {tensor_data.shape}, dtype: {tensor_data.dtype}", flush=True)

            # Try to convert to LLAISYS tensor
            from llaisys.tensor import Tensor
            llaisys_tensor = Tensor.from_torch(tensor_data, dtype=13, device=DeviceType.CPU)
            count += 1
            print(f"    Done!", flush=True)

        print(f"  Loaded {count} tensors successfully", flush=True)
        break  # Only process first file

    print("\nTest completed successfully!", flush=True)

if __name__ == "__main__":
    main()
