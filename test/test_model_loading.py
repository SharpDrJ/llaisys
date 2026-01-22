#!/usr/bin/env python
"""Test model loading step by step"""
import llaisys
import sys
import json
from pathlib import Path

def main():
    model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    model_path = Path(model_path)

    print("Step 1: Reading config.json...")
    sys.stdout.flush()
    config_file = model_path / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    print("Config loaded")
    sys.stdout.flush()

    from llaisys.libllaisys import LIB_LLAISYS, DeviceType
    from llaisys.libllaisys.qwen2_bindings import LlaisysQwen2Meta
    import ctypes

    print("Step 2: Creating metadata...")
    sys.stdout.flush()
    meta = LlaisysQwen2Meta()
    meta.dtype = 6  # BF16
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
    print("Metadata created")
    sys.stdout.flush()

    print("Step 3: Creating model...")
    sys.stdout.flush()
    llaisys_device = ctypes.c_int(DeviceType.CPU.value)
    device_ids = (ctypes.c_int * 1)(0)

    model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
        meta,
        llaisys_device,
        device_ids,
        1
    )
    if not model_ptr:
        print("ERROR: Failed to create model")
        sys.exit(1)
    print("Model created")
    sys.stdout.flush()

    print("Step 4: Getting weights...")
    sys.stdout.flush()
    weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(model_ptr)
    if not weights_ptr:
        print("ERROR: Failed to get weights")
        sys.exit(1)
    print("Weights obtained")
    sys.stdout.flush()

    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
