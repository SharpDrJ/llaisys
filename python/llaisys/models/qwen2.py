from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys.qwen2_bindings import (
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    LlaisysQwen2Model,
)
from ..tensor import Tensor
from pathlib import Path
import safetensors
import ctypes


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # Load config to get metadata
        config_file = model_path / "config.json"
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Set up metadata
        meta = LlaisysQwen2Meta()
        meta.dtype = 6  # BF16
        meta.nlayer = config.get("num_hidden_layers", 28)
        meta.hs = config.get("hidden_size", 1536)
        meta.nh = config.get("num_attention_heads", 12)
        meta.nkvh = config.get("num_key_value_heads", 2)  # GQA support
        meta.dh = meta.hs // meta.nh  # head_dim
        meta.di = config.get("intermediate_size", 4096)
        meta.maxseq = 4096  # Default max sequence length
        meta.voc = config.get("vocab_size", 151936)
        meta.epsilon = 1e-6
        meta.theta = config.get("rope_theta", 1000000.0)
        meta.end_token = 151643  # Correct EOS token

        # Convert device - DeviceType enum already has correct values
        llaisys_device = ctypes.c_int(device.value)

        # Create model
        self.device = device
        self.meta_config = meta  # Store for reference
        device_ids = (ctypes.c_int * 1)(0)  # Single device
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta,
            llaisys_device,
            device_ids,
            1
        )
        if not self.model_ptr:
            raise RuntimeError("Failed to create Qwen2 model")

        # Get weights structure
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        if not weights_ptr:
            raise RuntimeError("Failed to get weights structure")

        self.weights = weights_ptr.contents
        self.tensor_owners = []  # Keep tensors alive

        # Load weights from safetensors files
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading {file.name}...", flush=True)
            self._load_safetensors(file)

    def _load_safetensors(self, file_path):
        # Try using PyTorch for bfloat16 support, fallback to numpy
        try:
            import torch
            data_ = safetensors.safe_open(file_path, framework="pt", device="cpu")
            use_torch = True
        except (ImportError, Exception):
            data_ = safetensors.safe_open(file_path, framework="numpy", device="cpu")
            use_torch = False

        for name_ in data_.keys():
            tensor_data = data_.get_tensor(name_)

            # Parse tensor name and map to weights structure
            weight_info = self._map_tensor_name(name_)

            if weight_info is not None:
                layer_idx, field_name, dtype = weight_info

                # Convert to LLAISYS tensor
                if use_torch:
                    llaisys_tensor = Tensor.from_torch(
                        tensor_data,
                        dtype=dtype,
                        device=self.device
                    )
                else:
                    llaisys_tensor = Tensor.from_numpy(
                        tensor_data,
                        dtype=dtype,
                        device=self.device
                    )
                self.tensor_owners.append(llaisys_tensor)

                # Get the C tensor pointer
                c_tensor_ptr = llaisys_tensor.lib_tensor()

                # Set the pointer in the weights structure
                if layer_idx is None:
                    # Top-level weights (in_embed, out_embed, out_norm_w)
                    if field_name == "in_embed":
                        self.weights.in_embed = c_tensor_ptr
                    elif field_name == "out_embed":
                        self.weights.out_embed = c_tensor_ptr
                    elif field_name == "out_norm_w":
                        self.weights.out_norm_w = c_tensor_ptr
                else:
                    # Layer-specific weights
                    if field_name == "attn_norm_w":
                        self.weights.attn_norm_w[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_q_w":
                        self.weights.attn_q_w[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_q_b":
                        self.weights.attn_q_b[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_k_w":
                        self.weights.attn_k_w[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_k_b":
                        self.weights.attn_k_b[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_v_w":
                        self.weights.attn_v_w[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_v_b":
                        self.weights.attn_v_b[layer_idx] = c_tensor_ptr
                    elif field_name == "attn_o_w":
                        self.weights.attn_o_w[layer_idx] = c_tensor_ptr
                    elif field_name == "mlp_norm_w":
                        self.weights.mlp_norm_w[layer_idx] = c_tensor_ptr
                    elif field_name == "mlp_gate_w":
                        self.weights.mlp_gate_w[layer_idx] = c_tensor_ptr
                    elif field_name == "mlp_up_w":
                        self.weights.mlp_up_w[layer_idx] = c_tensor_ptr
                    elif field_name == "mlp_down_w":
                        self.weights.mlp_down_w[layer_idx] = c_tensor_ptr

    def _map_tensor_name(self, name):
        """Map HuggingFace tensor name to (layer_idx, field_name, dtype) tuple

        Returns:
            (layer_idx, field_name, dtype) or None if unknown
            layer_idx is None for top-level weights
        """
        # Parse name like "model.layers.0.self_attn.q_proj.weight"
        parts = name.split('.')

        # Use F32 since we convert bfloat16 weights to float32
        dtype = 13  # F32 (llaisysDataType_t value)

        # Embedding layers
        if name == "model.embed_tokens.weight":
            return (None, "in_embed", dtype)
        elif name == "lm_head.weight":
            return (None, "out_embed", dtype)
        elif name == "model.norm.weight":
            return (None, "out_norm_w", dtype)

        # Layer weights
        if parts[0] == "model" and parts[1] == "layers":
            layer_idx = int(parts[2])

            if parts[3] == "input_layernorm" and parts[4] == "weight":
                return (layer_idx, "attn_norm_w", dtype)

            elif parts[3] == "self_attn":
                if parts[4] == "q_proj":
                    if parts[5] == "weight":
                        return (layer_idx, "attn_q_w", dtype)
                    elif parts[5] == "bias":
                        return (layer_idx, "attn_q_b", dtype)
                elif parts[4] == "k_proj":
                    if parts[5] == "weight":
                        return (layer_idx, "attn_k_w", dtype)
                    elif parts[5] == "bias":
                        return (layer_idx, "attn_k_b", dtype)
                elif parts[4] == "v_proj":
                    if parts[5] == "weight":
                        return (layer_idx, "attn_v_w", dtype)
                    elif parts[5] == "bias":
                        return (layer_idx, "attn_v_b", dtype)
                elif parts[4] == "o_proj" and parts[5] == "weight":
                    return (layer_idx, "attn_o_w", dtype)

            elif parts[3] == "post_attention_layernorm" and parts[4] == "weight":
                return (layer_idx, "mlp_norm_w", dtype)

            elif parts[3] == "mlp":
                if parts[4] == "gate_proj" and parts[5] == "weight":
                    return (layer_idx, "mlp_gate_w", dtype)
                elif parts[4] == "up_proj" and parts[5] == "weight":
                    return (layer_idx, "mlp_up_w", dtype)
                elif parts[4] == "down_proj" and parts[5] == "weight":
                    return (layer_idx, "mlp_down_w", dtype)

        # Unknown tensor name
        print(f"Warning: Unknown tensor name: {name}")
        return None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        # Convert inputs to list
        tokens = list(inputs)
        output_tokens = []

        # Generate tokens
        for _ in range(max_new_tokens):
            # Prepare token array
            token_array = (ctypes.c_int64 * len(tokens))(*tokens)

            # Run inference
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr,
                token_array,
                len(tokens)
            )

            output_tokens.append(next_token)

            if next_token == self.meta_config.end_token:
                break
            tokens = [next_token]  # Next iteration only process new token

        return output_tokens

    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)
