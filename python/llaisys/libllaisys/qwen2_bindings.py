import ctypes
from .llaisys_types import llaisysDeviceType_t
from .tensor import llaisysTensor_t


# Define the Qwen2 structures
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),       # hidden_size
        ("nh", ctypes.c_size_t),       # num_heads
        ("nkvh", ctypes.c_size_t),     # num_kv_heads
        ("dh", ctypes.c_size_t),       # head_dim (unused, computed)
        ("di", ctypes.c_size_t),       # intermediate (unused)
        ("maxseq", ctypes.c_size_t),   # max_seq_len (unused)
        ("voc", ctypes.c_size_t),      # vocab_size
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),     # rope_theta
        ("end_token", ctypes.c_int64),
    ]


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]


LlaisysQwen2Model = ctypes.c_void_p  # Opaque pointer


def load_qwen2(lib):
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = ctypes.POINTER(LlaisysQwen2Model)

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [ctypes.POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        ctypes.POINTER(LlaisysQwen2Model),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64
