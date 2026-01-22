#ifndef LLAISYS_MODELS_QWEN2_ATTENTION_HPP
#define LLAISYS_MODELS_QWEN2_ATTENTION_HPP

#include "tensor/tensor.hpp"
#include "core/llaisys_core.hpp"
#include "kv_cache.hpp"
#include <memory>

namespace llaisys {
namespace models {
namespace qwen2 {

class Qwen2Attention {
public:
    Qwen2Attention(
        tensor_t q_proj_w, tensor_t q_proj_b,
        tensor_t k_proj_w, tensor_t k_proj_b,
        tensor_t v_proj_w, tensor_t v_proj_b,
        tensor_t o_proj_w,
        size_t num_heads, size_t num_kv_heads,
        double rope_theta,
        size_t layer_idx);
    ~Qwen2Attention() = default;

    // Forward pass through attention
    // Input: [seq_len, hidden_size]
    // Output: [seq_len, hidden_size]
    tensor_t forward(tensor_t hidden_state, size_t position, KVCache* cache);

private:
    tensor_t q_proj_w_, k_proj_w_, v_proj_w_, o_proj_w_;
    tensor_t q_proj_b_, k_proj_b_, v_proj_b_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    double rope_theta_;
    size_t layer_idx_;  // Layer index for KV cache
    llaisysDeviceType_t device_;
    int device_id_;

    // Helper to reshape and rearrange Q/K/V for multi-head attention
    tensor_t reshapeAndRearrange(tensor_t x, size_t num_heads, size_t head_dim);
};

} // namespace qwen2
} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_ATTENTION_HPP
