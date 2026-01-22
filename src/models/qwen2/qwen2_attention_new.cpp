#include "qwen2_attention.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/concat/op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys {
namespace models {
namespace qwen2 {

tensor_t Qwen2Attention::forward(tensor_t hidden_state, size_t position, KVCache* cache) {
    size_t seq_len = hidden_state->shape()[0];
    size_t hidden_size = hidden_state->shape()[1];
    size_t num_heads_q = num_heads_;
    size_t num_heads_kv = num_kv_heads_;
    size_t head_dim = head_dim_;

    // Compute Q, K, V projections
    auto q = Tensor::create({seq_len, num_heads_q * head_dim},
                            hidden_state->dtype(), device_, device_id_);
    ops::linear(q, hidden_state, q_proj_w_, q_proj_b_);

    auto k = Tensor::create({seq_len, num_heads_kv * head_dim},
                            hidden_state->dtype(), device_, device_id_);
    ops::linear(k, hidden_state, k_proj_w_, k_proj_b_);

    auto v = Tensor::create({seq_len, num_heads_kv * head_dim},
                            hidden_state->dtype(), device_, device_id_);
    ops::linear(v, hidden_state, v_proj_w_, v_proj_b_);

    // Apply RoPE to Q and K
    auto q_reshaped = q->view({seq_len, num_heads_q, head_dim});
    auto q_rope = Tensor::create({seq_len, num_heads_q, head_dim},
                                 q->dtype(), device_, device_id_);

    auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64,
                                   device_, device_id_);
    int64_t* pos_data = reinterpret_cast<int64_t*>(pos_ids->data());
    for (size_t i = 0; i < seq_len; ++i) {
        pos_data[i] = static_cast<int64_t>(position + i);
    }
    ops::rope(q_rope, q_reshaped, pos_ids, rope_theta_);

    auto k_reshaped = k->view({seq_len, num_heads_kv, head_dim});
    auto k_rope = Tensor::create({seq_len, num_heads_kv, head_dim},
                                 k->dtype(), device_, device_id_);
    ops::rope(k_rope, k_reshaped, pos_ids, rope_theta_);

    // Handle KV cache
    auto cached_k = cache->getK(layer_idx_);
    auto cached_v = cache->getV(layer_idx_);

    tensor_t k_for_attn, v_for_attn;
    if (cached_k == nullptr) {
        // First call - initialize cache with current K/V
        k_for_attn = k_rope;
        v_for_attn = v;
        cache->set(layer_idx_, k_rope, v);
    } else {
        // Subsequent calls - concatenate cache with current K/V
        k_for_attn = ops::concat({cached_k, k_rope}, 0);
        v_for_attn = ops::concat({cached_v, v}, 0);
        // Update cache with new concatenated tensors
        cache->set(layer_idx_, k_for_attn, v_for_attn);
    }

    // Compute attention
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto attn_out = Tensor::create({seq_len, num_heads_q, head_dim},
                                   hidden_state->dtype(), device_, device_id_);
    ops::self_attention(attn_out, q_rope, k_for_attn, v_for_attn, scale);

    // Merge heads: reshape from [seq_len, num_heads, head_dim] to [seq_len, num_heads * head_dim]
    auto attn_merged = attn_out->view({seq_len, num_heads_q * head_dim});

    // Output projection
    auto output = Tensor::create({seq_len, hidden_size},
                                 hidden_state->dtype(), device_, device_id_);
    ops::linear(output, attn_merged, o_proj_w_, nullptr);

    return output;
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
