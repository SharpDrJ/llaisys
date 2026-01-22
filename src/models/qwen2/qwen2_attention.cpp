#include "qwen2_attention.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/concat/op.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstring>

namespace llaisys {
namespace models {
namespace qwen2 {

Qwen2Attention::Qwen2Attention(
    tensor_t q_proj_w, tensor_t q_proj_b,
    tensor_t k_proj_w, tensor_t k_proj_b,
    tensor_t v_proj_w, tensor_t v_proj_b,
    tensor_t o_proj_w,
    size_t num_heads, size_t num_kv_heads,
    double rope_theta,
    size_t layer_idx)
    : q_proj_w_(q_proj_w), k_proj_w_(k_proj_w), v_proj_w_(v_proj_w), o_proj_w_(o_proj_w),
      q_proj_b_(q_proj_b), k_proj_b_(k_proj_b), v_proj_b_(v_proj_b),
      num_heads_(num_heads), num_kv_heads_(num_kv_heads),
      head_dim_(q_proj_w->shape()[0] / num_heads),
      rope_theta_(rope_theta),
      layer_idx_(layer_idx),
      device_(q_proj_w->deviceType()),
      device_id_(q_proj_w->deviceId()) {
}

tensor_t Qwen2Attention::reshapeAndRearrange(tensor_t x, size_t num_heads, size_t head_dim) {
    // x shape: [seq_len, num_heads * head_dim]
    // Reshape to: [seq_len, num_heads, head_dim]
    size_t seq_len = x->shape()[0];
    auto reshaped = x->view({seq_len, num_heads, head_dim});

    // We need to rearrange from [seq_len, num_heads, head_dim] to [num_heads, seq_len, head_dim]
    // Then view as [num_heads * seq_len, head_dim] for self_attention op
    //
    // self_attention expects memory layout where for index [total_idx, dim_idx]:
    //   head_idx = total_idx / seq_len
    //   seq_idx = total_idx % seq_len
    // So the memory should be: [h0_s0_d0, h0_s1_d0, ..., h1_s0_d0, h1_s1_d0, ...]

    // First permute from [seq_len, num_heads, head_dim] to [num_heads, seq_len, head_dim]
    auto permuted = reshaped->permute({1, 0, 2});  // Now shape is [num_heads, seq_len, head_dim]

    // permuted is not contiguous, so we need to make it contiguous before view
    auto permuted_contiguous = permuted->isContiguous() ? permuted : permuted->contiguous();

    // Then view as [num_heads * seq_len, head_dim] which has the correct memory layout
    auto result = permuted_contiguous->view({num_heads * seq_len, head_dim});

    return result;
}

tensor_t Qwen2Attention::forward(tensor_t hidden_state, size_t position, KVCache* cache) {
    size_t seq_len = hidden_state->shape()[0];
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

    // Apply RoPE to Q and K (requires shape [seq_len, num_heads, head_dim])
    auto q_reshaped = q->view({seq_len, num_heads_q, head_dim});
    auto q_rope = Tensor::create({seq_len, num_heads_q, head_dim},
                                 q->dtype(), device_, device_id_);

    auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_, device_id_);
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
        // Rearrange from [seq_len, num_heads, head_dim] to [num_heads * seq_len, head_dim]
        k_for_attn = reshapeAndRearrange(k_rope, num_heads_kv, head_dim);
        v_for_attn = reshapeAndRearrange(v, num_heads_kv, head_dim);
        cache->set(layer_idx_, k_for_attn, v_for_attn);
    } else {
        // Subsequent calls - concatenate cache with current K/V
        // cached_k/v have shape [num_heads * cached_seq_len, head_dim] (heads-major)
        // We need to reshape them back to [cached_seq_len, num_heads, head_dim] for concat
        size_t cached_seq_len = cached_k->shape()[0] / num_heads_kv;
        auto k_cached_seq = cached_k->view({cached_seq_len, num_heads_kv, head_dim});
        auto v_cached_seq = cached_v->view({cached_seq_len, num_heads_kv, head_dim});

        // Concatenate: [cached_seq_len, num_heads, head_dim] + [seq_len, num_heads, head_dim]
        // along sequence dimension (dim=0)
        auto k_concat = ops::concat({k_cached_seq, k_rope}, 0);
        auto v_concat = ops::concat({v_cached_seq, v}, 0);

        // Now rearrange the concatenated tensors back to heads-major layout
        k_for_attn = reshapeAndRearrange(k_concat, num_heads_kv, head_dim);
        v_for_attn = reshapeAndRearrange(v_concat, num_heads_kv, head_dim);

        // Update cache with new concatenated tensors
        cache->set(layer_idx_, k_for_attn, v_for_attn);
    }

    // Rearrange Q for self_attention: [seq_len, num_heads, head_dim] -> [num_heads * seq_len, head_dim]
    auto q_for_attn = reshapeAndRearrange(q_rope, num_heads_q, head_dim);

    // Self attention (output shape: [num_heads_q * seq_len, head_dim])
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto attn_out = Tensor::create({num_heads_q * seq_len, head_dim},
                                   hidden_state->dtype(), device_, device_id_);
    ops::self_attention(attn_out, q_for_attn, k_for_attn, v_for_attn, scale);

    // Merge heads: reshape from [num_heads * seq_len, head_dim] back to [seq_len, num_heads * head_dim]
    auto attn_merged = attn_out->view({seq_len, num_heads_q * head_dim});

    // Output projection
    auto output = Tensor::create({seq_len, hidden_state->shape()[1]},
                                 hidden_state->dtype(), device_, device_id_);
    ops::linear(output, attn_merged, o_proj_w_, nullptr);

    return output;
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
