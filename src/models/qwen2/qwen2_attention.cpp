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
    auto q_contiguous = q_reshaped->isContiguous() ? q_reshaped : q_reshaped->contiguous();
    auto q_rope = Tensor::create({seq_len, num_heads_q, head_dim},
                                 q->dtype(), device_, device_id_);

    auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_, device_id_);
    int64_t* pos_data = reinterpret_cast<int64_t*>(pos_ids->data());
    for (size_t i = 0; i < seq_len; ++i) {
        pos_data[i] = static_cast<int64_t>(position + i);
    }
    ops::rope(q_rope, q_contiguous, pos_ids, static_cast<float>(rope_theta_));

    auto k_reshaped = k->view({seq_len, num_heads_kv, head_dim});
    auto k_contiguous = k_reshaped->isContiguous() ? k_reshaped : k_reshaped->contiguous();
    auto k_rope = Tensor::create({seq_len, num_heads_kv, head_dim},
                                 k->dtype(), device_, device_id_);
    ops::rope(k_rope, k_contiguous, pos_ids, static_cast<float>(rope_theta_));

    // Handle KV cache - store 3D tensors [seq_len, num_heads, head_dim]
    auto cached_k = cache->getK(layer_idx_);
    auto cached_v = cache->getV(layer_idx_);

    // Ensure V is contiguous and in 3D format
    auto v_reshaped = v->view({seq_len, num_heads_kv, head_dim});
    auto v_contiguous = v_reshaped->isContiguous() ? v_reshaped : v_reshaped->contiguous();

    tensor_t k_for_attn, v_for_attn;
    if (cached_k == nullptr) {
        // First call - use current K/V directly (3D format)
        k_for_attn = k_rope;  // [seq_len, num_heads_kv, head_dim]
        v_for_attn = v_contiguous;  // [seq_len, num_heads_kv, head_dim]
        cache->set(layer_idx_, k_for_attn, v_for_attn);
    } else {
        // Subsequent calls - concatenate cache with current K/V
        k_for_attn = ops::concat({cached_k, k_rope}, 0);
        v_for_attn = ops::concat({cached_v, v_contiguous}, 0);
        // Update cache with new concatenated tensors (3D format)
        cache->set(layer_idx_, k_for_attn, v_for_attn);
    }

    // Q is already in the correct format for self_attention
    auto q_for_attn = q_rope;  // [seq_len, num_heads_q, head_dim]

    // Self attention (output shape: [seq_len, num_heads_q, head_dim])
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto attn_out = Tensor::create({seq_len, num_heads_q, head_dim},
                                   hidden_state->dtype(), device_, device_id_);
    ops::self_attention(attn_out, q_for_attn, k_for_attn, v_for_attn, static_cast<float>(scale));

    // Merge heads: reshape from [seq_len, num_heads_q, head_dim] to [seq_len, num_heads_q * head_dim]
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
