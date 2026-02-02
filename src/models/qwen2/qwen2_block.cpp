#include "qwen2_block.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/add/op.hpp"
#include <cmath>

namespace llaisys {
namespace models {
namespace qwen2 {

Qwen2Block::Qwen2Block(
    tensor_t attn_norm_w,
    tensor_t mlp_norm_w,
    std::unique_ptr<Qwen2Attention> attention,
    std::unique_ptr<Qwen2MLP> mlp)
    : attn_norm_w_(attn_norm_w),
      mlp_norm_w_(mlp_norm_w),
      attention_(std::move(attention)),
      mlp_(std::move(mlp)),
      epsilon_(1e-6f),
      device_(attn_norm_w->deviceType()),
      device_id_(attn_norm_w->deviceId()) {
}

tensor_t Qwen2Block::forward(tensor_t hidden_state, size_t position, KVCache* cache) {
    size_t seq_len = hidden_state->shape()[0];
    size_t hidden_size = hidden_state->shape()[1];

    // Attention block
    // 1. RMS norm before attention
    auto hidden_norm = Tensor::create({seq_len, hidden_size},
                                      hidden_state->dtype(), device_, device_id_);
    ops::rms_norm(hidden_norm, hidden_state, attn_norm_w_, epsilon_);

    // 2. Attention
    auto attn_output = attention_->forward(hidden_norm, position, cache);

    // 3. Residual connection
    auto residual = Tensor::create({seq_len, hidden_size},
                                   hidden_state->dtype(), device_, device_id_);
    ops::add(residual, hidden_state, attn_output);

    // MLP block
    // 1. RMS norm before MLP
    auto mlp_norm = Tensor::create({seq_len, hidden_size},
                                   residual->dtype(), device_, device_id_);
    ops::rms_norm(mlp_norm, residual, mlp_norm_w_, epsilon_);

    // 2. MLP
    auto mlp_output = mlp_->forward(mlp_norm);

    // 3. Residual connection
    auto output = Tensor::create({seq_len, hidden_size},
                                 residual->dtype(), device_, device_id_);
    ops::add(output, residual, mlp_output);

    return output;
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
