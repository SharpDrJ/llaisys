#include "qwen2_model.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys {
namespace models {
namespace qwen2 {

Qwen2Model::Qwen2Model(const Meta& meta, llaisysDeviceType_t device, int device_id)
    : meta_(meta),
      device_(device),
      device_id_(device_id),
      position_(0),
      weights_{},
      blocks_(meta.num_layers) {

    // Create KV cache
    kv_cache_ = std::make_unique<KVCache>(meta.num_layers);
}

void Qwen2Model::setWeights(const Weights& weights) {
    weights_ = weights;

    // Create layer blocks
    for (size_t i = 0; i < meta_.num_layers; ++i) {
        // Wrap C tensor pointers in C++ tensor_t for block creation
        tensor_t attn_q_w = Tensor::wrap(weights.attn_q_w[i]);
        tensor_t attn_q_b = Tensor::wrap(weights.attn_q_b[i]);
        tensor_t attn_k_w = Tensor::wrap(weights.attn_k_w[i]);
        tensor_t attn_k_b = Tensor::wrap(weights.attn_k_b[i]);
        tensor_t attn_v_w = Tensor::wrap(weights.attn_v_w[i]);
        tensor_t attn_v_b = Tensor::wrap(weights.attn_v_b[i]);
        tensor_t attn_o_w = Tensor::wrap(weights.attn_o_w[i]);
        tensor_t attn_norm_w = Tensor::wrap(weights.attn_norm_w[i]);

        tensor_t mlp_gate_w = Tensor::wrap(weights.mlp_gate_w[i]);
        tensor_t mlp_up_w = Tensor::wrap(weights.mlp_up_w[i]);
        tensor_t mlp_down_w = Tensor::wrap(weights.mlp_down_w[i]);
        tensor_t mlp_norm_w = Tensor::wrap(weights.mlp_norm_w[i]);

        // Create attention module
        auto attention = std::make_unique<Qwen2Attention>(
            attn_q_w, attn_q_b,
            attn_k_w, attn_k_b,
            attn_v_w, attn_v_b,
            attn_o_w,
            meta_.num_heads,
            meta_.num_kv_heads,
            meta_.rope_theta,
            i  // layer index
        );

        // Create MLP module
        auto mlp = std::make_unique<Qwen2MLP>(
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w
        );

        // Create block
        blocks_[i] = std::make_unique<Qwen2Block>(
            attn_norm_w,
            mlp_norm_w,
            std::move(attention),
            std::move(mlp)
        );
    }
}

int64_t Qwen2Model::forward(const std::vector<int64_t>& token_ids) {
    size_t seq_len = token_ids.size();
    ASSERT(weights_.in_embed != nullptr, "Embedding weights not set");
    ASSERT(weights_.out_embed != nullptr, "Output embedding weights not set");
    ASSERT(weights_.out_norm_w != nullptr, "Output norm weights not set");

    // Convert token_ids to tensor
    auto tokens = Tensor::create({seq_len}, LLAISYS_DTYPE_I64,
                                 device_, device_id_);
    int64_t* token_data = reinterpret_cast<int64_t*>(tokens->data());
    for (size_t i = 0; i < seq_len; ++i) {
        token_data[i] = token_ids[i];
    }

    // Wrap C tensor pointers for use in operations
    auto in_embed = Tensor::wrap(weights_.in_embed);
    auto out_embed = Tensor::wrap(weights_.out_embed);
    auto out_norm_w = Tensor::wrap(weights_.out_norm_w);

    // Use dtype from weights for consistency
    llaisysDataType_t compute_dtype = in_embed->dtype();

    // Embedding lookup
    // Input: [seq_len], Output: [seq_len, hidden_size]
    auto hidden = Tensor::create({seq_len, meta_.hidden_size},
                                 compute_dtype,
                                 device_, device_id_);
    ops::embedding(hidden, tokens, in_embed);

    // Pass through all layers
    for (size_t i = 0; i < meta_.num_layers; ++i) {
        ASSERT(blocks_[i] != nullptr, "Layer " + std::to_string(i) + " not initialized");
        hidden = blocks_[i]->forward(hidden, position_, kv_cache_.get());
    }

    // Update position for next call
    position_ += seq_len;

    // Final layer norm
    auto hidden_norm = Tensor::create({seq_len, meta_.hidden_size},
                                      hidden->dtype(), device_, device_id_);
    ops::rms_norm(hidden_norm, hidden, out_norm_w, meta_.epsilon);

    // Get only the last token's hidden state for next token prediction
    // Slice to get [1, hidden_size]
    auto last_hidden = hidden->slice(0, seq_len - 1, seq_len);

    // LM head projection: [1, hidden_size] @ [vocab_size, hidden_size]^T = [1, vocab_size]
    auto logits = Tensor::create({1, meta_.vocab_size},
                                 last_hidden->dtype(), device_, device_id_);
    ops::linear(logits, last_hidden, out_embed, nullptr);

    // Find argmax to get next token
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64,
                                  device_, device_id_);
    auto max_val = Tensor::create({1}, logits->dtype(), device_, device_id_);
    ops::argmax(max_idx, max_val, logits);

    int64_t* result = reinterpret_cast<int64_t*>(max_idx->data());
    return result[0];
}

void Qwen2Model::clearCache() {
    kv_cache_->clear();
    position_ = 0;
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
