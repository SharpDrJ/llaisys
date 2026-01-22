#ifndef LLAISYS_MODELS_QWEN2_MODEL_HPP
#define LLAISYS_MODELS_QWEN2_MODEL_HPP

#include "llaisys.h"
#include "tensor/tensor.hpp"
#include "core/llaisys_core.hpp"
#include "kv_cache.hpp"
#include "qwen2_block.hpp"
#include <memory>
#include <vector>

namespace llaisys {
namespace models {
namespace qwen2 {

class Qwen2Model {
public:
    struct Meta {
        size_t num_layers;
        size_t hidden_size;
        size_t num_heads;
        size_t num_kv_heads;
        size_t vocab_size;
        double rope_theta;
        double epsilon;
        int64_t eos_token;
    };

    struct Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;
        llaisysTensor_t* attn_norm_w;  // Pointer to array
        llaisysTensor_t* attn_q_w;
        llaisysTensor_t* attn_q_b;
        llaisysTensor_t* attn_k_w;
        llaisysTensor_t* attn_k_b;
        llaisysTensor_t* attn_v_w;
        llaisysTensor_t* attn_v_b;
        llaisysTensor_t* attn_o_w;
        llaisysTensor_t* mlp_norm_w;
        llaisysTensor_t* mlp_gate_w;
        llaisysTensor_t* mlp_up_w;
        llaisysTensor_t* mlp_down_w;
    };

    Qwen2Model(const Meta& meta,
               llaisysDeviceType_t device,
               int device_id = 0);
    ~Qwen2Model() = default;

    // Load weights (called by Python wrapper)
    void setWeights(const Weights& weights);

    // Forward pass: returns next token
    int64_t forward(const std::vector<int64_t>& token_ids);

    // Clear KV cache
    void clearCache();

    // Get metadata
    const Meta& getMeta() const { return meta_; }

private:
    Meta meta_;
    llaisysDeviceType_t device_;
    int device_id_;
    size_t position_;

    // Weights (stored as C pointers)
    Weights weights_;

    // Layers
    std::vector<std::unique_ptr<Qwen2Block>> blocks_;
    std::unique_ptr<KVCache> kv_cache_;
};

} // namespace qwen2
} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_MODEL_HPP
