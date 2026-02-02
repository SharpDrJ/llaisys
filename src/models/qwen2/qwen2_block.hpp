#ifndef LLAISYS_MODELS_QWEN2_BLOCK_HPP
#define LLAISYS_MODELS_QWEN2_BLOCK_HPP

#include "tensor/tensor.hpp"
#include "core/llaisys_core.hpp"
#include "qwen2_attention.hpp"
#include "qwen2_mlp.hpp"
#include "kv_cache.hpp"
#include <memory>

namespace llaisys {
namespace models {
namespace qwen2 {

class Qwen2Block {
public:
    Qwen2Block(
        tensor_t attn_norm_w,
        tensor_t mlp_norm_w,
        std::unique_ptr<Qwen2Attention> attention,
        std::unique_ptr<Qwen2MLP> mlp);
    ~Qwen2Block() = default;

    // Forward pass through transformer block
    // Input: [seq_len, hidden_size]
    // Output: [seq_len, hidden_size]
    tensor_t forward(tensor_t hidden_state, size_t position, KVCache* cache);

private:
    tensor_t attn_norm_w_;
    tensor_t mlp_norm_w_;
    std::unique_ptr<Qwen2Attention> attention_;
    std::unique_ptr<Qwen2MLP> mlp_;
    float epsilon_;
    llaisysDeviceType_t device_;
    int device_id_;
};

} // namespace qwen2
} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_BLOCK_HPP
