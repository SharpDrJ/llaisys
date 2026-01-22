#ifndef LLAISYS_MODELS_QWEN2_MLP_HPP
#define LLAISYS_MODELS_QWEN2_MLP_HPP

#include "tensor/tensor.hpp"
#include "core/llaisys_core.hpp"
#include <memory>

namespace llaisys {
namespace models {
namespace qwen2 {

class Qwen2MLP {
public:
    Qwen2MLP(tensor_t gate_proj_w, tensor_t up_proj_w, tensor_t down_proj_w);
    ~Qwen2MLP() = default;

    // Forward pass through MLP
    // Input: [seq_len, hidden_size]
    // Output: [seq_len, hidden_size]
    tensor_t forward(tensor_t hidden_state);

private:
    tensor_t gate_proj_w_;
    tensor_t up_proj_w_;
    tensor_t down_proj_w_;
    llaisysDeviceType_t device_;
    int device_id_;
};

} // namespace qwen2
} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_MLP_HPP
