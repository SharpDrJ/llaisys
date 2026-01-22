#include "qwen2_mlp.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/rearrange/op.hpp"

namespace llaisys {
namespace models {
namespace qwen2 {

Qwen2MLP::Qwen2MLP(tensor_t gate_proj_w, tensor_t up_proj_w, tensor_t down_proj_w)
    : gate_proj_w_(gate_proj_w),
      up_proj_w_(up_proj_w),
      down_proj_w_(down_proj_w),
      device_(gate_proj_w->deviceType()),
      device_id_(gate_proj_w->deviceId()) {
}

tensor_t Qwen2MLP::forward(tensor_t hidden_state) {
    size_t seq_len = hidden_state->shape()[0];
    size_t hidden_size = hidden_state->shape()[1];
    size_t intermediate_size = gate_proj_w_->shape()[0];

    // Compute gate projection: hidden_state @ gate_proj_w^T
    // gate_proj_w shape: [intermediate_size, hidden_size]
    // Output shape: [seq_len, intermediate_size]
    auto gate = Tensor::create({seq_len, intermediate_size},
                               hidden_state->dtype(), device_, device_id_);
    ops::linear(gate, hidden_state, gate_proj_w_, nullptr);

    // Compute up projection: hidden_state @ up_proj_w^T
    // up_proj_w shape: [intermediate_size, hidden_size]
    // Output shape: [seq_len, intermediate_size]
    auto up = Tensor::create({seq_len, intermediate_size},
                             hidden_state->dtype(), device_, device_id_);
    ops::linear(up, hidden_state, up_proj_w_, nullptr);

    // SwiGLU activation: SiLU(gate) * up
    auto intermediate = Tensor::create({seq_len, intermediate_size},
                                       hidden_state->dtype(), device_, device_id_);
    ops::swiglu(intermediate, gate, up);

    // Down projection: intermediate @ down_proj_w^T
    // down_proj_w shape: [hidden_size, intermediate_size]
    // Output shape: [seq_len, hidden_size]
    auto output = Tensor::create({seq_len, hidden_size},
                                 hidden_state->dtype(), device_, device_id_);
    ops::linear(output, intermediate, down_proj_w_, nullptr);

    return output;
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
