#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be I64.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    ASSERT(in->ndim() == 3, "RoPE: in must be 3D [seq_len, n_heads, head_dim].");
    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(pos_ids->numel() == seq_len, "RoPE: pos_ids size mismatch.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, num_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, num_heads, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
