#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    ASSERT(in->ndim() == 2, "Linear: in must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Linear: out must be 2D.");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K, "Linear: weight shape mismatch.");
    ASSERT(out->shape()[0] == M, "Linear: out shape mismatch.");
    ASSERT(out->shape()[1] == N, "Linear: out shape mismatch.");
    if (bias) {
        ASSERT(bias->numel() == N, "Linear: bias shape mismatch.");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), M, N, K);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), M, N, K);
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
