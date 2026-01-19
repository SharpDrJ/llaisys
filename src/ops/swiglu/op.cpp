#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::swiglu(out, gate, up);
    } else {
        TO_BE_IMPLEMENTED();
    }
}
} // namespace llaisys::ops
