#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "self_attention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(attn_val, q, k, v, scale);
    } else {
        TO_BE_IMPLEMENTED();
    }
}
} // namespace llaisys::ops
