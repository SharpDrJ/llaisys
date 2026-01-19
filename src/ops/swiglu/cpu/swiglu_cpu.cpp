#include "swiglu_cpu.hpp"
#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void swiglu_(tensor_t out, tensor_t gate, tensor_t up) {
    const size_t numel = out->numel();
    T *out_ptr = reinterpret_cast<T *>(out->data());
    const T *gate_ptr = reinterpret_cast<const T *>(gate->data());
    const T *up_ptr = reinterpret_cast<const T *>(up->data());

    for (size_t i = 0; i < numel; ++i) {
        float g = llaisys::utils::cast<float>(gate_ptr[i]);
        float u = llaisys::utils::cast<float>(up_ptr[i]);
        // swiglu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float res = u * (g / (1.0f + std::exp(-g)));
        out_ptr[i] = llaisys::utils::cast<T>(res);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    if (out->dtype() == LLAISYS_DTYPE_F32) {
        swiglu_<float>(out, gate, up);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        swiglu_<llaisys::fp16_t>(out, gate, up);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        swiglu_<llaisys::bf16_t>(out, gate, up);
    } else {
        ASSERT(false, "Unsupported data type");
    }
}

} // namespace llaisys::ops::cpu
