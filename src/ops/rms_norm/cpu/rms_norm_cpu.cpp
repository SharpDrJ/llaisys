#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch, size_t dim, float eps) {
    for (size_t i = 0; i < batch; ++i) {
        float sum_sq = 0.0f;
        const T *in_row = in + i * dim;
        T *out_row = out + i * dim;

        // 1. Calculate sum of squares
        for (size_t j = 0; j < dim; ++j) {
            float val = llaisys::utils::cast<float>(in_row[j]);
            sum_sq += val * val;
        }

        // 2. Calculate scale factor: 1 / sqrt(mean(x^2) + eps)
        float scale = 1.0f / std::sqrt(sum_sq / dim + eps);

        // 3. Apply normalization and weight
        for (size_t j = 0; j < dim; ++j) {
            float val = llaisys::utils::cast<float>(in_row[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            out_row[j] = llaisys::utils::cast<T>(val * scale * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t batch, size_t dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), batch, dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), batch, dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), batch, dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
