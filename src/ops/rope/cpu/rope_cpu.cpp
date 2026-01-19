#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;

    for (size_t i = 0; i < seq_len; ++i) {
        int64_t p_i = pos_ids[i];
        for (size_t j = 0; j < half_dim; ++j) {
            float phi = p_i / std::pow(theta, (2.0f * j) / head_dim);
            float cos_phi = std::cos(phi);
            float sin_phi = std::sin(phi);

            for (size_t h = 0; h < num_heads; ++h) {
                // Indices for the current i, h, j
                size_t idx_a = i * num_heads * head_dim + h * head_dim + j;
                size_t idx_b = idx_a + half_dim;

                float a = llaisys::utils::cast<float>(in[idx_a]);
                float b = llaisys::utils::cast<float>(in[idx_b]);

                out[idx_a] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out[idx_b] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    const int64_t *pos_ids_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ids_ptr, seq_len, num_heads,
                     head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ids_ptr, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ids_ptr, seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
