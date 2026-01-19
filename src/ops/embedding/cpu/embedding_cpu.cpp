#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t numel_index, size_t hidden_size) {
    size_t element_size = llaisys::utils::dsize(type);
    size_t row_size_in_bytes = hidden_size * element_size;
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    for (size_t i = 0; i < numel_index; ++i) {
        int64_t row_idx = idx_ptr[i];
        const std::byte *src = weight + row_idx * row_size_in_bytes;
        std::byte *dst = out + i * row_size_in_bytes;
        std::memcpy(dst, src, row_size_in_bytes);
    }
}
} // namespace llaisys::ops::cpu
