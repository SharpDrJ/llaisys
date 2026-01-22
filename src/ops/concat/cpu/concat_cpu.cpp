#include "concat_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {
void concat(std::byte* out_data,
            const std::vector<std::pair<const std::byte*, size_t>>& inputs_data_size,
            size_t dtype_size) {
    // Simple concatenation - just copy data from each input to output
    size_t offset = 0;
    for (const auto& [data, size] : inputs_data_size) {
        std::memcpy(out_data + offset, data, size * dtype_size);
        offset += size * dtype_size;
    }
}
} // namespace llaisys::ops::cpu
