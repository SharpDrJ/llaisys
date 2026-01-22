#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
// Concatenate tensors along a given dimension
// This is a simplified CPU implementation that works with contiguous tensors
// For production use, this would need to handle strides properly
void concat(std::byte* out_data,
            const std::vector<std::pair<const std::byte*, size_t>>& inputs_data_size,
            size_t dtype_size);
}
