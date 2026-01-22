#pragma once

#include "../../tensor/tensor.hpp"
#include <vector>

namespace llaisys::ops {
tensor_t concat(const std::vector<tensor_t>& tensors, int64_t dim);
}
