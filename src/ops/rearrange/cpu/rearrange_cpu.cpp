#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <vector>

namespace llaisys::ops::cpu {

// Helper function to compute linear index from multi-dimensional indices
template <typename T>
void rearrange_impl(
    T *out,
    const T *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &in_strides,
    size_t dim,
    size_t out_offset,
    size_t in_offset) {

    if (dim == shape.size()) {
        // Base case: copy single element
        out[out_offset] = in[in_offset];
        return;
    }

    for (size_t i = 0; i < shape[dim]; i++) {
        rearrange_impl(
            out,
            in,
            shape,
            out_strides,
            in_strides,
            dim + 1,
            out_offset + i * out_strides[dim],
            in_offset + i * in_strides[dim]);
    }
}

// Template for different data types
template <typename T>
void rearrange_(
    std::byte *out,
    const std::byte *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &in_strides) {

    rearrange_impl(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        shape,
        out_strides,
        in_strides,
        0,
        0,
        0);
}

void rearrange(
    std::byte *out,
    const std::byte *in,
    llaisysDataType_t type,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &in_strides) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_<float>(out, in, shape, out_strides, in_strides);
    case LLAISYS_DTYPE_BF16:
        return rearrange_<llaisys::bf16_t>(out, in, shape, out_strides, in_strides);
    case LLAISYS_DTYPE_F16:
        return rearrange_<llaisys::fp16_t>(out, in, shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I64:
        return rearrange_<int64_t>(out, in, shape, out_strides, in_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
