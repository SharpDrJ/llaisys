#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/concat_cpu.hpp"

namespace llaisys::ops {
tensor_t concat(const std::vector<tensor_t>& tensors, int64_t dim) {
    ASSERT(!tensors.empty(), "Concat: input tensor list cannot be empty");
    ASSERT(dim >= 0 && dim < static_cast<int64_t>(tensors[0]->ndim()),
           "Concat: dim out of range");

    // Check all tensors have same device
    llaisysDeviceType_t device_type = tensors[0]->deviceType();
    int device_id = tensors[0]->deviceId();
    for (const auto& t : tensors) {
        CHECK_SAME_DEVICE(tensors[0], t);
    }

    // Check all tensors have same dtype
    llaisysDataType_t dtype = tensors[0]->dtype();
    for (const auto& t : tensors) {
        CHECK_SAME_DTYPE(dtype, t->dtype());
    }

    // Check all tensors are contiguous
    for (const auto& t : tensors) {
        ASSERT(t->isContiguous(), "Concat: all tensors must be contiguous.");
    }

    // Calculate output shape
    size_t ndim = tensors[0]->ndim();
    std::vector<size_t> out_shape(ndim);

    // All dimensions except concat dim must be the same
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int64_t>(i) == dim) {
            // Sum along concat dimension
            out_shape[i] = 0;
            for (const auto& t : tensors) {
                out_shape[i] += t->shape()[i];
            }
        } else {
            // Check all other dimensions are the same
            size_t dim_size = tensors[0]->shape()[i];
            for (const auto& t : tensors) {
                ASSERT(t->shape()[i] == dim_size,
                       "Concat: all tensors must have same size in non-concat dimensions");
            }
            out_shape[i] = dim_size;
        }
    }

    // Create output tensor
    auto out = Tensor::create(out_shape, dtype, device_type, device_id);

    // Dispatch to device implementation
    if (device_type == LLAISYS_DEVICE_CPU) {
        // Prepare data pointers and sizes for CPU implementation
        std::vector<std::pair<const std::byte*, size_t>> inputs_data_size;
        for (const auto& t : tensors) {
            inputs_data_size.push_back({t->data(), t->numel()});
        }

        cpu::concat(out->data(), inputs_data_size, llaisys::utils::dsize(dtype));
        return out;
    }

    llaisys::core::context().setDevice(device_type, device_id);

    switch (device_type) {
    case LLAISYS_DEVICE_CPU: {
        // Prepare data pointers and sizes for CPU implementation
        std::vector<std::pair<const std::byte*, size_t>> inputs_data_size;
        for (const auto& t : tensors) {
            inputs_data_size.push_back({t->data(), t->numel()});
        }

        cpu::concat(out->data(), inputs_data_size, llaisys::utils::dsize(dtype));
        return out;
    }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return out;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
