#include "kv_cache.hpp"
#include "../../tensor/tensor.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../ops/concat/op.hpp"
#include <stdexcept>

namespace llaisys {
namespace models {
namespace qwen2 {

KVCache::KVCache(size_t num_layers)
    : cached_k_(num_layers, nullptr), cached_v_(num_layers, nullptr) {
}

void KVCache::update(size_t layer_idx, tensor_t new_k, tensor_t new_v) {
    if (layer_idx >= cached_k_.size()) {
        throw std::out_of_range("Layer index out of range");
    }

    // If cache is empty, just store the new tensors
    if (cached_k_[layer_idx] == nullptr) {
        cached_k_[layer_idx] = new_k;
        cached_v_[layer_idx] = new_v;
    } else {
        // Concatenate existing cache with new tensors
        tensor_t old_k = cached_k_[layer_idx];
        tensor_t old_v = cached_v_[layer_idx];

        // Concatenate along sequence dimension (dim 0)
        cached_k_[layer_idx] = ops::concat({old_k, new_k}, 0);
        cached_v_[layer_idx] = ops::concat({old_v, new_v}, 0);
    }
}

void KVCache::set(size_t layer_idx, tensor_t k, tensor_t v) {
    if (layer_idx >= cached_k_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    cached_k_[layer_idx] = k;
    cached_v_[layer_idx] = v;
}

tensor_t KVCache::getK(size_t layer_idx) const {
    if (layer_idx >= cached_k_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return cached_k_[layer_idx];
}

tensor_t KVCache::getV(size_t layer_idx) const {
    if (layer_idx >= cached_v_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return cached_v_[layer_idx];
}

size_t KVCache::getSeqLen() const {
    // All layers should have the same sequence length
    // Return 0 if cache is empty
    if (cached_k_.empty() || cached_k_[0] == nullptr) {
        return 0;
    }

    // Get shape from first layer's K tensor
    return cached_k_[0]->shape()[0];  // First dimension is sequence length
}

void KVCache::clear() {
    // Reset all cached tensors to nullptr
    std::fill(cached_k_.begin(), cached_k_.end(), nullptr);
    std::fill(cached_v_.begin(), cached_v_.end(), nullptr);
}

bool KVCache::isEmpty() const {
    return cached_k_.empty() || cached_k_[0] == nullptr;
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
