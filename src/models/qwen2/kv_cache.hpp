#ifndef LLAISYS_MODELS_QWEN2_KV_CACHE_HPP
#define LLAISYS_MODELS_QWEN2_KV_CACHE_HPP

#include "tensor/tensor.hpp"
#include <vector>

namespace llaisys {
namespace models {
namespace qwen2 {

class KVCache {
public:
    explicit KVCache(size_t num_layers);
    ~KVCache() = default;

    // Update cache with new K and V tensors for a specific layer (concatenates with existing)
    void update(size_t layer_idx, tensor_t new_k, tensor_t new_v);

    // Set cache directly (for already-concatenated tensors)
    void set(size_t layer_idx, tensor_t k, tensor_t v);

    // Get cached K and V for a specific layer
    tensor_t getK(size_t layer_idx) const;
    tensor_t getV(size_t layer_idx) const;

    // Get current sequence length (from cache size)
    size_t getSeqLen() const;

    // Clear all cached tensors
    void clear();

    // Check if cache is empty
    bool isEmpty() const;

private:
    std::vector<tensor_t> cached_k_;
    std::vector<tensor_t> cached_v_;
};

} // namespace qwen2
} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_KV_CACHE_HPP
