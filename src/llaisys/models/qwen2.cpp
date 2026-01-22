#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/qwen2_model.hpp"
#include <vector>
#include <memory>

// Opaque struct for C API
struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::qwen2::Qwen2Model> model;
    llaisys::models::qwen2::Qwen2Model::Weights weights;
    bool weights_loaded = false;
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    (void)device_ids; // Unused for now
    (void)ndevice;    // Unused for now

    // Create C++ model
    llaisys::models::qwen2::Qwen2Model::Meta cpp_meta;
    cpp_meta.num_layers = meta->nlayer;
    cpp_meta.hidden_size = meta->hs;
    cpp_meta.num_heads = meta->nh;
    cpp_meta.num_kv_heads = meta->nkvh;
    cpp_meta.vocab_size = meta->voc;
    cpp_meta.rope_theta = meta->theta;
    cpp_meta.epsilon = meta->epsilon;
    cpp_meta.eos_token = meta->end_token;

    int device_id = (device == LLAISYS_DEVICE_CPU) ? 0 : (device_ids ? device_ids[0] : 0);

    auto model = std::make_unique<llaisys::models::qwen2::Qwen2Model>(
        cpp_meta, device, device_id);

    // Create C API wrapper
    auto c_model = new LlaisysQwen2Model();
    c_model->model = std::move(model);
    c_model->weights = {};
    // Allocate arrays for layer weights
    c_model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer]();
    c_model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer]();

    return c_model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        // Free weight arrays
        delete[] model->weights.attn_norm_w;
        delete[] model->weights.attn_q_w;
        delete[] model->weights.attn_q_b;
        delete[] model->weights.attn_k_w;
        delete[] model->weights.attn_k_b;
        delete[] model->weights.attn_v_w;
        delete[] model->weights.attn_v_b;
        delete[] model->weights.attn_o_w;
        delete[] model->weights.mlp_norm_w;
        delete[] model->weights.mlp_gate_w;
        delete[] model->weights.mlp_up_w;
        delete[] model->weights.mlp_down_w;

        delete model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (model) {
        // Return pointer to weights structure so Python can populate it
        return reinterpret_cast<LlaisysQwen2Weights*>(&model->weights);
    }
    return nullptr;
}

int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {

    if (!model || !model->model) {
        return -1;  // Error
    }

    // Load weights if not already loaded
    if (!model->weights_loaded) {
        model->model->setWeights(model->weights);
        model->weights_loaded = true;
    }

    // Convert token IDs to vector
    std::vector<int64_t> tokens(token_ids, token_ids + ntoken);

    // Run forward pass
    return model->model->forward(tokens);
}

} // extern "C"
