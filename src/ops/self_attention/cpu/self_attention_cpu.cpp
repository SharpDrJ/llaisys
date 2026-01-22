#include "self_attention_cpu.hpp"
#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t hd = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    const size_t group_size = nh / nkvh;

    const T *q_ptr = reinterpret_cast<const T *>(q->data());
    const T *k_ptr = reinterpret_cast<const T *>(k->data());
    const T *v_ptr = reinterpret_cast<const T *>(v->data());
    T *out_ptr = reinterpret_cast<T *>(attn_val->data());

    for (size_t i = 0; i < qlen; ++i) {
        for (size_t h = 0; h < nh; ++h) {
            size_t h_kv = h / group_size;
            size_t j_end = (kvlen - qlen) + i;

            std::vector<float> scores(j_end + 1);
            float max_score = -std::numeric_limits<float>::infinity();

            // Compute scores
            for (size_t j = 0; j <= j_end; ++j) {
                float sum = 0.0f;
                for (size_t d = 0; d < hd; ++d) {
                    float q_val = llaisys::utils::cast<float>(q_ptr[i * nh * hd + h * hd + d]);
                    float k_val = llaisys::utils::cast<float>(k_ptr[j * nkvh * hd + h_kv * hd + d]);
                    sum += q_val * k_val;
                }
                scores[j] = sum * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            // Stable Softmax
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= j_end; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j <= j_end; ++j) {
                scores[j] /= sum_exp;
            }
            
            // Weighted sum of V
            for (size_t d = 0; d < hd; ++d) {
                float res = 0.0f;
                for (size_t j = 0; j <= j_end; ++j) {
                    float v_val = llaisys::utils::cast<float>(v_ptr[j * nkvh * hd + h_kv * hd + d]);
                    res += scores[j] * v_val;
                }
                out_ptr[i * nh * hd + h * hd + d] = llaisys::utils::cast<T>(res);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (q->dtype() == LLAISYS_DTYPE_F32) {
        self_attention_<float>(attn_val, q, k, v, scale);
    } else if (q->dtype() == LLAISYS_DTYPE_F16) {
        self_attention_<llaisys::fp16_t>(attn_val, q, k, v, scale);
    } else if (q->dtype() == LLAISYS_DTYPE_BF16) {
        self_attention_<llaisys::bf16_t>(attn_val, q, k, v, scale);
    } else {
        ASSERT(false, "Unsupported data type");
    }
}

} // namespace llaisys::ops::cpu
