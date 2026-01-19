#include "op.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), k->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), v->dtype());

    if (q->ndim() != 3 || k->ndim() != 3 || v->ndim() != 3 || attn_val->ndim() != 3)
        throw std::runtime_error("self_attention: all tensors must be 3D");

    size_t seqlen_q = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t dq = q->shape()[2];

    size_t seqlen_k = k->shape()[0];
    size_t nkvhead = k->shape()[1];

    size_t dk = k->shape()[2];
    size_t dv = v->shape()[2];

    if (dq != dk)
        throw std::runtime_error("self_attention: q and k last dim must match");
    if (attn_val->shape()[0] != seqlen_q || attn_val->shape()[1] != nhead || attn_val->shape()[2] != dv)
        throw std::runtime_error("self_attention: attn_val shape mismatch");

    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous() && attn_val->isContiguous(),
           "self_attention: all tensors must be contiguous");

    auto get_val = [](const std::byte* data, size_t idx, llaisysDataType_t dtype) -> float {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  return reinterpret_cast<const float*>(data)[idx];
            case LLAISYS_DTYPE_F16:  return utils::_f16_to_f32(reinterpret_cast<const fp16_t*>(data)[idx]);
            case LLAISYS_DTYPE_BF16: return utils::_bf16_to_f32(reinterpret_cast<const bf16_t*>(data)[idx]);
            default: throw std::runtime_error("self_attention: unsupported dtype");
        }
    };

    auto set_val = [&](std::byte* data, size_t idx, float val, llaisysDataType_t dtype) {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  reinterpret_cast<float*>(data)[idx] = val; break;
            case LLAISYS_DTYPE_F16:  reinterpret_cast<fp16_t*>(data)[idx] = utils::_f32_to_f16(val); break;
            case LLAISYS_DTYPE_BF16: reinterpret_cast<bf16_t*>(data)[idx] = utils::_f32_to_bf16(val); break;
            default: throw std::runtime_error("self_attention: unsupported dtype");
        }
    };

    llaisysDataType_t dtype = q->dtype();

    if (nhead % nkvhead != 0)
        throw std::runtime_error("self_attention: nhead must be a multiple of nkvhead");
    size_t head_repeat = nhead / nkvhead;

    int64_t shift = static_cast<int64_t>(seqlen_k) - static_cast<int64_t>(seqlen_q);

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / head_repeat;
        for (size_t i = 0; i < seqlen_q; i++) {
            std::vector<float> scores(seqlen_k, 0.0f);

            int64_t max_j = static_cast<int64_t>(i) + shift;
            if (max_j < 0) {
                for (size_t d = 0; d < dv; d++) {
                    size_t idx_out = i * nhead * dv + h * dv + d;
                    set_val(attn_val->data(), idx_out, 0.0f, dtype);
                }
                continue;
            }
            if (max_j >= static_cast<int64_t>(seqlen_k))
                max_j = static_cast<int64_t>(seqlen_k) - 1;

            for (size_t j = 0; j < seqlen_k; j++) {
                float s = 0.0f;
                for (size_t d = 0; d < dq; d++) {
                    size_t idx_q = i * nhead * dq + h * dq + d;
                    size_t idx_k = j * nkvhead * dk + kv_h * dk + d;
                    float q_val = get_val(q->data(), idx_q, dtype);
                    float k_val = get_val(k->data(), idx_k, dtype);
                    s += q_val * k_val;
                }
                scores[j] = s * scale;
            }

            float max_score = -1e9f;
            for (size_t j = 0; j <= static_cast<size_t>(max_j); j++)
                max_score = std::max(max_score, scores[j]);
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= static_cast<size_t>(max_j); j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j <= static_cast<size_t>(max_j); j++) scores[j] /= sum_exp;
            for (size_t j = static_cast<size_t>(max_j) + 1; j < seqlen_k; j++) scores[j] = 0.0f; 

            for (size_t d = 0; d < dv; d++) {
                float val = 0.0f;
                for (size_t j = 0; j < seqlen_k; j++) {
                    size_t idx_v = j * nkvhead * dv + kv_h * dv + d;
                    val += scores[j] * get_val(v->data(), idx_v, dtype);
                }
                size_t idx_out = i * nhead * dv + h * dv + d;
                set_val(attn_val->data(), idx_out, val, dtype);
            }
        }
    }
}

} // namespace llaisys::ops
