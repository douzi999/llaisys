#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    if (in->dtype() != out->dtype())
        throw std::runtime_error("rope: in and out must have same dtype");
    if (in->ndim() != 3 || out->ndim() != 3)
        throw std::runtime_error("rope: in and out must be 3D");
    if (pos_ids->ndim() != 1 || pos_ids->shape()[0] != in->shape()[0])
        throw std::runtime_error("rope: pos_ids must be 1D and match seqlen");
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64)
        throw std::runtime_error("rope: pos_ids must be int64");
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t dim = in->shape()[2];
    if (dim % 2 != 0)
        throw std::runtime_error("rope: last dimension must be even");
    if (out->shape()[0] != seqlen || out->shape()[1] != nhead || out->shape()[2] != dim)
        throw std::runtime_error("rope: output shape mismatch");
    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");

    auto get_val = [](const std::byte* data, size_t idx, llaisysDataType_t dtype) -> float {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  return reinterpret_cast<const float*>(data)[idx];
            case LLAISYS_DTYPE_F16:  return utils::_f16_to_f32(reinterpret_cast<const fp16_t*>(data)[idx]);
            case LLAISYS_DTYPE_BF16: return utils::_bf16_to_f32(reinterpret_cast<const bf16_t*>(data)[idx]);
            default: throw std::runtime_error("rope: unsupported dtype");
        }
    };
    auto set_val = [&](std::byte* data, size_t idx, float val, llaisysDataType_t dtype) {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  reinterpret_cast<float*>(data)[idx] = val; break;
            case LLAISYS_DTYPE_F16:  reinterpret_cast<fp16_t*>(data)[idx] = utils::_f32_to_f16(val); break;
            case LLAISYS_DTYPE_BF16: reinterpret_cast<bf16_t*>(data)[idx] = utils::_f32_to_bf16(val); break;
            default: throw std::runtime_error("rope: unsupported dtype");
        }
    };

    size_t half_dim = dim / 2;

    for (size_t i = 0; i < seqlen; i++) {
        int64_t pos = reinterpret_cast<const int64_t*>(pos_ids->data())[i];

        for (size_t h = 0; h < nhead; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                size_t idx_a = i * nhead * dim + h * dim + j;
                size_t idx_b = idx_a + half_dim;

                float a = get_val(in->data(), idx_a, in->dtype());
                float b = get_val(in->data(), idx_b, in->dtype());

                float phi = static_cast<float>(pos) / std::pow(theta, 2.0f * j / static_cast<float>(dim));
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);

                float a_ = a * cos_phi - b * sin_phi;
                float b_ = b * cos_phi + a * sin_phi;

                set_val(out->data(), idx_a, a_, out->dtype());
                set_val(out->data(), idx_b, b_, out->dtype());
            }
        }
    }
}
} // namespace llaisys::ops
