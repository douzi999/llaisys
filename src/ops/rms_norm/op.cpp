#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (in->ndim() != 2)
        throw std::runtime_error("rms_norm: input must be 2D");
    if (weight->ndim() != 1)
        throw std::runtime_error("rms_norm: weight must be 1D");
    if (out->ndim() != 2)
        throw std::runtime_error("rms_norm: output must be 2D");
    size_t batch = in->shape()[0];
    size_t features = in->shape()[1];    
    if (weight->shape()[0] != features)
        throw std::runtime_error("rms_norm: weight length must match input feature size");
    if (out->shape()[0] != batch || out->shape()[1] != features)
        throw std::runtime_error("rms_norm: output shape must match input shape");
    ASSERT(in->isContiguous() && out->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");
    
    auto get_val = [](const std::byte* data, size_t idx, llaisysDataType_t dtype) -> float {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  return reinterpret_cast<const float*>(data)[idx];
            case LLAISYS_DTYPE_F16:  return utils::_f16_to_f32(reinterpret_cast<const fp16_t*>(data)[idx]);
            case LLAISYS_DTYPE_BF16: return utils::_bf16_to_f32(reinterpret_cast<const bf16_t*>(data)[idx]);
            default: throw std::runtime_error("rms_norm: unsupported dtype");
        }
    };

    auto set_val = [&](std::byte* data, size_t idx, float val, llaisysDataType_t dtype) {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  reinterpret_cast<float*>(data)[idx] = val; break;
            case LLAISYS_DTYPE_F16:  reinterpret_cast<fp16_t*>(data)[idx] = utils::_f32_to_f16(val); break;
            case LLAISYS_DTYPE_BF16: reinterpret_cast<bf16_t*>(data)[idx] = utils::_f32_to_bf16(val); break;
            default: throw std::runtime_error("rms_norm: unsupported dtype");
        }
    };

    for (size_t i = 0; i < batch; i++) {
        float sum_squares = 0.0f;

        for (size_t j = 0; j < features; j++) {
            float x = get_val(in->data(), i*features + j, in->dtype());
            sum_squares += x * x;
        }

        float rms = std::sqrt(sum_squares / features + eps);

    
        for (size_t j = 0; j < features; j++) {
            float x = get_val(in->data(), i*features + j, in->dtype());
            float w = get_val(weight->data(), j, weight->dtype());
            set_val(out->data(), i*features + j, (x / rms) * w, out->dtype());
        }
    }

}
} // namespace llaisys::ops
