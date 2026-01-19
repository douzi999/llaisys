#include "op.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias); 
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (bias) CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    if (in->ndim() != 2)
        throw std::runtime_error("linear: input must be 2D");
    if (weight->ndim() != 2)
        throw std::runtime_error("linear: weight must be 2D");
    if (out->ndim() != 2)
        throw std::runtime_error("linear: output must be 2D");
    if (bias && bias->ndim() != 1)
        throw std::runtime_error("linear: bias must be 1D");
    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0]; 
    if (weight->shape()[1] != in_features)
        throw std::runtime_error("linear: weight and input feature size mismatch");
    if (out->shape()[0] != batch || out->shape()[1] != out_features)
        throw std::runtime_error("linear: output shape mismatch");
    if (bias && bias->shape()[0] != out_features)
        throw std::runtime_error("linear: bias shape mismatch");
    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "linear: all tensors must be contiguous");
    if (bias) ASSERT(bias->isContiguous(), "linear: bias must be contiguous");

    auto get_val = [](const std::byte* data, size_t idx, llaisysDataType_t dtype) -> float {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  return reinterpret_cast<const float*>(data)[idx];
            case LLAISYS_DTYPE_F16:  return utils::_f16_to_f32(reinterpret_cast<const fp16_t*>(data)[idx]);
            case LLAISYS_DTYPE_BF16: return utils::_bf16_to_f32(reinterpret_cast<const bf16_t*>(data)[idx]);
            default: throw std::runtime_error("linear: unsupported dtype");
        }
    };

    auto set_val = [&](std::byte* data, size_t idx, float val, llaisysDataType_t dtype) {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  reinterpret_cast<float*>(data)[idx] = val; break;
            case LLAISYS_DTYPE_F16:  reinterpret_cast<fp16_t*>(data)[idx] = utils::_f32_to_f16(val); break;
            case LLAISYS_DTYPE_BF16: reinterpret_cast<bf16_t*>(data)[idx] = utils::_f32_to_bf16(val); break;
            default: throw std::runtime_error("linear: unsupported dtype");
        }
    };

    for (size_t i = 0; i < batch; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; k++) {
                sum += get_val(in->data(), i*in_features + k, in->dtype()) *
                    get_val(weight->data(), j*in_features + k, weight->dtype());
            }
            if (bias) sum += get_val(bias->data(), j, bias->dtype());
            set_val(out->data(), i*out_features + j, sum, out->dtype());
        }
    }

}
} // namespace llaisys::ops
