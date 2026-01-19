#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    if (out->ndim() != 2 || gate->ndim() != 2 || up->ndim() != 2)
        throw std::runtime_error("swiglu: all tensors must be 2D");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");

    auto get_val = [](const std::byte* data, size_t idx, llaisysDataType_t dtype) -> float {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  return reinterpret_cast<const float*>(data)[idx];
            case LLAISYS_DTYPE_F16:  return utils::_f16_to_f32(reinterpret_cast<const fp16_t*>(data)[idx]);
            case LLAISYS_DTYPE_BF16: return utils::_bf16_to_f32(reinterpret_cast<const bf16_t*>(data)[idx]);
            default: throw std::runtime_error("swiglu: unsupported dtype");
        }
    };

    auto set_val = [&](std::byte* data, size_t idx, float val, llaisysDataType_t dtype) {
        switch(dtype) {
            case LLAISYS_DTYPE_F32:  reinterpret_cast<float*>(data)[idx] = val; break;
            case LLAISYS_DTYPE_F16:  reinterpret_cast<fp16_t*>(data)[idx] = utils::_f32_to_f16(val); break;
            case LLAISYS_DTYPE_BF16: reinterpret_cast<bf16_t*>(data)[idx] = utils::_f32_to_bf16(val); break;
            default: throw std::runtime_error("swiglu: unsupported dtype");
        }
    };

    size_t n = out->numel();
    llaisysDataType_t dtype = out->dtype();
    for (size_t i = 0; i < n; i++) {
        float g = get_val(gate->data(), i, dtype);
        float u = get_val(up->data(), i, dtype);
        float swish = g / (1.0f + std::exp(-g));
        set_val(out->data(), i, u * swish, dtype);
    }
}
} // namespace llaisys::ops
