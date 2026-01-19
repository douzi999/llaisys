#include "op.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be of type int64.");
    if (vals->ndim() != 1) 
        throw std::runtime_error("argmax: vals must be 1D");
    if (max_val->numel() != 1 || max_idx->numel() != 1)
        throw std::runtime_error("argmax: outputs must have 1 element");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");
    
    float max_value;
    int64_t max_index = 0;
    llaisysDataType_t dtype = vals->dtype();
    int64_t n = vals->numel();
    const std::byte* vals_ptr = vals->data();

    switch (dtype){
        case LLAISYS_DTYPE_F32:{
            const float* data = reinterpret_cast<const float*>(vals->data());
            max_value = data[0];
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const fp16_t* data = reinterpret_cast<const fp16_t*>(vals->data());
            max_value = utils::_f16_to_f32(data[0]);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const bf16_t* data = reinterpret_cast<const bf16_t*>(vals->data());
            max_value = utils::_bf16_to_f32(data[0]);
            break;
        }
        default:
            throw std::runtime_error("argmax: unsupported datatype");
    }

    for (int64_t i = 1; i < n; i++) {
        float val_i;
        switch(vals->dtype()) {
            case LLAISYS_DTYPE_F32:
                val_i = reinterpret_cast<const float*>(vals_ptr)[i];
                break;
            case LLAISYS_DTYPE_F16:
                val_i = utils::_f16_to_f32(reinterpret_cast<const fp16_t*>(vals_ptr)[i]);
                break;
            case LLAISYS_DTYPE_BF16:
                val_i = utils::_bf16_to_f32(reinterpret_cast<const bf16_t*>(vals_ptr)[i]);
                break;
            default:
                throw std::runtime_error("argmax: unsupported datatype");
        }
        if (val_i > max_value) {
            max_value = val_i;
            max_index = static_cast<int64_t>(i);
        }
    }

    switch(max_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            reinterpret_cast<float*>(max_val->data())[0] = max_value;
            break;
        case LLAISYS_DTYPE_F16:
            reinterpret_cast<fp16_t*>(max_val->data())[0] = utils::_f32_to_f16(max_value);
            break;
        case LLAISYS_DTYPE_BF16:
            reinterpret_cast<bf16_t*>(max_val->data())[0] = utils::_f32_to_bf16(max_value);
            break;
        default:
            throw std::runtime_error("argmax: unsupported max_val dtype");
    }

    reinterpret_cast<int64_t*>(max_idx->data())[0] = max_index;

}
} // namespace llaisys::ops
