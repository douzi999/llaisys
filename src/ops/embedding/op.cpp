#include "op.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64");
    if (weight->ndim() != 2)
        throw std::runtime_error("embedding: weight must be 2D");
    if (index->ndim() != 1)
        throw std::runtime_error("embedding: index must be 1D");
    if (out->ndim() != 2)
        throw std::runtime_error("embedding: out must be 2D");
    if (out->shape()[0] != index->numel() || out->shape()[1] != weight->shape()[1])
        throw std::runtime_error("embedding: out shape mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous");
    
    size_t n_rows = index->numel();
    size_t embedding_dim = weight->shape()[1];

    const int64_t* index_ptr = reinterpret_cast<const int64_t*>(index->data());
    std::byte* out_ptr = out->data();
    const std::byte* weight_ptr = weight->data();
    for (size_t i = 0; i < n_rows; i++) {
        int64_t idx = index_ptr[i];
        switch (weight->dtype()) {
            case LLAISYS_DTYPE_F32: {
                float* out_row = reinterpret_cast<float*>(out_ptr) + i * embedding_dim;
                const float* weight_row = reinterpret_cast<const float*>(weight_ptr) + idx * embedding_dim;

                for (size_t j = 0; j < embedding_dim; j++)
                    out_row[j] = weight_row[j];
                break;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t* out_row = reinterpret_cast<fp16_t*>(out_ptr) + i * embedding_dim;
                const fp16_t* weight_row = reinterpret_cast<const fp16_t*>(weight_ptr) + idx * embedding_dim;

                for (size_t j = 0; j < embedding_dim; j++)
                    out_row[j] = weight_row[j];
                break;
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t* out_row = reinterpret_cast<bf16_t*>(out_ptr) + i * embedding_dim;
                const bf16_t* weight_row = reinterpret_cast<const bf16_t*>(weight_ptr) + idx * embedding_dim;

                for (size_t j = 0; j < embedding_dim; j++)
                    out_row[j] = weight_row[j];
                break;
            }
            default:
                throw std::runtime_error("embedding: unsupported dtype");
        }
    }
}
} // namespace llaisys::ops
