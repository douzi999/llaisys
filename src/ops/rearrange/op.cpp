#include "op.hpp"

#include <cstring>
#include <vector>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    if (out->ndim() != in->ndim())
        throw std::runtime_error("rearrange: input and output must have same ndim");

    size_t numel = in->numel();
    if (numel == 0)
        return;

    if (in->isContiguous() && out->isContiguous()) {
        std::memcpy(out->data(), in->data(), numel * in->elementSize());
        return;
    }

    const auto &shape = in->shape();
    const auto &in_strides = in->strides();
    const auto &out_strides = out->strides();
    size_t ndim = shape.size();

    std::vector<size_t> idx(ndim, 0);

    auto calc_offset = [&](const std::vector<size_t> &indices,
                           const std::vector<ptrdiff_t> &strides) -> ptrdiff_t {
        ptrdiff_t off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            off += static_cast<ptrdiff_t>(indices[d]) * strides[d];
        }
        return off;
    };

    llaisysDataType_t dtype = in->dtype();
    const std::byte *in_ptr = in->data();
    std::byte *out_ptr = out->data();

    for (size_t n = 0; n < numel; ++n) {
        ptrdiff_t in_off = calc_offset(idx, in_strides);
        ptrdiff_t out_off = calc_offset(idx, out_strides);

        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            *(reinterpret_cast<float *>(out_ptr) + out_off) =
                *(reinterpret_cast<const float *>(in_ptr) + in_off);
            break;
        case LLAISYS_DTYPE_F16:
            *(reinterpret_cast<fp16_t *>(out_ptr) + out_off) =
                *(reinterpret_cast<const fp16_t *>(in_ptr) + in_off);
            break;
        case LLAISYS_DTYPE_BF16:
            *(reinterpret_cast<bf16_t *>(out_ptr) + out_off) =
                *(reinterpret_cast<const bf16_t *>(in_ptr) + in_off);
            break;
        default:
            throw std::runtime_error("rearrange: unsupported dtype");
        }

        for (size_t d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}
} // namespace llaisys::ops
