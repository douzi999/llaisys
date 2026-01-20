#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int *device_ids = nullptr;
    int ndevice = 0;

    // KV-cache
    size_t cache_len = 0;
    size_t maxseq = 0;
    llaisys::tensor_t *k_cache = nullptr;
    llaisys::tensor_t *v_cache = nullptr;
};

static llaisys::tensor_t require_tensor(llaisysTensor_t t, const char *name) {
    if (!t) throw std::runtime_error(std::string("qwen2: missing weight ") + name);
    return t->tensor;
}

static llaisys::tensor_t optional_tensor(llaisysTensor_t t) {
    return t ? t->tensor : nullptr;
}

static float read_val(const std::byte *data, size_t idx, llaisysDataType_t dtype) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32: return reinterpret_cast<const float*>(data)[idx];
        case LLAISYS_DTYPE_F16: return llaisys::utils::_f16_to_f32(reinterpret_cast<const llaisys::fp16_t*>(data)[idx]);
        case LLAISYS_DTYPE_BF16: return llaisys::utils::_bf16_to_f32(reinterpret_cast<const llaisys::bf16_t*>(data)[idx]);
        default: throw std::runtime_error("qwen2: unsupported dtype");
    }
}

__C {

__export struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) 
{
    if (!meta) return nullptr;

    auto *model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->ndevice = ndevice;
    model->maxseq = meta->maxseq;
    model->cache_len = 0;

    if (device_ids && ndevice > 0) {
        model->device_ids = new int[ndevice];
        std::memcpy(model->device_ids, device_ids, sizeof(int) * ndevice);
    }

    std::memset(&model->weights, 0, sizeof(model->weights));
    size_t nlayer = meta->nlayer;
    if (nlayer > 0) {
        model->weights.attn_norm_w = new llaisysTensor_t[nlayer]();
        model->weights.attn_q_w = new llaisysTensor_t[nlayer]();
        model->weights.attn_q_b = new llaisysTensor_t[nlayer]();
        model->weights.attn_k_w = new llaisysTensor_t[nlayer]();
        model->weights.attn_k_b = new llaisysTensor_t[nlayer]();
        model->weights.attn_v_w = new llaisysTensor_t[nlayer]();
        model->weights.attn_v_b = new llaisysTensor_t[nlayer]();
        model->weights.attn_o_w = new llaisysTensor_t[nlayer]();
        model->weights.mlp_norm_w = new llaisysTensor_t[nlayer]();
        model->weights.mlp_gate_w = new llaisysTensor_t[nlayer]();
        model->weights.mlp_up_w = new llaisysTensor_t[nlayer]();
        model->weights.mlp_down_w = new llaisysTensor_t[nlayer]();

        // allocate KV-cache
        model->k_cache = new llaisys::tensor_t[nlayer]();
        model->v_cache = new llaisys::tensor_t[nlayer]();
        int device_id = (model->device_ids && model->ndevice > 0) ? model->device_ids[0] : 0;
        for (size_t i = 0; i < nlayer; ++i) {
            model->k_cache[i] = llaisys::Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, model->device, device_id);
            model->v_cache[i] = llaisys::Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, model->device, device_id);
        }
    }

    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;

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

    delete[] model->k_cache;
    delete[] model->v_cache;

    delete[] model->device_ids;
    delete model;
}

__export struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model) return nullptr;
    return &model->weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0) return model->meta.end_token;

    const auto &meta = model->meta;
    int device_id = (model->device_ids && model->ndevice > 0) ? model->device_ids[0] : 0;
    if (model->maxseq > 0 && ntoken > model->maxseq) throw std::runtime_error("qwen2: ntoken exceeds maxseq");

    bool use_cache = (model->cache_len > 0 && ntoken == model->cache_len + 1);
    if (!use_cache) {
        model->cache_len = 0;
    }
    size_t q_len = use_cache ? 1 : ntoken;

    std::vector<size_t> tok_shape = {q_len};
    auto tok = llaisys::Tensor::create(tok_shape, LLAISYS_DTYPE_I64, model->device, device_id);
    if (use_cache) tok->load(&token_ids[ntoken - 1]);
    else tok->load(token_ids);

    std::vector<int64_t> pos_data(q_len);
    if (use_cache) pos_data[0] = static_cast<int64_t>(model->cache_len);
    else for (size_t i = 0; i < ntoken; ++i) pos_data[i] = static_cast<int64_t>(i);
    auto pos = llaisys::Tensor::create(tok_shape, LLAISYS_DTYPE_I64, model->device, device_id);
    pos->load(pos_data.data());

    auto in_embed = require_tensor(model->weights.in_embed, "in_embed");
    auto x = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
    llaisys::ops::embedding(x, tok, in_embed);

    float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        auto attn_norm_w = require_tensor(model->weights.attn_norm_w[layer], "attn_norm_w");
        auto q_w = require_tensor(model->weights.attn_q_w[layer], "attn_q_w");
        auto k_w = require_tensor(model->weights.attn_k_w[layer], "attn_k_w");
        auto v_w = require_tensor(model->weights.attn_v_w[layer], "attn_v_w");
        auto o_w = require_tensor(model->weights.attn_o_w[layer], "attn_o_w");

        auto q_b = optional_tensor(model->weights.attn_q_b[layer]);
        auto k_b = optional_tensor(model->weights.attn_k_b[layer]);
        auto v_b = optional_tensor(model->weights.attn_v_b[layer]);

        auto x1 = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::rms_norm(x1, x, attn_norm_w, meta.epsilon);

        auto q2d = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::linear(q2d, x1, q_w, q_b);

        llaisys::tensor_t k3d, v3d;
        if (!use_cache) {
            auto k2d = llaisys::Tensor::create({q_len, meta.nkvh*meta.dh}, meta.dtype, model->device, device_id);
            auto v2d = llaisys::Tensor::create({q_len, meta.nkvh*meta.dh}, meta.dtype, model->device, device_id);
            llaisys::ops::linear(k2d, x1, k_w, k_b);
            llaisys::ops::linear(v2d, x1, v_w, v_b);

            k3d = llaisys::Tensor::create({q_len, meta.nkvh, meta.dh}, meta.dtype, model->device, device_id);
            v3d = llaisys::Tensor::create({q_len, meta.nkvh, meta.dh}, meta.dtype, model->device, device_id);
            llaisys::ops::rearrange(k3d, k2d->view({q_len, meta.nkvh, meta.dh}));
            llaisys::ops::rearrange(v3d, v2d->view({q_len, meta.nkvh, meta.dh}));

            auto k_cache = model->k_cache[layer];
            auto v_cache = model->v_cache[layer];
            auto k_slice = k_cache->slice(0, model->cache_len, model->cache_len+q_len);
            auto v_slice = v_cache->slice(0, model->cache_len, model->cache_len+q_len);
            llaisys::ops::rearrange(k_slice, k3d);
            llaisys::ops::rearrange(v_slice, v3d);

            k3d = k_cache->slice(0, 0, model->cache_len+q_len);
            v3d = v_cache->slice(0, 0, model->cache_len+q_len);
        } else {
            // decode: compute k/v for the new token and append to cache
            auto k2d = llaisys::Tensor::create({q_len, meta.nkvh*meta.dh}, meta.dtype, model->device, device_id);
            auto v2d = llaisys::Tensor::create({q_len, meta.nkvh*meta.dh}, meta.dtype, model->device, device_id);
            llaisys::ops::linear(k2d, x1, k_w, k_b);
            llaisys::ops::linear(v2d, x1, v_w, v_b);

            k3d = llaisys::Tensor::create({q_len, meta.nkvh, meta.dh}, meta.dtype, model->device, device_id);
            v3d = llaisys::Tensor::create({q_len, meta.nkvh, meta.dh}, meta.dtype, model->device, device_id);
            llaisys::ops::rearrange(k3d, k2d->view({q_len, meta.nkvh, meta.dh}));
            llaisys::ops::rearrange(v3d, v2d->view({q_len, meta.nkvh, meta.dh}));
            // apply RoPE on new k before caching
            llaisys::ops::rope(k3d, k3d, pos, meta.theta);

            auto k_cache = model->k_cache[layer];
            auto v_cache = model->v_cache[layer];
            auto k_slice = k_cache->slice(0, model->cache_len, model->cache_len + q_len);
            auto v_slice = v_cache->slice(0, model->cache_len, model->cache_len + q_len);
            llaisys::ops::rearrange(k_slice, k3d);
            llaisys::ops::rearrange(v_slice, v3d);

            k3d = k_cache->slice(0, 0, model->cache_len + q_len);
            v3d = v_cache->slice(0, 0, model->cache_len + q_len);
        }

        auto q3d = llaisys::Tensor::create({q_len, meta.nh, meta.dh}, meta.dtype, model->device, device_id);
        llaisys::ops::rearrange(q3d, q2d->view({q_len, meta.nh, meta.dh}));
        llaisys::ops::rope(q3d, q3d, pos, meta.theta);
        // Only apply RoPE to k of the newly computed tokens.
        if (!use_cache) {
            llaisys::ops::rope(k3d, k3d, pos, meta.theta);
        }

        auto attn = llaisys::Tensor::create({q_len, meta.nh, meta.dh}, meta.dtype, model->device, device_id);
        llaisys::ops::self_attention(attn, q3d, k3d, v3d, scale);

        auto attn2d = attn->view({q_len, meta.hs});
        auto attn_out = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::linear(attn_out, attn2d, o_w, nullptr);

        auto x_attn = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::add(x_attn, x, attn_out);
        x = x_attn;

        // MLP
        auto mlp_norm_w = require_tensor(model->weights.mlp_norm_w[layer], "mlp_norm_w");
        auto gate_w = require_tensor(model->weights.mlp_gate_w[layer], "mlp_gate_w");
        auto up_w = require_tensor(model->weights.mlp_up_w[layer], "mlp_up_w");
        auto down_w = require_tensor(model->weights.mlp_down_w[layer], "mlp_down_w");

        auto x2 = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::rms_norm(x2, x, mlp_norm_w, meta.epsilon);

        auto gate = llaisys::Tensor::create({q_len, meta.di}, meta.dtype, model->device, device_id);
        auto up = llaisys::Tensor::create({q_len, meta.di}, meta.dtype, model->device, device_id);
        llaisys::ops::linear(gate, x2, gate_w, nullptr);
        llaisys::ops::linear(up, x2, up_w, nullptr);

        auto mlp = llaisys::Tensor::create({q_len, meta.di}, meta.dtype, model->device, device_id);
        llaisys::ops::swiglu(mlp, gate, up);

        auto mlp_out = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::linear(mlp_out, mlp, down_w, nullptr);

        auto x_mlp = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
        llaisys::ops::add(x_mlp, x, mlp_out);
        x = x_mlp;
    }

    auto out_norm_w = require_tensor(model->weights.out_norm_w, "out_norm_w");
    auto out_embed = require_tensor(model->weights.out_embed, "out_embed");

    auto x_norm = llaisys::Tensor::create({q_len, meta.hs}, meta.dtype, model->device, device_id);
    llaisys::ops::rms_norm(x_norm, x, out_norm_w, meta.epsilon);

    auto x_last = x_norm->slice(0, q_len-1, q_len);
    auto logits = llaisys::Tensor::create({1, meta.voc}, meta.dtype, model->device, device_id);
    llaisys::ops::linear(logits, x_last, out_embed, nullptr);

    const std::byte *logits_ptr = logits->data();
    size_t vocab = meta.voc;
    float max_val = read_val(logits_ptr, 0, meta.dtype);
    int64_t max_idx = 0;
    for (size_t j = 1; j < vocab; ++j) {
        float v = read_val(logits_ptr, j, meta.dtype);
        if (v > max_val) { max_val = v; max_idx = static_cast<int64_t>(j); }
    }

    model->cache_len += q_len;
    return max_idx;
}

} // extern "C"
