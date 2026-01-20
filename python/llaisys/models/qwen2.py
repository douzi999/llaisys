from typing import Sequence
from ctypes import byref, c_int, c_int64
import json

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysQwen2Model_t,
)
from .. import Tensor, RuntimeAPI, MemcpyKind

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._device = device
        self._model_path = Path(model_path)

        config_path = self._model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in: {self._model_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        meta = self._make_meta(config)
        self._meta = meta

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), self._device, device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create LLAISYS Qwen2 model")

        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not self._weights:
            raise RuntimeError("Failed to get LLAISYS Qwen2 weights handle")

        self._api = RuntimeAPI(self._device)
        self._weight_tensors = {}

        try:
            import torch  
        except Exception as exc:
            raise RuntimeError("torch is required to load bfloat16 safetensors") from exc

        for file in sorted(self._model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_)
                self._load_weight(name_, tensor)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if max_new_tokens is None:
            max_new_tokens = 0

        tokens = list(inputs)
        for _ in range(max_new_tokens):
            arr = (c_int64 * len(tokens))(*tokens)
            next_id = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(tokens))
            tokens.append(int(next_id))
            if int(next_id) == int(self._meta.end_token):
                break

        return tokens

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)

    @staticmethod
    def _map_dtype(dtype_str: str) -> DataType:
        if not dtype_str:
            return DataType.BF16
        s = str(dtype_str).lower()
        if s in ("bfloat16", "bf16"):
            return DataType.BF16
        if s in ("float16", "fp16", "f16"):
            return DataType.F16
        if s in ("float32", "fp32", "f32"):
            return DataType.F32
        return DataType.BF16

    @staticmethod
    def _get_eos_id(config) -> int:
        eos = config.get("eos_token_id", config.get("end_token_id", -1))
        if isinstance(eos, list) and eos:
            return int(eos[-1])
        return int(eos) if eos is not None else -1

    def _make_meta(self, config) -> LlaisysQwen2Meta:
        meta = LlaisysQwen2Meta()
        meta.dtype = self._map_dtype(config.get("torch_dtype"))
        meta.nlayer = int(config.get("num_hidden_layers", 0))
        meta.hs = int(config.get("hidden_size", 0))
        meta.nh = int(config.get("num_attention_heads", 0))
        meta.nkvh = int(config.get("num_key_value_heads", meta.nh))
        meta.dh = int(meta.hs // meta.nh) if meta.nh else 0
        meta.di = int(config.get("intermediate_size", 0))
        meta.maxseq = int(
            config.get(
                "max_position_embeddings",
                config.get("max_seq_len", 0),
            )
        )
        meta.voc = int(config.get("vocab_size", 0))
        meta.epsilon = float(config.get("rms_norm_eps", 1e-5))
        meta.theta = float(config.get("rope_theta", 10000.0))
        meta.end_token = int(self._get_eos_id(config))
        return meta

    @staticmethod
    def _torch_dtype_to_llaisys(dtype) -> DataType:
        import torch

        if dtype == torch.bfloat16:
            return DataType.BF16
        if dtype == torch.float16:
            return DataType.F16
        if dtype == torch.float32:
            return DataType.F32
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    def _make_llaisys_tensor(self, torch_tensor):
        if not torch_tensor.is_contiguous():
            torch_tensor = torch_tensor.contiguous()
        ll_dtype = self._torch_dtype_to_llaisys(torch_tensor.dtype)
        ll_tensor = Tensor(torch_tensor.shape, dtype=ll_dtype, device=self._device)
        bytes_ = torch_tensor.numel() * torch_tensor.element_size()
        self._api.memcpy_sync(
            ll_tensor.data_ptr(),
            torch_tensor.data_ptr(),
            bytes_,
            MemcpyKind.D2D,
        )
        return ll_tensor

    def _load_weight(self, name: str, torch_tensor):
        ll_tensor = self._make_llaisys_tensor(torch_tensor)
        self._weight_tensors[name] = ll_tensor

        w = self._weights.contents
        if name == "model.embed_tokens.weight":
            w.in_embed = ll_tensor.lib_tensor()
            return
        if name == "lm_head.weight":
            w.out_embed = ll_tensor.lib_tensor()
            return
        if name == "model.norm.weight":
            w.out_norm_w = ll_tensor.lib_tensor()
            return

        if not name.startswith("model.layers."):
            return

        parts = name.split(".")
        if len(parts) < 5:
            return

        try:
            layer_idx = int(parts[2])
        except ValueError:
            return

        if layer_idx < 0 or layer_idx >= int(self._meta.nlayer):
            return

        if parts[3] == "input_layernorm" and parts[4] == "weight":
            w.attn_norm_w[layer_idx] = ll_tensor.lib_tensor()
            return
        if parts[3] == "post_attention_layernorm" and parts[4] == "weight":
            w.mlp_norm_w[layer_idx] = ll_tensor.lib_tensor()
            return

        if parts[3] == "self_attn":
            proj = parts[4]
            kind = parts[5] if len(parts) > 5 else ""
            if proj == "q_proj" and kind == "weight":
                w.attn_q_w[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "q_proj" and kind == "bias":
                w.attn_q_b[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "k_proj" and kind == "weight":
                w.attn_k_w[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "k_proj" and kind == "bias":
                w.attn_k_b[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "v_proj" and kind == "weight":
                w.attn_v_w[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "v_proj" and kind == "bias":
                w.attn_v_b[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "o_proj" and kind == "weight":
                w.attn_o_w[layer_idx] = ll_tensor.lib_tensor()
                return

        if parts[3] == "mlp":
            proj = parts[4]
            kind = parts[5] if len(parts) > 5 else ""
            if proj == "gate_proj" and kind == "weight":
                w.mlp_gate_w[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "up_proj" and kind == "weight":
                w.mlp_up_w[layer_idx] = ll_tensor.lib_tensor()
                return
            if proj == "down_proj" and kind == "weight":
                w.mlp_down_w[layer_idx] = ll_tensor.lib_tensor()
                return
