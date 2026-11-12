# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE engine on FlashInfer's BF16 trtllm-gen path (declared by ``MoEDecodeOptions.bf16_draft_engine``): an
un-finalized logits-routed SwiGLU run of up to ``M_MAX`` rows takes one launch (``moe_bf16_engine.py``)
instead of ``trtllm_bf16_moe``'s routing + two GEMMs and returns its ``do_finalize=False`` outputs: the same gemm2 row
and BF16 top-k weight per (token, k) bit for bit, through the engine's own expanded -> permuted index (rows packed).
"""

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import (
    M_MAX,
    supports_local_experts,
    supports_routing,
)
from tensorrt_llm._torch.utils import ActivationType

from ..quantization import TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K
from ..routing import RoutingMethodType

if TYPE_CHECKING:
    from .flashinfer_bf16 import FlashinferTrtllmGenBf16Impl
    from .kernel_inputs import KernelInputs


def applies(impl: "FlashinferTrtllmGenBf16Impl", k: "KernelInputs") -> bool:
    """Whether this run fits the engine (host-side checks only)."""
    if k.do_finalize or k.router_logits is None or k.routing_bias is not None:
        return False
    if impl.activation_type != ActivationType.Swiglu or impl.bias:
        return False
    if impl.act_alpha is not None or impl.act_beta is not None or impl.act_clamp is not None:
        return False
    if impl.routing_method.routing_method_type != RoutingMethodType.Renormalize:
        return False
    if not supports_routing(impl.num_slots, k.top_k) or not supports_local_experts(
        impl.num_slots, impl.expert_size_per_partition
    ):
        return False
    qm = impl.quant_method
    if not qm.use_shuffled_weight or qm.weight_layout != TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K:
        return False
    x, lg = k.x, k.router_logits
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    m = x.shape[0]
    if not 1 <= m <= M_MAX or x.stride(1) != 1 or (x.stride(0) * 2) % 16:
        return False
    if (
        lg.dtype != torch.bfloat16
        or tuple(lg.shape) != (m, impl.num_slots)
        or not lg.is_contiguous()
    ):
        return False
    w13, w2 = impl.w3_w1_weight, impl.w2_weight
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16 or w13.dim() != 4 or w2.dim() != 4:
        return False
    e, hb, two_i, kb = w13.shape
    hidden, inter = hb * kb, two_i // 2
    if (
        kb != 64
        or e != impl.expert_size_per_partition
        or tuple(w2.shape) != (e, inter // 64, hidden, 64)
        or x.shape[1] != hidden
    ):
        return False
    return hidden % 128 == 0 and inter % 128 == 0


def run(
    impl: "FlashinferTrtllmGenBf16Impl", k: "KernelInputs", state: torch.Tensor
) -> list[torch.Tensor]:
    from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import moe_bf16_engine

    x = k.x
    m = x.shape[0]
    top_k = k.top_k
    hidden, inter = impl.w2_weight.shape[2], impl.w3_w1_weight.shape[2] // 2
    dev = x.device
    h = torch.empty((m * top_k, inter), dtype=torch.bfloat16, device=dev)
    out = torch.empty((m * top_k, hidden), dtype=torch.bfloat16, device=dev)
    wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
    e2p = torch.empty((m, top_k), dtype=torch.int32, device=dev)
    moe_bf16_engine(
        x,
        k.router_logits,
        impl.w3_w1_weight,
        impl.w2_weight,
        h,
        out,
        wts,
        e2p,
        state,
        top_k=top_k,
        local_start=impl.slot_start,
    )
    # trtllm_bf16_moe's un-finalized ABI: gemm2 rows, top-k weights [m, top_k], expanded -> permuted index [m, top_k]
    return [out, wts, e2p]
