# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch packing of precomputed top-k routing for the TRTLLM-gen routed MoE ops.

The routed entry points take ``(expert_id << 16) | bf16_bits(weight)`` per (token, k) as int32. The torch expression
``(ids.to(int32) << 16) | weights.to(bfloat16).view(int16).to(int32)`` costs four elementwise launches on the decode
critical path; ``pack_topk_bf16`` writes the same words in one launch: BF16 round-to-nearest-even of the weight, its
bits sign-extended to 32 bits, OR'ed with the shifted id.
"""

import torch
import triton
import triton.language as tl

from ....flashinfer_utils import get_env_enable_pdl
from ....modules.fla.op import gdc_wait_with_memory_clobber

# Words per program (a decode step's top-k fits one program).
_BLOCK = 1024


# ``n`` only bounds the mask: not specialized, so one compiled variant serves every word count (Triton would
# otherwise compile n == 1, n % 16 == 0 and other n apart, the first in-step call of a new class compiling).
@triton.jit(do_not_specialize=["n"])
def _pack_topk_bf16_kernel(ids_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        gdc_wait_with_memory_clobber()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    ids = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    wbits = w.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    tl.store(out_ptr + offs, (ids << 16) | wbits, mask=mask)
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def pack_topk_bf16(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """int32 ``(topk_ids << 16) | bf16_bits(topk_weights)`` (bit-identical to the torch expression in the module
    docstring); falls back to that expression for inputs the kernel does not take."""
    if (
        not topk_ids.is_cuda
        or topk_ids.shape != topk_weights.shape
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or topk_ids.numel() == 0
    ):
        return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).contiguous().view(
            torch.int16
        ).to(torch.int32)
    ids = topk_ids.contiguous()
    w = topk_weights.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    n = ids.numel()
    use_pdl = get_env_enable_pdl()
    _pack_topk_bf16_kernel[(triton.cdiv(n, _BLOCK),)](
        ids, w, out, n, BLOCK=_BLOCK, USE_PDL=use_pdl, num_warps=4, launch_pdl=use_pdl
    )
    return out
