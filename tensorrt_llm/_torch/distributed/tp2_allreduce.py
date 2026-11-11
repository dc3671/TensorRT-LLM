# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace and launcher of the TP2 large-message all-reduce + residual RMSNorm (+ FP8) kernel
(``kernels/blackwell/tp2_allreduce_norm.py``), one per TP2 mapping.

``AllReduce`` builds it (one per mapping, ``AllReduce.tp2_fused_norm_kernels``) when the model declares
``allreduce_tp2_fused_norm``, the strategy is AUTO and the TP2 group has no MNNVL all-reduce; ``AllReduce.forward``
routes the residual RMSNorm fusions of BF16 calls of ``MIN_TOKENS`` to ``max_tokens`` tokens here. The workspace (each
rank's two landing buffers of max_tokens / 2 rows and its flags: max_tokens x hidden x 2 bytes, e.g. 128 MiB for
16384 tokens at hidden 4096) is a McastGPUBuffer allocated when the first AllReduce of the mapping is built, before
the KV cache is sized and any CUDA graph is captured.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Fusion op -> (quantized output, normalized output), as MNNVLAllReduce._FUSED_NORM_OUTPUTS.
_OUTPUTS = {
    AllReduceFusionOp.RESIDUAL_RMS_NORM: (False, True),
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8: (True, False),
    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8: (True, True),
}


class Tp2AllReduceNorm:
    """The TP2 all-reduce + residual RMSNorm of one mapping, for calls of ``MIN_TOKENS`` to ``max_tokens`` tokens.
    Construction is collective and raises RuntimeError on every rank where the workspace cannot be set up."""

    # Smaller calls stay on the library all-reduce, whose one-shot kernel is faster there.
    MIN_TOKENS = 2048

    @classmethod
    def is_supported(cls, mapping: Mapping, hidden: int, max_tokens: int) -> bool:
        """CuTe DSL, a TP2 mapping, calls that reach MIN_TOKENS, and a row the kernel's reduction tree spans."""
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not (IS_CUTLASS_DSL_AVAILABLE and mapping.tp_size == 2 and max_tokens >= cls.MIN_TOKENS):
            return False
        from ..kernels.blackwell.tp2_allreduce_norm import Tp2WarpAllReduceNormKernel

        return Tp2WarpAllReduceNormKernel.supports_hidden(hidden)

    def __init__(self, mapping: Mapping, hidden: int, max_tokens: int) -> None:
        from .ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            use_mnnvl_fabric_handle,
        )

        self.mapping = mapping
        self.hidden = hidden
        self.max_tokens = max_tokens
        dev = torch.device("cuda", torch.cuda.current_device())
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.warps = 8
        # The TP pair's communicator (a split of the session communicator; the buffer only borrows it). Not
        # get_or_scale_allreduce_mnnvl_workspace: that would also allocate an MNNVL workspace nothing here uses.
        comm = self._comm = _get_mnnvl_workspace_comm(mapping)
        half = (max_tokens + 1) // 2
        lbytes = half * hidden * 2
        flag_bytes = 2 * self.sms * self.warps * 4
        size = 2 * lbytes + flag_bytes + 4096
        # Collective; a failure (e.g. no multicast support) raises on every rank.
        self.handle = _make_mnnvl_mcast_buffer(
            comm, size, mapping, use_mnnvl_fabric_handle(mapping)
        )
        error = None
        try:
            r, p = mapping.tp_rank, 1 - mapping.tp_rank
            own = self.handle.get_uc_buffer(r, (size // 4,), torch.int32, 0)
            own.zero_()
            peer = self.handle.get_uc_buffer(p, (size // 4,), torch.int32, 0)
            self.ctl = torch.zeros((4,), dtype=torch.int32, device=dev)  # epoch, arrival counter
            o, q = own.data_ptr(), peer.data_ptr()
            self.table = torch.tensor(
                [
                    o,
                    o + lbytes,
                    o + 2 * lbytes,
                    q,
                    q + lbytes,
                    q + 2 * lbytes,
                    self.ctl.data_ptr(),
                    self.ctl.data_ptr() + 4,
                ],
                dtype=torch.int64,
                device=dev,
            )
            self._bufs = (own, peer)
            self._fns = {}
            self._dummy_scale = torch.ones((1,), dtype=torch.float32, device=dev)
            torch.cuda.synchronize()
        except RuntimeError as e:  # e.g. out of memory on this rank only
            error = e
        # Every rank's flags are zero before any rank can raise one; a failure on one rank fails all of them.
        if comm.allreduce(int(error is not None)):
            raise RuntimeError(f"TP2 all-reduce workspace setup failed on a rank: {error}")
        logger.info(
            f"[TP2 all-reduce] workspace {size >> 20} MiB for {max_tokens} tokens, H {hidden}"
        )

    def supports(
        self,
        input: torch.Tensor,
        fusion_op: AllReduceFusionOp,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> bool:
        """BF16 input, residual and norm weight of the kernel's shapes, for one of its fusions."""
        return (
            fusion_op in _OUTPUTS
            and input.dim() == 2
            and input.dtype == torch.bfloat16
            and input.shape[1] == self.hidden
            and self.MIN_TOKENS <= input.shape[0] <= self.max_tokens
            and input.is_contiguous()
            and residual.dtype == torch.bfloat16
            and residual.shape == input.shape
            and norm_weight.dtype == torch.bfloat16
            and norm_weight.numel() == self.hidden
        )

    def __call__(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        fusion_op,
        scale: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Outputs of the library op for ``fusion_op``: [norm, residual], [fp8, residual] or [norm, fp8, residual]."""
        import cuda.bindings.driver as cuda
        import cutlass

        from ..cute_dsl_utils import graph_safe_cute_tensor
        from ..flashinfer_utils import get_env_enable_pdl
        from ..kernels.blackwell.tp2_allreduce_norm import Tp2WarpAllReduceNormKernel

        quant, norm_out = _OUTPUTS[fusion_op]
        key = (norm_out, quant, float(eps))
        t = lambda a: graph_safe_cute_tensor(a, dynamic_leading_dim=a.dim() - 1)  # noqa: E731
        residual = residual.contiguous()
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        q = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if quant
            else None
        )
        sc = scale.reshape(1).float() if quant else self._dummy_scale
        fn = self._fns.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "compile the TP2 all-reduce + norm kernel before CUDA graph capture"
                )
            k = Tp2WarpAllReduceNormKernel(
                hidden=self.hidden,
                norm_eps=float(eps),
                ctas=self.sms,
                warps=self.warps,
                norm_out=norm_out,
                quant=quant,
                use_pdl=get_env_enable_pdl(),
            )
            ex = torch.empty((2, self.hidden), dtype=torch.bfloat16, device=input.device)
            exq = torch.empty((2, self.hidden), dtype=torch.uint8, device=input.device)
            fn = self._fns[key] = cutlass.cute.compile(
                k,
                t(ex),
                t(ex),
                graph_safe_cute_tensor(norm_weight.reshape(-1)),
                graph_safe_cute_tensor(sc),
                t(ex),
                t(ex),
                t(exq),
                graph_safe_cute_tensor(self.table),
                cutlass.Int32(0),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        qv = q.view(torch.uint8) if quant else res_out.view(torch.uint8)[:, : self.hidden]
        fn(
            t(input),
            t(residual),
            graph_safe_cute_tensor(norm_weight.reshape(-1)),
            graph_safe_cute_tensor(sc),
            t(res_out),
            t(norm),
            t(qv),
            graph_safe_cute_tensor(self.table),
            cutlass.Int32(self.mapping.tp_rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if quant:
            return (norm, q, res_out) if norm_out else (q, res_out)
        return norm, res_out
