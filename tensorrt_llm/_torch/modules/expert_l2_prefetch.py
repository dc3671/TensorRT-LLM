# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-step L2 prefetch of the routed experts a MoE layer is likely to use at the next decode step.

At small decode batches DRAM idles through most of a layer's dependency chain, while the experts a layer touches
change slowly between steps; so their FC1 weights are read into L2 while the chain runs, on a side stream forked from
the current one (the main chain's kernels and PDL edges are untouched):

- record: after the decode front has built a layer's routing tables, a 1-CTA kernel stores the ``num_experts`` local
  experts with the most token tiles (ties: lower id) for the next step;
- prefetch: early in the layer, ``num_ctas`` 1-warp CTAs issue ``cp.async.bulk.prefetch.L2`` of those experts'
  ``w3_w1_weight`` / ``w3_w1_weight_scale`` in 64 KiB pieces;
- join: the model's forward joins the side stream once, after its last decoder layer.

The selection only steers prefetches, so outputs are unchanged by construction. Forwards outside
``min_rows``..``max_rows`` rows neither record nor prefetch (decided per captured batch size).
"""

from types import MappingProxyType
from typing import Mapping, Optional

import torch
import triton
import triton.language as tl

PIECE = 65536  # bytes per bulk prefetch
NUM_LOCAL_MAX = 128


@triton.jit
def _record_kernel(
    batch_ptr, nct_ptr, sel_ptr, K: tl.constexpr, NL: tl.constexpr, BLOCK: tl.constexpr
):
    """sel[0:K] = the K local experts with the most token tiles in this call (-1 where fewer are touched)."""
    n = tl.load(nct_ptr)
    offs = tl.arange(0, BLOCK)
    e = tl.load(batch_ptr + offs, mask=offs < n, other=NL)
    e = tl.where((e >= 0) & (e < NL), e, NL)
    cnt = tl.histogram(e, 2 * NL)  # bin NL holds the padding
    ids = tl.arange(0, 2 * NL)
    # key: tiles high, then the lower id first; experts with no tile and the padding bins get key 0
    key = tl.where((ids < NL) & (cnt > 0), cnt * 1024 + (1023 - ids), 0)
    key = tl.sort(key, descending=True)
    sel = tl.where(key > 0, 1023 - (key % 1024), -1)
    tl.store(sel_ptr + ids, sel, mask=ids < K)


@triton.jit
def _prefetch_kernel(
    w_ptr,
    sf_ptr,
    sel_ptr,
    sink_ptr,
    per_w,
    per_sf,
    piece,
    n_items,
    NPW: tl.constexpr,
    NPP: tl.constexpr,
    G: tl.constexpr,
):
    """CTA c issues items c, c + G, ...: item k = (selection slot k // NPP, piece k % NPP) of that expert's [FC1
    weight (NPW pieces), FC1 block scales (NPP - NPW pieces)], one fire-and-forget bulk L2 prefetch of ``piece``
    bytes from thread 0 (one thread per CTA in a loop issues faster than several lanes of one instruction)."""
    c = tl.program_id(0)
    lane = tl.arange(0, 32)
    acc = tl.zeros([32], tl.int32)
    for k in range(c, n_items, G):
        i = k // NPP
        p = k % NPP
        e = tl.load(sel_ptr + i)
        e64 = tl.maximum(e, 0).to(tl.int64)
        pw = tl.minimum(p, NPW - 1).to(tl.int64)
        ps = tl.maximum(p - NPW, 0).to(tl.int64)
        aw = w_ptr + (e64 * per_w + pw * piece) + lane * 0
        asf = sf_ptr + (e64 * per_sf + ps * piece) + lane * 0
        go = (lane == 0) & (e >= 0)
        sz = lane * 0 + piece
        rw = tl.inline_asm_elementwise(
            "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q cp.async.bulk.prefetch.L2.global [$1], $3; mov.u32 $0, 0; }",
            "=r,l,r,r",
            [aw, (go & (p < NPW)).to(tl.int32), sz],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        rs = tl.inline_asm_elementwise(
            "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q cp.async.bulk.prefetch.L2.global [$1], $3; mov.u32 $0, 0; }",
            "=r,l,r,r",
            [asf, (go & (p >= NPW)).to(tl.int32), sz],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        acc += rw + rs
    tl.store(sink_ptr + c * 32 + lane, acc)


class _DeviceState:
    def __init__(self, device: torch.device, num_layers: int, num_experts: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the expert prefetch state must exist before CUDA-graph capture (run one eager "
                "forward first)"
            )
        self.sel = torch.full((num_layers, num_experts), -1, dtype=torch.int32, device=device)
        self.sink = torch.zeros(1 << 12, dtype=torch.int32, device=device)


class ExpertL2Prefetch:
    """Cross-step expert prefetch of one model (module docstring), for forwards of ``min_rows``..``max_rows`` rows
    of a router picking ``top_k`` experts per row, on the model's side ``stream``.

    Layers ``register`` the MoE backend that holds their routed FC1 weights (``w3_w1_weight``,
    ``w3_w1_weight_scale``: [local experts, ...], contiguous)."""

    # Each layer prefetches the experts of this many rows' picks, from one CTA per two experts: a slower, gentler
    # burst than the whole grid, so the prefetch competes less with the latency-bound chain for DRAM.
    PICK_ROWS = 4

    _state: Optional[_DeviceState] = None
    _forked = False  # this forward forked the side stream; join() clears it
    num_layers = 0
    min_rows = 0
    max_rows = 0
    num_experts = 0
    num_ctas = 0
    stream: Optional[torch.cuda.Stream] = None
    _backends: Mapping[int, torch.nn.Module] = MappingProxyType({})

    def __init__(
        self, num_layers: int, min_rows: int, max_rows: int, top_k: int, stream: torch.cuda.Stream
    ):
        self.num_layers = num_layers
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.num_experts = min(self.PICK_ROWS * top_k, NUM_LOCAL_MAX)
        self.num_ctas = -(-self.num_experts // 2)
        self.stream = stream
        self._backends = {}

    def register(self, layer_idx: int, backend: torch.nn.Module) -> None:
        if 0 <= layer_idx < self.num_layers:
            self._backends[layer_idx] = backend

    def takes_rows(self, rows: int) -> bool:
        # Side-stream forks and Triton launches: not under torch.compile.
        return self.min_rows <= rows <= self.max_rows and not torch.compiler.is_compiling()

    def _device_state(self, device: torch.device) -> _DeviceState:
        if self._state is None:
            self._state = _DeviceState(device, self.num_layers, self.num_experts)
        return self._state

    def _fork(self) -> None:
        self.stream.wait_stream(torch.cuda.current_stream())
        self._forked = True

    def prefetch(self, layer_idx: int, rows: int) -> None:
        """Issue the FC1 prefetch of the experts recorded for ``layer_idx`` at the previous step (on the side
        stream), for a forward of ``rows`` rows."""
        backend = self._backends.get(layer_idx)
        # NVFP4 experts only: the pieces are FC1 weights and their block scales.
        if backend is None or not self.takes_rows(rows) or not backend.has_nvfp4:
            return
        w, sf = backend.w3_w1_weight, backend.w3_w1_weight_scale
        if (
            w.dim() != 3
            or w.shape[0] > NUM_LOCAL_MAX
            or not w.is_contiguous()
            or not sf.is_contiguous()
        ):
            return
        wb, sb = (
            w.view(torch.uint8).view(w.shape[0], -1),
            sf.view(torch.uint8).view(sf.shape[0], -1),
        )
        per_w, per_sf = wb.shape[1], sb.shape[1]
        if per_w % PIECE or per_sf % PIECE:
            return
        state = self._device_state(w.device)
        npw, nps = per_w // PIECE, per_sf // PIECE
        self._fork()
        with torch.cuda.stream(self.stream):
            _prefetch_kernel[(self.num_ctas,)](
                wb,
                sb,
                state.sel[layer_idx],
                state.sink,
                per_w,
                per_sf,
                PIECE,
                self.num_experts * (npw + nps),
                NPW=npw,
                NPP=npw + nps,
                G=self.num_ctas,
                num_warps=1,
            )

    def record(
        self,
        layer_idx: int,
        cta_batch: torch.Tensor,
        num_ctas: torch.Tensor,
        num_local: int,
        keep: torch.Tensor,
        rows: int,
    ) -> None:
        """Store the top experts by token tiles of this call's routing tables (``cta_batch``: local expert per token
        tile, ``num_ctas``: tile count) for ``layer_idx``'s next prefetch. ``keep``: the tensor that owns the tables,
        held for the side stream's read."""
        if (
            layer_idx not in self._backends
            or not self.takes_rows(rows)
            or num_local > NUM_LOCAL_MAX
        ):
            return
        state = self._device_state(cta_batch.device)
        block = max(256, triton.next_power_of_2(cta_batch.numel()))
        self._fork()
        with torch.cuda.stream(self.stream):
            _record_kernel[(1,)](
                cta_batch,
                num_ctas,
                state.sel[layer_idx],
                K=self.num_experts,
                NL=NUM_LOCAL_MAX,
                BLOCK=block,
                num_warps=4,
            )
        keep.record_stream(self.stream)

    def join(self) -> None:
        """Join the side stream into the current stream (once per forward, after the last decoder layer). Only a
        stream this forward forked: a CUDA-graph capture may not wait on a stream it never forked."""
        if self._forked:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._forked = False
