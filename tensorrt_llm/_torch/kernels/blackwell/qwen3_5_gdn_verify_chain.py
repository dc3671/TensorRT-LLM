# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 GDN target-verify chain in one CuTe DSL launch (SM100 / SM103).

For a decode-only MTP target-verify step of one Qwen3.5 GDN layer rank (8 value / 2 key heads of 128, hidden 4096,
1..16 requests x 7 rows) the kernel runs in_proj_qkvz (E4M3 x E4M3, static per-tensor scales), in_proj_ba (BF16), the
causal conv1d update in its spec-decoding form (per-row windows and the new conv state), the cached-replay verify with
the in-layer checkpoint fold, the per-head gated RMSNorm and the E4M3 out_proj input.

GEMM: 20 clusters x 6 CTAs own in_proj_qkvz, one 128-channel tile (one head's q, k, v or z) per cluster, split-K 6 over
32 k-blocks of 128, swap-AB tcgen05 M128 x N(rows). A CTA's whole weight slice is TMA-staged into SMEM before
griddepcontrol.wait; only the activations load after it. Split-K partials leave TMEM as [row][channel] FP32 blocks and
reach the owner rank of a request (request % 6) as one cp.async.bulk SMEM -> DSMEM copy per owner. The owner reduces,
applies the scales, the causal conv1d and SiLU (q / k / v tiles), and publishes 7 BF16 rows per channel with a 16-bit
tag in one 16-byte store; its conv windows and conv state leave from an SMEM image.

Verify: one CTA per (request, value head) also runs the verify, off the q / k clusters and below CTA 108 where the
grid allows (12 extra CTAs past 15 requests). Before the wait it loads the slot tables, its state slice, the replay
history and the in_proj_ba weight rows, and folds the history into the checkpoint hc = e^g0 S + (u * cd)^T k in
registers; hc is also the verify's effective state, so after the wait no history term remains. After the wait its
warps 4..7 compute one 512-wide K slice of all 16 in_proj_ba channels and publish it; the CTA of each head sums the
8 slices of its b / a rows and derives the gates. It then polls the k / q and v / z words, runs the replay on BF16
hi / lo mma.sync, and writes the folded checkpoint (when the history overflows), the new history rows and the gated
RMSNorm's E4M3 output.

Cross-CTA hand-offs (published rows, in_proj_ba slices) go through a global buffer and carry a tag derived from a
device-side launch index: every warp takes a ticket from a per-(request count, CTA) counter right after the PDL wait,
so the tags advance on every launch, including CUDA-graph replays, without host involvement. Launches of this kernel
must not overlap (one stream). The kernel is a PDL dependent of the producer of its activations and triggers its own
dependents once its MMAs complete.

Precision: not bitwise against the stock chain (low-M GEMM + conv1d update + trtllm::gdn_verify). The GEMMs accumulate
in FP32 in a different order, the replay runs on BF16 hi / lo mma.sync with FP32 accumulation, and the gates use
MUFU approximations.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda_driver
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T as dsl_T
from cutlass.cutlass_dsl import dsl_user_op

# Shape specialization: one rank of a Qwen3.5 GDN layer under 8-way tensor parallelism.
TOKENS = 7  # rows per request (the MTP verify window)
V_HEADS = 8
K_HEADS = 2
HEAD_DIM = 128
HIDDEN = 4096
QKVZ_DIM = 2560  # in_proj_qkvz channels: q | k | v | z
BA_DIM = 16  # in_proj_ba channels: b | a
CONV_DIM = 1536  # q | k | v channels of the causal conv1d
CONV_WIDTH = 4
OUT_DIM = 1024
HISTORY = 16  # replay history rows per cache buffer
MAX_REQUESTS = 16
MAX_SLOTS = 64  # cache slots the in-kernel slot-table lookup covers
QKVZ_TILES = QKVZ_DIM // 128
PARTIAL_BYTES = TOKENS * 128 * 4  # one request's split-K partial block: [row][channel] fp32
CLUSTER = 6
GEMM_CTAS = QKVZ_TILES * CLUSTER
THREADS = 256
SMEM_BUDGET = 188 * 1024
TMEM_COLS = 256
TICKET_CTAS = 148  # ticket counters per request count (one 128-byte line per CTA)
# Bits of the launch index a hand-off tag takes: 17 * index + N stays within Int32 while the counter wraps.
TAG_LAUNCH_BITS = 26

# verify scratch (bytes, aliased over the weight ring once the MMAs have drained)
O_XA = 0  # bf16 [16][136]: rows 0..6 raw k, 8..14 raw q, 7 / 15 zero
VSS = 136  # row stride (floats) of the v / z rows: 8 mod 32, so the 64-bit reads of rows gid = 0..6 are conflict-free
O_VS = 4352  # f32 [7][VSS]
O_ZS = 8192  # f32 [7][VSS]
O_MS = 12288  # f32 misc
# bf16 conv_window / conv_state images, 6 KB per owned request (past every split-K staging block, so an owner may
# write them while its later-slot bulk pushes still read the staging)
O_CV = 65536
M_KG = 0  # [2][16][KGS]: X kh^T (cols j < 16), then X X^T (cols 0..7 k_j, 8..15 q_j)
KGS = 20  # padded row stride (floats): the 8 gid rows of a warp read their rows bank-conflict free (8- / 16-byte reads)
M_RED = 2 * 16 * KGS  # [8][8]
RS = 272  # row stride (bytes) of the [16][136] bf16 arrays
RT = 48  # row stride (bytes) of the [*][24] bf16 arrays
# pre-wait history region (dedicated): old_k rows, old_k^T, old_u^T, old_g
H_KHS = 0  # bf16 [16][136]
H_KHT = 4352  # bf16 [128][24]
H_UHT = 10496  # bf16 [128][24]
H_GH = 16640  # f32 [16]
HIST_BYTES = 16704
# early scratch (dedicated, f32 words): BARED [8][16], BETA [8], CS [8], EG [8], DN [8][8], 128 unused words
E_BETA = 128
E_CS = 136
E_EG = 144
E_DN = 152
E_BAW = (
    344  # f32 [4 warps][16 channels][8 rows]: this CTA's in_proj_ba K-slice partials, per warp 4..7
)
E_BAC = 856  # f32 [8 src heads][16]: the consumed slice sums (b rows 0..6, a rows 7..13) of this CTA's head
E_TOTAL = 984
BAW = 5  # 16-byte publish words per (request, dest head, src head): (tag, 3 values) x 5 = 14 values


def handoff_tag(ticket, n):
    """The cross-CTA hand-off tag of a launch with ``n`` requests from a warp's ticket (8 per launch): in [1, 65535],
    0 being the zero-filled buffer's; consecutive launches of one ``n`` differ, also across the index wrap."""
    return (((ticket >> 3) & ((1 << TAG_LAUNCH_BITS) - 1)) * 17 + n) % 65535 + 1


def _asm(ret, args, s, cons, side, loc, ip):
    return llvm.inline_asm(
        ret,
        args,
        s,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _i32(x, loc, ip):
    return cutlass.Int32(x).ir_value(loc=loc, ip=ip)


def _f32(x, loc, ip):
    return cutlass.Float32(x).ir_value(loc=loc, ip=ip)


def _i64(x, loc, ip):
    return cutlass.Int64(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def bf16r(x, *, loc=None, ip=None):
    """fp32 -> bf16 (RN) -> fp32."""
    return cutlass.Float32(
        _asm(
            dsl_T.f32(),
            [_f32(x, loc, ip)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.f32.bf16 $0, h;\n\t}",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def pack_bf16(lo, hi, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            dsl_T.i32(),
            [_f32(hi, loc, ip), _f32(lo, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def i2f(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_i32(w, loc, ip)], "mov.b32 $0, $1;", "=f,r", False, loc, ip)
    )


@dsl_user_op
def f2i(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(dsl_T.i32(), [_f32(x, loc, ip)], "mov.b32 $0, $1;", "=r,f", False, loc, ip)
    )


@dsl_user_op
def bf_lo(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            dsl_T.f32(),
            [_i32(w, loc, ip)],
            "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def bf_hi(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            dsl_T.f32(),
            [_i32(w, loc, ip)],
            "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xFFFF0000;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def tag_of(w, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(dsl_T.i32(), [_i32(w, loc, ip)], "and.b32 $0, $1, 0xFFFF;", "=r,r", False, loc, ip)
    )


@dsl_user_op
def or_i32(a, b, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            dsl_T.i32(),
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            "or.b32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def e4m3x2(lo, hi, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            dsl_T.i32(),
            [_f32(hi, loc, ip), _f32(lo, loc, ip)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, $1, $2;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate (m16n8k16)."""
    f32 = dsl_T.f32()
    ops = [_i32(x, loc, ip) for x in (a0, a1, a2, a3, b0, b1)] + [
        _f32(x, loc, ip) for x in (c0, c1, c2, c3)
    ]
    res = _asm(
        llvm.StructType.get_literal([f32, f32, f32, f32]),
        ops,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return [cutlass.Float32(llvm.extractvalue(f32, res, [i], loc=loc, ip=ip)) for i in range(4)]


# ---------------- global memory (volatile asm so prefetches stay where they are written) ----------------
@dsl_user_op
def ldg_u32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(dsl_T.i32(), [_i64(addr, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", True, loc, ip)
    )


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_i64(addr, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", True, loc, ip)
    )


@dsl_user_op
def ldg_u16(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            dsl_T.i32(),
            [_i64(addr, loc, ip)],
            "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


def _v4(res, loc, ip):
    i32 = dsl_T.i32()
    return [cutlass.Int32(llvm.extractvalue(i32, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ldg_v4(addr, *, loc=None, ip=None):
    i32 = dsl_T.i32()
    res = _asm(
        llvm.StructType.get_literal([i32, i32, i32, i32]),
        [_i64(addr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        True,
        loc,
        ip,
    )
    return _v4(res, loc, ip)


@dsl_user_op
def ldg_v4_rlx(addr, *, loc=None, ip=None):
    i32 = dsl_T.i32()
    res = _asm(
        llvm.StructType.get_literal([i32, i32, i32, i32]),
        [_i64(addr, loc, ip)],
        "ld.relaxed.gpu.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        True,
        loc,
        ip,
    )
    return _v4(res, loc, ip)


@dsl_user_op
def stg_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def stg_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _f32(v, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def stg_u16(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def stg_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip)] + [_i32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def stg_v4_rlx(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip)] + [_i32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "st.relaxed.gpu.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def stg_2u32(addr, lo, hi, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(lo, loc, ip), _i32(hi, loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        True,
        loc,
        ip,
    )


# ---------------- shared memory ----------------
@dsl_user_op
def lds_u32(a, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(dsl_T.i32(), [_i32(a, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", True, loc, ip)
    )


@dsl_user_op
def lds_f32(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_i32(a, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", True, loc, ip)
    )


@dsl_user_op
def lds_v4(a, *, loc=None, ip=None):
    i32 = dsl_T.i32()
    res = _asm(
        llvm.StructType.get_literal([i32, i32, i32, i32]),
        [_i32(a, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return _v4(res, loc, ip)


@dsl_user_op
def lds_v2(a, *, loc=None, ip=None):
    i32 = dsl_T.i32()
    res = _asm(
        llvm.StructType.get_literal([i32, i32]),
        [_i32(a, loc, ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        True,
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(i32, res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def sts_u32(a, v, *, loc=None, ip=None):
    _asm(
        None, [_i32(a, loc, ip), _i32(v, loc, ip)], "st.shared.b32 [$0], $1;", "r,r", True, loc, ip
    )


@dsl_user_op
def sts_f32(a, v, *, loc=None, ip=None):
    _asm(
        None, [_i32(a, loc, ip), _f32(v, loc, ip)], "st.shared.f32 [$0], $1;", "r,f", True, loc, ip
    )


@dsl_user_op
def sts_u16(a, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(a, loc, ip), _i32(v, loc, ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.shared.b16 [$0], h;\n\t}",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def sts_v4(a, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(a, loc, ip)] + [_i32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mapa_u32(addr, rank, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            dsl_T.i32(),
            [_i32(addr, loc, ip), _i32(rank, loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def remote_arrive(addr, *, loc=None, ip=None):
    """Mbarrier arrive (relaxed, cluster scope) on another CTA's barrier (shared::cluster address from mapa)."""
    _asm(
        None,
        [_i32(addr, loc, ip)],
        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [$0];",
        "r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def bulk_s2s(dst, src, nbytes, mbar, *, loc=None, ip=None):
    """cp.async.bulk shared::cta -> shared::cluster with complete_tx on the destination CTA's mbarrier."""
    _asm(
        None,
        [_i32(dst, loc, ip), _i32(src, loc, ip), _i32(nbytes, loc, ip), _i32(mbar, loc, ip)],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def bulk_commit_wait_read(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;",
        "",
        True,
        loc,
        ip,
    )


@dsl_user_op
def fence_proxy_async_smem(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", True, loc, ip)


@dsl_user_op
def prefetch_l2a(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(nbytes, loc, ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def bulk_wait_all(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.wait_group.read 0;", "", True, loc, ip)


@dsl_user_op
def tmem_st32(taddr, vals, *, loc=None, ip=None):
    """tcgen05.st 32x32b.x32: this warp's 32 TMEM lanes x 32 columns from 32 registers per thread (+ wait::st)."""
    regs = ", ".join("$%d" % (i + 1) for i in range(32))
    _asm(
        None,
        [_i32(taddr, loc, ip)] + [_i32(vals[i], loc, ip) for i in range(32)],
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {"
        + regs
        + "};\n\ttcgen05.wait::st.sync.aligned;",
        "r" + ",r" * 32,
        True,
        loc,
        ip,
    )


@dsl_user_op
def tmem_ld32(taddr, *, loc=None, ip=None):
    """tcgen05.ld 32x32b.x32 (+ wait::ld): 32 registers per thread back from this warp's TMEM lanes."""
    i32 = dsl_T.i32()
    regs = ", ".join("$%d" % i for i in range(32))
    res = _asm(
        llvm.StructType.get_literal([i32] * 32),
        [_i32(taddr, loc, ip)],
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {"
        + regs
        + "}, [$32];\n\ttcgen05.wait::ld.sync.aligned;",
        ",".join(["=r"] * 32) + ",r",
        True,
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(i32, res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def tmem_ld_x16s(taddrs, *, loc=None, ip=None):
    """len(taddrs) tcgen05.ld 32x32b.x16 (16 columns each) and one wait::ld.

    The whole accumulator row of this thread in one TMEM round trip instead of one per 16-column chunk.
    """
    i32 = dsl_T.i32()
    n = len(taddrs)
    nout = 16 * n
    body = "tcgen05.fence::after_thread_sync;\n\t"
    for c in range(n):
        regs = ", ".join("$%d" % (16 * c + i) for i in range(16))
        body += "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%s}, [$%d];\n\t" % (regs, nout + c)
    body += "tcgen05.wait::ld.sync.aligned;"
    res = _asm(
        llvm.StructType.get_literal([i32] * nout),
        [_i32(a, loc, ip) for a in taddrs],
        body,
        ",".join(["=r"] * nout) + "," + ",".join(["r"] * n),
        True,
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(i32, res, [i], loc=loc, ip=ip)) for i in range(nout)]


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            dsl_T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def rcp_approx(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_f32(a, loc, ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@dsl_user_op
def atom_add_relaxed(addr, v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            dsl_T.i32(),
            [_i64(addr, loc, ip), _i32(v, loc, ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def ex2_approx(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_f32(a, loc, ip)], "ex2.approx.ftz.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@dsl_user_op
def lg2_approx(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_f32(a, loc, ip)], "lg2.approx.ftz.f32 $0, $1;", "=f,f", False, loc, ip)
    )


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@cute.jit
def exp_fast(x: cutlass.Float32) -> cutlass.Float32:
    return ex2_approx(x * cutlass.Float32(LOG2E))


@dsl_user_op
def tanh_approx(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(dsl_T.f32(), [_f32(a, loc, ip)], "tanh.approx.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@cute.jit
def silu_tanh(x: cutlass.Float32) -> cutlass.Float32:
    """SiLU as x * (0.5 + 0.5 tanh(x / 2)): one MUFU op instead of ex2 + rcp."""
    h = cutlass.Float32(0.5) * x
    return h + h * tanh_approx(h)


@cute.jit
def sigmoid_fast(x: cutlass.Float32) -> cutlass.Float32:
    return rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-x))


def _rslot(rbase, rxb, slot, split):
    """Base of receive slot `slot` (compile-time): [slot][src rank] blocks.

    Split mode keeps slot 0 in the dedicated buffer and slots >= 1 over the x ring.
    """
    if split:
        return rbase if slot == 0 else rxb + (slot - 1) * CLUSTER * PARTIAL_BYTES
    return rbase + slot * CLUSTER * PARTIAL_BYTES


def _compile_options(N):
    # Register cap: THREADS x registers per CTA must fit on an SM beside the CTAs of the producer kernel it overlaps.
    return f"--ptxas-options -maxrregcount={136 if N <= 11 else 112}"


class _GdnVerifyChain:
    def __init__(self, N: int):
        self.N = N
        self.R = N * TOKENS
        self.n_mma = max(16, ((self.R + 15) // 16) * 16)
        self.kb = 128
        self.nkb = HIDDEN // self.kb
        self.nstage = 6
        self.nro = (N + CLUSTER - 1) // CLUSTER
        self.grid = GEMM_CTAS if V_HEADS * N <= GEMM_CTAS else GEMM_CTAS + 2 * CLUSTER
        self.nver = V_HEADS * N
        # where the grid allows, verify CTAs sit on the V / Z clusters (the Q / K clusters 0..3 stay GEMM-only, so
        # q / k publish first) and below CTA 108 (clusters that fit beside the producer's CTAs start first)
        self.vbase = (
            108 - V_HEADS * N
            if V_HEADS * N <= 108 - 4 * CLUSTER
            else (108 - V_HEADS * N if V_HEADS * N <= 108 else 0)
        )
        # N 7 / 8: ranks 0 .. N - 7 own two requests (b and b + 6). Verify units of requests 0..5 (published first) sit
        # on the single-owner ranks of clusters 4.. only, units of the late requests 6.. on the double-owner ranks, so
        # no early request's verify waits behind its host's second reduce
        self.vmap = 7 <= N <= 8
        self.vm_c0 = 4
        self.vm_d = max(N - CLUSTER, 0)
        self.vm_s = CLUSTER - self.vm_d

        self.recv_bytes = CLUSTER * self.nro * PARTIAL_BYTES
        a_b = self.nstage * 128 * self.kb
        b_b = self.nstage * self.n_mma * self.kb
        # Split-K receive layout [slot][src rank][row][channel] within SMEM_BUDGET (above it the CTA cannot share an
        # SM with the producer's CTAs): dedicated (N <= 6), split: slot 0 dedicated, slots >= 1 over the x ring once
        # the destination's MMAs are done (N 7 .. 11), alias: every slot over the x ring (N >= 12).
        rded = CLUSTER * PARTIAL_BYTES

        def fits(extra, hist):
            return 4096 + a_b + b_b + extra + hist <= SMEM_BUDGET

        self.rsplit = False
        if fits(self.recv_bytes, HIST_BYTES):
            self.alias_recv = False
            self.hx = False
        elif self.nro == 2 and fits(rded, 128):
            self.alias_recv = False
            self.rsplit = True
            self.hx = not fits(rded, HIST_BYTES)
        else:
            self.alias_recv = True
            self.hx = not fits(0, HIST_BYTES)
        # hx: the history rows are staged in the still-empty x ring before the wait (only the decays G_h keep a
        # dedicated slot) and the two fragments the verify needs after the wait (k-history rows for pass 1, u-history
        # for pass 2) are parked in TMEM
        self.tmem_cols = 512 if self.hx else TMEM_COLS
        self.smem_bytes = (
            4096
            + a_b
            + b_b
            + (0 if self.alias_recv else (rded if self.rsplit else self.recv_bytes))
            + (128 if self.hx else HIST_BYTES)
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        x_bf16: cute.Tensor,
        w_ba: cute.Tensor,
        conv_w: cute.Tensor,
        conv_state: cute.Tensor,
        conv_window: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        norm_w: cute.Tensor,
        fp8_scale: cute.Tensor,
        x_scale: cute.Tensor,
        w_scale: cute.Tensor,
        state: cute.Tensor,
        slots: cute.Tensor,
        old_u: cute.Tensor,
        old_k: cute.Tensor,
        old_g: cute.Tensor,
        cache_buf_idx: cute.Tensor,
        prev_acc: cute.Tensor,
        y: cute.Tensor,
        tickets: cute.Tensor,
        qkvz: cute.Tensor,
        eps: cutlass.Float32,
        commit: cutlass.Int32,
        st_stride: cutlass.Int64,
        cs_stride: cutlass.Int64,
        stream: cuda_driver.CUstream,
    ):
        a_dtype = mA.element_type
        self.a_dtype = a_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            a_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (128, self.n_mma),
        )
        mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (128, self.n_mma, mma_inst_k * (self.kb // mma_inst_k))
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.nstage
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, a_dtype, self.nstage
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        self.ab_bytes = cute.size_in_bytes(a_dtype, a_smem_layout) + cute.size_in_bytes(
            a_dtype, b_smem_layout
        )
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(
            tiled_mma.make_fragment_C(tiled_mma.partition_shape_C(self.mma_tiler[:2]))
        )

        @cute.struct
        class SharedStorage:
            full_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            recv_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            xfree_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding_buf: cutlass.Int32
            early: cute.struct.MemRange[cutlass.Float32, E_TOTAL]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            a_smem_layout_staged,
            b_smem_layout_staged,
            x_bf16,
            w_ba,
            conv_w,
            conv_state,
            conv_window,
            A_log,
            dt_bias,
            norm_w,
            fp8_scale,
            x_scale,
            w_scale,
            state,
            slots,
            old_u,
            old_k,
            old_g,
            cache_buf_idx,
            prev_acc,
            y,
            tickets,
            qkvz,
            eps,
            commit,
            st_stride,
            cs_stride,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        x_bf16: cute.Tensor,
        w_ba: cute.Tensor,
        conv_w: cute.Tensor,
        conv_state: cute.Tensor,
        conv_window: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        norm_w: cute.Tensor,
        fp8_scale: cute.Tensor,
        x_scale: cute.Tensor,
        w_scale: cute.Tensor,
        state: cute.Tensor,
        slots: cute.Tensor,
        old_u: cute.Tensor,
        old_k: cute.Tensor,
        old_g: cute.Tensor,
        cache_buf_idx: cute.Tensor,
        prev_acc: cute.Tensor,
        y: cute.Tensor,
        tickets: cute.Tensor,
        qkvz: cute.Tensor,
        eps: cutlass.Float32,
        commit: cutlass.Int32,
        st_stride: cutlass.Int64,
        cs_stride: cutlass.Int64,
    ):
        N = self.N
        NMMA = self.n_mma
        NRO = self.nro
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        gid = lane // 4
        tig = lane % 4

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        a_bytes = cute.cosize(a_smem_layout_staged.outer)
        b_bytes = cute.cosize(b_smem_layout_staged.outer)
        pool = smem.allocate(a_bytes + b_bytes, 1024)
        recv = pool + a_bytes
        recvx = pool + a_bytes
        if cutlass.const_expr(not self.alias_recv):
            recv = smem.allocate(CLUSTER * PARTIAL_BYTES if self.rsplit else self.recv_bytes, 128)
        hist = smem.allocate(128 if self.hx else HIST_BYTES, 128)
        HB = hist.toint()
        HBG = HB  # G_h decays (always dedicated)
        HBX = HB  # history rows (pre-wait), x ring base when self.hx
        if cutlass.const_expr(self.hx):
            HBG = HB - H_GH
            HBX = (pool + a_bytes).toint()
        sA = cute.make_tensor(
            cute.recast_ptr(pool, a_smem_layout_staged.inner, dtype=self.a_dtype),
            a_smem_layout_staged.outer,
        )
        sB = cute.make_tensor(
            cute.recast_ptr(pool + a_bytes, b_smem_layout_staged.inner, dtype=self.a_dtype),
            b_smem_layout_staged.outer,
        )
        full_mbar = storage.full_mbar.data_ptr()
        acc_mbar = storage.acc_mbar.data_ptr()
        recv_mbar = storage.recv_mbar.data_ptr()
        # xfree[o]: rank o's MMA is done, its x ring may receive split-K pushes (point-to-point, no cluster barrier)
        xfree_mbar = storage.xfree_mbar.data_ptr()
        SB = pool.toint()  # verify scratch base (aliases the weight ring)
        SE = storage.early.data_ptr().toint()
        SMS = SB + O_MS

        # global base addresses
        g_xbf = x_bf16.iterator.toint()
        g_wba = w_ba.iterator.toint()
        g_convw = conv_w.iterator.toint()
        g_cs = conv_state.iterator.toint()
        g_cw = conv_window.iterator.toint()
        g_state = state.iterator.toint()
        g_ou = old_u.iterator.toint()
        g_ok = old_k.iterator.toint()
        g_og = old_g.iterator.toint()
        g_y = y.iterator.toint()
        g_tickets = tickets.iterator.toint()
        g_q = qkvz.iterator.toint()
        g_ba = qkvz.iterator.toint() + cutlass.Int64(MAX_REQUESTS * QKVZ_TILES * 128 * 16)
        # Launch index for the cross-CTA tags: each warp takes a ticket from a per-(N, CTA) counter after the PDL
        # wait (8 per launch). Every CTA of an N's grid runs in every launch and launches never overlap, so all
        # CTAs derive the same index.
        ctr_old = cutlass.Int32(0)
        # this CTA's ticket counter: its own 128-byte line, pulled into L2 now (a read) so the post-wait atomic is an
        # uncontended L2 hit
        g_tk = g_tickets + cutlass.Int64(((N * TICKET_CTAS + cta) * 32) * 4)
        if tidx == 0:
            prefetch_l2a(g_tk, cutlass.Int32(16))
        # CTAs past the first 108 may start only when the producer ends: their counter lines are pulled in by CTA - 108
        if cutlass.const_expr(self.grid > 108):
            if tidx == 32:
                if cta + 108 < self.grid:
                    prefetch_l2a(g_tk + cutlass.Int64(108 * 32 * 4), cutlass.Int32(16))
        is_gemm = cta < GEMM_CTAS
        vflag = cutlass.Int32(0)
        vrel = cutlass.Int32(0)
        if cutlass.const_expr(self.vmap):
            # (request, head) unit of this CTA: requests 0..5 on the single-owner ranks, 6.. on the double-owner ones
            v_cl = cta // CLUSTER
            v_rk = cta % CLUSTER
            if v_cl >= self.vm_c0:
                if v_cl < 18:
                    if v_rk >= self.vm_d:
                        v_u = (v_cl - self.vm_c0) * self.vm_s + (v_rk - self.vm_d)
                        if v_u < CLUSTER * V_HEADS:
                            vrel = v_u
                            vflag = cutlass.Int32(1)
                    else:
                        v_u = (v_cl - self.vm_c0) * self.vm_d + v_rk
                        if v_u < V_HEADS * self.vm_d:
                            vrel = CLUSTER * V_HEADS + v_u
                            vflag = cutlass.Int32(1)
        else:
            vrel = cta - self.vbase
            if vrel >= 0:
                if vrel < self.nver:
                    vflag = cutlass.Int32(1)
            vrel = cutlass.max(vrel, 0)
        is_ver = vflag != 0
        # slot metadata first (before any setup or weight traffic): the slot of this verify CTA's request and the
        # whole prev_acc / cache_buf_idx tables (64 slots, two words per lane), resolved by shuffles once the slot
        # is known, so the state / history loads need one dependent HBM round trip instead of two
        vs_e = cutlass.Int32(-1)
        pa0 = cutlass.Int32(0)
        pa1 = cutlass.Int32(0)
        cb0 = cutlass.Int32(0)
        cb1 = cutlass.Int32(0)
        # per-tensor scales: raw loads now, combined only after the wait (no stall on them before it)
        xs_raw = ldg_f32(x_scale.iterator.toint())
        ws_raw = ldg_f32(w_scale.iterator.toint())
        f8s = ldg_f32(fp8_scale.iterator.toint())
        sbo_e = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
        for jj in cutlass.range_constexpr(2):
            sbo_e[jj] = cutlass.Int32(-1)
            if cta < GEMM_CTAS:
                bo_e = cta % CLUSTER + CLUSTER * ((tidx // 128) + 2 * jj)
                if bo_e < N:
                    sbo_e[jj] = ldg_u32(slots.iterator.toint() + cutlass.Int64(bo_e * 4))
        if is_ver:
            vs_e = ldg_u32(slots.iterator.toint() + cutlass.Int64((vrel // V_HEADS) * 4))
            pa0 = ldg_u32(prev_acc.iterator.toint() + cutlass.Int64(lane * 4))
            pa1 = ldg_u32(prev_acc.iterator.toint() + cutlass.Int64((32 + lane) * 4))
            cb0 = ldg_u32(cache_buf_idx.iterator.toint() + cutlass.Int64(lane * 4))
            cb1 = ldg_u32(cache_buf_idx.iterator.toint() + cutlass.Int64((32 + lane) * 4))
        t_tile = cta // CLUSTER
        s_split = cta % CLUSTER
        n_own = (N - s_split + CLUSTER - 1) // CLUSTER
        kb_start = s_split * (self.nkb // CLUSTER) + cutlass.max(
            s_split - (CLUSTER - self.nkb % CLUSTER), 0
        )
        kb_cnt = (self.nkb // CLUSTER) + cutlass.min(
            cutlass.max(s_split - (CLUSTER - self.nkb % CLUSTER - 1), 0), 1
        )

        # ---------------- setup: barriers + weight TMA (before griddepcontrol.wait) ----------------
        gA = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.nstage):
                    cute.arch.mbarrier_init(full_mbar + i, 1)
                cute.arch.mbarrier_init(acc_mbar, 1)
                # one receive barrier per slot (request s_split + 6 slot): a slot's reduce starts as soon as its own
                # pushes land (slot 0 never waits for the later slots' second cluster-barrier phase)
                for sl_ in cutlass.range_constexpr(NRO):
                    cute.arch.mbarrier_init(recv_mbar + sl_, 129)
                    if sl_ < n_own:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            recv_mbar + sl_, (CLUSTER - 1) * PARTIAL_BYTES
                        )
                    else:
                        cute.arch.mbarrier_arrive(recv_mbar + sl_)
                for p_ in cutlass.range_constexpr(CLUSTER):
                    cute.arch.mbarrier_init(xfree_mbar + p_, 1)
            cute.arch.mbarrier_init_fence()
            if is_gemm:
                for kk in cutlass.range_constexpr(self.nstage):
                    if kk < kb_cnt:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(full_mbar + kk, self.ab_bytes)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, t_tile, kb_start + kk, 0)],
                            tAsA[(None, kk)],
                            tma_bar_ptr=full_mbar + kk,
                        )
        cute.arch.mbarrier_init_fence()
        # the one cluster barrier of this kernel: every rank's barriers are initialised before any DSMEM access (arrive
        # here, wait right after griddepcontrol.wait, where it has long completed)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.barrier()

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier
        )
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        # verify CTAs park their long-lived pre-wait registers (state fragments, folded checkpoint) in TMEM columns
        # [128, 256): warp w owns lanes 32 (w % 4) .. +31, columns + 32 (w // 4); keeps registers within the budget
        # beside the producer (threads x regs per CTA)
        tm_base = lds_u32(storage.tmem_holding_buf.ptr.toint())
        tm_sb = tm_base + ((32 * (warp_idx % 4)) << 16) + 128 + 32 * (warp_idx // 4)
        tm_cm = tm_sb + 64
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # ---------------- pre-wait prefetch (nothing here is written by the producer) ----------------

        # GEMM-owner conv inputs: requests bo = s_split + CLUSTER * (hf + 2 jj)
        hf = tidx // 128
        ch_i = tidx % 128
        conv_ch = t_tile * 128 + ch_i
        cwr = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        csr = cute.make_rmem_tensor(cute.make_layout(6), cutlass.Float32)
        sbo = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
        for j in cutlass.range_constexpr(4):
            cwr[j] = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(6):
            csr[j] = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(2):
            sbo[j] = cutlass.Int32(-1)
        # conv owner inputs: raw words only (decoded after the wait, so no thread stalls on them here)
        cwq = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
        csq = cute.make_rmem_tensor(cute.make_layout(6), cutlass.Int32)
        for j in cutlass.range_constexpr(2):
            cwq[j] = cutlass.Int32(0)
        for j in cutlass.range_constexpr(6):
            csq[j] = cutlass.Int32(0)
        if is_gemm:
            for jj in cutlass.range_constexpr(2):
                sbo[jj] = sbo_e[jj]
            if t_tile < 12:
                cwq[0] = ldg_u32(g_convw + cutlass.Int64(conv_ch * 8))
                cwq[1] = ldg_u32(g_convw + cutlass.Int64(conv_ch * 8 + 4))
                for jj in cutlass.range_constexpr(2):
                    if sbo[jj] >= 0:
                        cbase = (
                            g_cs + cutlass.Int64(sbo[jj]) * cs_stride + cutlass.Int64(conv_ch * 6)
                        )
                        for j in cutlass.range_constexpr(3):
                            csq[jj * 3 + j] = ldg_u16(cbase + cutlass.Int64(2 * j))

        # verify-CTA inputs
        vi = vrel // V_HEADS
        hv = vrel % V_HEADS
        hk = hv // 4
        vs = cutlass.Int32(-1)
        pn = cutlass.Int32(0)
        vbuf = cutlass.Int32(0)
        g0 = cutlass.Float32(0.0)
        sb = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Int32)
        cmo = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Int32)
        wbr = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Int32)
        nwr = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        for j in cutlass.range_constexpr(32):
            sb[j] = cutlass.Int32(0)
            cmo[j] = cutlass.Int32(0)
        for j in cutlass.range_constexpr(32):
            wbr[j] = cutlass.Int32(0)
        sraw = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Int32)
        for j in cutlass.range_constexpr(32):
            sraw[j] = cutlass.Int32(0)
        for j in cutlass.range_constexpr(4):
            nwr[j] = cutlass.Float32(0.0)
        a_log = cutlass.Float32(0.0)
        dtb = cutlass.Float32(0.0)
        do_commit = cutlass.Int32(0)
        if is_ver:
            vs = vs_e
            if vs >= 0:
                pasel = pa0
                cbsel = cb0
                if vs >= 32:
                    pasel = pa1
                    cbsel = cb1
                pn = cute.arch.shuffle_sync(pasel, vs & 31)
                vbuf = cute.arch.shuffle_sync(cbsel, vs & 31)
                # state fragments sb[(nt*8 + ks)*2 + r] = state[v = 16w + 8nt + gid][16ks + 8r + 2tig .. +1]: loaded as
                # whole 16-byte chunks (lane tig of a quad takes chunks 4j + tig of its two rows: 8 vector loads
                # instead of 32 scalar ones), redistributed by a quad transpose after the barrier below
                sbase = g_state + cutlass.Int64(vs) * st_stride + cutlass.Int64(hv * 128 * 128 * 2)
                for nt in cutlass.range_constexpr(2):
                    vrow = warp_idx * 16 + nt * 8 + gid
                    for j in cutlass.range_constexpr(4):
                        w4s = ldg_v4(sbase + cutlass.Int64((vrow * 128 + (4 * j + tig) * 8) * 2))
                        for e in cutlass.range_constexpr(4):
                            sraw[(nt * 4 + j) * 4 + e] = w4s[e]
                if pn + TOKENS > HISTORY:
                    if commit != 0:
                        do_commit = cutlass.Int32(1)
                gbase = g_og + cutlass.Int64((((vs * 2 + vbuf) * V_HEADS + hv) * HISTORY) * 4)
                if pn > 0:
                    g0 = ldg_f32(gbase + cutlass.Int64((pn - 1) * 4))
                # history rows -> SMEM (warps 4..7): thread q owns 16 elements of kh and of uh
                if tidx >= 128:
                    q = tidx - 128
                    hj = q // 8
                    hc = (q % 8) * 16
                    hkr = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Int32)
                    hur = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Int32)
                    for j in cutlass.range_constexpr(8):
                        hkr[j] = cutlass.Int32(0)
                        hur[j] = cutlass.Int32(0)
                    if hj < pn:
                        kaddr = g_ok + cutlass.Int64(
                            ((((vs * 2 + vbuf) * HISTORY + hj) * K_HEADS + hk) * 128 + hc) * 2
                        )
                        uaddr = g_ou + cutlass.Int64(
                            ((((vs * 2 + vbuf) * HISTORY + hj) * V_HEADS + hv) * 128 + hc) * 2
                        )
                        k4a = ldg_v4(kaddr)
                        k4b = ldg_v4(kaddr + cutlass.Int64(16))
                        u4a = ldg_v4(uaddr)
                        u4b = ldg_v4(uaddr + cutlass.Int64(16))
                        for e in cutlass.range_constexpr(4):
                            hkr[e] = k4a[e]
                            hkr[4 + e] = k4b[e]
                            hur[e] = u4a[e]
                            hur[4 + e] = u4b[e]
                    sts_v4(HBX + H_KHS + hj * RS + hc * 2, hkr[0], hkr[1], hkr[2], hkr[3])
                    sts_v4(HBX + H_KHS + hj * RS + hc * 2 + 16, hkr[4], hkr[5], hkr[6], hkr[7])
                    for e in cutlass.range_constexpr(8):
                        sts_u16(HBX + H_KHT + (hc + 2 * e) * RT + hj * 2, hkr[e])
                        sts_u16(HBX + H_KHT + (hc + 2 * e + 1) * RT + hj * 2, hkr[e] >> 16)
                        sts_u16(HBX + H_UHT + (hc + 2 * e) * RT + hj * 2, hur[e])
                        sts_u16(HBX + H_UHT + (hc + 2 * e + 1) * RT + hj * 2, hur[e] >> 16)
                    if tidx < 144:
                        sts_f32(
                            HBG + H_GH + (tidx - 128) * 4,
                            ldg_f32(gbase + cutlass.Int64((tidx - 128) * 4)),
                        )
                # in_proj_ba K-slice of this head CTA (warps 4..7): all 16 w_ba rows over K [512 hv + 128 (warp - 4),
                # +128) as mma.sync A fragments; lane (gid, tig) holds the 16-byte chunks [8 tig, +8) of rows gid and
                # gid + 8 of every 32-wide block (the mma k order is permuted the same way on both operands)
                if tidx >= 128:
                    if tidx < 128 + BA_DIM:
                        prefetch_l2a(
                            g_wba + cutlass.Int64(((tidx - 128) * HIDDEN + hv * 512) * 2),
                            cutlass.Int32(1024),
                        )
                for nt in cutlass.range_constexpr(2):
                    nw2 = ldg_u32(
                        norm_w.iterator.toint()
                        + cutlass.Int64((warp_idx * 16 + nt * 8 + tig * 2) * 2)
                    )
                    nwr[nt * 2] = bf_lo(nw2)
                    nwr[nt * 2 + 1] = bf_hi(nw2)
                a_log = ldg_f32(A_log.iterator.toint() + cutlass.Int64(hv * 4))
                dtb = ldg_f32(dt_bias.iterator.toint() + cutlass.Int64(hv * 4))
                cute.arch.barrier()
                # quad transpose: lane tig needs word tig of every chunk; chunk 4j + q sits in lane q
                for nt in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(4):
                        for rr in cutlass.range_constexpr(4):
                            snd = sraw[(nt * 4 + j) * 4]
                            for q in cutlass.range_constexpr(1, 4):
                                if (tig ^ rr) == q:
                                    snd = sraw[(nt * 4 + j) * 4 + q]
                            rcv = snd
                            if cutlass.const_expr(rr > 0):
                                rcv = cute.arch.shuffle_sync_bfly(snd, rr)
                            for q in cutlass.range_constexpr(4):
                                if (tig ^ rr) == q:
                                    sb[nt * 16 + 4 * j + q] = rcv
                # ---- fold from pre-wait data: hc = e^g0 S + (uh * cd)^T kh, cd_j = e^(g0 - Gh_j) ----
                # hc is the committed checkpoint and the verify's effective state: e^G_t (k_t S) + k-history_t
                # = e^(cs_t) (k_t hc) with cs_t = G_t - g0 (likewise q), so the verify carries no history term.
                eg0 = cute.math.exp(g0)
                cdv = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                for e in cutlass.range_constexpr(4):
                    jc = tig * 2 + (e % 2) + (e // 2) * 8
                    cdv[e] = cutlass.Float32(0.0)
                    if jc < pn:
                        cdv[e] = cute.math.exp(g0 - lds_f32(HBG + H_GH + jc * 4))
                ah = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
                al = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
                for rg in cutlass.range_constexpr(4):
                    rv = warp_idx * 16 + gid + (rg % 2) * 8
                    cj = tig * 2 + (rg // 2) * 8
                    uw = lds_u32(HBX + H_UHT + rv * RT + cj * 2)
                    x0 = bf_lo(uw) * cdv[(rg // 2) * 2]
                    x1 = bf_hi(uw) * cdv[(rg // 2) * 2 + 1]
                    ah[rg] = pack_bf16(x0, x1)
                    al[rg] = pack_bf16(x0 - bf16r(x0), x1 - bf16r(x1))
                cmt = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Int32)
                for nt in cutlass.range_constexpr(16):
                    kb0 = lds_u32(HBX + H_KHT + (nt * 8 + gid) * RT + tig * 4)
                    kb1 = lds_u32(HBX + H_KHT + (nt * 8 + gid) * RT + tig * 4 + 16)
                    s0 = sb[(nt // 2) * 2 + (nt % 2)]
                    s1 = sb[(8 + nt // 2) * 2 + (nt % 2)]
                    d = mma_bf16(
                        ah[0],
                        ah[1],
                        ah[2],
                        ah[3],
                        kb0,
                        kb1,
                        eg0 * bf_lo(s0),
                        eg0 * bf_hi(s0),
                        eg0 * bf_lo(s1),
                        eg0 * bf_hi(s1),
                    )
                    d = mma_bf16(al[0], al[1], al[2], al[3], kb0, kb1, d[0], d[1], d[2], d[3])
                    cmt[nt * 2] = pack_bf16(d[0], d[1])
                    cmt[nt * 2 + 1] = pack_bf16(d[2], d[3])
                # hc in the pass-1 B-operand layout (D tile nt of rows gid / gid + 8 is sb[nt] / sb[16 + nt])
                for nt in cutlass.range_constexpr(16):
                    sb[nt] = cmt[nt * 2]
                    sb[16 + nt] = cmt[nt * 2 + 1]
                if do_commit != 0:
                    # 4x4 transpose inside each tig quad: lane tig gets k = 8 (4g + tig) .. +7 (one 16-byte store)
                    for g in cutlass.range_constexpr(4):
                        for h in cutlass.range_constexpr(2):
                            for r in cutlass.range_constexpr(4):
                                sidx = tig ^ r
                                snd = cmt[(4 * g) * 2 + h]
                                for i in cutlass.range_constexpr(1, 4):
                                    if sidx == i:
                                        snd = cmt[(4 * g + i) * 2 + h]
                                rcv = snd
                                if cutlass.const_expr(r > 0):
                                    rcv = cute.arch.shuffle_sync_bfly(snd, r)
                                for j in cutlass.range_constexpr(4):
                                    if sidx == j:
                                        cmo[(g * 2 + h) * 4 + j] = rcv
                    tmem_st32(tm_cm, cmo)
                tmem_st32(tm_sb, sb)

        if cutlass.const_expr(self.hx):
            # the history rows staged in the x ring must be consumed before the x TMA overwrites them
            fence_proxy_async_smem()
            cute.arch.barrier()
        # ================= wait for the producer (x_fp8 / x_bf16) =================
        cute.arch.griddepcontrol_wait()
        cute.arch.cluster_wait()
        cwr[0] = bf_lo(cwq[0])
        cwr[1] = bf_hi(cwq[0])
        cwr[2] = bf_lo(cwq[1])
        cwr[3] = bf_hi(cwq[1])
        for j in cutlass.range_constexpr(6):
            csr[j] = bf_lo(csq[j])
        if cute.arch.lane_idx() == 0:
            ctr_old = atom_add_relaxed(g_tk, cutlass.Int32(1))
        qscale = xs_raw * ws_raw
        inv_f8s = div_rn(cutlass.Float32(1.0), f8s)

        if is_gemm:
            if warp_idx == 0:
                for kk in cutlass.range_constexpr(self.nstage):
                    if kk < kb_cnt:
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, kb_start + kk, 0)],
                            tBsB[(None, kk)],
                            tma_bar_ptr=full_mbar + kk,
                        )
                # ---------------- MMA issue (warp 0) ----------------
                for kk in cutlass.range(kb_cnt, unroll=1):
                    cute.arch.mbarrier_wait(full_mbar + kk, 0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, kk != 0)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, None, kk)],
                        tCrB[(None, None, None, kk)],
                        tCtAcc,
                    )
                with cute.arch.elect_one():
                    tcgen05.commit(acc_mbar)
        # ---- in_proj_ba K-slice (verify CTAs, warps 4..7, while the x TMA is in flight): head CTA (i, h) computes
        # all 16 ba channels of request i over K [512 h, +512) and publishes them as tagged 16-byte words; the CTA
        # of (i, h') sums the 8 slices of its two channels during the split-K push.
        ba_bar = cutlass.pipeline.NamedBarrier(barrier_id=3, num_threads=128)
        if is_ver:
            if vs >= 0:
                if tidx >= 128:
                    for j in cutlass.range_constexpr(4):
                        kc = hv * 512 + (warp_idx - 4) * 128 + j * 32 + tig * 8
                        w0 = ldg_v4(g_wba + cutlass.Int64((gid * HIDDEN + kc) * 2))
                        w1 = ldg_v4(g_wba + cutlass.Int64(((gid + 8) * HIDDEN + kc) * 2))
                        for e in cutlass.range_constexpr(4):
                            wbr[j * 8 + e] = w0[e]
                            wbr[j * 8 + 4 + e] = w1[e]
                    xb = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Int32)
                    xrow = vi * TOKENS + cutlass.min(gid, TOKENS - 1)
                    for j in cutlass.range_constexpr(4):
                        kc = hv * 512 + (warp_idx - 4) * 128 + j * 32 + tig * 8
                        xv = ldg_v4(g_xbf + cutlass.Int64((xrow * HIDDEN + kc) * 2))
                        for e in cutlass.range_constexpr(4):
                            xb[j * 4 + e] = xv[e]
                    if gid == TOKENS:
                        for e in cutlass.range_constexpr(16):
                            xb[e] = cutlass.Int32(0)
                    dba = [
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                    ]
                    dbb = [
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                    ]
                    for j in cutlass.range_constexpr(4):
                        for s in cutlass.range_constexpr(2):
                            if cutlass.const_expr(s == 0):
                                dba = mma_bf16(
                                    wbr[j * 8 + 2 * s],
                                    wbr[j * 8 + 4 + 2 * s],
                                    wbr[j * 8 + 2 * s + 1],
                                    wbr[j * 8 + 4 + 2 * s + 1],
                                    xb[j * 4 + 2 * s],
                                    xb[j * 4 + 2 * s + 1],
                                    dba[0],
                                    dba[1],
                                    dba[2],
                                    dba[3],
                                )
                            else:
                                dbb = mma_bf16(
                                    wbr[j * 8 + 2 * s],
                                    wbr[j * 8 + 4 + 2 * s],
                                    wbr[j * 8 + 2 * s + 1],
                                    wbr[j * 8 + 4 + 2 * s + 1],
                                    xb[j * 4 + 2 * s],
                                    xb[j * 4 + 2 * s + 1],
                                    dbb[0],
                                    dbb[1],
                                    dbb[2],
                                    dbb[3],
                                )
                    # D[c][t]: lane (gid, tig) holds c = gid / gid + 8, t = 2 tig / 2 tig + 1
                    bw = SE + (E_BAW + (warp_idx - 4) * 128) * 4
                    sts_f32(bw + (gid * 8 + 2 * tig) * 4, dba[0] + dbb[0])
                    sts_f32(bw + (gid * 8 + 2 * tig + 1) * 4, dba[1] + dbb[1])
                    sts_f32(bw + ((gid + 8) * 8 + 2 * tig) * 4, dba[2] + dbb[2])
                    sts_f32(bw + ((gid + 8) * 8 + 2 * tig + 1) * 4, dba[3] + dbb[3])
                    ba_bar.arrive_and_wait()
                    tag_b = handoff_tag(cute.arch.shuffle_sync(ctr_old, 0), N)
                    if tidx < 128 + V_HEADS * BAW:
                        hd = (tidx - 128) // BAW
                        wq_ = (tidx - 128) % BAW
                        pv = cute.make_rmem_tensor(cute.make_layout(3), cutlass.Float32)
                        for e in cutlass.range_constexpr(3):
                            kk_ = wq_ * 3 + e
                            ch_ = hd + V_HEADS * (kk_ // TOKENS)
                            tt_ = kk_ % TOKENS
                            acc_ = cutlass.Float32(0.0)
                            if kk_ < 2 * TOKENS:
                                for w in cutlass.range_constexpr(4):
                                    acc_ = acc_ + lds_f32(
                                        SE + (E_BAW + w * 128 + ch_ * 8 + tt_) * 4
                                    )
                            pv[e] = acc_
                        stg_v4_rlx(
                            g_ba
                            + cutlass.Int64(
                                (((vi * V_HEADS + hd) * V_HEADS + hv) * BAW + wq_) * 16
                            ),
                            tag_b,
                            f2i(pv[0]),
                            f2i(pv[1]),
                            f2i(pv[2]),
                        )

        # ---------------- GEMM epilogue: split-K push over DSMEM, owner reduce + conv + publish ----------------
        tag = handoff_tag(cute.arch.shuffle_sync(ctr_old, 0), N)
        # committed checkpoint (folded before the wait, parked in TMEM): every warp stores its 16 rows straight from
        # TMEM-loaded registers once its post-wait issue work is out (warp 0 after the MMA issue, warps 4..7 after the
        # in_proj_ba slice), so nothing of it is left on the verify path
        if is_ver:
            if do_commit != 0:
                cml0 = tmem_ld32(tm_cm)
                gcs = g_state + cutlass.Int64(vs) * st_stride + cutlass.Int64(hv * 128 * 128 * 2)
                for g in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(2):
                        vr = warp_idx * 16 + gid + 8 * h
                        stg_v4(
                            gcs + cutlass.Int64((vr * 128 + 32 * g + 8 * tig) * 2),
                            cml0[(g * 2 + h) * 4],
                            cml0[(g * 2 + h) * 4 + 1],
                            cml0[(g * 2 + h) * 4 + 2],
                            cml0[(g * 2 + h) * 4 + 3],
                        )
        if tidx < 128:
            if is_gemm:
                cute.arch.mbarrier_wait(acc_mbar, 0)
        cute.arch.griddepcontrol_launch_dependents()
        # This rank's MMAs are done: if it receives split-K partials in its x ring (split: slots >= 1, alias: every
        # slot), signal each peer's xfree[this rank]; a pusher waits only for its destination's signal.
        if tidx == 0:
            if is_gemm:
                xsig = cutlass.Int32(0)
                if cutlass.const_expr(self.rsplit):
                    if s_split + CLUSTER < N:
                        xsig = cutlass.Int32(1)
                if cutlass.const_expr(self.alias_recv):
                    if s_split < N:
                        xsig = cutlass.Int32(1)
                if xsig != 0:
                    xfb = xfree_mbar.toint() + s_split * 8
                    for p_ in cutlass.range_constexpr(CLUSTER):
                        if s_split != p_:
                            remote_arrive(mapa_u32(xfb, p_))
        # ---------------- in_proj_ba: sum the 8 published K-slices of this head's b / a rows, then the gates --------
        if is_ver:
            if vs >= 0:
                if tidx >= 128:
                    if tidx < 128 + V_HEADS * BAW:
                        srch = (tidx - 128) // BAW
                        wq_ = (tidx - 128) % BAW
                        baddr = g_ba + cutlass.Int64(
                            (((vi * V_HEADS + hv) * V_HEADS + srch) * BAW + wq_) * 16
                        )
                        bwv = ldg_v4_rlx(baddr)
                        bt0 = bwv[0]
                        bt1 = bwv[1]
                        bt2 = bwv[2]
                        bt3 = bwv[3]
                        while bt0 != tag:
                            bwv2 = ldg_v4_rlx(baddr)
                            bt0 = bwv2[0]
                            bt1 = bwv2[1]
                            bt2 = bwv2[2]
                            bt3 = bwv2[3]
                        sts_f32(SE + (E_BAC + srch * 16 + wq_ * 3) * 4, i2f(bt1))
                        sts_f32(SE + (E_BAC + srch * 16 + wq_ * 3 + 1) * 4, i2f(bt2))
                        sts_f32(SE + (E_BAC + srch * 16 + wq_ * 3 + 2) * 4, i2f(bt3))
                if tidx >= 128:
                    ba_bar.arrive_and_wait()
                # ---- warp 7: gates, cumulative decays and every exp factor of the verify ----
                if warp_idx == 7:
                    tl = cutlass.min(lane, TOKENS - 1)
                    braw = cutlass.Float32(0.0)
                    araw = cutlass.Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        braw = braw + lds_f32(SE + (E_BAC + w * 16 + tl) * 4)
                        araw = araw + lds_f32(SE + (E_BAC + w * 16 + TOKENS + tl) * 4)
                    braw = bf16r(braw)
                    araw = bf16r(araw)
                    beta = rcp_approx(cutlass.Float32(1.0) + exp_fast(-braw))
                    xa_ = araw + dtb
                    sp = xa_
                    if xa_ <= cutlass.Float32(20.0):
                        sp = lg2_approx(cutlass.Float32(1.0) + exp_fast(xa_)) * cutlass.Float32(LN2)
                    gd = -exp_fast(a_log) * sp
                    csum = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(TOKENS):
                        gj = cute.arch.shuffle_sync(gd, j)
                        if j <= lane:
                            csum = csum + gj
                    if lane < TOKENS:
                        sts_f32(SE + (E_BETA + lane) * 4, beta)
                        sts_f32(SE + (E_CS + lane) * 4, csum)
                    egv = cutlass.Float32(0.0)
                    if lane < TOKENS:
                        egv = exp_fast(csum)
                    if lane < 8:
                        sts_f32(SE + (E_EG + lane) * 4, egv)
                    for rr in cutlass.range_constexpr(2):
                        q2 = rr * 32 + lane
                        dt_ = q2 // 8
                        dj = q2 % 8
                        cst_ = cute.arch.shuffle_sync(csum, dt_)
                        csj = cute.arch.shuffle_sync(csum, dj)
                        dnv = cutlass.Float32(0.0)
                        if dt_ < TOKENS:
                            if dj <= dt_:
                                dnv = exp_fast(cst_ - csj)
                        sts_f32(SE + (E_DN + q2) * 4, dnv)

        if cutlass.const_expr(self.rsplit):
            # split receive (N 7 .. 11): slot 0 goes to the dedicated buffers right away; slot 1 lands on the drained
            # x rings, so every thread of the cluster first waits for the second barrier phase (all ranks' MMAs done)
            accs = cute.make_rmem_tensor(cute.make_layout(NMMA), cutlass.Float32)
            rbs = recv.toint()
            rxs = recvx.toint()
            mbs = recv_mbar.toint()
            push_bs = cutlass.pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            for j in cutlass.range_constexpr(NMMA):
                accs[j] = cutlass.Float32(0.0)
            if tidx < 128:
                tm_acc = tm_base + ((32 * warp_idx) << 16)
                accw = tmem_ld_x16s([tm_acc + 16 * c for c in range(NMMA // 16)])
                for j in cutlass.range_constexpr(NMMA):
                    accs[j] = i2f(accw[j])
            for slot in cutlass.range_constexpr(NRO):
                if tidx < 128:
                    for r in cutlass.range_constexpr(CLUSTER):
                        if cutlass.const_expr(CLUSTER * slot + r < N):
                            b = CLUSTER * slot + r
                            pdst = SB + (slot * CLUSTER + r) * PARTIAL_BYTES + tidx * 4
                            if s_split == r:
                                pdst = (
                                    _rslot(rbs, rxs, slot, True)
                                    + s_split * PARTIAL_BYTES
                                    + tidx * 4
                                )
                            for tt in cutlass.range_constexpr(TOKENS):
                                sts_u32(pdst + tt * 512, f2i(accs[7 * b + tt]))
                    fence_proxy_async_smem()
                    push_bs.arrive_and_wait()
                    if tidx == 0:
                        for r in cutlass.range_constexpr(CLUSTER):
                            if cutlass.const_expr(CLUSTER * slot + r < N):
                                if s_split != r:
                                    if cutlass.const_expr(slot >= 1):
                                        # (destination's x ring: wait for its MMA-done signal)
                                        cute.arch.mbarrier_wait(xfree_mbar + r, 0)
                                    bulk_s2s(
                                        mapa_u32(
                                            _rslot(rbs, rxs, slot, True) + s_split * PARTIAL_BYTES,
                                            r,
                                        ),
                                        SB + (slot * CLUSTER + r) * PARTIAL_BYTES,
                                        PARTIAL_BYTES,
                                        mapa_u32(mbs + 8 * slot, r),
                                    )
                    # this rank's own partial of the slot is in place
                    cute.arch.mbarrier_arrive(recv_mbar + slot)
        else:
            if is_gemm:
                if tidx < 128:
                    accv = cute.make_rmem_tensor(cute.make_layout(NMMA), cutlass.Float32)
                    push_bar = cutlass.pipeline.NamedBarrier(barrier_id=2, num_threads=128)
                    rbase = recv.toint()
                    rxb = recvx.toint()
                    mbase = recv_mbar.toint()
                    if cutlass.const_expr(NMMA <= 80):
                        # whole row (NMMA columns) in one TMEM round trip
                        tm_acc = tm_base + ((32 * warp_idx) << 16)
                        accw = tmem_ld_x16s([tm_acc + 16 * c for c in range(NMMA // 16)])
                        for j in cutlass.range_constexpr(NMMA):
                            accv[j] = i2f(accw[j])
                    if cutlass.const_expr(NMMA <= 80 and N > 8):
                        # slot by slot (N > 8): stage requests 6k .. 6k + 5 for all owners, push them, then the next
                        for slot in cutlass.range_constexpr(NRO):
                            for r in cutlass.range_constexpr(CLUSTER):
                                if cutlass.const_expr(CLUSTER * slot + r < N):
                                    b = CLUSTER * slot + r
                                    pdst = SB + (slot * CLUSTER + r) * PARTIAL_BYTES + tidx * 4
                                    if s_split == r:
                                        pdst = (
                                            _rslot(rbase, rxb, slot, self.rsplit)
                                            + s_split * PARTIAL_BYTES
                                            + tidx * 4
                                        )
                                    for tt in cutlass.range_constexpr(TOKENS):
                                        sts_u32(pdst + tt * 512, f2i(accv[7 * b + tt]))
                            fence_proxy_async_smem()
                            push_bar.arrive_and_wait()
                            if tidx == 0:
                                for r in cutlass.range_constexpr(CLUSTER):
                                    if cutlass.const_expr(CLUSTER * slot + r < N):
                                        if s_split != r:
                                            bulk_s2s(
                                                mapa_u32(
                                                    _rslot(rbase, rxb, slot, self.rsplit)
                                                    + s_split * PARTIAL_BYTES,
                                                    r,
                                                ),
                                                SB + (slot * CLUSTER + r) * PARTIAL_BYTES,
                                                PARTIAL_BYTES,
                                                mapa_u32(mbase + 8 * slot, r),
                                            )
                            cute.arch.mbarrier_arrive(recv_mbar + slot)
                        if tidx == 0:
                            bulk_commit_wait_read()
                    else:
                        # N > 11 (accumulator read chunk by chunk): each slot's blocks (requests 6k .. 6k + 5) are
                        # pushed as soon as the TMEM chunk that completes the slot is staged
                        tm_acc_b = tm_base + ((32 * warp_idx) << 16)
                        for chk in cutlass.range_constexpr(NMMA // 16):
                            if cutlass.const_expr(NMMA > 80):
                                # accumulator columns in two batched TMEM round trips (chunks 0..3, then the rest)
                                # instead of one blocking load + wait per 16-column chunk
                                if cutlass.const_expr(chk == 0 or chk == 4):
                                    c_hi = 4 if chk == 0 else NMMA // 16
                                    accw_b = tmem_ld_x16s(
                                        [tm_acc_b + 16 * c for c in range(chk, c_hi)]
                                    )
                                    for j in cutlass.range_constexpr(16 * (c_hi - chk)):
                                        accv[chk * 16 + j] = i2f(accw_b[j])
                            for b in cutlass.range_constexpr(N):
                                if (7 * b + 6) // 16 == chk:
                                    own = b % CLUSTER
                                    slot = b // CLUSTER
                                    # partial block [row t][channel] fp32 (conflict-free 128-byte rows)
                                    pdst = SB + (slot * CLUSTER + own) * PARTIAL_BYTES + tidx * 4
                                    if s_split == own:
                                        pdst = (
                                            _rslot(rbase, rxb, slot, self.rsplit)
                                            + s_split * PARTIAL_BYTES
                                            + tidx * 4
                                        )
                                    for tt in cutlass.range_constexpr(TOKENS):
                                        sts_u32(pdst + tt * 512, f2i(accv[7 * b + tt]))
                            for slot in cutlass.range_constexpr(NRO):
                                b_last = min(CLUSTER * slot + CLUSTER - 1, N - 1)
                                if cutlass.const_expr((7 * b_last + 6) // 16 == chk):
                                    fence_proxy_async_smem()
                                    push_bar.arrive_and_wait()
                                    if tidx == 0:
                                        for r in cutlass.range_constexpr(CLUSTER):
                                            if cutlass.const_expr(CLUSTER * slot + r < N):
                                                if s_split != r:
                                                    if cutlass.const_expr(self.alias_recv):
                                                        cute.arch.mbarrier_wait(xfree_mbar + r, 0)
                                                    bulk_s2s(
                                                        mapa_u32(
                                                            _rslot(rbase, rxb, slot, self.rsplit)
                                                            + s_split * PARTIAL_BYTES,
                                                            r,
                                                        ),
                                                        SB + (slot * CLUSTER + r) * PARTIAL_BYTES,
                                                        PARTIAL_BYTES,
                                                        mapa_u32(mbase + 8 * slot, r),
                                                    )
                                    cute.arch.mbarrier_arrive(recv_mbar + slot)
        yvk = cute.make_rmem_tensor(cute.make_layout(2 * (TOKENS + 3)), cutlass.Float32)
        okj = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
        gpre = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Int32)
        for k in cutlass.range_constexpr(2):
            okj[k] = cutlass.Int32(0)
        for k in cutlass.range_constexpr(8):
            gpre[k] = cutlass.Int32(0)
        gflag = cutlass.Int32(0)
        aA_e = g_q + cutlass.Int64(
            ((vi * QKVZ_TILES + 2 + hk - 2 * (tidx // 128)) * 128 + tidx % 128) * 16
        )
        aB_e = g_q + cutlass.Int64(
            ((vi * QKVZ_TILES + 4 + hv + 8 * (tidx // 128)) * 128 + tidx % 128) * 16
        )
        if is_gemm:
            rb2 = recv.toint()
            rx2 = recvx.toint()
            for jj in cutlass.range_constexpr(2):
                qo = hf + 2 * jj
                bo = s_split + CLUSTER * qo
                if bo < N:
                    # this slot's partials (own + 5 pushed) have landed
                    cute.arch.mbarrier_wait(recv_mbar + qo, 0)
                    sl = sbo[jj]
                    if sl >= 0:
                        yv = cute.make_rmem_tensor(cute.make_layout(TOKENS + 3), cutlass.Float32)
                        for tt in cutlass.range_constexpr(TOKENS + 3):
                            yv[tt] = cutlass.Float32(0.0)
                        rq_ = rb2 + qo * (CLUSTER * PARTIAL_BYTES)
                        if cutlass.const_expr(self.rsplit):
                            rq_ = rx2 + (qo - 1) * (CLUSTER * PARTIAL_BYTES)
                            if qo == 0:
                                rq_ = rb2
                        for r in cutlass.range_constexpr(CLUSTER):
                            for tt in cutlass.range_constexpr(TOKENS):
                                yv[3 + tt] = yv[3 + tt] + lds_f32(
                                    rq_ + r * PARTIAL_BYTES + tt * 512 + ch_i * 4
                                )
                        for tt in cutlass.range_constexpr(TOKENS):
                            yv[3 + tt] = bf16r(yv[3 + tt] * qscale)
                        pub = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                        if t_tile < 12:
                            for j in cutlass.range_constexpr(3):
                                yv[j] = csr[jj * 3 + j]
                            for tt in cutlass.range_constexpr(TOKENS):
                                acc = (
                                    cwr[0] * yv[tt]
                                    + cwr[1] * yv[tt + 1]
                                    + cwr[2] * yv[tt + 2]
                                    + cwr[3] * yv[tt + 3]
                                )
                                pub[tt] = bf16r(acc * sigmoid_fast(acc))
                        else:
                            for tt in cutlass.range_constexpr(TOKENS):
                                pub[tt] = yv[3 + tt]
                        p0 = or_i32(pack_bf16(cutlass.Float32(0.0), pub[0]), tag)
                        p1 = pack_bf16(pub[1], pub[2])
                        p2 = pack_bf16(pub[3], pub[4])
                        p3 = pack_bf16(pub[5], pub[6])
                        stg_v4_rlx(
                            g_q + cutlass.Int64(((bo * QKVZ_TILES + t_tile) * 128 + ch_i) * 16),
                            p0,
                            p1,
                            p2,
                            p3,
                        )
                        for k in cutlass.range_constexpr(TOKENS + 3):
                            yvk[jj * (TOKENS + 3) + k] = yv[k]
                        okj[jj] = cutlass.Int32(1)
            # verify CTAs: first try of the gather words goes out right after the publish, so its round trip
            # overlaps the conv window / state stores below
            if is_ver:
                if vs >= 0:
                    wA0 = ldg_v4_rlx(aA_e)
                    wB0 = ldg_v4_rlx(aB_e)
                    for e in cutlass.range_constexpr(4):
                        gpre[e] = wA0[e]
                        gpre[4 + e] = wB0[e]
                    gflag = cutlass.Int32(1)
            for jj in cutlass.range_constexpr(2):
                qo = hf + 2 * jj
                bo = s_split + CLUSTER * qo
                sl = sbo[jj]
                if okj[jj] != 0:
                    yv = cute.make_rmem_tensor(cute.make_layout(TOKENS + 3), cutlass.Float32)
                    for k in cutlass.range_constexpr(TOKENS + 3):
                        yv[k] = yvk[jj * (TOKENS + 3) + k]
                    if t_tile < 12:
                        # conv_window[bo, t] = seq[t+1 : t+4]; conv_state[slot] = seq[7:10]: an SMEM image of
                        # this tile's 128-channel rows (7 x 768 B + 768 B), stored below as 16-byte words
                        cvs = SB + O_CV + (hf * 2 + jj) * 6144
                        for tt in cutlass.range_constexpr(TOKENS):
                            for j in cutlass.range_constexpr(3):
                                sts_u16(
                                    cvs + tt * 768 + ch_i * 6 + j * 2,
                                    pack_bf16(yv[tt + 1 + j], cutlass.Float32(0.0)),
                                )
                        for j in cutlass.range_constexpr(3):
                            sts_u16(
                                cvs + 5376 + ch_i * 6 + j * 2,
                                pack_bf16(yv[7 + j], cutlass.Float32(0.0)),
                            )
                        cutlass.pipeline.NamedBarrier(
                            barrier_id=4 + hf, num_threads=128
                        ).arrive_and_wait()
                        # the 8 rows (7 windows + the new conv_state, 768 B each) leave as 16-byte stores of the
                        # half's 128 threads
                        for r in cutlass.range_constexpr(3):
                            wd = ch_i + 128 * r
                            chk = wd // 48
                            off = (wd % 48) * 16
                            v4 = lds_v4(cvs + chk * 768 + off)
                            ga_ = (
                                g_cs
                                + cutlass.Int64(sl) * cs_stride
                                + cutlass.Int64(t_tile * 128 * 6 + off)
                            )
                            if chk < TOKENS:
                                ga_ = g_cw + cutlass.Int64(
                                    ((bo * TOKENS + chk) * CONV_DIM + t_tile * 128) * 6 + off
                                )
                            stg_v4(ga_, v4[0], v4[1], v4[2], v4[3])

        # the split-K staging blocks (bulk-copy sources) are reused only by the verify scratch: thread 0 waits for
        # their reads here (and at teardown), not on the owner-reduce path of warp 0
        if tidx == 0:
            bulk_commit_wait_read()
        # ---------------- verify ----------------
        if is_ver:
            cute.arch.barrier()
            if vs >= 0:
                overflow = pn + TOKENS > HISTORY
                wbuf = vbuf
                woff = pn
                if overflow:
                    wbuf = 1 - vbuf
                    woff = cutlass.Int32(0)
                if tidx < TOKENS:
                    gval = lds_f32(SE + (E_CS + tidx) * 4)
                    if pn + TOKENS <= HISTORY:
                        gval = gval + g0
                    stg_f32(
                        g_og
                        + cutlass.Int64(
                            (((vs * 2 + wbuf) * V_HEADS + hv) * HISTORY + woff + tidx) * 4
                        ),
                        gval,
                    )
                isq = tidx // 128
                gch = tidx % 128
                tA = 2 + hk - 2 * isq
                aA = g_q + cutlass.Int64(((vi * QKVZ_TILES + tA) * 128 + gch) * 16)
                aB = g_q + cutlass.Int64(((vi * QKVZ_TILES + 4 + hv + 8 * isq) * 128 + gch) * 16)
                # first try of every word this thread needs (k or q; v or z): issued after the publish when this
                # CTA is a GEMM owner, here otherwise
                if gflag == 0:
                    wA1 = ldg_v4_rlx(aA)
                    wB1 = ldg_v4_rlx(aB)
                    for e in cutlass.range_constexpr(4):
                        gpre[e] = wA1[e]
                        gpre[4 + e] = wB1[e]
                # ---- gather k / q (all threads) and v (threads < 128); z (threads >= 128) is checked later ----
                ga0 = gpre[0]
                ga1 = gpre[1]
                ga2 = gpre[2]
                ga3 = gpre[3]
                gb0 = gpre[4]
                gb1 = gpre[5]
                gb2 = gpre[6]
                gb3 = gpre[7]
                # state fragments (parked in TMEM before the wait) come back while the k / q words are polled
                sbl = tmem_ld32(tm_sb)
                for j in cutlass.range_constexpr(32):
                    sb[j] = sbl[j]
                while tag_of(ga0) != tag:
                    wA2 = ldg_v4_rlx(aA)
                    ga0 = wA2[0]
                    ga1 = wA2[1]
                    ga2 = wA2[2]
                    ga3 = wA2[3]
                # fresh try of the v / z word now that k / q are in: its round trip overlaps pass 1 (the first try went
                # out right after this CTA's own publish and is usually too early)
                wB3 = ldg_v4_rlx(aB)
                gb0 = wB3[0]
                gb1 = wB3[1]
                gb2 = wB3[2]
                gb3 = wB3[3]
                # raw bf16 rows: XA rows 0..6 = k, 8..14 = q (exact operands of the bf16 MMAs)
                xrow = SB + O_XA + (isq * 8) * RS + gch * 2
                sts_u16(xrow, ga0 >> 16)
                sts_u16(xrow + RS, ga1)
                sts_u16(xrow + 2 * RS, ga1 >> 16)
                sts_u16(xrow + 3 * RS, ga2)
                sts_u16(xrow + 4 * RS, ga2 >> 16)
                sts_u16(xrow + 5 * RS, ga3)
                sts_u16(xrow + 6 * RS, ga3 >> 16)
                sts_u16(xrow + 7 * RS, cutlass.Int32(0))
                cute.arch.barrier()
                # ---- pass 1 on raw rows: P = X S^T (all warps); KH = X kh^T (warps 0, 1); Gram X X^T (warps 2, 3) ----
                # all fragment loads first (independent), then the MMAs: four accumulator chains per column tile
                af = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Int32)
                for ks in cutlass.range_constexpr(8):
                    c0 = ks * 16 + tig * 2
                    af[ks * 4 + 0] = lds_u32(SB + O_XA + gid * RS + c0 * 2)
                    af[ks * 4 + 1] = lds_u32(SB + O_XA + (gid + 8) * RS + c0 * 2)
                    af[ks * 4 + 2] = lds_u32(SB + O_XA + gid * RS + c0 * 2 + 16)
                    af[ks * 4 + 3] = lds_u32(SB + O_XA + (gid + 8) * RS + c0 * 2 + 16)
                kf = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Int32)
                for e in cutlass.range_constexpr(16):
                    kf[e] = cutlass.Int32(0)
                gram_w = (warp_idx >> 1) == 1
                if gram_w:
                    kbase = SB + O_XA + ((warp_idx - 2) * 8 + gid) * RS
                    for ks in cutlass.range_constexpr(8):
                        c0 = ks * 16 + tig * 2
                        kf[ks * 2] = lds_u32(kbase + c0 * 2)
                        kf[ks * 2 + 1] = lds_u32(kbase + c0 * 2 + 16)
                acc4 = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                ack = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                for e in cutlass.range_constexpr(32):
                    acc4[e] = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(8):
                    ack[e] = cutlass.Float32(0.0)
                for ks in cutlass.range_constexpr(8):
                    par = ks % 4
                    for nt in cutlass.range_constexpr(2):
                        o_ = par * 8 + nt * 4
                        d = mma_bf16(
                            af[ks * 4],
                            af[ks * 4 + 1],
                            af[ks * 4 + 2],
                            af[ks * 4 + 3],
                            sb[(nt * 8 + ks) * 2],
                            sb[(nt * 8 + ks) * 2 + 1],
                            acc4[o_],
                            acc4[o_ + 1],
                            acc4[o_ + 2],
                            acc4[o_ + 3],
                        )
                        for e in cutlass.range_constexpr(4):
                            acc4[o_ + e] = d[e]
                    if gram_w:
                        p2_ = ks % 2
                        d = mma_bf16(
                            af[ks * 4],
                            af[ks * 4 + 1],
                            af[ks * 4 + 2],
                            af[ks * 4 + 3],
                            kf[ks * 2],
                            kf[ks * 2 + 1],
                            ack[p2_ * 4],
                            ack[p2_ * 4 + 1],
                            ack[p2_ * 4 + 2],
                            ack[p2_ * 4 + 3],
                        )
                        for e in cutlass.range_constexpr(4):
                            ack[p2_ * 4 + e] = d[e]
                # fold parities 2 / 3 into 0 / 1: acc[par * 8 + nt * 4 + e], par in {0, 1}
                acc = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float32)
                for e in cutlass.range_constexpr(16):
                    acc[e] = acc4[e] + acc4[16 + e]
                if gram_w:
                    cb = (
                        SMS + (M_KG + (warp_idx // 2) * 16 * KGS + (warp_idx % 2) * 8 + tig * 2) * 4
                    )
                    sts_f32(cb + (gid * KGS) * 4, ack[0] + ack[4])
                    sts_f32(cb + (gid * KGS + 1) * 4, ack[1] + ack[5])
                    sts_f32(cb + ((gid + 8) * KGS) * 4, ack[2] + ack[6])
                    sts_f32(cb + ((gid + 8) * KGS + 1) * 4, ack[3] + ack[7])
                cute.arch.barrier()
                # ---- pass 2 (per thread: rows t = gid of the k and q halves) ----
                tt7 = cutlass.min(gid, TOKENS - 1)
                GRB = SMS + (M_KG + 16 * KGS) * 4
                # every SMEM operand of pass 2 first (independent loads, padded rows read as 8- / 16-byte vectors), then
                # the math
                gko = lds_f32(GRB + (gid * KGS + gid) * 4)
                gqt = lds_f32(GRB + ((tt7 + 8) * KGS + tt7 + 8) * 4)
                egt = lds_f32(SE + (E_EG + tt7) * 4)
                bet = lds_f32(SE + (E_BETA + tt7) * 4)
                dnr = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                for j in cutlass.range_constexpr(TOKENS):
                    dnr[j] = lds_f32(SE + (E_DN + tt7 * 8 + j) * 4)
                gkr = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                gqr = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                for h4 in cutlass.range_constexpr(2):
                    w4 = lds_v4(GRB + (tt7 * KGS + h4 * 4) * 4)
                    for e in cutlass.range_constexpr(4):
                        gkr[h4 * 4 + e] = i2f(w4[e])
                    w4 = lds_v4(GRB + ((tt7 + 8) * KGS + h4 * 4) * 4)
                    for e in cutlass.range_constexpr(4):
                        gqr[h4 * 4 + e] = i2f(w4[e])
                # inverse k-row norms: lane (gid, tig) computes row gid's, the others come by shuffle (one MUFU op
                # per thread instead of eight)
                rk_t = cute.math.rsqrt(gko + cutlass.Float32(1e-6))
                rkv = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                for j in cutlass.range_constexpr(TOKENS):
                    rkv[j] = cute.arch.shuffle_sync(rk_t, j * 4)
                rq_t = cute.math.rsqrt(gqt + cutlass.Float32(1e-6)) * cutlass.Float32(
                    1.0 / math.sqrt(HEAD_DIM)
                )
                if gid == TOKENS:
                    rk_t = cutlass.Float32(0.0)
                    rq_t = cutlass.Float32(0.0)
                amr = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                qkr = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                for j in cutlass.range_constexpr(TOKENS):
                    am_ = bet * gkr[j] * rk_t * rkv[j] * dnr[j]
                    if j >= gid:
                        am_ = cutlass.Float32(0.0)
                    amr[j] = am_
                    qkr[j] = gqr[j] * rq_t * rkv[j] * dnr[j]
                ur = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                orr = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                # e^(cs_t) (k_t hc) = e^G_t (k_t S) + k-history_t (then -> U); likewise q -> o
                for nt in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(2):
                        pk_ = acc[nt * 4 + e] + acc[8 + nt * 4 + e]
                        pq_ = acc[nt * 4 + 2 + e] + acc[8 + nt * 4 + 2 + e]
                        ur[nt * 2 + e] = egt * rk_t * pk_
                        orr[nt * 2 + e] = egt * rq_t * pq_
                # ---- v of this request (threads < 128): needed only from here on (the pass-2 prep overlaps its wait)
                if isq == 0:
                    while tag_of(gb0) != tag:
                        wB2 = ldg_v4_rlx(aB)
                        gb0 = wB2[0]
                        gb1 = wB2[1]
                        gb2 = wB2[2]
                        gb3 = wB2[3]
                    vb = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                    vb[0] = bf_hi(gb0)
                    vb[1] = bf_lo(gb1)
                    vb[2] = bf_hi(gb1)
                    vb[3] = bf_lo(gb2)
                    vb[4] = bf_hi(gb2)
                    vb[5] = bf_lo(gb3)
                    vb[6] = bf_hi(gb3)
                    for t in cutlass.range_constexpr(TOKENS):
                        sts_f32(SB + O_VS + (t * VSS + gch) * 4, vb[t])
                cute.arch.barrier()
                for nt in cutlass.range_constexpr(2):
                    vw = lds_v2(SB + O_VS + (tt7 * VSS + warp_idx * 16 + nt * 8 + tig * 2) * 4)
                    for e in cutlass.range_constexpr(2):
                        ur[nt * 2 + e] = bet * (i2f(vw[e]) - ur[nt * 2 + e])
                for j in cutlass.range_constexpr(TOKENS):
                    for c in cutlass.range_constexpr(4):
                        uj = cute.arch.shuffle_sync(ur[c], j * 4 + tig)
                        ur[c] = ur[c] - amr[j] * uj
                        orr[c] = orr[c] + qkr[j] * uj
                ss = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(4):
                    orr[c] = bf16r(orr[c])
                    ss = ss + orr[c] * orr[c]
                ss = ss + cute.arch.shuffle_sync_bfly(ss, 1)
                ss = ss + cute.arch.shuffle_sync_bfly(ss, 2)
                if tig == 0:
                    sts_f32(SMS + (M_RED + warp_idx * 8 + gid) * 4, ss)
                # z of this request (threads >= 128): first try was issued with the gather
                if isq == 1:
                    while tag_of(gb0) != tag:
                        wz2 = ldg_v4_rlx(aB)
                        gb0 = wz2[0]
                        gb1 = wz2[1]
                        gb2 = wz2[2]
                        gb3 = wz2[3]
                    zb = cute.make_rmem_tensor(cute.make_layout(TOKENS), cutlass.Float32)
                    zb[0] = bf_hi(gb0)
                    zb[1] = bf_lo(gb1)
                    zb[2] = bf_hi(gb1)
                    zb[3] = bf_lo(gb2)
                    zb[4] = bf_hi(gb2)
                    zb[5] = bf_lo(gb3)
                    zb[6] = bf_hi(gb3)
                    for t in cutlass.range_constexpr(TOKENS):
                        sts_f32(SB + O_ZS + (t * VSS + gch) * 4, zb[t])
                cute.arch.barrier()
                tot = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    tot = tot + lds_f32(SMS + (M_RED + w * 8 + tt7) * 4)
                rstd = cute.math.rsqrt(tot * cutlass.Float32(1.0 / HEAD_DIM) + eps)
                if gid < TOKENS:
                    row = vi * TOKENS + gid
                    for nt in cutlass.range_constexpr(2):
                        vcb = warp_idx * 16 + nt * 8 + tig * 2
                        yy = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Float32)
                        zw = lds_v2(SB + O_ZS + (gid * VSS + vcb) * 4)
                        for e in cutlass.range_constexpr(2):
                            zz = i2f(zw[e])
                            yv_ = orr[nt * 2 + e] * rstd * nwr[nt * 2 + e] * silu_tanh(zz)
                            yy[e] = bf16r(yv_) * inv_f8s
                        stg_u16(
                            g_y + cutlass.Int64(row * OUT_DIM + hv * 128 + vcb),
                            e4m3x2(yy[0], yy[1]),
                        )
                # U rows -> old_u, after y (off the norm path)
                if gid < TOKENS:
                    for nt in cutlass.range_constexpr(2):
                        vcb = warp_idx * 16 + nt * 8 + tig * 2
                        stg_u32(
                            g_ou
                            + cutlass.Int64(
                                (
                                    (((vs * 2 + wbuf) * HISTORY + woff + gid) * V_HEADS + hv) * 128
                                    + vcb
                                )
                                * 2
                            ),
                            pack_bf16(ur[nt * 2], ur[nt * 2 + 1]),
                        )
                # normalised k rows -> old_k (heads 0 / 4 only), after y (off the pass-2 path)
                if hv % 4 == 0:
                    if tidx < 32 * TOKENS:
                        kt = tidx // 32
                        kc = (tidx % 32) * 4
                        rkk = cute.math.rsqrt(
                            lds_f32(GRB + (kt * KGS + kt) * 4) + cutlass.Float32(1e-6)
                        )
                        kw0 = lds_u32(SB + O_XA + kt * RS + kc * 2)
                        kw1 = lds_u32(SB + O_XA + kt * RS + kc * 2 + 4)
                        kaddr2 = g_ok + cutlass.Int64(
                            ((((vs * 2 + wbuf) * HISTORY + woff + kt) * K_HEADS + hk) * 128 + kc)
                            * 2
                        )
                        stg_2u32(
                            kaddr2,
                            pack_bf16(bf_lo(kw0) * rkk, bf_hi(kw0) * rkk),
                            pack_bf16(bf_lo(kw1) * rkk, bf_hi(kw1) * rkk),
                        )
            else:
                # padded request: its y rows are zero
                if tidx < 56:
                    za = g_y + cutlass.Int64(
                        (vi * TOKENS + tidx // 8) * OUT_DIM + hv * 128 + (tidx % 8) * 16
                    )
                    stg_v4(
                        za, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)
                    )

        # outstanding bulk async-groups must have read their SMEM before the CTA exits (the kernel's completion
        # covers the global writes themselves)
        if tidx % 128 == 0:
            bulk_wait_all()
        # ---------------- teardown ----------------
        # (no teardown cluster barrier: every DSMEM write into this CTA is covered by its own receive waits, and every
        # peer that signals xfree into it is waited on before this CTA's pushes, so nothing reaches it after it exits)
        cute.arch.barrier()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------
_COMPILED = {}
_SCRATCH = {}


def _device_index(device) -> int:
    device = torch.device(device)
    return torch.cuda.current_device() if device.index is None else device.index


def prepare_qwen3_5_gdn_verify_chain(device) -> dict:
    """Allocate (once per device) the zeroed launch-ticket counters and the cross-CTA hand-off buffer.

    Call before any CUDA-graph capture that launches the kernel (e.g. at module construction): both buffers are
    shared by every launch on the device and must live outside a graph's memory pool.
    """
    index = _device_index(device)
    if index not in _SCRATCH:
        dev = torch.device("cuda", index)
        _SCRATCH[index] = dict(
            tickets=torch.zeros(
                (MAX_REQUESTS + 1) * TICKET_CTAS * 32, dtype=torch.int32, device=dev
            ),
            handoff=torch.zeros(
                MAX_REQUESTS * QKVZ_TILES * 128 * 4 + MAX_REQUESTS * V_HEADS * V_HEADS * BAW * 4,
                dtype=torch.int32,
                device=dev,
            ),
        )
    return _SCRATCH[index]


def _scratch(device) -> dict:
    index = _device_index(device)
    if index not in _SCRATCH and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("call prepare_qwen3_5_gdn_verify_chain() before capturing a CUDA graph")
    return prepare_qwen3_5_gdn_verify_chain(device)


def _slot_view(t):
    """1-D view from t's base pointer over all its slots (slot stride t.stride(0), each slot contiguous).

    Never .reshape(-1) / .contiguous(): on a strided pool view they would copy and lose the writes.
    """
    span = (t.shape[0] - 1) * t.stride(0) + t[0].numel()
    return t.as_strided((span,), (1,))


def _fd(t):
    return from_dlpack(t, assumed_align=16)


def _is(t: Optional[torch.Tensor], dtype: torch.dtype, shape: Optional[tuple] = None) -> bool:
    """T is a contiguous, 16-byte aligned tensor of this dtype (and shape)."""
    return (
        t is not None
        and t.dtype == dtype
        and (shape is None or tuple(t.shape) == shape)
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


def _is_slot_pool(t: Optional[torch.Tensor], slot_shape: tuple) -> bool:
    """T is a bf16 pool [slots, *slot_shape]: each slot packed, 16-byte aligned slots."""
    packed = tuple(math.prod(slot_shape[i + 1 :]) for i in range(len(slot_shape)))
    return (
        t is not None
        and t.dtype == torch.bfloat16
        and t.dim() == len(slot_shape) + 1
        and tuple(t.shape[1:]) == slot_shape
        and tuple(t.stride()[1:]) == packed
        and (t.stride(0) * t.element_size()) % 16 == 0
        and t.data_ptr() % 16 == 0
    )


def qwen3_5_gdn_verify_chain_supported(
    x_fp8: torch.Tensor,
    x_bf16: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    w_qkvz: torch.Tensor,
    w_qkvz_scale: Optional[torch.Tensor],
    w_ba: torch.Tensor,
    conv_w: torch.Tensor,
    conv_state: Optional[torch.Tensor],
    conv_window: Optional[torch.Tensor],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    fp8_scale: Optional[torch.Tensor],
    state: Optional[torch.Tensor],
    slots: torch.Tensor,
    old_u: Optional[torch.Tensor],
    old_k: Optional[torch.Tensor],
    old_g: Optional[torch.Tensor],
    cache_buf_idx: Optional[torch.Tensor],
    prev_acc: Optional[torch.Tensor],
    rows_per_request: int,
    qkvz_bias: Optional[torch.Tensor] = None,
    ba_bias: Optional[torch.Tensor] = None,
    conv_bias: Optional[torch.Tensor] = None,
) -> bool:
    """Whether qwen3_5_gdn_verify_chain() takes these operands (same names and order; host-side checks only).

    The kernel's shape specialization (TOKENS rows per request, 1..MAX_REQUESTS requests, the layer shapes above),
    dtypes, layouts and 16-byte alignment, no biases, and at most MAX_SLOTS cache slots.
    """
    if rows_per_request != TOKENS or x_fp8 is None or x_fp8.dim() != 2:
        return False
    rows = x_fp8.shape[0]
    if rows % TOKENS or not TOKENS <= rows <= TOKENS * MAX_REQUESTS:
        return False
    requests = rows // TOKENS
    f32, bf16, e4m3, i32 = torch.float32, torch.bfloat16, torch.float8_e4m3fn, torch.int32
    return (
        qkvz_bias is None
        and ba_bias is None
        and conv_bias is None
        and _is(x_fp8, e4m3, (rows, HIDDEN))
        and _is(x_bf16, bf16, (rows, HIDDEN))
        and _is(w_qkvz, e4m3, (QKVZ_DIM, HIDDEN))
        and _is(w_ba, bf16, (BA_DIM, HIDDEN))
        and _is(conv_w, bf16, (CONV_DIM, CONV_WIDTH))
        and all(_is(s, f32) and s.numel() == 1 for s in (x_scale, w_qkvz_scale, fp8_scale))
        and _is(A_log, f32, (V_HEADS,))
        and _is(dt_bias, f32, (V_HEADS,))
        and _is(norm_w, bf16, (HEAD_DIM,))
        and _is_slot_pool(state, (V_HEADS, HEAD_DIM, HEAD_DIM))
        and _is_slot_pool(conv_state, (CONV_DIM, CONV_WIDTH - 1))
        and _is(conv_window, bf16)
        and conv_window.dim() == 4
        and tuple(conv_window.shape[1:]) == (TOKENS, CONV_DIM, CONV_WIDTH - 1)
        and conv_window.shape[0] >= requests
        and _is(slots, i32)
        and slots.dim() == 1
        and slots.shape[0] >= requests
        and _is(old_u, bf16)
        and tuple(old_u.shape[1:]) == (2, HISTORY, V_HEADS, HEAD_DIM)
        and _is(old_k, bf16)
        and tuple(old_k.shape[1:]) == (2, HISTORY, K_HEADS, HEAD_DIM)
        and _is(old_g, f32)
        and tuple(old_g.shape[1:]) == (2, V_HEADS, HISTORY)
        and all(
            _is(t, i32) and t.dim() == 1 and t.numel() <= MAX_SLOTS
            for t in (cache_buf_idx, prev_acc)
        )
    )


@torch.no_grad()
def qwen3_5_gdn_verify_chain(
    x_fp8: torch.Tensor,
    x_bf16: torch.Tensor,
    x_scale: torch.Tensor,
    w_qkvz: torch.Tensor,
    w_qkvz_scale: torch.Tensor,
    w_ba: torch.Tensor,
    conv_w: torch.Tensor,
    conv_state: torch.Tensor,
    conv_window: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    fp8_scale: torch.Tensor,
    eps: float,
    state: torch.Tensor,
    slots: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_g: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_acc: torch.Tensor,
    commit: int,
    y: torch.Tensor,
) -> torch.Tensor:
    """Run the verify chain of one decode-only MTP target-verify step; returns y.

    Operands must satisfy qwen3_5_gdn_verify_chain_supported(); R = 7 N rows of N requests.

    Args:
        x_fp8: [R, 4096] E4M3, the layer input quantized with x_scale (the PDL producer's output).
        x_bf16: [R, 4096] bf16, the same layer input.
        x_scale, w_qkvz_scale: float32 scalars, the static per-tensor scales of in_proj_qkvz.
        w_qkvz: [2560, 4096] E4M3 in_proj_qkvz weight (rows q | k | v | z).
        w_ba: [16, 4096] bf16 in_proj_ba weight (rows b | a).
        conv_w: [1536, 4] bf16 conv1d weight (no bias).
        conv_state: [slots, 1536, 3] bf16 pool (slot stride a multiple of 16 bytes); updated per request slot.
        conv_window: [>= N, 7, 1536, 3] bf16, written: the conv window after each row of request i at index i.
        A_log, dt_bias: [8] float32.
        norm_w: [128] bf16 gated RMSNorm weight; fp8_scale: float32 scalar out_proj input scale; eps: norm epsilon.
        state: [slots, 8, 128, 128] bf16 recurrent-state pool (slot stride a multiple of 16 bytes).
        slots: [>= N] int32 cache slot per request (< 64; negative: padded request, its y rows are zeroed).
        old_u, old_k: [slots, 2, 16, 8 | 2, 128] bf16 replay history; old_g: [slots, 2, 8, 16] float32.
        cache_buf_idx, prev_acc: [slots <= 64] int32 history buffer and accepted-token count per slot (read only).
        commit: nonzero folds the history into state when it would overflow.
        y: [R, 1024] E4M3 output (out_proj input).
    """
    (
        x_fp8,
        x_bf16,
        x_scale,
        w_qkvz,
        w_qkvz_scale,
        w_ba,
        conv_w,
        conv_state,
        conv_window,
        A_log,
        dt_bias,
        norm_w,
        fp8_scale,
        state,
        slots,
        old_u,
        old_k,
        old_g,
        cache_buf_idx,
        prev_acc,
        y,
    ) = (
        t.detach()
        for t in (
            x_fp8,
            x_bf16,
            x_scale,
            w_qkvz,
            w_qkvz_scale,
            w_ba,
            conv_w,
            conv_state,
            conv_window,
            A_log,
            dt_bias,
            norm_w,
            fp8_scale,
            state,
            slots,
            old_u,
            old_k,
            old_g,
            cache_buf_idx,
            prev_acc,
            y,
        )
    )
    assert tuple(state.stride()[1:]) == (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1), state.stride()
    assert tuple(conv_state.stride()[1:]) == (CONV_WIDTH - 1, 1), conv_state.stride()
    sts_b = state.stride(0) * state.element_size()
    css_b = conv_state.stride(0) * conv_state.element_size()
    assert sts_b % 16 == 0 and css_b % 16 == 0, (sts_b, css_b)
    N = x_fp8.shape[0] // TOKENS
    sc = _scratch(x_fp8.device)
    args = [
        _fd(w_qkvz.unsqueeze(2)),
        _fd(x_fp8.unsqueeze(2)),
        _fd(x_bf16.reshape(-1)),
        _fd(w_ba.reshape(-1)),
        _fd(conv_w.reshape(-1)),
        _fd(_slot_view(conv_state)),
        _fd(conv_window.reshape(-1)),
        _fd(A_log.reshape(-1)),
        _fd(dt_bias.reshape(-1)),
        _fd(norm_w.reshape(-1)),
        _fd(fp8_scale.reshape(-1)),
        _fd(x_scale.reshape(-1)),
        _fd(w_qkvz_scale.reshape(-1)),
        _fd(_slot_view(state)),
        _fd(slots.reshape(-1)),
        _fd(old_u.reshape(-1)),
        _fd(old_k.reshape(-1)),
        _fd(old_g.reshape(-1)),
        _fd(cache_buf_idx.reshape(-1)),
        _fd(prev_acc.reshape(-1)),
        _fd(y.view(torch.uint8).reshape(-1)),
        _fd(sc["tickets"]),
        _fd(sc["handoff"]),
    ]
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    epsv = cutlass.Float32(float(eps))
    sts = cutlass.Int64(sts_b)
    css = cutlass.Int64(css_b)
    cm = cutlass.Int32(int(commit))
    if N not in _COMPILED:
        _COMPILED[N] = cute.compile(
            _GdnVerifyChain(N), *args, epsv, cm, sts, css, stream, options=_compile_options(N)
        )
    _COMPILED[N](*args, epsv, cm, sts, css, stream)
    return y
