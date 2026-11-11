# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen3_5_gdn_verify_chain against the stock target-verify chain.

Reference: in_proj_qkvz as an FP32 GEMM of the dequantized E4M3 operands rounded to BF16 (the low-M GEMM's output),
in_proj_ba as F.linear in BF16, causal_conv1d_update with intermediate windows, then trtllm::gdn_verify. The fused
kernel accumulates the GEMMs in another order and runs the replay on BF16 hi / lo mma.sync, so it is compared within
rounding bounds; it is bitwise deterministic, also under CUDA-graph replay.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.kernels.blackwell.qwen3_5_gdn_verify_chain import (
    BA_DIM,
    CONV_DIM,
    CONV_WIDTH,
    HEAD_DIM,
    HIDDEN,
    HISTORY,
    K_HEADS,
    MAX_REQUESTS,
    MAX_SLOTS,
    OUT_DIM,
    QKVZ_DIM,
    TAG_LAUNCH_BITS,
    TOKENS,
    V_HEADS,
    handoff_tag,
    prepare_qwen3_5_gdn_verify_chain,
    qwen3_5_gdn_verify_chain,
    qwen3_5_gdn_verify_chain_supported,
)

skip_no_sm100f = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)

EPS = 1e-6
E4M3 = torch.float8_e4m3fn
MUTABLE = ("conv_state", "conv_window", "state", "old_u", "old_k", "old_g")


def _weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    def scalar(v):
        return torch.tensor([v], dtype=torch.float32, device="cuda")

    return SimpleNamespace(
        x_scale=scalar(0.05),
        # E4M3 weights of std 8 at scale 1 / 400: in_proj_qkvz outputs of std ~1.3
        w_qkvz=(randn(QKVZ_DIM, HIDDEN) * 8).to(E4M3),
        w_qkvz_scale=scalar(0.0025),
        w_ba=(randn(BA_DIM, HIDDEN) * 0.02).to(torch.bfloat16),
        conv_w=(randn(CONV_DIM, CONV_WIDTH) * 0.4).to(torch.bfloat16),
        A_log=torch.log(torch.rand(V_HEADS, generator=g, device="cuda") * 7 + 1).float(),
        dt_bias=(torch.rand(V_HEADS, generator=g, device="cuda") * 2 - 1).float(),
        norm_w=(torch.rand(HEAD_DIM, generator=g, device="cuda") + 0.5).to(torch.bfloat16),
        fp8_scale=scalar(0.02),
    )


def _step(w, n, pnat, seed):
    """Activations, slot tables and the initial cache buffers of one verify step of n requests."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    x_bf16 = randn(n * TOKENS, HIDDEN).to(torch.bfloat16)
    buf = torch.zeros(MAX_SLOTS, dtype=torch.int32, device="cuda")
    buf[1::2] = 1
    kh = randn(MAX_SLOTS, 2, HISTORY, K_HEADS, HEAD_DIM)
    dec = torch.rand(MAX_SLOTS, 2, V_HEADS, HISTORY, generator=g, device="cuda") * 0.49 + 0.01
    init = dict(
        conv_state=randn(MAX_SLOTS, CONV_DIM, CONV_WIDTH - 1).to(torch.bfloat16),
        # rows past n are not written
        conv_window=randn(MAX_REQUESTS, TOKENS, CONV_DIM, CONV_WIDTH - 1).to(torch.bfloat16),
        state=(randn(MAX_SLOTS, V_HEADS, HEAD_DIM, HEAD_DIM) * 0.05).to(torch.bfloat16),
        old_u=(randn(MAX_SLOTS, 2, HISTORY, V_HEADS, HEAD_DIM) * 0.5).to(torch.bfloat16),
        old_k=(kh / kh.norm(dim=-1, keepdim=True)).to(torch.bfloat16),
        old_g=(-torch.cumsum(dec, dim=-1)).float().contiguous(),
    )
    return SimpleNamespace(
        n=n,
        pnat=pnat,
        x_bf16=x_bf16,
        x_fp8=(x_bf16.float() / w.x_scale).to(E4M3),
        slots=torch.randperm(MAX_SLOTS, generator=g, device="cuda")[:n].to(torch.int32),
        prev_acc=torch.full((MAX_SLOTS,), pnat, dtype=torch.int32, device="cuda"),
        cache_buf_idx=buf,
        init=init,
    )


def _clone(init):
    return {k: v.clone() for k, v in init.items()}


def _fused(w, s, m):
    y = torch.empty(s.n * TOKENS, OUT_DIM, dtype=E4M3, device="cuda")
    return qwen3_5_gdn_verify_chain(
        s.x_fp8,
        s.x_bf16,
        w.x_scale,
        w.w_qkvz,
        w.w_qkvz_scale,
        w.w_ba,
        w.conv_w,
        m["conv_state"],
        m["conv_window"],
        w.A_log,
        w.dt_bias,
        w.norm_w,
        w.fp8_scale,
        EPS,
        m["state"],
        s.slots,
        m["old_u"],
        m["old_k"],
        m["old_g"],
        s.cache_buf_idx,
        s.prev_acc,
        1,
        y,
    )


def _stock(w, s, m):
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update

    n = s.n
    qkvz = ((s.x_fp8.float() @ w.w_qkvz.float().t()) * (w.x_scale * w.w_qkvz_scale)).to(
        torch.bfloat16
    )
    ba = F.linear(s.x_bf16, w.w_ba)
    conv_out = causal_conv1d_update(
        qkvz[:, :CONV_DIM].reshape(n, TOKENS, CONV_DIM).transpose(1, 2),
        m["conv_state"],
        w.conv_w,
        None,
        "silu",
        conv_state_indices=s.slots,
        intermediate_conv_window=m["conv_window"],
        intermediate_state_indices=torch.arange(n, dtype=torch.int32, device="cuda"),
    )
    return torch.ops.trtllm.gdn_verify(
        conv_out.transpose(1, 2).reshape(n * TOKENS, CONV_DIM),
        qkvz[:, CONV_DIM:].view(n * TOKENS, V_HEADS, HEAD_DIM),
        ba[:, V_HEADS:],
        ba[:, :V_HEADS],
        w.A_log,
        w.dt_bias,
        w.norm_w,
        w.fp8_scale,
        EPS,
        m["state"],
        s.slots,
        m["old_u"],
        m["old_k"],
        m["old_g"],
        s.cache_buf_idx,
        s.prev_acc,
    )


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.fixture(scope="module")
def weights():
    prepare_qwen3_5_gdn_verify_chain(torch.device("cuda", torch.cuda.current_device()))
    return _weights()


@skip_no_sm100f
@pytest.mark.parametrize("pnat", [0, 9, 12])
@pytest.mark.parametrize("n", [1, 2, 5, 8, 16])
def test_matches_stock_chain(weights, n, pnat):
    w = weights
    s = _step(w, n, pnat, seed=100 * n + pnat)
    m_ref = _clone(s.init)
    y_ref = _stock(w, s, m_ref)
    m = _clone(s.init)
    y = _fused(w, s, m)
    torch.cuda.synchronize()

    # E4M3 output: equal up to FP8 rounding flips of near-tie values (the bound of the stock verify's own test)
    assert _rel(y, y_ref) < 5e-2
    assert (y.view(torch.uint8) != y_ref.view(torch.uint8)).float().mean().item() < 0.2

    touched = s.slots.long()
    kept = torch.ones(MAX_SLOTS, dtype=torch.bool, device="cuda")
    kept[touched] = False
    for name in ("conv_state", "state", "old_u", "old_k", "old_g"):
        assert torch.equal(m[name][kept], s.init[name][kept]), (
            f"{name}: write outside the request slots"
        )
    assert torch.equal(m["conv_window"][n:], s.init["conv_window"][n:])

    # Conv state and windows copy BF16 in_proj_qkvz outputs: they differ only where the FP32 sums of the two
    # GEMMs round to neighbouring BF16 values.
    for got, ref in (
        (m["conv_state"][touched], m_ref["conv_state"][touched]),
        (m["conv_window"][:n], m_ref["conv_window"][:n]),
    ):
        assert _rel(got, ref) < 1e-3
        assert (got != ref).float().mean().item() < 1e-2

    if pnat + TOKENS > HISTORY:
        assert not torch.equal(m["state"][touched], s.init["state"][touched]), "commit expected"
        # BF16 checkpoint: within one BF16 rounding step (2^-9) of the stock fold
        assert _rel(m["state"][touched], m_ref["state"][touched]) < 2e-3
    else:
        assert torch.equal(m["state"], s.init["state"])
    assert _rel(m["old_u"][touched], m_ref["old_u"][touched]) < 2e-3
    assert _rel(m["old_k"][touched], m_ref["old_k"][touched]) < 1e-2
    assert _rel(m["old_g"][touched], m_ref["old_g"][touched]) < 1e-3


@skip_no_sm100f
@pytest.mark.parametrize("n", [1, 8, 16])
def test_deterministic(weights, n):
    w = weights
    s = _step(w, n, 12, seed=7 + n)
    m0, m1 = _clone(s.init), _clone(s.init)
    y0 = _fused(w, s, m0)
    y1 = _fused(w, s, m1)
    torch.cuda.synchronize()
    assert torch.equal(y0.view(torch.uint8), y1.view(torch.uint8))
    for name in MUTABLE:
        assert torch.equal(m0[name], m1[name]), name


@skip_no_sm100f
@pytest.mark.parametrize("n", [5, 16])
def test_cuda_graph_replay(weights, n):
    """Captured once, replayed on new inputs: equal to eager launches (the hand-off tags advance on the device)."""
    w = weights
    static = _step(w, n, 12, seed=11)
    m_static = _clone(static.init)
    _fused(w, static, _clone(static.init))  # compiles outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = _fused(w, static, m_static)

    for seed in (12, 13):
        s = _step(w, n, 12, seed=seed)
        for name in ("x_fp8", "x_bf16", "slots", "prev_acc", "cache_buf_idx"):
            getattr(static, name).copy_(getattr(s, name))
        for name in MUTABLE:
            m_static[name].copy_(s.init[name])
        graph.replay()
        m_eager = _clone(s.init)
        y_eager = _fused(w, s, m_eager)
        torch.cuda.synchronize()
        assert torch.equal(y_static.view(torch.uint8), y_eager.view(torch.uint8))
        for name in MUTABLE:
            assert torch.equal(m_static[name], m_eager[name]), name


def _as_int32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= 1 << 31 else x


def test_handoff_tag_wraps_without_reaching_zero():
    """The tag is (17 * launch + N) % 65535 + 1 for the first 2^TAG_LAUNCH_BITS launches; past them, and past the
    Int32 wrap of the ticket counter, it stays in [1, 65535] and differs between consecutive launches of one N."""
    launches = list(range(4096)) + list(range((1 << TAG_LAUNCH_BITS) - 4096, 1 << TAG_LAUNCH_BITS))
    for launch in launches:
        for n in (1, 7, MAX_REQUESTS):
            assert handoff_tag(8 * launch, n) == (17 * launch + n) % 65535 + 1
    for start in (8 << TAG_LAUNCH_BITS, 1 << 31, 1 << 32):
        prev = {}
        for launch in range(-8, 8):
            ticket = _as_int32(start + 8 * launch)
            for n in range(1, MAX_REQUESTS + 1):
                tag = handoff_tag(ticket, n)
                assert 1 <= tag <= 65535
                assert prev.get(n) != tag
                prev[n] = tag


@skip_no_sm100f
@pytest.mark.parametrize("ticket", [8 * ((1 << TAG_LAUNCH_BITS) - 2), (1 << 31) - 16])
def test_ticket_counter_wrap(weights, ticket):
    """Launches across the tag's launch-index wrap and the Int32 wrap of the ticket counters: same outputs."""
    w = weights
    s = _step(w, 8, 12, seed=21)
    m_ref = _clone(s.init)
    y_ref = _fused(w, s, m_ref)
    scratch = prepare_qwen3_5_gdn_verify_chain(torch.device("cuda", torch.cuda.current_device()))
    torch.cuda.synchronize()
    scratch["tickets"].fill_(ticket)
    scratch["handoff"].zero_()
    for _ in range(4):
        m = _clone(s.init)
        y = _fused(w, s, m)
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.uint8), y_ref.view(torch.uint8))
        for name in MUTABLE:
            assert torch.equal(m[name], m_ref[name]), name


def _operands(n=2, pool=8, **overrides):
    """CPU operands of the declared shapes (contents unused by the predicate)."""
    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    ops = dict(
        x_fp8=torch.empty(n * TOKENS, HIDDEN, dtype=E4M3),
        x_bf16=torch.empty(n * TOKENS, HIDDEN, dtype=bf16),
        x_scale=torch.ones(1, dtype=f32),
        w_qkvz=torch.empty(QKVZ_DIM, HIDDEN, dtype=E4M3),
        w_qkvz_scale=torch.ones(1, dtype=f32),
        w_ba=torch.empty(BA_DIM, HIDDEN, dtype=bf16),
        conv_w=torch.empty(CONV_DIM, CONV_WIDTH, dtype=bf16),
        conv_state=torch.empty(pool, CONV_DIM, CONV_WIDTH - 1, dtype=bf16),
        conv_window=torch.empty(MAX_REQUESTS, TOKENS, CONV_DIM, CONV_WIDTH - 1, dtype=bf16),
        A_log=torch.empty(V_HEADS, dtype=f32),
        dt_bias=torch.empty(V_HEADS, dtype=f32),
        norm_w=torch.empty(HEAD_DIM, dtype=bf16),
        fp8_scale=torch.ones(1, dtype=f32),
        state=torch.empty(pool, V_HEADS, HEAD_DIM, HEAD_DIM, dtype=bf16),
        slots=torch.zeros(n, dtype=i32),
        old_u=torch.empty(pool, 2, HISTORY, V_HEADS, HEAD_DIM, dtype=bf16),
        old_k=torch.empty(pool, 2, HISTORY, K_HEADS, HEAD_DIM, dtype=bf16),
        old_g=torch.empty(pool, 2, V_HEADS, HISTORY, dtype=f32),
        cache_buf_idx=torch.zeros(pool, dtype=i32),
        prev_acc=torch.zeros(pool, dtype=i32),
        rows_per_request=TOKENS,
    )
    ops.update(overrides)
    return ops


def test_supported_predicate():
    ok = qwen3_5_gdn_verify_chain_supported
    bf16, f32 = torch.bfloat16, torch.float32
    pool = 8
    for n in (1, 5, MAX_REQUESTS):
        assert ok(**_operands(n))
    # per-block pool views: slots strided, each slot packed
    assert ok(
        **_operands(state=torch.empty(pool, 2, V_HEADS, HEAD_DIM, HEAD_DIM, dtype=bf16)[:, 0])
    )
    padded = torch.empty(pool, CONV_DIM * 3 + 8, dtype=bf16)[:, : CONV_DIM * 3]
    assert ok(**_operands(conv_state=padded.view(pool, CONV_DIM, 3)))
    assert ok(**_operands(pool=MAX_SLOTS))

    # rows
    assert not ok(**_operands(rows_per_request=TOKENS - 1))
    assert not ok(**_operands(MAX_REQUESTS + 1))
    rows8 = dict(
        x_fp8=torch.empty(8, HIDDEN, dtype=E4M3), x_bf16=torch.empty(8, HIDDEN, dtype=bf16)
    )
    assert not ok(**_operands(**rows8))
    assert not ok(**_operands(x_bf16=torch.empty(3 * TOKENS, HIDDEN, dtype=bf16)))
    assert not ok(**_operands(5, conv_window=torch.empty(4, TOKENS, CONV_DIM, 3, dtype=bf16)))
    assert not ok(**_operands(5, slots=torch.zeros(4, dtype=torch.int32)))
    # heads
    assert not ok(**_operands(state=torch.empty(pool, 4, HEAD_DIM, HEAD_DIM, dtype=bf16)))
    assert not ok(**_operands(old_u=torch.empty(pool, 2, HISTORY, 4, HEAD_DIM, dtype=bf16)))
    assert not ok(**_operands(old_k=torch.empty(pool, 2, HISTORY, 4, HEAD_DIM, dtype=bf16)))
    assert not ok(**_operands(old_g=torch.empty(pool, 2, 4, HISTORY, dtype=f32)))
    assert not ok(**_operands(A_log=torch.empty(2 * V_HEADS, dtype=f32)))
    assert not ok(**_operands(w_qkvz=torch.empty(2 * QKVZ_DIM, HIDDEN, dtype=E4M3)))
    # dtypes
    assert not ok(**_operands(x_bf16=torch.empty(2 * TOKENS, HIDDEN, dtype=torch.float16)))
    assert not ok(**_operands(w_qkvz=torch.empty(QKVZ_DIM, HIDDEN, dtype=bf16)))
    assert not ok(**_operands(state=torch.empty(pool, V_HEADS, HEAD_DIM, HEAD_DIM, dtype=f32)))
    assert not ok(**_operands(old_g=torch.empty(pool, 2, V_HEADS, HISTORY, dtype=bf16)))
    assert not ok(**_operands(slots=torch.zeros(2, dtype=torch.int64)))
    assert not ok(**_operands(fp8_scale=None))
    assert not ok(**_operands(x_scale=None))
    # layout, alignment, slot count, biases
    assert not ok(**_operands(x_fp8=torch.empty(HIDDEN, 2 * TOKENS, dtype=E4M3).t()))
    assert not ok(
        **_operands(
            state=torch.empty(pool, V_HEADS, HEAD_DIM, HEAD_DIM, dtype=bf16).transpose(2, 3)
        )
    )
    shifted = torch.empty(pool * CONV_DIM * 3 + 1, dtype=bf16)[1:].view(pool, CONV_DIM, 3)
    assert not ok(**_operands(conv_state=shifted))
    assert not ok(**_operands(pool=MAX_SLOTS + 1))
    assert not ok(**_operands(conv_bias=torch.empty(CONV_DIM, dtype=bf16)))
    assert not ok(**_operands(qkvz_bias=torch.empty(QKVZ_DIM, dtype=bf16)))
