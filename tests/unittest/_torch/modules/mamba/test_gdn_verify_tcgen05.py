# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gdn_verify_tcgen05 (the tcgen05 GDN MTP target-verify kernel) against trtllm::gdn_verify with FP8 output.

The two differ only in the MMA accumulation order and the gated norm's sum order: y agrees up to FP8 rounding flips,
the committed state and the histories to bf16 rounding. A replay from the same inputs is bitwise identical.
"""

import pytest
import torch

from tensorrt_llm._torch.kernels.blackwell.gdn_verify_tcgen05 import (
    compile_gdn_verify_tcgen05,
    gdn_verify_tcgen05,
    gdn_verify_tcgen05_supported,
)

skip_no_sm100f = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)
_DIM, _HIST, _SLOTS = 128, 16, 256
_T = 7


def _verify_case(heads, n, t, pnat, seed=1000):
    """Per-rank shapes of a 16 k-head / 64 v-head GDN layer: heads = (k heads, v heads)."""
    K = V = _DIM
    H, HV = heads
    g = torch.Generator(device="cuda").manual_seed(seed)
    cd = 2 * H * K + HV * V
    proj = (torch.randn(n * t, cd + HV * V, generator=g, device="cuda") * 0.7).to(torch.bfloat16)
    ba = torch.randn(n * t, 2 * HV, generator=g, device="cuda").to(torch.bfloat16)
    st = dict(
        a_log=torch.log(torch.rand(HV, generator=g, device="cuda") * 7 + 1).float(),
        dt_bias=(torch.rand(HV, generator=g, device="cuda") * 2 - 1).float(),
        norm_w=(torch.rand(V, generator=g, device="cuda") + 0.5).to(torch.bfloat16),
        fp8_scale=torch.tensor(0.02, dtype=torch.float32, device="cuda"),
        state_indices=torch.randperm(_SLOTS, generator=g, device="cuda")[:n].to(torch.int32),
        pnat=torch.full((_SLOTS,), pnat, dtype=torch.int32, device="cuda"),
        buf=torch.zeros(_SLOTS, dtype=torch.int32, device="cuda"),
    )
    st["buf"][2::3] = 1
    kh = torch.randn(_SLOTS, 2, _HIST, H, K, generator=g, device="cuda")
    dec = torch.rand(_SLOTS, 2, HV, _HIST, generator=g, device="cuda") * 0.49 + 0.01
    mut = dict(
        ssm=(torch.randn(_SLOTS, HV, V, K, generator=g, device="cuda") * 0.05).to(torch.bfloat16),
        old_k=(kh / kh.norm(dim=-1, keepdim=True)).to(torch.bfloat16),
        old_u=(torch.randn(_SLOTS, 2, _HIST, HV, V, generator=g, device="cuda") * 0.5).to(
            torch.bfloat16
        ),
        old_g=(-torch.cumsum(dec, dim=-1)).float().contiguous(),
    )
    return HV, cd, proj, ba, st, mut


def _verify_op(op, HV, cd, n, t, proj, ba, st, mut):
    """Run op (trtllm::gdn_verify's argument list, FP8 output) on clones of the mutable state."""
    m = {k: v.clone() for k, v in mut.items()}
    y = op(
        proj[:, :cd],
        proj[:, cd:].view(n * t, HV, _DIM),
        ba[:, HV:],
        ba[:, :HV],
        st["a_log"],
        st["dt_bias"],
        st["norm_w"],
        st["fp8_scale"],
        1e-6,
        m["ssm"],
        st["state_indices"],
        m["old_u"],
        m["old_k"],
        m["old_g"],
        st["buf"],
        st["pnat"],
    )
    return y, m


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize(
    "T,HV,HK,fp8_out,expected",
    [
        (7, 8, 2, True, True),
        (7, 64, 16, True, True),
        (2, 4, 1, True, True),
        (8, 8, 2, True, True),
        (7, 8, 2, False, False),
        (1, 8, 2, True, False),
        (9, 8, 2, True, False),
        (7, 8, 3, True, False),
        (7, 0, 2, True, False),
    ],
)
def test_gdn_verify_tcgen05_supported(T, HV, HK, fp8_out, expected):
    assert gdn_verify_tcgen05_supported(T, HV, HK, fp8_out) is expected


@skip_no_sm100f
@pytest.mark.parametrize(
    "heads,n",
    [
        ((2, 8), 1),
        ((2, 8), 16),
        ((2, 8), 64),
        ((2, 8), 128),
        ((16, 64), 8),
        ((16, 64), 52),
    ],
)
@pytest.mark.parametrize("pnat", [0, 12])
def test_gdn_verify_tcgen05_matches_gdn_verify(heads, n, pnat):
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, _T, pnat)
    y_ref, m_ref = _verify_op(torch.ops.trtllm.gdn_verify, HV, cd, n, _T, proj, ba, st, mut)
    y, m = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    torch.cuda.synchronize()
    assert y.dtype == torch.float8_e4m3fn and y.shape == y_ref.shape
    # identical up to FP8 rounding flips of near-tie values
    assert _rel(y, y_ref) < 5e-2
    assert (y.view(torch.uint8) != y_ref.view(torch.uint8)).float().mean().item() < 0.2
    assert _rel(m["old_u"], m_ref["old_u"]) < 1e-3
    assert _rel(m["old_k"], m_ref["old_k"]) < 1e-2
    assert torch.allclose(m["old_g"], m_ref["old_g"], rtol=1e-5, atol=1e-6)
    assert _rel(m["ssm"], m_ref["ssm"]) < 1e-3
    if pnat + _T > _HIST:
        assert not torch.equal(m["ssm"], mut["ssm"]), "commit expected (history full)"


@skip_no_sm100f
@pytest.mark.parametrize("heads,n", [((2, 8), 128), ((16, 64), 52)])
def test_gdn_verify_tcgen05_replay_is_bitwise(heads, n):
    """Two runs from the same inputs (one committing slot in three requests) give bitwise-identical results."""
    H, HV = heads
    compile_gdn_verify_tcgen05(_T, HV, H, torch.device("cuda"))
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, _T, 0)
    st["pnat"][st["state_indices"][::3].long()] = 12
    y0, m0 = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    y1, m1 = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    torch.cuda.synchronize()
    assert torch.equal(y0.view(torch.uint8), y1.view(torch.uint8))
    for k in m0:
        assert torch.equal(m0[k], m1[k]), k
    assert not torch.equal(m0["ssm"], mut["ssm"]), "commit expected (history full)"


@skip_no_sm100f
@pytest.mark.parametrize("t", [2, 4, 7, 8])
@pytest.mark.parametrize("pnat", ["zero", "offset", "fills", "mixed"])
@pytest.mark.parametrize("n", [16, 200])
def test_gdn_verify_tcgen05_matches_gdn_verify_across_draft_lengths(t, pnat, n):
    """Every admitted draft length, appends at an offset into the history, a history the step fills exactly
    (16 - T), batches mixing appends and commits, and more requests than one CTA wave of 128."""
    heads = (2, 8)
    start = {"zero": 0, "offset": 3, "fills": _HIST - t, "mixed": 0}[pnat]
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, t, start, seed=2000 + t)
    if pnat == "mixed":
        # Every third request commits (its history overflows), the others append at offset 3.
        st["pnat"].fill_(3)
        st["pnat"][st["state_indices"][::3].long()] = 12
    y_ref, m_ref = _verify_op(torch.ops.trtllm.gdn_verify, HV, cd, n, t, proj, ba, st, mut)
    y, m = _verify_op(gdn_verify_tcgen05, HV, cd, n, t, proj, ba, st, mut)
    torch.cuda.synchronize()
    assert y.dtype == torch.float8_e4m3fn and y.shape == y_ref.shape
    assert _rel(y, y_ref) < 5e-2
    assert (y.view(torch.uint8) != y_ref.view(torch.uint8)).float().mean().item() < 0.2
    assert _rel(m["old_u"], m_ref["old_u"]) < 1e-3
    assert _rel(m["old_k"], m_ref["old_k"]) < 1e-2
    assert torch.allclose(m["old_g"], m_ref["old_g"], rtol=1e-5, atol=1e-6)
    assert _rel(m["ssm"], m_ref["ssm"]) < 1e-3
