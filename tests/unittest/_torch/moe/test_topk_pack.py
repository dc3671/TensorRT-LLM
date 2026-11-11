# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pack_topk_bf16 (one launch) against the torch expression of the same packed words."""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.topk_pack import pack_topk_bf16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (ids.to(torch.int32) << 16) | weights.to(torch.bfloat16).contiguous().view(
        torch.int16
    ).to(torch.int32)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 10), (7, 8), (56, 10), (300, 8)])
def test_pack_topk_bf16_matches_torch(ids_dtype, w_dtype, shape):
    torch.manual_seed(0)
    ids = torch.randint(0, 512, shape, dtype=ids_dtype, device="cuda")
    weights = torch.rand(shape, dtype=torch.float32, device="cuda")
    # Rounding edges (ties, denormals, zeros, signs, infinities) next to the random weights.
    edge = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.0 + 2.0**-8,
            1.0 + 3 * 2.0**-9,
            1e-40,
            float("inf"),
            -float("inf"),
        ],
        device="cuda",
    )
    flat = weights.view(-1)
    n = min(edge.numel(), flat.numel())
    flat[:n] = edge[:n]
    weights = weights.to(w_dtype)
    torch.testing.assert_close(
        pack_topk_bf16(ids, weights), _reference(ids, weights), rtol=0, atol=0
    )
