# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ExpertL2Prefetch: the record kernel keeps the experts with the most token tiles, only registered layers and
forwards in the row range record or prefetch, and the prefetch leaves every tensor unchanged."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.expert_l2_prefetch import PIECE, ExpertL2Prefetch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="cp.async.bulk.prefetch.L2 needs SM90+",
)

NUM_LOCAL = 16
TOP_K = 10


def _selection(prefetch: ExpertL2Prefetch, layer_idx: int) -> list:
    torch.cuda.synchronize()
    return prefetch._device_state(torch.device("cuda")).sel[layer_idx].tolist()


def _backend(num_local: int = NUM_LOCAL) -> SimpleNamespace:
    return SimpleNamespace(
        has_nvfp4=True,
        w3_w1_weight=torch.randint(0, 255, (num_local, 2, PIECE), dtype=torch.uint8, device="cuda"),
        w3_w1_weight_scale=torch.randint(
            0, 255, (num_local, PIECE), dtype=torch.uint8, device="cuda"
        ),
    )


def test_record_keeps_the_experts_with_most_tiles() -> None:
    prefetch = ExpertL2Prefetch(
        num_layers=2, min_rows=2, max_rows=64, top_k=TOP_K, stream=torch.cuda.Stream()
    )
    prefetch.register(0, _backend())
    k = prefetch.num_experts
    # Tiles per expert: 5 -> 3, 2 -> 3, 9 -> 1 (ties: lower id first); the entries past num_ctas do not count.
    cta_batch = torch.tensor([5, 2, 9, 5, 2, 5, 2, 7, 7, 7, 7], dtype=torch.int32, device="cuda")
    num_ctas = torch.tensor([7], dtype=torch.int32, device="cuda")
    prefetch.record(0, cta_batch, num_ctas, NUM_LOCAL, cta_batch, rows=8)
    prefetch.join()
    assert _selection(prefetch, 0) == ([2, 5, 9] + [-1] * k)[:k]

    # Outside the row range, or for an unregistered layer: nothing is recorded.
    other = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    prefetch.record(0, other, num_ctas.new_tensor([2]), NUM_LOCAL, other, rows=1)
    prefetch.record(0, other, num_ctas.new_tensor([2]), NUM_LOCAL, other, rows=65)
    prefetch.record(1, other, num_ctas.new_tensor([2]), NUM_LOCAL, other, rows=8)
    prefetch.join()
    assert _selection(prefetch, 0) == ([2, 5, 9] + [-1] * k)[:k]
    assert _selection(prefetch, 1) == [-1] * k


def test_prefetch_changes_no_tensor_and_joins() -> None:
    prefetch = ExpertL2Prefetch(
        num_layers=1, min_rows=2, max_rows=64, top_k=TOP_K, stream=torch.cuda.Stream()
    )
    backend = _backend()
    weight, scale = backend.w3_w1_weight.clone(), backend.w3_w1_weight_scale.clone()
    prefetch.register(0, backend)
    tiles = torch.arange(NUM_LOCAL, dtype=torch.int32, device="cuda")
    prefetch.record(0, tiles, tiles.new_tensor([NUM_LOCAL]), NUM_LOCAL, tiles, rows=8)
    prefetch.join()
    assert not prefetch._forked
    prefetch.prefetch(0, rows=8)
    assert prefetch._forked  # forked by prefetch() itself
    prefetch.join()
    assert not prefetch._forked
    torch.cuda.synchronize()
    assert torch.equal(backend.w3_w1_weight, weight)
    assert torch.equal(backend.w3_w1_weight_scale, scale)


def test_prefetch_skips_experts_without_nvfp4_scales() -> None:
    """A layer whose experts are not NVFP4 (no FC1 block scales) issues no prefetch and forks nothing."""
    prefetch = ExpertL2Prefetch(
        num_layers=1, min_rows=2, max_rows=64, top_k=TOP_K, stream=torch.cuda.Stream()
    )
    prefetch.register(0, SimpleNamespace(has_nvfp4=False))
    prefetch.prefetch(0, rows=8)
    assert not prefetch._forked
