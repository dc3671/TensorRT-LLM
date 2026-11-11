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
"""The {GPU} token in remote_cache_config.ucx_devices (the remote level's NIXL devices). No GPU needed."""

import os
import unittest
from unittest import mock

from tensorrt_llm._torch.pyexecutor.kv_cache import remote_donor
from tensorrt_llm._torch.pyexecutor.kv_cache.remote_donor import resolve_ucx_devices


def _visible(value=None):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if value is not None:
        env["CUDA_VISIBLE_DEVICES"] = value
    return mock.patch.dict(os.environ, env, clear=True)


class TestRemoteUcxDevices(unittest.TestCase):
    def test_unset_and_literal_values_pass_through(self):
        self.assertIsNone(resolve_ucx_devices(None, device=0))
        self.assertEqual(
            resolve_ucx_devices("rdma_vf_rail0:1,rdma_vf_rail2:1", device=3),
            "rdma_vf_rail0:1,rdma_vf_rail2:1",
        )

    def test_gpu_token_maps_the_device_through_cuda_visible_devices(self):
        # A TP2 CTX instance on GPUs 2,3: its second rank (CUDA device 1) is GPU 3 and uses rail 3.
        with _visible("2,3"):
            self.assertEqual(
                resolve_ucx_devices("rdma_vf_rail{GPU}:1", device=1), "rdma_vf_rail3:1"
            )
        with _visible():
            self.assertEqual(
                resolve_ucx_devices("rdma_vf_rail{GPU}:1", device=2), "rdma_vf_rail2:1"
            )

    def test_gpu_token_maps_a_uuid_through_the_pci_bus_id(self):
        with (
            _visible("GPU-1a2b,GPU-3c4d"),
            mock.patch.object(remote_donor, "_nvml_gpu_index", return_value=5) as nvml,
        ):
            self.assertEqual(
                resolve_ucx_devices("rdma_vf_rail{GPU}:1", device=1), "rdma_vf_rail5:1"
            )
        nvml.assert_called_once_with(1)

    def test_unresolvable_tokens_raise(self):
        cases = [
            ("rdma_vf_rail{GPU}:1", "0"),  # device 1 not visible
            ("rdma_vf_rail{RANK}:1", None),
        ]
        for devices, visible in cases:
            with self.subTest(devices=devices, visible=visible), _visible(visible):
                with self.assertRaises(ValueError):
                    resolve_ucx_devices(devices, device=1)


if __name__ == "__main__":
    unittest.main()
