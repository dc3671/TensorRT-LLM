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
"""The remote-level lender makes all of its memory resident before it publishes it. Linux only, no GPU."""

import ctypes
import errno
import mmap
import sys
import unittest
from unittest import mock

from tensorrt_llm._torch.pyexecutor.kv_cache import remote_lender

_SIZE = 96 << 20


def _resident_bytes(address: int, size: int) -> int:
    """Bytes of [address, address + size) in memory, page by page with mincore(2).

    /proc/self/smaps is not exact here: the kernel may merge the test mapping with a neighbouring anonymous one.
    """
    page = mmap.PAGESIZE
    pages = (size + page - 1) // page
    vec = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(v & 1 for v in vec) * page


@unittest.skipUnless(sys.platform.startswith("linux"), "uses mincore(2)")
class TestPopulate(unittest.TestCase):
    def _region(self):
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | remote_lender._MAP_NORESERVE
        region = mmap.mmap(-1, _SIZE, flags=flags)
        self.addCleanup(region.close)
        return region, ctypes.addressof(ctypes.c_char.from_buffer(region))

    def test_every_page_is_resident(self):
        region, address = self._region()
        self.assertEqual(_resident_bytes(address, _SIZE), 0)
        remote_lender._populate(region, address, _SIZE)
        self.assertEqual(_resident_bytes(address, _SIZE), _SIZE)

    def test_fallback_without_madv_populate_write(self):
        region, address = self._region()
        self.assertEqual(_resident_bytes(address, _SIZE), 0)
        # A kernel without MADV_POPULATE_WRITE (EINVAL) takes the zero-fill path.
        with mock.patch.object(
            remote_lender, "madvise_range", side_effect=OSError(errno.EINVAL, "EINVAL")
        ):
            remote_lender._populate(region, address, _SIZE)
        self.assertEqual(_resident_bytes(address, _SIZE), _SIZE)


if __name__ == "__main__":
    unittest.main()
