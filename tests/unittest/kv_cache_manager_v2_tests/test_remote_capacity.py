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
"""The remote level's capacity check, donor failure markers and fabric chunk placement.

No GPU, except TestFabricPlacement (the fabric transport).

check_remote_capacity waits until the expected donors have published (donor.<name>.ready + .meta) or failed
(donor.<name>.failed), then compares what they offer with what the clients claim. It must fail as soon as the
last donor reports with a shortfall, pass as soon as the offer covers the need, and name failed and short donors.
"""

import contextlib
import errno
import gc
import mmap
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tensorrt_llm._torch.pyexecutor.kv_cache import remote_donor
from tensorrt_llm._torch.pyexecutor.kv_cache.remote_donor import (
    FabricDonor,
    NixlDonor,
    check_remote_capacity,
    scan_donors,
)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from remote_test_utils import fabric_available
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

_GIB = 1 << 30


def _publish(directory, name, gib, chunk_gib=1, short_gib=0, host="hostA", extra=()):
    lines = [
        "version 1",
        f"name {name}",
        f"host {host}",
        "pid 1",
        f"chunk_bytes {chunk_gib * _GIB}",
        "transport fabric",
        "numa 0",
        f"segment 0 {gib * _GIB}",
        *extra,
    ]
    if short_gib:
        lines.append(f"short_bytes {short_gib * _GIB}")
    with open(os.path.join(directory, f"donor.{name}.meta"), "w") as f:
        f.write("\n".join(lines) + "\n")
    open(os.path.join(directory, f"donor.{name}.ready"), "w").close()


class TestCapacityCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_capacity_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_scan_counts_whole_chunks_and_failures(self):
        _publish(self.dir, "a", 160)
        _publish(self.dir, "b", 100, short_gib=60)
        donor = NixlDonor(self.dir, 160 * _GIB, chunk_bytes=_GIB, name="c")
        donor.publish_failure(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
        open(os.path.join(self.dir, "claim.a.0"), "w").close()  # claims are not donors
        published, failed = scan_donors(self.dir)
        self.assertEqual(sorted(p["name"] for p in published), ["a", "b"])
        self.assertEqual(sum(p["offered_bytes"] for p in published), 260 * _GIB)
        self.assertEqual([p["short_bytes"] for p in published if p["name"] == "b"], [60 * _GIB])
        self.assertEqual([f["name"] for f in failed], ["c"])
        self.assertEqual(failed[0]["planned_bytes"], 160 * _GIB)
        self.assertIn("OUT_OF_MEMORY", failed[0]["error"])

    def test_passes_when_offer_covers_need(self):
        for i in range(8):
            _publish(self.dir, f"l{i}", 160)
        result = check_remote_capacity(self.dir, 8, 1218 * _GIB, timeout_s=10, log=lambda _: None)
        self.assertEqual(result["offered_bytes"], 1280 * _GIB)

    def test_fails_fast_when_all_donors_reported_short(self):
        # 5 of 8 donors published 160 GiB, 3 failed; the clients claim 6 x 203 GiB.
        for i in range(5):
            _publish(self.dir, f"ok{i}", 160)
        for i in range(3):
            NixlDonor(self.dir, 160 * _GIB, chunk_bytes=_GIB, name=f"bad{i}").publish_failure(
                RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
            )
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                8,
                1218 * _GIB,
                timeout_s=600,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertLess(
            time.monotonic() - t0, 5.0
        )  # no wait for the timeout: every donor has reported
        message = str(ctx.exception)
        self.assertIn("5 of 8 planned donors published 800 GiB, 3 failed", message)
        self.assertIn("claim 1218 GiB", message)
        self.assertIn("bad0", message)
        self.assertIn("OUT_OF_MEMORY", message)

    def test_names_short_donors(self):
        _publish(self.dir, "a", 100, short_gib=60)
        _publish(self.dir, "b", 160)
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                2,
                300 * _GIB,
                timeout_s=10,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertIn("a (hostA) is 60 GiB short", str(ctx.exception))

    def test_short_donor_note_names_its_numa_placement(self):
        _publish(self.dir, "a", 100, short_gib=60, extra=["numa_chunks 0:60,1:40"])
        _publish(self.dir, "b", 160)
        published, _ = scan_donors(self.dir)
        self.assertEqual(
            {p["name"]: p["numa_chunks"] for p in published}, {"a": "0:60,1:40", "b": ""}
        )
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                2,
                300 * _GIB,
                timeout_s=10,
                poll_s=0.05,
                log=lambda _: None,
            )
        message = str(ctx.exception)
        self.assertIn("a (hostA) is 60 GiB short (chunks per NUMA node 0:60,1:40)", message)
        self.assertNotIn("\n", message)

    def test_times_out_when_donors_do_not_report(self):
        _publish(self.dir, "a", 10)
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                3,
                20 * _GIB,
                timeout_s=0.3,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)
        self.assertIn("2 not reported", str(ctx.exception))

    def test_passes_before_all_report_once_covered(self):
        _publish(self.dir, "a", 160)
        _publish(self.dir, "b", 160)
        t0 = time.monotonic()
        check_remote_capacity(
            self.dir,
            8,
            300 * _GIB,
            timeout_s=600,
            poll_s=0.05,
            log=lambda _: None,
        )
        self.assertLess(time.monotonic() - t0, 5.0)

    def test_waits_for_a_donor_that_publishes_later(self):
        _publish(self.dir, "a", 100)
        threading.Timer(0.3, lambda: _publish(self.dir, "b", 100)).start()
        result = check_remote_capacity(
            self.dir,
            2,
            200 * _GIB,
            timeout_s=10,
            poll_s=0.05,
            log=lambda _: None,
        )
        self.assertEqual(result["offered_bytes"], 200 * _GIB)

    def test_failure_marker_survives_close(self):
        donor = NixlDonor(self.dir, 4 * _GIB, chunk_bytes=_GIB, name="gone")
        donor.close()
        donor.publish_failure(RuntimeError("boom\nsecond line"))
        path = os.path.join(self.dir, "donor.gone.failed")
        self.assertTrue(os.path.exists(path))
        text = open(path).read()
        self.assertIn("planned_bytes 4294967296", text)
        self.assertNotIn("\nsecond line", text)  # the error stays on one line

    def test_close_waits_for_a_failed_background_start_closing_itself(self):
        # The start thread closes the donor itself on failure; a concurrent close() must not withdraw alongside it.
        donor = NixlDonor(self.dir, 4 * _GIB, chunk_bytes=_GIB, name="racy")
        entered, release = threading.Event(), threading.Event()
        active, depths = [0], []
        real_withdraw = donor._withdraw

        def withdraw():
            active[0] += 1
            depths.append(active[0])
            if threading.current_thread().name == "kvcm2-remote-donor":
                entered.set()
                release.wait(10)
            real_withdraw()
            active[0] -= 1

        start_threads = []

        def failing_start():
            start_threads.append(threading.current_thread())
            raise RuntimeError("no memory")

        donor._withdraw = withdraw
        donor.start = failing_start
        donor.start_in_background()
        self.assertTrue(entered.wait(10))
        closer = threading.Thread(target=donor.close)
        closer.start()
        closer.join(0.3)
        self.assertTrue(
            closer.is_alive(), "close() returned while the start thread was still closing"
        )
        release.set()
        closer.join(10)
        self.assertFalse(closer.is_alive())
        self.assertEqual(depths, [1, 1])
        start_threads[0].join(10)  # it still writes the failure marker after its close
        self.assertFalse(start_threads[0].is_alive())

    def _started(self, name):
        donor = FabricDonor(self.dir, 4 * _GIB, chunk_bytes=_GIB, name=name)
        with mock.patch.object(
            FabricDonor,
            "_start_transport",
            lambda self: self._publish([], ["segment 0 1"], "test"),
        ):
            donor.start()
        return donor

    def test_a_donor_nobody_closed_withdraws_and_leaves_a_marker(self):
        # E.g. startup failed between the donation start and the manager that would own it.
        donor = self._started("orphan")
        ready = os.path.join(self.dir, "donor.orphan.ready")
        self.assertTrue(os.path.exists(ready))
        del donor
        gc.collect()
        self.assertFalse(os.path.exists(ready))
        _, failed = scan_donors(self.dir)
        self.assertEqual([f["name"] for f in failed], ["orphan"])

    def test_a_closed_donor_leaves_no_marker(self):
        donor = self._started("done")
        donor.close()
        del donor
        gc.collect()
        self.assertEqual(scan_donors(self.dir), ([], []))

    def test_close_with_an_error_leaves_a_marker(self):
        donor = self._started("aborted")
        donor.close(RuntimeError("initialization failed on another rank"))
        _, failed = scan_donors(self.dir)
        self.assertIn("another rank", failed[0]["error"])

    def test_donor_names_are_unique_within_a_process(self):
        names = {NixlDonor(self.dir, _GIB, chunk_bytes=_GIB).name for _ in range(64)}
        self.assertEqual(len(names), 64)


class TestRendezvousPermissions(unittest.TestCase):
    """Donor files hold NIXL registration keys and fabric handles: the directory and its files are the user's."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_perm_")

    def tearDown(self):
        os.chmod(self.dir, 0o700)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_new_directory_is_private(self):
        path = os.path.join(self.dir, "rendezvous")
        remote_donor.ensure_rendezvous_directory(path)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o700)

    def test_a_directory_others_can_write_is_refused(self):
        os.chmod(self.dir, 0o777)
        with self.assertRaises(PermissionError):
            remote_donor.ensure_rendezvous_directory(self.dir)
        os.chmod(self.dir, 0o755)
        remote_donor.ensure_rendezvous_directory(self.dir)

    def test_donor_files_are_private(self):
        NixlDonor(self.dir, _GIB, chunk_bytes=_GIB, name="x").publish_failure(RuntimeError("e"))
        self.assertEqual(os.stat(os.path.join(self.dir, "donor.x.failed")).st_mode & 0o077, 0)


class TestPopulate(unittest.TestCase):
    """Populate errors other than a missing MADV_POPULATE_WRITE (test_remote_donor_memory) raise."""

    def _populate_with(self, error):
        region = mmap.mmap(-1, 1 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        try:
            with mock.patch.object(remote_donor, "madvise_range", side_effect=error):
                remote_donor._populate(region, 0, 1 << 20)
        finally:
            region.close()

    def test_other_errors_raise(self):
        for code in (errno.ENOMEM, getattr(errno, "EHWPOISON", 133)):
            with self.assertRaises(OSError):
                self._populate_with(OSError(code, os.strerror(code)))


class TestPlaceChunks(unittest.TestCase):
    """Chunk placement when the driver refuses chunks on a node that still has room.

    Pure Python: the allocator, the node memory, the reclaim-and-compact pass, the per-node lock and the clock are
    fakes. The policy is the module's (2 s of retries, passes of at least 8 GiB) with a 24 GiB reserve per node.
    """

    def setUp(self):
        self.t = [0.0]
        self.room = {0: 200.0, 1: 200.0}  # GiB free per node (no page cache)
        # GiB free in >= 2 MiB blocks; passes raise it only where a test says so.
        self.contig = {0: 0.0, 1: 0.0}
        self.passes = []
        self.locked = set()  # nodes whose compaction lock is held
        self.on_lock = lambda node: True  # runs as the lock is taken; False refuses it

    @contextlib.contextmanager
    def _node_lock(self, node):
        granted = self.on_lock(node)
        if granted:
            self.locked.add(node)
        try:
            yield granted
        finally:
            self.locked.discard(node)

    def _sleep(self, seconds):
        self.t[0] += seconds

    def _meminfo(self, node):
        return {"MemFree": self.room[node], "MemTotal": 480.0}

    def _compact(self, node, gib):
        assert node in self.locked, "a pass ran without its node's lock"
        self.passes.append((node, gib))

    def _place(self, n_chunks, create, **overrides):
        args = dict(
            chunk_gib=1.0,
            reserve_gib=24.0,
            meminfo=self._meminfo,
            compact=self._compact,
            contiguous=lambda node: self.contig[node],
            sleep=self._sleep,
            clock=lambda: self.t[0],
            node_lock=self._node_lock,
        )
        args.update(overrides)
        return remote_donor._place_chunks(n_chunks, [0, 1], create, **args)

    def _allocator(self, refuse):
        """create(node) refuses when refuse(node, call number) says so; an accepted chunk takes 1 GiB of the node."""
        calls = []

        def create(node):
            calls.append(node)
            if refuse(node, len(calls)):
                return None
            self.room[node] -= 1.0
            return object()

        return create, calls

    def _passes_on(self, node):
        return [gib for n, gib in self.passes if n == node]

    def test_all_accepted(self):
        create, calls = self._allocator(lambda node, i: False)
        handles, placement = self._place(10, create)
        self.assertEqual(len(handles), 10)
        self.assertEqual(placement, {0: 10, 1: 0})
        self.assertEqual(calls, [0] * 10)

    def test_transient_refusal_is_retried(self):
        # Refuse chunk 3's first try and its first retry.
        create, calls = self._allocator(lambda node, i: i in (4, 5))
        handles, placement = self._place(10, create)
        self.assertEqual(placement, {0: 10, 1: 0})
        self.assertEqual(calls, [0] * 12)
        self.assertEqual(self.passes, [])
        self.assertGreater(self.t[0], 0.0)  # backed off before the retries

    def test_pass_recovers_a_refusal(self):
        # Node 0 refuses from chunk 5 on until one reclaim-and-compact pass has run on it.
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        handles, placement = self._place(10, create)
        self.assertEqual(placement, {0: 10, 1: 0})
        self.assertEqual(self.passes, [(0, 8.0)])  # the smallest pass: the node has plenty of room
        self.assertGreaterEqual(self.t[0], 2.0)  # the retries came first

    def test_pass_is_sized_to_keep_the_reserve_and_margin(self):
        self.room[0] = 24.0 + 1.0 + 8.0 + 3.5  # reserve + chunk + margin + 3.5 GiB
        create, _ = self._allocator(lambda node, i: node == 0 and not self.passes)
        handles, placement = self._place(1, create)
        self.assertEqual(placement[0], 1)
        self.assertEqual(len(self.passes), 1)
        self.assertAlmostEqual(self.passes[0][1], 3.5)

    def test_no_room_for_a_pass_moves_on_and_stops_trying_the_node(self):
        self.room[0] = 24.0 + 1.0 + 8.0 + 0.5  # less than 1 GiB left for a pass
        create, calls = self._allocator(lambda node, i: node == 0)
        handles, placement = self._place(5, create)
        self.assertEqual(placement, {0: 0, 1: 5})
        self.assertEqual(self.passes, [])
        # Refused (and retried) for the first chunk only, then skipped for the later ones.
        self.assertNotIn(0, calls[calls.index(1) :])

    def test_short_when_every_node_refuses(self):
        create, _ = self._allocator(lambda node, i: i > 3)  # three chunks, then nothing anywhere
        handles, placement = self._place(10, create)
        self.assertEqual(len(handles), 3)
        self.assertEqual(placement, {0: 3, 1: 0})
        self.assertEqual(len(self._passes_on(0)), remote_donor._MAX_PASSES_PER_REFUSAL)

    def test_a_failing_pass_stops_passes_on_the_node(self):
        attempts = []

        def broken(node, gib):
            attempts.append(node)
            raise RuntimeError("libnuma is unavailable")

        create, _ = self._allocator(lambda node, i: node == 0 and i > 2)
        handles, placement = self._place(4, create, compact=broken)
        self.assertEqual(placement, {0: 2, 1: 2})
        self.assertEqual(attempts, [0])

    def test_a_pass_is_sized_to_the_donation_still_missing(self):
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(40, create)
        self.assertEqual(self.passes, [(0, 35.0)])  # 40 - 5 chunks, more than the smallest pass

    def test_a_pass_stays_within_its_cap(self):
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(150, create)
        self.assertEqual(self.passes, [(0, 64.0)])  # _MAX_PASS_CHUNKS chunks

    def test_pass_floor_and_margin_follow_the_reserve(self):
        # A third of a 48 GiB reserve: passes of at least 16 GiB, leaving 16 GiB above the reserve and the chunk.
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(10, create, reserve_gib=48.0)
        self.assertEqual(self.passes, [(0, 16.0)])
        self.passes.clear()
        self.room[0] = 48.0 + 1.0 + 16.0 + 3.5
        create, _ = self._allocator(lambda node, i: node == 0 and not self.passes)
        self._place(1, create, reserve_gib=48.0)
        self.assertEqual(len(self.passes), 1)
        self.assertAlmostEqual(self.passes[0][1], 3.5)

    def test_passes_go_on_while_they_raise_contiguous_memory(self):
        # Node 0 refuses until five passes have run; each pass leaves its size free in >= 2 MiB blocks.
        def compact(node, gib):
            self.passes.append((node, gib))
            self.contig[node] += gib

        create, _ = self._allocator(lambda node, i: node == 0 and i > 2 and len(self.passes) < 5)
        handles, placement = self._place(4, create, compact=compact)
        self.assertEqual(placement, {0: 4, 1: 0})
        self.assertEqual(len(self.passes), 5)

    def test_a_node_given_up_on_is_swept_once_more(self):
        # Node 0 is fragmented while node 1 still has room; once node 1 is full, node 0 holds memory in >= 2 MiB blocks
        # again after the donor left it.
        self.room[1] = 24.0 + 4.0  # four chunks above the reserve

        def contiguous(node):
            return 10.0 if node == 0 and self.room[1] < 25.0 else 0.0

        create, _ = self._allocator(lambda node, i: node == 0 and contiguous(0) < 1.0)
        handles, placement = self._place(8, create, contiguous=contiguous)
        self.assertEqual(placement, {0: 4, 1: 4})
        self.assertEqual(len(self._passes_on(0)), remote_donor._MAX_PASSES_PER_REFUSAL)

    def test_the_sweep_runs_once(self):
        # Nodes that look recovered but refuse every chunk: the donation ends short after one sweep, without looping.
        create, calls = self._allocator(lambda node, i: i > 3)
        self.contig = {0: 5.0, 1: 5.0}
        handles, placement = self._place(10, create)
        self.assertEqual(len(handles), 3)
        self.assertLess(len(calls), 200)

    def test_a_pass_is_sized_from_the_memory_left_once_it_holds_the_lock(self):
        # Another donor on the node fills 160 GiB of it while this one waits for the lock.
        def other_donor_ran(node):
            self.room[node] -= 160.0
            return True

        self.on_lock = other_donor_ran
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(150, create)
        self.assertEqual(len(self.passes), 1)
        self.assertAlmostEqual(self.passes[0][1], 200.0 - 5 - 160.0 - 24.0 - 1.0 - 8.0)

    def test_no_pass_without_the_lock(self):
        self.on_lock = lambda node: False
        create, _ = self._allocator(lambda node, i: node == 0 and i > 2)
        handles, placement = self._place(4, create)
        self.assertEqual(placement, {0: 2, 1: 2})
        self.assertEqual(self.passes, [])

    def test_reserve_skips_a_node_without_trying_it(self):
        self.room[0] = 24.5  # below reserve + chunk
        create, calls = self._allocator(lambda node, i: False)
        handles, placement = self._place(3, create)
        self.assertEqual(placement, {0: 0, 1: 3})
        self.assertNotIn(0, calls)


class TestNodeCompactionLock(unittest.TestCase):
    """The flock that lets one donor per host and NUMA node run reclaim-and-compact passes at a time."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_compaction_lock_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_donors_on_one_node_take_turns(self):
        order = []
        holding = threading.Event()

        def second():
            holding.wait(10)
            with remote_donor._node_compaction_lock(self.dir, "hostA", 0) as locked:
                order.append(("second", locked))

        thread = threading.Thread(target=second)
        thread.start()
        with remote_donor._node_compaction_lock(self.dir, "hostA", 0) as locked:
            holding.set()
            thread.join(0.3)
            self.assertTrue(thread.is_alive(), "the second donor passed a held lock")
            order.append(("first", locked))
        thread.join(10)
        self.assertEqual(order, [("first", True), ("second", True)])

    def test_other_nodes_and_hosts_do_not_wait(self):
        with remote_donor._node_compaction_lock(self.dir, "hostA", 0) as locked:
            self.assertTrue(locked)
            for host, node in (("hostA", 1), ("hostB", 0)):
                with remote_donor._node_compaction_lock(self.dir, host, node) as other:
                    self.assertTrue(other)

    def test_unusable_directory_refuses_the_lock(self):
        missing = os.path.join(self.dir, "missing")
        with remote_donor._node_compaction_lock(missing, "hostA", 0) as locked:
            self.assertFalse(locked)


class TestBuddyInfo(unittest.TestCase):
    TEXT = (
        "Node 0, zone   Normal     10      5      3      2      1      0      0      0      0      4      2\n"
        "Node 1, zone   Normal    100     50     30     20     10      5      0      0      0      0      0\n"
    )

    def test_free_memory_in_blocks_of_2_mib_or_more(self):
        f = remote_donor._contiguous_free_gib
        # 4 KiB pages: order 9 is 2 MiB, order 10 is 4 MiB: 4 x 2 + 2 x 4 = 16 MiB on node 0, none on node 1.
        self.assertAlmostEqual(f(0, text=self.TEXT, page=4096), 16 / 1024)
        self.assertEqual(f(1, text=self.TEXT, page=4096), 0.0)
        # 64 KiB pages (aarch64): order 5 is 2 MiB. Node 0: 4 x 32 + 2 x 64 MiB; node 1: 5 x 2 MiB.
        self.assertAlmostEqual(f(0, text=self.TEXT, page=65536), 256 / 1024)
        self.assertAlmostEqual(f(1, text=self.TEXT, page=65536), 10 / 1024)
        self.assertEqual(f(7, text=self.TEXT, page=4096), 0.0)


@unittest.skipUnless(
    len(remote_donor._host_numa_nodes()) >= 2 and fabric_available(),
    "needs the fabric transport (a Grace GPU with an IMEX channel) and two NUMA nodes with memory",
)
class TestFabricPlacement(unittest.TestCase):
    """Where a fabric donation's chunks go when its node is full or the driver refuses them.

    A HOST_NUMA allocation never leaves its node, and the driver can refuse chunks with room left. Node fullness and
    refusals are faked; the allocations are real.
    """

    CHUNK = 16 << 20

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_fabric_placement_")
        self.real = remote_donor._node_meminfo

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _full(self, node):
        return dict(self.real(node), MemFree=0.0, **{"Active(file)": 0.0, "Inactive(file)": 0.0})

    def _meta(self, donor):
        return remote_donor._meta_fields(os.path.join(self.dir, f"donor.{donor.name}.meta"))

    def test_full_preferred_node_spills_to_the_other(self):
        donor = FabricDonor(
            self.dir,
            4 * self.CHUNK,
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )
        fake = lambda n: self._full(n) if n == 0 else self.real(n)  # noqa: E731
        with mock.patch.object(remote_donor, "_node_meminfo", fake):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["numa_chunks"], "1:4")
            self.assertEqual(fields["segments"], [[0, 4 * self.CHUNK]])
            self.assertNotIn("short_bytes", fields)
            fabric = os.path.join(self.dir, f"donor.{donor.name}.fabric")
            self.assertEqual(os.path.getsize(fabric), 4 * 64)
        finally:
            donor.close()

    def test_donation_that_only_partly_fits_is_published_short(self):
        donor = FabricDonor(
            self.dir,
            4 * self.CHUNK,
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )
        # Node 0 is full, node 1 has room for two chunks.
        fake = lambda n: self._full(n) if n == 0 or len(donor._fabric) >= 2 else self.real(n)  # noqa: E731
        with mock.patch.object(remote_donor, "_node_meminfo", fake):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["segments"], [[0, 2 * self.CHUNK]])
            self.assertEqual(int(fields["short_bytes"]), 2 * self.CHUNK)
            published, _ = scan_donors(self.dir)
            self.assertEqual(published[0]["offered_bytes"], 2 * self.CHUNK)
        finally:
            donor.close()

    def test_refused_chunks_are_retried_with_real_allocations(self):
        # The driver refuses two calls (chunk 1's first try and its first retry); the donation still completes on
        # node 0.
        real_create = FabricDonor._try_create
        calls = [0]

        def flaky(donor, drv, prop):
            calls[0] += 1
            return None if calls[0] in (2, 3) else real_create(donor, drv, prop)

        donor = FabricDonor(
            self.dir,
            4 * self.CHUNK,
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )
        with (
            mock.patch.object(FabricDonor, "_try_create", flaky),
            mock.patch.object(remote_donor, "_FABRIC_RETRY_S", 0.5),
        ):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["numa_chunks"], "0:4")
            self.assertEqual(calls[0], 6)
        finally:
            donor.close()

    def test_a_real_reclaim_pass_recovers_a_refusal(self):
        # Refuse until one real reclaim-and-compact pass (64 chunks of 16 MiB, libnuma-bound, huge pages) has run on
        # node 0.
        real_create = FabricDonor._try_create
        real_compact = remote_donor._compact_node
        passes = []

        def counted(node, gib):
            passes.append((node, gib))
            return real_compact(node, gib)

        def refusing(donor, drv, prop):
            return real_create(donor, drv, prop) if passes or not donor._fabric else None

        donor = FabricDonor(
            self.dir,
            3 * self.CHUNK,
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )
        with (
            mock.patch.object(FabricDonor, "_try_create", refusing),
            mock.patch.object(remote_donor, "_compact_node", counted),
            mock.patch.object(remote_donor, "_FABRIC_RETRY_S", 0.2),
        ):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["numa_chunks"], "0:3")
            self.assertEqual(passes, [(0, 1.0)])
        finally:
            donor.close()

    def test_no_room_anywhere_leaves_a_failure_marker(self):
        donor = FabricDonor(self.dir, 4 * self.CHUNK, chunk_bytes=self.CHUNK)
        with mock.patch.object(remote_donor, "_node_meminfo", self._full):
            donor.start_or_report()
        self.assertFalse(os.path.exists(os.path.join(self.dir, f"donor.{donor.name}.ready")))
        published, failed = scan_donors(self.dir)
        self.assertEqual(published, [])
        self.assertEqual([f["name"] for f in failed], [donor.name])
        self.assertIn("no NUMA node can hold", failed[0]["error"])


if __name__ == "__main__":
    unittest.main()
