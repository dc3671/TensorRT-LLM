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
"""Lend host memory to the remote cache levels of KV cache manager v2.

A lender publishes host memory in a rendezvous directory on a shared file system; a remote cache level
(``KvCacheConfig.remote_cache_config``; C++ ``storage/remoteMem.h``, which also describes the directory layout) claims
whole chunks of it with O_EXCL claim files and moves cold pages itself, so the lender does no work after publishing.

* ``nixl``: anonymous memory registered with a NIXL agent; ``lender.<name>.desc`` holds the agent's ``AgentDesc``.
* ``fabric``: Grace DRAM allocated with cuMemCreate (``CU_MEM_LOCATION_TYPE_HOST_NUMA``), one allocation per chunk
  because an imported fabric allocation maps only whole; ``lender.<name>.fabric`` holds one 64-byte handle per chunk.
  Chunks fill the GPU's host NUMA node first, then the other nodes; a lend that only partly fits is published with
  ``short_bytes``. Lenders on one host take turns at reclaim-and-compact passes per NUMA node by flock on
  ``compact.<host>.node<N>.lock``.

A lender that cannot start leaves ``lender.<name>.failed``, which clients ignore and capacity checks count. In a
serving process ``start_lend`` starts the lend (``remote_cache_config.lend_size``) before the KV cache estimation, and
the final ``KVCacheManagerV2`` owns it.
"""

import contextlib
import ctypes
import fcntl
import mmap
import os
import socket
import threading
import time
from typing import TYPE_CHECKING, Optional

from tensorrt_llm.llmapi.llm_args import REMOTE_CACHE_CHUNK_BYTES, REMOTE_CACHE_LEND_RESERVE_BYTES
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import RemoteCacheConfig

TRANSPORTS = ("nixl", "fabric")


def _nvml_gpu_index(device: int) -> int:
    """NVML's index (nvidia-smi's numbering) of CUDA device `device`, found by its PCI bus id."""
    import pynvml
    from cuda.bindings import driver as drv

    _cuda_check(drv.cuInit(0))
    bus_id = _cuda_check(drv.cuDeviceGetPCIBusId(32, _cuda_check(drv.cuDeviceGet(device))))
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode()
    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetIndex(
            pynvml.nvmlDeviceGetHandleByPciBusId(bus_id.rstrip("\x00"))
        )
    finally:
        pynvml.nvmlShutdown()


def _physical_gpu_index(device: Optional[int]) -> int:
    """The physical index of CUDA device `device` (default: the current one): a plain index in CUDA_VISIBLE_DEVICES
    directly, any other entry (a UUID or MIG id) through the device's PCI bus id."""
    if device is None:
        import torch

        device = torch.cuda.current_device()
    visible = [
        v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()
    ]
    if not visible:
        return device
    if device >= len(visible):
        raise ValueError(
            f"cannot map CUDA device {device} to a GPU index through "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    if visible[device].isdigit():
        return int(visible[device])
    return _nvml_gpu_index(device)


def resolve_ucx_devices(devices: Optional[str], device: Optional[int] = None) -> Optional[str]:
    """Replace ``{GPU}`` in `devices` with this rank's physical GPU index (CUDA device `device`, default the current
    one, through CUDA_VISIBLE_DEVICES), so that "rdma_vf_rail{GPU}:1" gives each rank the rail next to its GPU."""
    if not devices or "{" not in devices:
        return devices
    resolved = devices.replace("{GPU}", str(_physical_gpu_index(device)))
    if "{" in resolved or "}" in resolved:
        raise ValueError(f"UCX devices {devices!r}: unknown token (known: {{GPU}})")
    return resolved


def _log(message: str) -> None:
    logger.info(f"KVCM2 remote lender: {message}")


def _write_atomically(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# Linux values (asm-generic, so x86_64 and aarch64 alike); Python's mmap module lacks some of them before 3.13.
_MAP_NORESERVE = 0x4000
_MADV_HUGEPAGE = 14
_MADV_POPULATE_WRITE = 23  # Linux 5.14+
_POPULATE_SLICE = 64 << 20


def _populate(region: mmap.mmap, nbytes: int) -> None:
    """Make every page of `region` resident: huge pages where the kernel has them (MADV_HUGEPAGE is set first), 4 KiB
    pages elsewhere. MADV_POPULATE_WRITE does it in the kernel; without it, zero the region slice by slice."""
    try:
        region.madvise(_MADV_POPULATE_WRITE)
        return
    except (OSError, ValueError):
        pass
    zeros = bytes(_POPULATE_SLICE)
    view = memoryview(region)
    try:
        for offset in range(0, nbytes, _POPULATE_SLICE):
            end = min(offset + _POPULATE_SLICE, nbytes)
            view[offset:end] = zeros[: end - offset]
    finally:
        view.release()


def _bind_to_numa_node(address: int, nbytes: int, node: int) -> bool:
    """Bind [address, address + nbytes) to NUMA node `node` with libnuma, before it is touched."""
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
    except OSError:
        return False
    if libnuma.numa_available() < 0:
        return False
    libnuma.numa_tonode_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_tonode_memory.restype = None
    libnuma.numa_tonode_memory(
        ctypes.c_void_p(address), ctypes.c_size_t(nbytes), ctypes.c_int(node)
    )
    return True


def _cuda_check(result):
    """Unwrap a cuda.bindings driver call's (CUresult, value...) tuple, raising on an error."""
    from cuda.bindings import driver as drv

    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        name = drv.cuGetErrorName(err)[1]
        raise RuntimeError(name.decode() if isinstance(name, bytes) else str(name))
    if isinstance(result, tuple):
        return result[1] if len(result) == 2 else result[1:]
    return None


_GIB = 1 << 30
# Seconds a refused fabric chunk is retried with backoff before a pass: the driver refuses transiently while reclaiming.
_FABRIC_RETRY_S = 2.0
# A reclaim-and-compact pass (_compact_node) covers the lend still missing, at least this share of the node reserve
# and at most _MAX_PASS_CHUNKS chunks, and leaves this share of the reserve free above the reserve and the chunk.
_PASS_RESERVE_SHARE = 1 / 3
_MAX_PASS_CHUNKS = 64
# Pass bounds: per node, and per refused chunk; past the first bound only while each pass adds a chunk's worth of
# >= 2 MiB blocks.
_MAX_PASSES_PER_NODE = 64
_MAX_PASSES_PER_REFUSAL = 3
_MAX_PASSES_PER_REFUSAL_PROGRESSING = 16


def _node_meminfo(node: int) -> dict:
    """GiB per field of /sys/devices/system/node/node<node>/meminfo; empty if unreadable."""
    out = {}
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:
                parts = line.split()  # "Node 0 MemFree:  123 kB"
                if len(parts) >= 4 and parts[3].isdigit():
                    out[parts[2].rstrip(":")] = int(parts[3]) * 1024 / _GIB
    except OSError:
        return {}
    return out


def _numa_nodes() -> list:
    try:
        names = os.listdir("/sys/devices/system/node")
    except OSError:
        return []
    return sorted(int(n[4:]) for n in names if n.startswith("node") and n[4:].isdigit())


def _reclaimable_gib(info: dict) -> float:
    """Clean file pages the kernel can drop, which cuMemCreate's host allocations reclaim (shmem sits on the
    anonymous LRU, outside these counts)."""
    return max(
        0.0,
        info.get("Active(file)", 0.0) + info.get("Inactive(file)", 0.0) - info.get("Dirty", 0.0),
    )


def _host_numa_nodes() -> list:
    """The NUMA nodes that have memory (GPU memory partitions show up as nodes with MemTotal 0)."""
    return [n for n in _numa_nodes() if _node_meminfo(n).get("MemTotal", 0.0) > 0]


def _contiguous_free_gib(
    node: int, min_bytes: int = 2 << 20, text: Optional[str] = None, page: Optional[int] = None
) -> float:
    """GiB of free memory on NUMA node `node` held in blocks of at least `min_bytes` (/proc/buddyinfo: free blocks of
    each order, per zone); -1 when unreadable. A refusal while this is near zero points at fragmentation."""
    try:
        if text is None:
            with open("/proc/buddyinfo") as f:
                text = f.read()
        page = page or os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return -1.0
    total = 0
    for line in text.splitlines():
        parts = line.replace(",", " ").split()  # "Node 0, zone   Normal   n0 n1 ..."
        if len(parts) < 5 or parts[0] != "Node" or not parts[1].isdigit() or int(parts[1]) != node:
            continue
        for order, count in enumerate(parts[4:]):
            if count.isdigit() and (page << order) >= min_bytes:
                total += int(count) * (page << order)
    return total / _GIB


def _host_numa_node_of(drv, device) -> int:
    """The host NUMA node closest to CUDA device `device`; the first node with memory if the driver reports none."""
    err, node = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
    )
    if err == drv.CUresult.CUDA_SUCCESS and node >= 0:
        return node
    nodes = _host_numa_nodes()
    return nodes[0] if nodes else 0


def _compact_node(node: int, gib: float) -> None:
    """Reclaim and compact memory on NUMA node `node`: fault in `gib` GiB of node-bound anonymous memory with huge
    pages (direct reclaim and compaction), then unmap it, leaving about that much free in 2 MiB blocks. Raises when
    libnuma cannot bind the region, since unbound it would land anywhere."""
    nbytes = int(gib * _GIB) // (2 << 20) * (2 << 20)
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        anchor = ctypes.c_char.from_buffer(region)
        address = ctypes.addressof(anchor)
        del anchor  # the buffer export must end before close()
        if not _bind_to_numa_node(address, nbytes, node):
            raise RuntimeError("libnuma is unavailable, so the region cannot be bound to the node")
        try:
            region.madvise(_MADV_HUGEPAGE)
        except OSError:
            pass
        _populate(region, nbytes)
    finally:
        region.close()


@contextlib.contextmanager
def _node_compaction_lock(directory: str, host: str, node: int):
    """Hold the exclusive flock of NUMA node `node` of `host` in `directory`; yields False, holding nothing, when the
    file system refuses it (no pass may then run on the node)."""
    path = os.path.join(directory, f"compact.{host}.node{node}.lock")
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as error:
        _log(f"cannot open {path} ({error}); no reclaim-and-compact pass on node {node}")
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as error:
            _log(f"cannot lock {path} ({error}); no reclaim-and-compact pass on node {node}")
            yield False
            return
        yield True
    finally:
        os.close(fd)  # drops the lock


def _place_chunks(
    n_chunks: int,
    nodes: list,
    create,
    *,
    chunk_gib: float,
    reserve_gib: float,
    node_lock,
    meminfo=None,
    compact=None,
    contiguous=None,
    sleep=time.sleep,
    clock=time.monotonic,
):
    """Place up to `n_chunks` chunks over `nodes`, the preferred node first; `create(node)` returns a handle, or None
    when the driver refuses. A node takes chunks while it keeps `reserve_gib`; a refused chunk is retried, then
    compacted for under `node_lock(node)` (see the constants), which lenders sharing the node take in turn; a node
    that still refuses is left, and swept once more when no node takes the chunk. Returns (handles, {node: placed})."""
    meminfo = meminfo or _node_meminfo
    compact = compact or _compact_node
    contiguous = contiguous or _contiguous_free_gib
    handles = []
    placement = {n: 0 for n in nodes}
    passes = {n: 0 for n in nodes}
    pass_failed = set()
    min_pass_gib = margin_gib = reserve_gib * _PASS_RESERVE_SHARE
    max_pass_gib = _MAX_PASS_CHUNKS * chunk_gib

    def room(node):
        info = meminfo(node)
        return info.get("MemFree", 0.0) + _reclaimable_gib(info) if info else float("inf")

    def recover(node):
        delay, deadline = 0.05, clock() + _FABRIC_RETRY_S
        while clock() < deadline:
            sleep(delay)
            delay = min(2 * delay, 0.5)
            handle = create(node)
            if handle is not None:
                return handle
        tries = 0
        while (
            tries < _MAX_PASSES_PER_REFUSAL_PROGRESSING
            and passes[node] < _MAX_PASSES_PER_NODE
            and node not in pass_failed
        ):
            tries += 1
            with node_lock(node) as locked:
                if not locked:
                    pass_failed.add(node)
                    return None
                # Sized from the node's memory once the lock is held: another lender may have used it meanwhile.
                missing = (n_chunks - len(handles)) * chunk_gib
                size = min(
                    max(min_pass_gib, missing),
                    max_pass_gib,
                    room(node) - reserve_gib - chunk_gib - margin_gib,
                )
                if size < chunk_gib:
                    return None
                before = contiguous(node)
                try:
                    compact(node, size)
                except Exception as error:  # noqa: BLE001 - no pass is possible here; the chunk moves on
                    _log(f"reclaim-and-compact pass on node {node} failed: {error!r}")
                    pass_failed.add(node)
                    return None
                passes[node] += 1
                gained = contiguous(node) - before
                handle = create(node)
            if handle is not None:
                return handle
            if tries >= _MAX_PASSES_PER_REFUSAL and gained < chunk_gib:
                return None  # the passes stopped bringing contiguous memory
        return None

    exhausted = set()
    swept = False
    while len(handles) < n_chunks:
        handle = None
        for node in nodes:
            if node in exhausted or room(node) - chunk_gib < reserve_gib:
                continue  # refused for good, or the node keeps its reserve for the serving process
            handle = create(node)
            if handle is None:
                handle = recover(node)
                if handle is None:
                    exhausted.add(node)
            if handle is not None:
                placement[node] += 1
                break
        if handle is None:
            if swept:
                break
            # A node given up on can recover after the lender left it (compaction and other lenders' frees go on).
            swept = True
            revived = {
                n
                for n in exhausted
                if contiguous(n) >= chunk_gib and room(n) - chunk_gib >= reserve_gib
            }
            if not revived:
                break
            exhausted -= revived
            continue
        handles.append(handle)
    if any(passes.values()):
        _log(f"reclaim-and-compact passes per NUMA node: {passes}")
    return handles, placement


def _meta_fields(path: str) -> dict:
    fields = {"segments": []}
    with open(path) as f:
        for line in f:
            key, _, value = line.strip().partition(" ")
            if key == "segment":
                fields["segments"].append([int(x) for x in value.split()])
            elif key:
                fields[key] = value
    return fields


def scan_lenders(directory: str):
    """The lenders of a rendezvous directory that have reported: (published, failed). A published lender offers the
    whole chunks of its segments; a failed one left lender.<name>.failed (RemoteLender.publish_failure)."""
    published, failed = [], []
    try:
        files = sorted(os.listdir(directory))
    except FileNotFoundError:
        return published, failed
    for file in files:
        if not file.startswith("lender."):
            continue
        name = file[len("lender.") :].rsplit(".", 1)[0]
        try:
            if file.endswith(".ready"):
                fields = _meta_fields(os.path.join(directory, f"lender.{name}.meta"))
                chunk = int(fields.get("chunk_bytes", 0))
                offered = sum(b - b % chunk for _, b in fields["segments"]) if chunk else 0
                published.append(
                    {
                        "name": name,
                        "host": fields.get("host", "?"),
                        "offered_bytes": offered,
                        "short_bytes": int(fields.get("short_bytes", 0)),
                        "numa_chunks": fields.get("numa_chunks", ""),
                    }
                )
            elif file.endswith(".failed"):
                fields = _meta_fields(os.path.join(directory, file))
                failed.append(
                    {
                        "name": name,
                        "host": fields.get("host", "?"),
                        "planned_bytes": int(fields.get("planned_bytes", 0)),
                        "error": fields.get("error", "?"),
                    }
                )
        except (OSError, ValueError):
            continue  # a file being replaced; the next scan sees it
    return published, failed


def check_remote_capacity(
    directory: str,
    num_lenders: int,
    need_bytes: int,
    *,
    timeout_s: float,
    poll_s: float = 2.0,
    log=None,
) -> dict:
    """Wait until `num_lenders` lenders have published or failed (at most `timeout_s`), or until they offer
    `need_bytes`, then raise if the offer falls short, naming the failed and short lenders. A job whose lenders cannot
    cover its clients thus stops when the last lender reports instead of after the clients' claim wait."""
    log = log or _log
    need = need_bytes
    expected = num_lenders
    deadline = time.monotonic() + timeout_s
    while True:
        published, failed = scan_lenders(directory)
        offered = sum(p["offered_bytes"] for p in published)
        reported = len(published) + len(failed)
        if offered >= need or reported >= expected or time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    summary = (
        f"{len(published)} of {expected} planned lenders published {offered / _GIB:.0f} GiB, {len(failed)} failed, "
        f"{max(0, expected - reported)} not reported; the plan's clients claim {need / _GIB:.0f} GiB"
    )
    if offered < need:
        notes = [f"{f['name']} ({f['host']}) failed: {f['error']}" for f in failed]
        notes += [
            f"{p['name']} ({p['host']}) is {p['short_bytes'] / _GIB:.0f} GiB short"
            + (f" (chunks per NUMA node {p['numa_chunks']})" if p["numa_chunks"] else "")
            for p in published
            if p["short_bytes"]
        ]
        raise RuntimeError(
            f"KVCM2 remote level capacity check failed in {directory}: {summary}"
            + (f". {'; '.join(notes)}" if notes else "")
        )
    log(f"KVCM2 remote level capacity check passed in {directory}: {summary}")
    return {"published": published, "failed": failed, "offered_bytes": offered, "need_bytes": need}


class RemoteLender:
    """One lent region: mapped, registered and published by ``start()``, withdrawn by ``close()``.

    With ``transport="fabric"`` the region is Grace DRAM allocated on NUMA node ``numa_node`` (default: the node of
    CUDA device ``device``), leaving ``reserve_bytes`` free per node, and published as CUDA fabric handles;
    ``ucx_devices`` applies to ``"nixl"`` only."""

    directory: str = ""
    nbytes: int = 0
    chunk_bytes: int = REMOTE_CACHE_CHUNK_BYTES
    reserve_bytes: int = REMOTE_CACHE_LEND_RESERVE_BYTES
    numa_node: Optional[int] = None
    device: Optional[int] = None
    transport: str = "nixl"
    ucx_devices: Optional[str] = None
    name: str = ""
    host: str = ""
    _map: Optional[mmap.mmap] = None
    _address: int = 0
    _agent = None
    _registered: bool = False
    _published: Optional[list] = None  # paths written, withdrawn by close()
    _fabric: Optional[list] = None  # a fabric lend's cuMemCreate handles, one per chunk
    _fabric_ctx = None
    # The device whose primary context _fabric_ctx retains; close() releases it.
    _fabric_device = None
    # start_in_background()'s thread, joined by close().
    _start_thread: Optional[threading.Thread] = None
    # Serializes close(): a failed background start closes on its own thread.
    _close_lock: Optional[threading.Lock] = None

    def __init__(
        self,
        directory: str,
        nbytes: int,
        *,
        transport: str = "nixl",
        ucx_devices: Optional[str] = None,
        chunk_bytes: int = REMOTE_CACHE_CHUNK_BYTES,
        reserve_bytes: int = REMOTE_CACHE_LEND_RESERVE_BYTES,
        name: Optional[str] = None,
        numa_node: Optional[int] = None,
        device: Optional[int] = None,
    ):
        if nbytes < chunk_bytes:
            raise ValueError(f"lend size {nbytes} is below one chunk ({chunk_bytes})")
        if transport not in TRANSPORTS:
            raise ValueError(f"transport {transport!r} is not one of {TRANSPORTS}")
        self.directory = directory
        # Whole chunks only: a client never sees a partial chunk.
        self.nbytes = nbytes - nbytes % chunk_bytes
        self.chunk_bytes = chunk_bytes
        self.reserve_bytes = reserve_bytes
        self.numa_node = numa_node
        self.device = device
        self.transport = transport
        self.ucx_devices = ucx_devices
        host = socket.gethostname().split(".")[0]
        self.name = name or f"kvcm2-lender-{host}-{os.getpid()}-{id(self) & 0xFFFF:04x}"
        self.host = host
        self._map: Optional[mmap.mmap] = None
        self._address = 0
        self._agent = None
        self._registered = False
        self._published = []
        self._fabric = []  # a fabric lend's cuMemCreate handles, one per chunk
        self._fabric_ctx = None
        self._fabric_device = None
        self._start_thread = None
        self._close_lock = threading.Lock()

    @property
    def address(self) -> int:
        return self._address

    def _path(self, suffix: str) -> str:
        return os.path.join(self.directory, f"lender.{self.name}.{suffix}")

    def _publish(self, extra_files, meta_lines, timing: str) -> None:
        meta = "\n".join(
            [
                "version 1",
                f"name {self.name}",
                f"host {self.host}",
                f"pid {os.getpid()}",
                f"chunk_bytes {self.chunk_bytes}",
                *meta_lines,
                "",
            ]
        ).encode()
        for suffix, data in extra_files:
            _write_atomically(self._path(suffix), data)
            self._published.append(self._path(suffix))
        _write_atomically(self._path("meta"), meta)
        self._published.append(self._path("meta"))
        _write_atomically(self._path("ready"), b"")
        self._published.append(self._path("ready"))
        _log(
            f"{self.name} lends {self.nbytes / (1 << 30):.2f} GiB in {self.nbytes // self.chunk_bytes} chunk(s) "
            f"of {self.chunk_bytes / (1 << 30):.2f} GiB via {self.directory} ({timing})"
        )

    def _release_fabric(self) -> None:
        from cuda.bindings import driver as drv

        for handle in self._fabric:
            try:
                _cuda_check(drv.cuMemRelease(handle))
            except Exception as error:  # noqa: BLE001 - teardown keeps going
                _log(f"{self.name}: releasing a fabric chunk failed: {error}")
        self._fabric = []

    @staticmethod
    def _fabric_prop(drv, node: int):
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = node
        prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        return prop

    def _try_create(self, drv, prop):
        """One chunk, or None when the node is out of memory (any other error raises)."""
        try:
            return _cuda_check(drv.cuMemCreate(self.chunk_bytes, prop, 0))
        except RuntimeError as error:
            if "OUT_OF_MEMORY" in str(error):
                return None
            raise

    def _start_fabric(self) -> None:
        """Allocate the lend as Grace DRAM (cuMemCreate, HOST_NUMA), one allocation per chunk placed by _place_chunks,
        and publish their fabric handles in chunk order. What fits is published with short_bytes; a lender that fits
        nothing raises."""
        from cuda.bindings import driver as drv

        os.makedirs(self.directory, exist_ok=True)
        _cuda_check(drv.cuInit(0))
        device = _cuda_check(drv.cuDeviceGet(0 if self.device is None else self.device))
        self._fabric_ctx = _cuda_check(drv.cuDevicePrimaryCtxRetain(device))
        self._fabric_device = device
        _cuda_check(drv.cuCtxPushCurrent(self._fabric_ctx))
        handles = []
        planned = self.nbytes
        chunk_gib = self.chunk_bytes / _GIB
        reserve_gib = self.reserve_bytes / _GIB
        try:
            numa = self.numa_node if self.numa_node is not None else _host_numa_node_of(drv, device)
            nodes = [numa] + [n for n in _host_numa_nodes() if n != numa]
            props = {n: self._fabric_prop(drv, n) for n in nodes}
            gran = _cuda_check(
                drv.cuMemGetAllocationGranularity(
                    props[numa],
                    drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            )
            if self.chunk_bytes % gran:
                raise ValueError(
                    f"chunk size {self.chunk_bytes} is not a multiple of the fabric allocation granularity {gran}"
                )

            def create(node):
                handle = self._try_create(drv, props[node])
                if handle is not None:
                    # Released by the except below if anything fails later.
                    self._fabric.append(handle)
                return handle

            t0 = time.monotonic()
            placed, per_node = _place_chunks(
                planned // self.chunk_bytes,
                nodes,
                create,
                chunk_gib=chunk_gib,
                reserve_gib=reserve_gib,
                node_lock=lambda node: _node_compaction_lock(self.directory, self.host, node),
            )
            for handle in placed:
                exported = _cuda_check(
                    drv.cuMemExportToShareableHandle(
                        handle, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0
                    )
                )
                handles.append(bytes(exported.data))
                # The C++ client reads 64-byte CUmemFabricHandles back to back.
                if len(handles[-1]) != 64:
                    raise RuntimeError(f"fabric handle of {len(handles[-1])} B, expected 64")
            t1 = time.monotonic()
            if not handles:
                raise RuntimeError(
                    f"no NUMA node can hold a {chunk_gib:.2f} GiB fabric chunk and keep its "
                    f"{reserve_gib:.0f} GiB reserve"
                )
        except BaseException:
            # Nothing is published: release what was made.
            self._release_fabric()
            raise
        finally:
            _cuda_check(drv.cuCtxPopCurrent())
        self.nbytes = len(handles) * self.chunk_bytes
        short = planned - self.nbytes
        placement = ",".join(f"{n}:{c}" for n, c in per_node.items() if c)
        meta = [
            "transport fabric",
            f"numa {numa}",
            f"numa_chunks {placement}",
            f"segment 0 {self.nbytes}",
        ]
        if short:
            meta.append(f"short_bytes {short}")
            _log(
                f"{self.name}: only {self.nbytes / _GIB:.0f} of {planned / _GIB:.0f} GiB fit as fabric memory "
                f"(reserve {reserve_gib:.0f} GiB per node)"
            )
        self._publish(
            [("fabric", b"".join(handles))],
            meta,
            f"fabric memory, chunks per NUMA node {placement}, {len(handles)} allocation(s) created in "
            f"{t1 - t0:.1f} s",
        )

    def publish_failure(self, error: BaseException) -> None:
        """Leave lender.<name>.failed (planned bytes and the error) so capacity checks count this lender as reported
        instead of waiting for it. Clients ignore the file: they list only .ready lenders."""
        try:
            os.makedirs(self.directory, exist_ok=True)
            text = "\n".join(
                [
                    "version 1",
                    f"name {self.name}",
                    f"host {self.host}",
                    f"pid {os.getpid()}",
                    f"transport {self.transport}",
                    f"planned_bytes {self.nbytes}",
                    f"error {error!r}".replace("\n", " "),
                    "",
                ]
            )
            _write_atomically(self._path("failed"), text.encode())
        except OSError as write_error:
            _log(f"{self.name}: cannot write its failure marker: {write_error}")

    def start(self) -> None:
        if self.transport == "fabric":
            self._start_fabric()
            return
        from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
            BaseAgentConfig,
            MemoryDesc,
            MemoryDescs,
            MemoryType,
            NixlTransferAgent,
        )

        os.makedirs(self.directory, exist_ok=True)
        t0 = time.monotonic()
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE
        self._map = mmap.mmap(-1, self.nbytes, flags=flags)
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        try:
            self._map.madvise(_MADV_HUGEPAGE)
        except OSError:
            pass
        if self.numa_node is not None and not _bind_to_numa_node(
            self._address, self.nbytes, self.numa_node
        ):
            _log(f"libnuma unavailable; {self.name} relies on first touch for NUMA placement")
        # Fault every page in now: registration does not pin memory where the NIC pages on demand, and a resident
        # lend fails here rather than mid-run and never makes a remote write wait on a page fault.
        _populate(self._map, self.nbytes)
        t1 = time.monotonic()

        config = BaseAgentConfig()
        config.name = self.name
        config.use_prog_thread = True
        config.multi_thread = False
        config.use_listen_thread = False
        # This agent's own UCX devices; the process-wide UCX_NET_DEVICES stays as the serving process needs it.
        if self.ucx_devices:
            config.backend_params = {"ucx_devices": self.ucx_devices}
        self._agent = NixlTransferAgent(config)
        self._agent.register_memory(
            MemoryDescs(MemoryType.DRAM, [MemoryDesc(self._address, self.nbytes, 0)])
        )
        self._registered = True
        t2 = time.monotonic()
        desc = self._agent.get_local_agent_desc().serialize()
        self._publish(
            [("desc", desc)],
            [f"segment {self._address} {self.nbytes}"],
            f"map + populate {t1 - t0:.1f} s, register {t2 - t1:.1f} s",
        )

    def start_or_report(self) -> None:
        """``start()``; on failure, release what was made and leave the failure marker (``publish_failure``)."""
        try:
            self.start()
        except Exception as error:  # noqa: BLE001 - a failed lender only removes capacity
            _log(f"{self.name} failed to start: {error!r}")
            self.close()
            self.publish_failure(error)

    def start_in_background(self) -> None:
        """``start_or_report()`` on a thread, so the caller does not wait for the memory to be faulted in and
        registered; clients wait for the ``.ready`` file."""
        self._start_thread = threading.Thread(
            target=self.start_or_report, name="kvcm2-remote-lender", daemon=True
        )
        self._start_thread.start()

    def wait_started(self) -> None:
        """Wait for a background start to publish or fail."""
        thread = self._start_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def close(self) -> None:
        """Wait for a background start, then withdraw the files, the registration and the memory. NIXL clients still
        holding chunks fail their next transfer at the NIC instead of reading stale bytes."""
        thread, self._start_thread = self._start_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._close_lock:
            self._withdraw()

    def _withdraw(self) -> None:
        for path in reversed(self._published):
            try:
                os.unlink(path)
            except OSError:
                pass
        self._published.clear()
        if self._fabric_ctx is not None:
            from cuda.bindings import driver as drv

            if self._fabric:
                try:
                    _cuda_check(drv.cuCtxPushCurrent(self._fabric_ctx))
                    try:
                        self._release_fabric()
                    finally:
                        _cuda_check(drv.cuCtxPopCurrent())
                except Exception as error:  # noqa: BLE001 - teardown keeps going
                    _log(f"{self.name}: releasing the fabric lend failed: {error}")
            try:
                _cuda_check(drv.cuDevicePrimaryCtxRelease(self._fabric_device))
            except Exception as error:  # noqa: BLE001 - teardown keeps going
                _log(f"{self.name}: releasing the primary context failed: {error}")
            self._fabric_ctx = None
            self._fabric_device = None
        self._fabric = []
        if self._agent is not None and self._registered:
            from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
                MemoryDesc,
                MemoryDescs,
                MemoryType,
            )

            try:
                self._agent.deregister_memory(
                    MemoryDescs(MemoryType.DRAM, [MemoryDesc(self._address, self.nbytes, 0)])
                )
            except Exception as error:  # noqa: BLE001 - teardown keeps going
                _log(f"{self.name}: deregistering failed: {error}")
            self._registered = False
        self._agent = None
        if self._map is not None:
            try:
                self._map.close()
            except BufferError:
                pass
            self._map = None


def start_lend(remote_config: "RemoteCacheConfig", device: int) -> RemoteLender:
    """Start lending ``remote_config.lend_size`` bytes of host memory near CUDA device `device`, in the background."""
    directory = os.path.expandvars(os.path.expanduser(remote_config.directory))
    lender = RemoteLender(
        directory,
        remote_config.lend_size,
        transport=remote_config.transport,
        ucx_devices=resolve_ucx_devices(remote_config.ucx_devices, device),
        reserve_bytes=remote_config.lend_reserve_size,
        device=device,
    )
    lender.start_in_background()
    logger.info(
        f"KV cache manager v2 lends {remote_config.lend_size / _GIB:.2f}GiB of host memory "
        f"to remote cache levels via {directory} ({remote_config.transport})"
    )
    return lender
