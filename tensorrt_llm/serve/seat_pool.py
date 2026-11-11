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
"""The queue of a pooled generation router (``router.PooledLoadBalancingRouter``).

Every generation server has ``seats`` seats; a request holds one from its placement until it
finishes. A request that finds no free seat waits here, and each freed seat goes to the first
waiter in order that may use that server: the highest request ``priority`` first (the
orchestrator's predicted output length with ``output_length_prediction``), ties in arrival order.
With ``max_wait_s``, a request that has waited that long goes ahead of every request that has
not, oldest first.
"""

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.logger import logger

STATS_LOG_PERIOD_S = 30.0


def priority_or_default(priority: Any) -> float:
    """A request priority as a number in [0, 1]; anything else is the default priority."""
    if isinstance(priority, (int, float)) and not isinstance(priority, bool):
        if math.isfinite(priority) and 0.0 <= priority <= 1.0:
            return float(priority)
    return DEFAULT_REQUEST_PRIORITY


@dataclass
class Waiter:
    """One request waiting for a seat; ``holders`` counts the callers waiting on its placement."""

    rid: int
    seq: int
    arrival: float
    priority: float
    exclude: Optional[str]
    future: Any = None
    holders: int = 0
    arriving: bool = True


def _pct(values: List[float], q: float) -> float:
    return values[min(len(values) - 1, int(q * len(values)))]


class SeatPool:
    """Waiters, their order and placement counters. Not thread safe: its router calls it under one lock."""

    def __init__(
        self,
        seats: int,
        max_wait_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(seats, bool) or not isinstance(seats, int) or seats < 1:
            raise ValueError(f"max_requests_per_server={seats!r} must be an integer >= 1")
        if max_wait_s is not None and not (math.isfinite(max_wait_s) and max_wait_s > 0.0):
            raise ValueError(f"max_queue_wait_s={max_wait_s!r} must be a positive number or None")
        self.seats = seats
        self.max_wait_s = max_wait_s
        self._clock = clock
        # rid -> waiter, in arrival order (a waiter is never re-inserted).
        self._waiters: dict[int, Waiter] = {}
        self._seq = itertools.count()
        self._pick_now: Optional[float] = None
        self.placed = 0
        self.placed_at_once = 0
        self.reordered = 0
        self.aged = 0
        self.dropped = 0
        self._win_start = clock()
        self._win_placed = 0
        self._win_waits: List[float] = []

    def describe(self, num_servers: int) -> str:
        cap = (
            f"a request that has waited {self.max_wait_s:g} s goes ahead, oldest first"
            if self.max_wait_s is not None
            else "no wait cap"
        )
        return (
            f"generation requests wait in one queue for a free seat ({self.seats} per server, "
            f"{num_servers} servers), highest request priority first, ties in arrival order; {cap}"
        )

    def __len__(self) -> int:
        return len(self._waiters)

    def waiter(self, rid: int) -> Optional[Waiter]:
        return self._waiters.get(rid)

    def add(self, rid: int, priority: Any, exclude: Optional[str], future: Any) -> Waiter:
        """A new waiter, arriving now."""
        if rid in self._waiters:
            raise ValueError(f"request {rid} is already waiting")
        waiter = Waiter(
            rid=rid,
            seq=next(self._seq),
            arrival=self._clock(),
            priority=priority_or_default(priority),
            exclude=exclude,
            future=future,
        )
        self._waiters[rid] = waiter
        return waiter

    def remove(self, rid: int) -> Optional[Waiter]:
        """Drop a waiter that finished or was cancelled before its placement."""
        waiter = self._waiters.pop(rid, None)
        if waiter is not None:
            self.dropped += 1
        return waiter

    def _is_aged(self, waiter: Waiter, now: float) -> bool:
        return self.max_wait_s is not None and now - waiter.arrival >= self.max_wait_s

    def sort_key(self, waiter: Waiter, now: float) -> Tuple:
        """Smaller goes first: aged waiters by arrival, then the rest by (-priority, arrival)."""
        if self._is_aged(waiter, now):
            return (0, 0.0, waiter.seq)
        return (1, -waiter.priority, waiter.seq)

    def pick(self, free_servers: Sequence[str]) -> Optional[Tuple[Waiter, List[str]]]:
        """The first waiter in order that may use one of ``free_servers``, with those servers."""
        if not self._waiters or not free_servers:
            return None
        now = self._pick_now = self._clock()
        best, best_key, best_servers = None, None, None
        for waiter in self._waiters.values():
            servers = [s for s in free_servers if s != waiter.exclude]
            if not servers:
                continue
            key = self.sort_key(waiter, now)
            if best_key is None or key < best_key:
                best, best_key, best_servers = waiter, key, servers
        if best is None:
            return None
        return best, best_servers

    def record_grant(self, waiter: Waiter) -> None:
        """``waiter`` (from :meth:`pick`) took a seat; it leaves the queue."""
        now = self._pick_now if self._pick_now is not None else self._clock()
        self._pick_now = None
        first_seq = next(iter(self._waiters.values())).seq
        self._waiters.pop(waiter.rid, None)
        self.placed += 1
        self._win_placed += 1
        self.reordered += waiter.seq != first_seq
        self.aged += self._is_aged(waiter, now)
        self.placed_at_once += waiter.arriving
        self._win_waits.append(max(0.0, now - waiter.arrival))

    def maybe_log_stats(self, seats_used: int, seats_total: int) -> Optional[str]:
        """The stats line every ``STATS_LOG_PERIOD_S`` with a placement (also returned)."""
        now = self._clock()
        if now - self._win_start < STATS_LOG_PERIOD_S or not self._win_placed:
            return None
        waits = sorted(self._win_waits)
        line = (
            f"[seat_pool] {self._win_placed} placed in {now - self._win_start:.0f} s, queue wait "
            f"p50 / p90 / max {_pct(waits, 0.5):.2f} / {_pct(waits, 0.9):.2f} / {waits[-1]:.2f} s; "
            f"{len(self._waiters)} waiting; seats in use {seats_used} / {seats_total}; total placed "
            f"{self.placed} ({self.placed_at_once} at once, {self.reordered} ahead of an earlier "
            f"arrival, {self.aged} past the cap), dropped {self.dropped}"
        )
        self._win_start = now
        self._win_placed = 0
        self._win_waits = []
        logger.info(line)
        return line
