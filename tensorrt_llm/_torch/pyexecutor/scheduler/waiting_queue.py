# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import heapq
import itertools
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Optional

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy
from tensorrt_llm.logger import logger

from ..executor_request_queue import RequestQueueItem


class WaitingQueue(ABC):
    """Abstract base class for waiting queues."""

    @abstractmethod
    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> RequestQueueItem:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        """Return up to ``k`` requests in pop order without removing them.

        Implementations must not copy or sort the whole queue for a small ``k``.
        """
        pass

    def admission_groups(self, requests: list[RequestQueueItem]) -> list[int]:
        """Admission group of each request from ``peek_requests``, numbered in pop order.

        Out-of-order admission must finish a group before the next one. Default:
        one group.
        """
        return [0] * len(requests)

    @abstractmethod
    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert a request that could not be scheduled.

        Implementations must ensure the request is served before any request
        (of equal priority in the case of a priority queue) that was added
        *after* it originally arrived (i.e. it does not lose
        its place to later-arriving requests of equal priority).  For FCFS
        queues this means inserting at the front; for priority queues this
        means using a tiebreaker that sorts ahead of normally-added entries.
        """
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert multiple requests that could not be scheduled.

        See prepend_request for the ordering contract.  Requests are
        re-inserted such that the first request in the iterable comes out
        first after re-insertion (i.e. original queue order is preserved).
        """
        pass

    def mark_admitted(self, requests: Iterable[RequestQueueItem]) -> None:
        """Called with the popped requests that were admitted, not put back."""

    @abstractmethod
    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSWaitingQueue(deque, WaitingQueue):
    """A first-come-first-served queue that supports deque operations."""

    @staticmethod
    def _warn_if_priority_set(request: RequestQueueItem) -> None:
        if request.request is not None and request.request.priority != DEFAULT_REQUEST_PRIORITY:
            logger.warning(
                "A request has a non-default priority but the FCFS waiting "
                "queue is in use; the priority value will be ignored. "
                "Use WaitingQueuePolicy.PRIORITY to enable priority scheduling."
            )

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to FCFS policy."""
        self._warn_if_priority_set(request)
        self.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
        requests = list(requests)
        for request in requests:
            self._warn_if_priority_set(request)
        self.extend(requests)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        """Return up to ``k`` requests from the front of the queue, O(k)."""
        return list(itertools.islice(self, max(k, 0)))

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue.

        Requests are inserted in input order: the first item in ``requests``
        ends up at the front of the queue.
        """
        self.extendleft(reversed(list(requests)))

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        filtered_requests = [req for req in self if req.id not in request_ids]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityWaitingQueue(WaitingQueue):
    """A priority queue that serves higher-priority requests first.

    Requests with equal priority are served in FCFS order (arrival order).
    Priority is read from ``item.request.priority``; the default is DEFAULT_REQUEST_PRIORITY.

    Heap entries are ``(neg_priority, insertion_counter, item)`` tuples.
    Because ``insertion_counter`` is strictly monotonically increasing, no two
    entries share the same first two elements, so ``item`` is never compared
    and does not need to implement ``__lt__``.

    Complexity:
        - add_request / add_requests: O(log n)
        - pop_request / peek_request: O(log n) / O(1)
        - peek_requests(k): O(n log k), without copying or sorting the heap
        - remove_by_ids: O(n) rebuild + O(n) heapify
        - __iter__: O(n log n) over a sorted copy (priority order)
    """

    def __init__(self) -> None:
        # Min-heap of (neg_priority, insertion_counter, RequestQueueItem).
        self._heap: list[tuple] = []
        # _counter assigns strictly increasing non-negative integers to
        # normally-added requests for FCFS tiebreaking within a priority cohort.
        # _prepend_counter assigns strictly decreasing negative integers to
        # prepended requests, guaranteeing they sort before any normally-added
        # request of the same priority.
        self._counter = itertools.count()
        self._prepend_counter = itertools.count(-1, -1)

    def _get_priority(self, item: RequestQueueItem) -> float:
        if item.request is not None:
            return item.request.priority
        return DEFAULT_REQUEST_PRIORITY

    def _push(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._counter), item)
        heapq.heappush(self._heap, entry)

    def _push_front(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._prepend_counter), item)
        heapq.heappush(self._heap, entry)

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue in priority order."""
        self._push(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue in priority order."""
        for request in requests:
            self._push(request)

    def pop_request(self) -> RequestQueueItem:
        """Pop the highest-priority request (FCFS tiebreak)."""
        if not self._heap:
            raise IndexError("pop from an empty queue")
        _, _, item = heapq.heappop(self._heap)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Return the highest-priority request without removing it."""
        if not self._heap:
            raise IndexError("peek from an empty queue")
        return self._heap[0][2]

    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        """Return up to ``k`` requests in priority order without removing them."""
        return [e[2] for e in heapq.nsmallest(max(k, 0), self._heap)]

    def admission_groups(self, requests: list[RequestQueueItem]) -> list[int]:
        """One group per priority."""
        groups = []
        group = 0
        for i, request in enumerate(requests):
            if i and self._get_priority(request) != self._get_priority(requests[i - 1]):
                group += 1
            groups.append(group)
        return groups

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert a request ahead of all normally-added requests of the same priority.

        Uses a strictly decreasing counter so the request sorts before any
        request added via add_request / add_requests, preserving FCFS order
        within a priority cohort when requests are returned to the queue.
        """
        self._push_front(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert requests ahead of all normally-added requests of the same priority.

        Requests are pushed in *reverse* iteration order so that the first
        item in ``requests`` ends up with the most-negative prepend counter
        and therefore sorts first within its priority cohort.  This means
        the first request in the iterable comes out first after re-insertion,
        preserving original queue order without any reversal by the caller.
        """
        for request in reversed(list(requests)):
            self._push_front(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        self._heap = [e for e in self._heap if e[2].id not in request_ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over requests in descending priority order.

        Returns a generator over a snapshot of the heap taken at call time.
        Modifications to the queue after iteration begins are not reflected.
        """
        return (e[2] for e in sorted(self._heap))


class ShortestDecodeFirstWaitingQueue(WaitingQueue):
    """Serves the request with the fewest ``max_tokens`` first.

    Ties, and requests without ``max_tokens`` (which sort last), go in arrival
    order. With ``max_wait_s``, a request that has waited that long goes ahead
    of every request that has not, oldest first.

    Every rank pops the same order: the keys are the broadcast request's
    ``max_tokens``, the arrival order and, for the cap, rank 0's arrival stamp
    (``RequestQueueItem.admit_arrival``) against rank 0's clock (``set_now``).

    Entries are ``(max_tokens, arrival_seq, item)`` keyed by the unique
    ``arrival_seq``, so items are never compared. A request put back keeps its
    arrival sequence, so it goes ahead of later arrivals with the same key. The
    pop order changes only with the entries or the clock, so it is sorted once
    and popped from until then.
    """

    _NO_MAX_TOKENS = float("inf")

    def __init__(self, max_wait_s: Optional[float] = None) -> None:
        if max_wait_s is not None and not (math.isfinite(max_wait_s) and max_wait_s > 0.0):
            raise ValueError(f"max_wait_s={max_wait_s!r} must be a positive number or None")
        self.max_wait_s = max_wait_s
        # Arrival sequence -> (max_tokens, sequence, item).
        self._entries: dict[int, tuple] = {}
        # Entries in reverse pop order; None when stale.
        self._order: Optional[list[tuple]] = None
        self._now: Optional[float] = None
        self._counter = itertools.count()
        # For a request put back that was never added.
        self._prepend_counter = itertools.count(-1, -1)
        # Request id -> arrival sequence, while queued or popped and not yet admitted.
        self._seq: dict[int, int] = {}

    def set_now(self, now: Optional[float]) -> None:
        """Rank 0's clock, set once per iteration before the pops; None keeps the last value."""
        if now is not None:
            self._now = now
            self._order = None

    @classmethod
    def _max_tokens_of(cls, item: RequestQueueItem) -> float:
        if item.request is None or item.request.max_tokens is None:
            return cls._NO_MAX_TOKENS
        return float(item.request.max_tokens)

    def _is_aged(self, item: RequestQueueItem) -> bool:
        if self.max_wait_s is None or self._now is None or item.admit_arrival is None:
            return False
        return self._now - item.admit_arrival >= self.max_wait_s

    def _key(self, entry: tuple) -> tuple:
        tokens, seq, item = entry
        return (0, seq) if self._is_aged(item) else (1, tokens, seq)

    def _get_order(self) -> list[tuple]:
        if self._order is None:
            self._order = sorted(self._entries.values(), key=self._key, reverse=True)
        return self._order

    def _push(self, item: RequestQueueItem, seq: int) -> None:
        self._entries[seq] = (self._max_tokens_of(item), seq, item)
        self._seq[item.id] = seq
        self._order = None

    def add_request(self, request: RequestQueueItem) -> None:
        self._push(request, next(self._counter))

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        if not self._entries:
            raise IndexError("pop from an empty queue")
        if self.max_wait_s is not None and self._now is None:
            raise RuntimeError(
                "ShortestDecodeFirstWaitingQueue with a wait cap needs set_now() before the first pop"
            )
        _, seq, item = self._get_order().pop()
        del self._entries[seq]
        return item

    def peek_request(self) -> RequestQueueItem:
        if not self._entries:
            raise IndexError("peek from an empty queue")
        return self._get_order()[-1][2]

    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        if k <= 0 or not self._entries:
            return []
        return [entry[2] for entry in self._get_order()[: -k - 1 : -1]]

    def prepend_request(self, request: RequestQueueItem) -> None:
        seq = self._seq.get(request.id)
        self._push(request, seq if seq is not None else next(self._prepend_counter))

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        # Reversed, so that among requests without an arrival sequence the first gets the lowest one.
        for request in reversed(list(requests)):
            self.prepend_request(request)

    def mark_admitted(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in requests:
            self._seq.pop(request.id, None)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        self._entries = {s: e for s, e in self._entries.items() if e[2].id not in request_ids}
        self._order = None
        for request_id in request_ids:
            self._seq.pop(request_id, None)

    def __bool__(self) -> bool:
        return len(self._entries) > 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Requests in pop order, over a snapshot taken at call time."""
        return iter([entry[2] for entry in reversed(self._get_order())])


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    gen_admission=None,
    max_wait_s: Optional[float] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use.
        gen_admission: Optional ``GenAdmissionConfig``; overrides ``policy`` when enabled.
        max_wait_s: The wait cap of ``WaitingQueuePolicy.SHORTEST_DECODE_FIRST``; None: no cap.

    Returns:
        A WaitingQueue instance.
    """
    if policy == WaitingQueuePolicy.SHORTEST_DECODE_FIRST:
        if gen_admission is not None and gen_admission.enabled:
            raise ValueError(
                f"waiting_queue_policy={policy} cannot be combined with GEN admission by original "
                f"arrival ({gen_admission.describe()})"
            )
        return ShortestDecodeFirstWaitingQueue(max_wait_s)
    if gen_admission is not None and gen_admission.enabled:
        from .gen_admission_queue import GenAdmissionWaitingQueue

        if policy != WaitingQueuePolicy.FCFS:
            logger.warning(
                f"[gen_admission] {gen_admission.describe()} overrides "
                f"waiting_queue_policy={policy}; request priorities are ignored."
            )
        return GenAdmissionWaitingQueue(gen_admission)
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.PRIORITY:
        return PriorityWaitingQueue()
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
