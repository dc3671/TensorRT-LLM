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
"""Pooled generation dispatch: serve/seat_pool.py and router.PooledLoadBalancingRouter."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
    extract_disagg_cfg,
)
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams
from tensorrt_llm.serve.router import (
    POOLED_SELECT_TIMEOUT_S,
    CoordinatorDelegatingRouter,
    LoadBalancingRouter,
    PooledLoadBalancingRouter,
    create_router,
)
from tensorrt_llm.serve.seat_pool import SeatPool, priority_or_default

pytestmark = pytest.mark.cpu_only


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------------- SeatPool


def test_priority_or_default():
    assert priority_or_default(0.25) == 0.25 and priority_or_default(1) == 1.0
    for bad in (None, True, "0.3", float("nan"), -0.1, 1.5):
        assert priority_or_default(bad) == DEFAULT_REQUEST_PRIORITY


@pytest.mark.parametrize("seats, cap", [(0, None), (True, None), (1.5, None), (1, 0.0), (1, -1.0)])
def test_pool_rejects_bad_options(seats, cap):
    with pytest.raises(ValueError):
        SeatPool(seats, cap)


def test_pool_order_highest_priority_then_arrival():
    pool = SeatPool(1, clock=_Clock())
    for rid, priority in [(1, 0.3), (2, 0.9), (3, 0.9), (4, None)]:
        pool.add(rid, priority, None, None)
    order = []
    while len(pool):
        waiter, servers = pool.pick(["s0"])
        assert servers == ["s0"]
        pool.record_grant(waiter)
        order.append(waiter.rid)
    assert order == [2, 3, 4, 1]
    assert pool.placed == 4 and pool.reordered == 3


def test_pool_cap_puts_aged_waiters_first_oldest_first():
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=60.0, clock=clock)
    pool.add(1, 0.1, None, None)
    clock.now = 10.0
    pool.add(2, 0.2, None, None)
    clock.now = 50.0
    pool.add(3, 0.9, None, None)
    clock.now = 69.0
    assert pool.pick(["s"])[0].rid == 1  # only request 1 has waited 60 s
    clock.now = 70.0
    first = pool.pick(["s"])[0]
    pool.record_grant(first)
    assert first.rid == 1 and pool.aged == 1
    assert pool.pick(["s"])[0].rid == 2


def test_pool_ages_from_the_orchestrator_arrival():
    """A request past its context phase has waited since it reached the orchestrator."""
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=60.0, clock=clock)
    pool.add(1, 0.9, None, None)
    pool.add(2, 0.1, None, None, orig_elapsed_s=55.0)
    assert pool.pick(["s"])[0].rid == 1
    clock.now = 5.0
    assert pool.pick(["s"])[0].rid == 2  # 60 s since its arrival, 5 s in the pool


def test_pool_expires_at_the_kv_transfer_timeout():
    """Only a request past its context phase gets the deadline, counted from its entry here."""
    clock = _Clock()
    pool = SeatPool(1, clock=clock, kv_transfer_timeout_s=60.0)
    pool.add(1, 0.5, None, None, orig_elapsed_s=10.0)
    pool.add(2, 0.5, None, None)
    clock.now = 59.9
    assert pool.expire() == [] and pool.remaining_s(pool.waiter(1)) == pytest.approx(0.1)
    clock.now = 60.0
    expired = pool.expire()
    assert [w.rid for w in expired] == [1] and pool.expired == 1
    assert "KV transfer timeout (60 s" in pool.expiry_message(expired[0])
    assert len(pool) == 1 and pool.remaining_s(pool.waiter(2)) is None
    # Without the timeout nothing expires on its own.
    plain = SeatPool(1, clock=clock)
    plain.add(3, 0.5, None, None, orig_elapsed_s=10.0)
    clock.now = 1e6
    assert plain.expire() == []


def test_pool_skips_excluded_server():
    pool = SeatPool(1, clock=_Clock())
    pool.add(1, 0.9, "s0", None)
    pool.add(2, 0.1, None, None)
    assert pool.pick(["s0"])[0].rid == 2
    waiter, servers = pool.pick(["s0", "s1"])
    assert waiter.rid == 1 and servers == ["s1"]
    assert pool.pick([]) is None


# ---------------------------------------------------------------------------------- router


def _router(servers=("s0", "s1"), seats=1, cap=None, kv_ms=None) -> PooledLoadBalancingRouter:
    args = {"max_requests_per_server": seats, "max_queue_wait_s": cap}
    if kv_ms is not None:
        args["kv_transfer_timeout_ms"] = kv_ms
    config = RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION, args=args)
    router = create_router(config, list(servers))
    assert isinstance(router, PooledLoadBalancingRouter)
    return router


def test_create_router():
    plain = create_router(
        RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION), ["s"]
    )
    assert type(plain) is LoadBalancingRouter and not getattr(plain, "pooled", False)
    with pytest.raises(ValueError, match="load_balancing"):
        create_router(
            RouterConfig(
                type="round_robin",
                server_role=ServerRole.GENERATION,
                args={"max_requests_per_server": 1},
            ),
            ["s"],
        )
    with pytest.raises(ValueError, match="generation"):
        create_router(
            RouterConfig(
                type="load_balancing",
                server_role=ServerRole.CONTEXT,
                args={"max_requests_per_server": 1},
            ),
            ["s"],
        )
    with pytest.raises(ValueError, match="use_tokens"):
        create_router(
            RouterConfig(
                type="load_balancing",
                server_role=ServerRole.GENERATION,
                args={"max_requests_per_server": 1, "use_tokens": True},
            ),
            ["s"],
        )


def test_routing_key_carries_priority():
    router = _router()
    assert router.routing_key(CompletionRequest(model="m", prompt="x", priority=0.25)) == {
        "num_tokens": 0,
        "priority": 0.25,
    }
    assert router.routing_key(CompletionRequest(model="m", prompt="x"))["priority"] == 0.5
    after_ctx = CompletionRequest(
        model="m",
        prompt="x",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", orig_elapsed_s=2.5
        ),
    )
    assert router.routing_key(after_ctx)["orig_elapsed_s"] == 2.5


def test_extract_disagg_cfg_gives_the_pooled_router_the_ctx_kv_timeout():
    pooled = {"router": {"type": "load_balancing", "max_requests_per_server": 1}}
    default = extract_disagg_cfg(generation_servers=dict(pooled, router=dict(pooled["router"])))
    assert default.gen_router_config.args["kv_transfer_timeout_ms"] == 60000
    config = extract_disagg_cfg(
        context_servers={"cache_transceiver_config": {"kv_transfer_timeout_ms": 30000}},
        generation_servers=dict(pooled, router=dict(pooled["router"])),
    )
    assert config.gen_router_config.args["kv_transfer_timeout_ms"] == 30000
    router = create_router(config.gen_router_config, ["s"])
    assert router._pool.kv_transfer_timeout_s == 30.0
    plain = extract_disagg_cfg(generation_servers={"router": {"type": "load_balancing"}})
    assert "kv_transfer_timeout_ms" not in plain.gen_router_config.args


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_free_seats_place_at_once_like_load_balancing():
    router = _router(seats=2)
    stock = LoadBalancingRouter(ServerRole.GENERATION, ["s0", "s1"])
    placed, expected = [], []
    for rid in range(4):
        placed.append((await router.get_next_server_by_key({}, req_id=rid))[0])
        expected.append((await stock.get_next_server_by_key({}, req_id=rid))[0])
    assert placed == expected
    assert router._pool.placed_at_once == 4


@pytest.mark.asyncio
async def test_freed_seat_goes_to_the_best_waiter():
    router = _router(servers=("s0",), seats=1)
    assert (await router.get_next_server_by_key({}, req_id=0))[0] == "s0"
    waits = {
        rid: asyncio.create_task(router.get_next_server_by_key({"priority": p}, req_id=rid))
        for rid, p in [(1, 0.2), (2, 0.8), (3, 0.8)]
    }
    await _settle()
    assert not any(t.done() for t in waits.values()) and len(router._pool) == 3
    order = []
    for done_rid in (0, 2, 3):
        await router.finish_request_by_id(done_rid)
        await _settle()
        ready = [rid for rid, t in waits.items() if t.done() and rid not in order]
        assert len(ready) == 1
        order.extend(ready)
    assert order == [2, 3, 1]
    assert (await waits[1])[0] == "s0"


@pytest.mark.asyncio
async def test_repeated_select_keeps_its_seat_and_place():
    router = _router(servers=("s0",), seats=1)
    first = await router.get_next_server_by_key({}, req_id=1)
    assert await router.get_next_server_by_key({}, req_id=1) == first
    a = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    b = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert len(router._pool) == 1 and router._pool.waiter(2).holders == 2
    await router.finish_request_by_id(1)
    assert (await a)[0] == (await b)[0] == "s0"
    assert router._get_server_load("s0") == 1


@pytest.mark.asyncio
async def test_finish_while_waiting_drops_the_waiter():
    router = _router(servers=("s0",), seats=1)
    await router.get_next_server_by_key({}, req_id=1)
    waiting = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    await router.finish_request_by_id(2)
    with pytest.raises(ValueError, match="finished while waiting"):
        await waiting
    assert len(router._pool) == 0 and router._get_server_load("s0") == 1


@pytest.mark.asyncio
async def test_cancellation_releases_queue_place_and_seat():
    router = _router(servers=("s0",), seats=1)
    await router.get_next_server_by_key({}, req_id=1)
    waiting = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert len(router._pool) == 0
    # A placement whose only caller was cancelled gives its seat back.
    late = asyncio.create_task(router.get_next_server_by_key({}, req_id=3))
    await _settle()
    await router.finish_request_by_id(1)
    late.cancel()
    with pytest.raises(asyncio.CancelledError):
        await late
    assert router._get_server_load("s0") == 0 and router._req_routing_table == {}


@pytest.mark.asyncio
async def test_waiter_fails_at_the_kv_transfer_timeout():
    """A request past its context phase that finds no seat within the timeout fails; others wait."""
    router = _router(servers=("s0",), seats=1, kv_ms=50)
    await router.get_next_server_by_key({}, req_id=0)
    doomed = asyncio.create_task(
        router.get_next_server_by_key({"priority": 0.9, "orig_elapsed_s": 3.0}, req_id=1)
    )
    patient = asyncio.create_task(router.get_next_server_by_key({"priority": 0.1}, req_id=2))
    with pytest.raises(ValueError, match="KV transfer timeout"):
        await asyncio.wait_for(doomed, timeout=5)
    assert len(router._pool) == 1 and router._pool.expired == 1 and not patient.done()
    await router.finish_request_by_id(0)
    assert (await asyncio.wait_for(patient, timeout=5))[0] == "s0"
    # The in-process path reads the elapsed time from the request.
    request = CompletionRequest(
        model="m",
        prompt="x",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", orig_elapsed_s=1.0
        ),
    )
    with pytest.raises(ValueError, match="KV transfer timeout"):
        await asyncio.wait_for(router.get_next_server(request), timeout=5)
    assert router._get_server_load("s0") == 1 and len(router._pool) == 0


@pytest.mark.asyncio
async def test_in_process_path_keys_by_request():
    router = _router(servers=("s0",), seats=1)
    first = CompletionRequest(model="m", prompt="x", priority=0.1)
    second = CompletionRequest(model="m", prompt="x", priority=0.9)
    assert (await router.get_next_server(first))[0] == "s0"
    waiting = asyncio.create_task(router.get_next_server(second))
    await _settle()
    assert not waiting.done()
    await router.finish_request(first)
    assert (await waiting)[0] == "s0"


# ---------------------------------------------------------------------------------- coordinator


def _client_factory(*_args, **_kwargs):
    return AsyncMock()


def _coordinator(seats=1):
    config = DisaggServerConfig(
        server_configs=[CtxGenServerConfig(type="gen", hostname="g", port=1)],
        ctx_router_config=RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(
            type="load_balancing",
            server_role=ServerRole.GENERATION,
            args={"max_requests_per_server": seats},
        ),
    )
    return DisaggCoordinatorService(config, _client_factory, reservation_timeout_secs=60)


@pytest.mark.asyncio
async def test_pooled_select_waits_outside_the_reservation_lock():
    """A waiting select must not block the finish (on the same lock stripe) that frees its seat."""
    coordinator = _coordinator()
    assert isinstance(coordinator.gen_router, PooledLoadBalancingRouter)
    stripe = coordinator._reservation_lock(("generation", 1))
    other = next(
        rid
        for rid in range(2, 100000)
        if coordinator._reservation_lock(("generation", rid)) is stripe
    )
    assert (await coordinator.select("generation", {}, 1, None))[0] == "g:1"
    waiting = asyncio.create_task(coordinator.select("generation", {"priority": 0.5}, other, None))
    await _settle()
    assert not waiting.done()
    await asyncio.wait_for(coordinator.finish("generation", 1), timeout=5)
    server, _, req_id = await asyncio.wait_for(waiting, timeout=5)
    assert (server, req_id) == ("g:1", other)
    # A repeated select renews the reservation and keeps the seat.
    await coordinator.select("generation", {}, other, None)
    assert coordinator.gen_router._get_server_load("g:1") == 1
    await coordinator.finish("generation", other)
    assert coordinator.gen_router._get_server_load("g:1") == 0
    assert coordinator._reservation_tasks == {}


def test_delegating_router_waits_longer_for_a_pooled_select():
    pooled = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0)
    plain = CoordinatorDelegatingRouter(
        "http://c:1", LoadBalancingRouter(ServerRole.GENERATION, ["s"]), "generation", 5.0
    )
    assert pooled._select_timeout_s == POOLED_SELECT_TIMEOUT_S
    assert plain._select_timeout_s == 5.0
