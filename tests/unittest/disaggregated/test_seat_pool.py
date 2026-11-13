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
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import msgpack
import pytest
from fastapi import HTTPException

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
    extract_disagg_cfg,
)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import (
    POOLED_RESERVATION_TIMEOUT_S,
    DisaggCoordinatorService,
)
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams
from tensorrt_llm.serve.router import (
    POOLED_SELECT_TIMEOUT_S,
    CoordinatorDelegatingRouter,
    LoadBalancingRouter,
    PooledLoadBalancingRouter,
    Router,
    build_disagg_routers,
    create_router,
)
from tensorrt_llm.serve.seat_pool import NoGenerationSeatError, SeatPool, priority_or_default

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
    """Every waiter gets the deadline, counted from its entry here (generation_first ones too)."""
    clock = _Clock()
    pool = SeatPool(1, clock=clock, kv_transfer_timeout_s=60.0)
    pool.add(1, 0.5, None, None, orig_elapsed_s=10.0)
    clock.now = 10.0
    pool.add(2, 0.5, None, None)  # no context phase yet: generation_first
    clock.now = 59.9
    assert pool.expire() == [] and pool.remaining_s(pool.waiter(1)) == pytest.approx(0.1)
    clock.now = 60.0
    expired = pool.expire()
    assert [w.rid for w in expired] == [1] and pool.expired == 1
    assert "KV transfer timeout (60 s" in pool.expiry_message(expired[0])
    assert len(pool) == 1 and pool.remaining_s(pool.waiter(2)) == pytest.approx(10.0)
    clock.now = 70.0
    assert [w.rid for w in pool.expire()] == [2] and len(pool) == 0
    # Without the timeout nothing expires on its own.
    plain = SeatPool(1, clock=clock)
    plain.add(3, 0.5, None, None, orig_elapsed_s=10.0)
    clock.now = 1e6
    assert plain.expire() == []


def test_pool_ages_by_arrival_then_seq():
    """Aged waiters go by arrival at the orchestrator, like AgingPriorityWaitingQueue."""
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=60.0, clock=clock)
    pool.add(1, 0.5, None, None)  # arrival 0
    clock.now = 1.0
    pool.add(2, 0.5, None, None, orig_elapsed_s=30.0)  # arrival -29, joined later
    clock.now = 60.0
    assert pool.pick(["s"])[0].rid == 2
    assert pool.sort_key(pool.waiter(2), 60.0) < pool.sort_key(pool.waiter(1), 60.0)


def test_pool_pick_does_not_scan_every_waiter(monkeypatch):
    """A placement looks at one waiter per excluded server, not at the whole queue."""
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=600.0, clock=clock, kv_transfer_timeout_s=60.0)
    for rid in range(2000):
        pool.add(rid, (rid % 97) / 97.0, "ctx0" if rid % 2 else None, None)
    calls = []
    real = pool.sort_key
    monkeypatch.setattr(pool, "sort_key", lambda w, now: calls.append(w) or real(w, now))
    order = []
    while len(pool) > 1990:
        waiter, _ = pool.pick(["s"])
        pool.record_grant(waiter)
        order.append(waiter.priority)
    assert len(calls) <= 2 * 10
    assert order == sorted(order, reverse=True) and order[0] == 96 / 97.0
    # Removed and placed waiters are never picked again, and expiry pops only the expired.
    for rid in range(0, 2000, 3):
        pool.remove(rid)
    live = len(pool)
    assert 0 < live == len(pool.waiters()) < 1990
    clock.now = 59.0
    assert pool.expire() == []
    clock.now = 60.0
    assert len(pool.expire()) == live and len(pool) == 0 and pool.pick(["s"]) is None


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
    """A request that finds no seat within the timeout fails, generation_first ones too."""
    router = _router(servers=("s0",), seats=1, kv_ms=50)
    await router.get_next_server_by_key({}, req_id=0)
    doomed = asyncio.create_task(
        router.get_next_server_by_key({"priority": 0.9, "orig_elapsed_s": 3.0}, req_id=1)
    )
    gen_first = asyncio.create_task(router.get_next_server_by_key({"priority": 0.1}, req_id=2))
    with pytest.raises(NoGenerationSeatError, match="KV transfer timeout"):
        await asyncio.wait_for(doomed, timeout=5)
    with pytest.raises(NoGenerationSeatError, match="KV transfer timeout"):
        await asyncio.wait_for(gen_first, timeout=5)
    assert len(router._pool) == 0 and router._pool.expired == 2
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


# ---------------------------------------------------------------------------------- review fixes


async def _set_servers(router, servers):
    """What add_server / remove_server / the cluster watch do to the server list."""
    async with router._lock:
        old = router._servers.copy()
        router._servers = list(servers)
        router._on_servers_updated(old, router._servers)


@pytest.mark.asyncio
async def test_seat_count_survives_a_server_leaving_and_rejoining():
    """A re-registered server keeps the seats its running requests hold (no -k drift)."""
    router = _router(servers=("s0",), seats=1)
    await router.get_next_server_by_key({}, req_id=1)
    await _set_servers(router, [])
    await _set_servers(router, ["s0"])
    assert router._get_server_load("s0") == 1  # request 1 still holds its seat
    waiting = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert not waiting.done()
    await router.finish_request_by_id(1)
    assert (await asyncio.wait_for(waiting, 5))[0] == "s0"
    await router.finish_request_by_id(2)
    assert router._get_server_load("s0") == 0 and router._departed_state == {}
    # A server that leaves for good drops its state once its last seat is released.
    await router.get_next_server_by_key({}, req_id=3)
    await _set_servers(router, [])
    await router.finish_request_by_id(3)
    assert router._departed_state == {} and router._seat_state == {}


def _ctx_router_with(infos):
    ctx = create_router(
        RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT), list(infos)
    )
    ctx._server_info.update(infos)
    return ctx


@pytest.mark.asyncio
async def test_seat_wait_bound_comes_from_the_context_servers():
    """The context servers' own kv_transfer_timeout_ms (/server_info) wins over the server YAML."""
    gen_config = RouterConfig(
        type="load_balancing",
        server_role=ServerRole.GENERATION,
        args={"max_requests_per_server": 1, "kv_transfer_timeout_ms": 60000},
    )
    ctx_config = RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT)
    ctx, gen = build_disagg_routers(ctx_config, gen_config, ["c0", "c1"], ["s0"])
    assert gen._context_router is ctx
    ctx._server_info.update(
        {"c0": {"kv_transfer_timeout_ms": 600000}, "c1": {"kv_transfer_timeout_ms": 300000}}
    )
    await gen.get_next_server_by_key({}, req_id=1)
    assert gen._pool.kv_transfer_timeout_s == 300.0
    # Without a reported value the configured one applies.
    gen.set_context_router(_ctx_router_with({"c0": {}}))
    waiting = asyncio.create_task(gen.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert gen._pool.kv_transfer_timeout_s == 60.0
    assert gen._pool.waiter(2).deadline - gen._pool.waiter(2).entered == pytest.approx(60.0)
    waiting.cancel()


def test_max_queue_wait_without_pooling_is_rejected():
    with pytest.raises(ValueError, match="max_requests_per_server"):
        extract_disagg_cfg(
            generation_servers={"router": {"type": "load_balancing", "max_queue_wait_s": 30}}
        )
    with pytest.raises(ValueError, match="max_requests_per_server"):
        extract_disagg_cfg(context_servers={"router": {"max_queue_wait_s": 30}})


def test_router_declares_pooled():
    assert Router.pooled is False and PooledLoadBalancingRouter.pooled is True
    assert not create_router(
        RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION), ["s"]
    ).pooled
    assert CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0).pooled


@pytest.mark.asyncio
async def test_coordinator_stop_fails_seat_waiters():
    coordinator = _coordinator()
    await coordinator.select("generation", {}, 1, None)
    waiting = asyncio.create_task(coordinator.select("generation", {}, 2, None))
    await _settle()
    await asyncio.wait_for(coordinator.stop(), timeout=5)
    with pytest.raises(NoGenerationSeatError, match="stopping"):
        await asyncio.wait_for(waiting, timeout=5)


@pytest.mark.asyncio
async def test_pooled_seat_is_not_reclaimed_at_the_reservation_ttl():
    """A granted seat outlives the coordinator's 180 s TTL: only /finish (or the 6 h net) frees it."""
    coordinator = _coordinator()
    coordinator._reservation_timeout_secs = 0.05
    await coordinator.select("generation", {}, 1, None)
    await asyncio.sleep(0.3)
    assert coordinator.gen_router._get_server_load("g:1") == 1
    assert coordinator._reservation_timeout_for(coordinator.gen_router) == (
        POOLED_RESERVATION_TIMEOUT_S
    )
    assert coordinator._reservation_timeout_for(coordinator.ctx_router) == 0.05
    await coordinator.finish("generation", 1)
    assert coordinator.gen_router._get_server_load("g:1") == 0


def test_seat_timeout_reaches_the_client_as_503(monkeypatch):
    from fastapi.testclient import TestClient

    coordinator = _coordinator()

    async def no_seat(*_args):
        raise NoGenerationSeatError("no seat within 60 s")

    monkeypatch.setattr(coordinator, "select", no_seat)
    http = TestClient(CoordinatorServer(coordinator).app)
    raw = http.post(
        "/select",
        content=msgpack.packb({"role": "generation", "routing_key": {}, "req_id": 1}),
    )
    assert raw.status_code == 503
    assert msgpack.unpackb(raw.content, raw=False)["no_generation_seat"] is True
    server = SimpleNamespace(_perf_metrics_collector=Mock())
    with pytest.raises(HTTPException) as error:
        OpenAIDisaggServer._handle_exception(server, NoGenerationSeatError("no seat"))
    assert error.value.status_code == 503


class _FakeCoordinatorHttp:
    """A coordinator whose pooled /select waits until a /finish for its req_id arrives."""

    def __init__(self, select_delay_s=None):
        self.finished = []
        self.select_delay_s = select_delay_s
        self._finish_events = {}

    async def select(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        if self.select_delay_s is not None:
            await asyncio.sleep(self.select_delay_s)
        else:
            event = self._finish_events.setdefault(body["req_id"], asyncio.Event())
            await event.wait()
            return web.Response(
                status=503,
                body=msgpack.packb({"error": "finished while waiting"}),
                content_type="application/msgpack",
            )
        return web.Response(
            body=msgpack.packb({"server": "s0", "info": {}, "req_id": body["req_id"]}),
            content_type="application/msgpack",
        )

    async def finish(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        self.finished.append(body["req_id"])
        self._finish_events.setdefault(body["req_id"], asyncio.Event()).set()
        return web.Response(body=msgpack.packb({}), content_type="application/msgpack")


async def _with_fake_coordinator(fake, body):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    app = web.Application()
    app.router.add_post("/select", fake.select)
    app.router.add_post("/finish", fake.finish)
    async with TestServer(app) as server:
        router = CoordinatorDelegatingRouter(
            f"http://{server.host}:{server.port}", _router(), "generation", 5.0
        )
        try:
            await body(router)
        finally:
            await router.close()


def _gen_request(rid):
    return CompletionRequest(
        model="m",
        prompt="x",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", ctx_request_id=rid, disagg_request_id=rid
        ),
    )


@pytest.mark.asyncio
async def test_delegating_router_finishes_a_cancelled_pooled_select_at_once():
    """A cancelled caller sends /finish at once; it does not wait for the seat first."""
    fake = _FakeCoordinatorHttp()

    async def body(router):
        selecting = asyncio.create_task(router.get_next_server(_gen_request(7)))
        await asyncio.sleep(0.2)
        selecting.cancel()
        done, _ = await asyncio.wait({selecting}, timeout=5)
        assert selecting in done and selecting.cancelled()
        await asyncio.wait_for(router._finish_queue.join(), timeout=5)
        assert 7 in fake.finished

    await _with_fake_coordinator(fake, body)


@pytest.mark.asyncio
async def test_delegating_router_finishes_a_timed_out_select():
    fake = _FakeCoordinatorHttp(select_delay_s=2.0)

    async def body(router):
        router._select_timeout_s = 0.1
        with pytest.raises(asyncio.TimeoutError):
            await router.get_next_server(_gen_request(9))
        await asyncio.wait_for(router._finish_queue.join(), timeout=5)
        assert fake.finished == [9]

    await _with_fake_coordinator(fake, body)


class _FlakyCoordinatorFinish:
    """A coordinator /finish that fails its first ``failures`` calls, then reaches ``coordinator``."""

    def __init__(self, coordinator, failures, status=503):
        self.coordinator = coordinator
        self.failures = failures
        self.status = status
        self.calls = 0
        self.finished = []

    async def finish(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        self.calls += 1
        if self.calls <= self.failures:
            return web.Response(
                status=self.status,
                body=msgpack.packb({"error": "unavailable"}),
                content_type="application/msgpack",
            )
        await self.coordinator.finish(body["role"], body["req_id"], body["success"])
        self.finished.append(body["req_id"])
        return web.Response(body=msgpack.packb({}), content_type="application/msgpack")


async def _finish_through_flaky_coordinator(local_router, fake, rids, monkeypatch):
    """Finish ``rids`` through a delegating router whose finish queue holds one entry."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    monkeypatch.setattr("tensorrt_llm.serve.router.COORDINATOR_FINISH_RETRY_DELAY_S", 0.001)
    app = web.Application()
    app.router.add_post("/finish", fake.finish)
    async with TestServer(app) as server:
        router = CoordinatorDelegatingRouter(
            f"http://{server.host}:{server.port}", local_router, "generation", 5.0
        )
        router._finish_queue = asyncio.Queue(maxsize=1)
        try:
            # No await between the puts: the workers have not run, so the second is QueueFull.
            for rid in rids:
                await router.finish_request(_gen_request(rid), req_id=rid)
            # Parked finishes enter the queue as workers take from it, so this covers them too.
            await asyncio.wait_for(router._finish_queue.join(), timeout=30)
        finally:
            await router.close()
        return router


@pytest.mark.asyncio
async def test_pooled_finish_is_parked_and_retried_until_the_seat_is_released(monkeypatch):
    """A pooled /finish is neither dropped on a full queue nor abandoned after 3 failures."""
    coordinator = _coordinator(seats=2)
    for rid in (1, 2):
        assert (await coordinator.select("generation", {}, rid, None))[0] == "g:1"
    assert coordinator.gen_router._get_server_load("g:1") == 2
    fake = _FlakyCoordinatorFinish(coordinator, failures=5)
    router = await _finish_through_flaky_coordinator(_router(), fake, (1, 2), monkeypatch)
    assert router._dropped_finishes == 0 and router._parked_finishes == 1
    assert sorted(fake.finished) == [1, 2]
    assert coordinator.gen_router._get_server_load("g:1") == 0
    assert coordinator._reservation_tasks == {}


@pytest.mark.asyncio
async def test_non_pooled_finish_still_drops_on_a_full_queue_and_gives_up(monkeypatch):
    """The non-pooled path is unchanged: overflow is dropped and a finish stops after 3 attempts."""
    local_router = LoadBalancingRouter(server_role=None, servers=["s0"])
    assert not local_router.pooled
    fake = _FlakyCoordinatorFinish(AsyncMock(), failures=5)
    router = await _finish_through_flaky_coordinator(local_router, fake, (1, 2), monkeypatch)
    assert router._dropped_finishes == 1 and getattr(router, "_parked_finishes", 0) == 0
    assert fake.calls == 3 and fake.finished == []


@pytest.mark.asyncio
async def test_pooled_finish_stops_on_a_malformed_reply_or_at_the_expiry(monkeypatch):
    fake = _FlakyCoordinatorFinish(AsyncMock(), failures=100, status=400)
    router = await _finish_through_flaky_coordinator(_router(), fake, (1,), monkeypatch)
    assert fake.calls == 1
    monkeypatch.setattr(
        CoordinatorDelegatingRouter,
        "__init__",
        _with_finish_window(CoordinatorDelegatingRouter.__init__, 0.05),
    )
    fake = _FlakyCoordinatorFinish(AsyncMock(), failures=10**6)
    router = await _finish_through_flaky_coordinator(_router(), fake, (1,), monkeypatch)
    assert router._pooled_finish_window_s == 0.05
    assert 3 < fake.calls < 10**6 and fake.finished == []


def _with_finish_window(init, window_s):
    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._pooled_finish_window_s = window_s

    return patched


def test_pooled_finish_window_follows_the_coordinator_expiry(monkeypatch):
    """The env timeout can lengthen the pooled expiry (and the finish retry window), not shorten it."""
    monkeypatch.setenv("TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT", "60")
    router = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0)
    assert router._pooled_finish_window_s == POOLED_RESERVATION_TIMEOUT_S
    coordinator = DisaggCoordinatorService(_coordinator()._config, _client_factory)
    assert coordinator._reservation_timeout_for(coordinator.gen_router) == (
        POOLED_RESERVATION_TIMEOUT_S
    )
    longer = 2 * POOLED_RESERVATION_TIMEOUT_S
    monkeypatch.setenv("TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT", str(longer))
    router = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0)
    assert router._pooled_finish_window_s == longer
