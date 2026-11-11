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
"""output_length_prediction wired through the disaggregated config, coordinator and service."""

import asyncio
import json
from unittest.mock import AsyncMock

import msgpack
import pytest
from fastapi.testclient import TestClient

from tensorrt_llm.llmapi.disagg_utils import (
    DisaggServerConfig,
    OutputLengthPredictionConfig,
    extract_disagg_cfg,
)
from tensorrt_llm.serve import output_length_predictor as olp
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import CoordinatorClient, DisaggCoordinatorService
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ConversationParams,
    DisaggregatedParams,
    UsageInfo,
)
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


def _client_factory(*_args, **_kwargs):
    return AsyncMock()


def _config(predictor="ridge"):
    config = DisaggServerConfig(server_configs=[])
    if predictor is not None:
        config.output_length_prediction = OutputLengthPredictionConfig(predictor=predictor)
    return config


def test_config_parsing():
    assert extract_disagg_cfg().output_length_prediction is None
    assert extract_disagg_cfg(output_length_prediction={}).output_length_prediction.predictor == (
        "ridge_gbm"
    )
    config = extract_disagg_cfg(output_length_prediction={"predictor": "ridge"})
    assert config.output_length_prediction.predictor == "ridge"
    with pytest.raises(ValueError):
        extract_disagg_cfg(output_length_prediction={"predictor": "ema"})
    with pytest.raises(ValueError):
        extract_disagg_cfg(output_length_prediction={"model": "file.json"})
    # Not a per-role option: it is not copied into the server sections.
    config = extract_disagg_cfg(
        output_length_prediction={"predictor": "ridge"}, context_servers={"num_instances": 0}
    )
    assert all("output_length_prediction" not in s.other_args for s in config.server_configs)


@pytest.mark.asyncio
async def test_coordinator_predicts_and_learns_in_process():
    coordinator = DisaggCoordinatorService(_config(), client_factory=_client_factory)
    tokens, key = await coordinator.predict_output_length("c", 1000)
    assert tokens == olp.PRIOR_TOKENS
    assert await coordinator.record_output_length("c", 1000, 64.0, key)
    assert (await coordinator.predict_output_length("c", 1200))[0] == 64.0
    off = DisaggCoordinatorService(_config(None), client_factory=_client_factory)
    with pytest.raises(ValueError, match="not configured"):
        await off.predict_output_length("c", 1000)


def test_coordinator_server_route():
    coordinator = DisaggCoordinatorService(_config(), client_factory=_client_factory)
    # No lifespan: the coordinator is not started (it has no servers to wait for).
    http = TestClient(CoordinatorServer(coordinator).app)

    def post(body):
        raw = http.post("/output_length", content=msgpack.packb(body, use_bin_type=True))
        return raw.status_code, msgpack.unpackb(raw.content, raw=False)

    status, reply = post({"op": "predict", "conv": "c", "isl": 1000})
    assert status == 200 and reply["tokens"] == olp.PRIOR_TOKENS
    status, reply = post({"op": "record", "conv": "c", "isl": 1000, "out": 80, "key": reply["key"]})
    assert status == 200 and reply == {"learned": True}
    assert post({"op": "predict", "conv": "c", "isl": 1100})[1]["tokens"] == 80.0
    for bad in (
        {"op": "fit"},
        {"op": "predict", "conv": 3, "isl": 1},
        {"op": "predict", "conv": "c", "isl": True},
        {"op": "record", "conv": "c", "isl": 1},
        {"op": "record", "conv": "c", "isl": 1, "out": 5, "key": "k"},
    ):
        assert post(bad)[0] == 400, bad


@pytest.mark.asyncio
async def test_coordinator_client_round_trip():
    """A fleet worker's CoordinatorClient reaches the coordinator's predictor over /output_length."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    coordinator = DisaggCoordinatorService(_config(), client_factory=_client_factory)
    seen = []

    async def handler(request):
        body = msgpack.unpackb(await request.read(), raw=False)
        seen.append(body)
        if body["op"] == "predict":
            tokens, key = await coordinator.predict_output_length(body["conv"], body["isl"])
            reply = {"tokens": tokens, "key": key}
        else:
            reply = {
                "learned": await coordinator.record_output_length(
                    body["conv"], body["isl"], body["out"], body["key"]
                )
            }
        return web.Response(body=msgpack.packb(reply), content_type="application/msgpack")

    app = web.Application()
    app.router.add_post("/output_length", handler)
    async with TestServer(app) as server:
        client = CoordinatorClient(f"http://{server.host}:{server.port}", _config())
        try:
            tokens, key = await client.predict_output_length("c", 1000)
            assert tokens == olp.PRIOR_TOKENS
            assert await client.record_output_length("c", 1000, 77.0, key)
            assert (await client.predict_output_length("c", 1100))[0] == 77.0
        finally:
            await client.stop()
    assert [b["op"] for b in seen] == ["predict", "record", "predict"]


def _ctx_response(prompt_tokens):
    return CompletionResponse(
        model="m",
        usage=UsageInfo(
            prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1
        ),
        prompt_token_ids=[1, 2, 3],
        choices=[
            CompletionResponseChoice(
                index=0,
                text="",
                finish_reason="length",
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only", disagg_request_id=42, ctx_request_id=42
                ),
            )
        ],
    )


@pytest.mark.asyncio
async def test_service_sets_priority_before_generation_placement(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config()
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    coordinator._ctx_router = AsyncMock(spec=Router)
    coordinator._gen_router = AsyncMock(spec=Router)
    coordinator._ctx_router.get_next_server.return_value = ("ctx:1", {})
    placed = []

    async def place_generation(req, **kwargs):
        placed.append(req.priority)
        return "gen:1", {}

    coordinator._gen_router.get_next_server.side_effect = place_generation
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_client.send_request.return_value = _ctx_response(5000)
    final = f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 300}})}\n\n"

    async def gen_stream():
        yield b'data: {"choices": [{"text": "x"}]}\n\n'
        yield final.encode()

    service._gen_client.send_request.return_value = gen_stream()

    async def send(conv):
        request = CompletionRequest(
            model="m",
            prompt="hi",
            max_tokens=7,
            stream=True,
            conversation_params=ConversationParams(conversation_id=conv),
        )
        response = await service._send_disagg_request_ctx_first(request)
        return [chunk async for chunk in response]

    chunks = await send("conv-1")
    assert chunks[-1] == final.encode()
    assert placed == [pytest.approx(olp.priority_of(olp.PRIOR_TOKENS))]
    await asyncio.sleep(0.01)  # the record task
    predictor = coordinator._output_length_predictor
    assert predictor.table.recorded == 1
    service._gen_client.send_request.return_value = gen_stream()
    await send("conv-1")
    # The conversation's learned reply (300 tokens) now sets the priority; max_tokens=7 does not.
    assert placed[-1] == pytest.approx(olp.priority_of(300.0))


@pytest.mark.asyncio
async def test_service_without_prediction_leaves_priority_alone(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config(None)
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    coordinator._ctx_router = AsyncMock(spec=Router)
    coordinator._gen_router = AsyncMock(spec=Router)
    coordinator._ctx_router.get_next_server.return_value = ("ctx:1", {})
    coordinator._gen_router.get_next_server.return_value = ("gen:1", {})
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    assert service._output_length_priority is None
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_client.send_request.return_value = _ctx_response(5000)
    service._gen_client.send_request.return_value = "response"
    request = CompletionRequest(model="m", prompt="hi")
    assert await service._send_disagg_request_ctx_first(request) == "response"
    assert service._gen_client.send_request.call_args.args[0].priority is None
