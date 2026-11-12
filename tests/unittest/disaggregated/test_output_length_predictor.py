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
"""Online output-length prediction (serve/output_length_predictor.py)."""

import asyncio
import gc
import json
import math
import random
import statistics
import threading
import time

import pytest

from tensorrt_llm.serve import output_length_predictor as olp
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
    UsageInfo,
)

pytestmark = pytest.mark.cpu_only


def _gen_request(conv_id="c0", prompt_tokens=1000, cls=CompletionRequest):
    kwargs = {"model": "m", "max_tokens": 4096}
    kwargs["prompt" if cls is CompletionRequest else "messages"] = (
        "hi" if cls is CompletionRequest else [{"role": "user", "content": "hi"}]
    )
    request = cls(**kwargs)
    if conv_id is not None:
        request.conversation_params = ConversationParams(conversation_id=conv_id)
    request.disaggregated_params = DisaggregatedParams(
        request_type="generation_only",
        ctx_usage=None if prompt_tokens is None else UsageInfo(prompt_tokens=prompt_tokens),
    )
    return request


class _FakeFitter:
    """Fits nothing; returns the result it is given on the poll after ``start``."""

    def __init__(self, result=None):
        self.result = result
        self.started = []
        self._busy = False
        self._ready = None

    def busy(self):
        return self._busy

    def start(self, rows, n, meta):
        self.started.append((n, len(rows), dict(meta)))
        self._ready = dict(self.result or {}, **meta)

    def poll(self):
        ready, self._ready = self._ready, None
        return ready


_ONE_LEAF = {"baseline": math.log1p(99.0), "trees": [[[True, 0.0, 0, 0.0, True, 0, 0]]]}


def _good_fit(**extra):
    return {**_ONE_LEAF, "n_iter": 1, "fit_s": 0.01, "export_max_abs_diff": 0.0, **extra}


# ---------------------------------------------------------------------------------- table and features


def test_priority_is_strictly_decreasing_in_tokens():
    values = [olp.priority_of(t) for t in (0, 1, 100, 375.77, 1024, 1e6)]
    assert values[0] == 1.0
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[3] == pytest.approx(0.7315, abs=1e-4)
    assert olp.priority_of(-5) == 1.0


def test_table_reset_and_skip_rules():
    table = olp.ConversationLengthTable()
    assert not table.record("c", 1000, 2)  # cache primer: not learned
    assert table.skipped == 1 and len(table) == 0
    assert not table.record(None, 1000, 50)  # no conversation id
    assert table.record("c", 1000, 50)
    # One reply with a short prompt, then a jump of more than 10,000 appended tokens: a new prefix.
    assert table.snapshot_for("c", 1000 + 50 + 10001) is None
    assert table.resets == 1 and len(table) == 0
    assert table.record("c", 1000, 50) and table.record("c", 1100, 70)
    snapshot = table.snapshot_for("c", 20000)  # two replies: never reset
    assert snapshot == (2.0, 70.0, 60.0, 120.0, 70.0, 1100.0)


def test_table_evicts_least_recently_used():
    table = olp.ConversationLengthTable(max_conversations=2)
    for conv in ("a", "b"):
        table.record(conv, 100, 10)
    table.snapshot_for("a", 100)
    table.record("c", 100, 10)
    assert table.snapshot_for("b", 100) is None and table.snapshot_for("a", 100) is not None


def test_features():
    first = olp.features(None, 1000)
    assert first[:2] == [0.0, math.log1p(1000)] and all(v != v for v in first[2:])
    row = olp.features((2.0, 70.0, 60.0, 120.0, 70.0, 1100.0), 1500)
    expected = [math.log1p(2), math.log1p(1500), math.log1p(70), math.log1p(60), math.log1p(60)]
    assert row[:5] == pytest.approx(expected)
    assert row[5] == pytest.approx(math.log1p(70))
    assert row[6] == pytest.approx(math.log1p(1500 - 1100 - 70))
    assert olp.features((1.0, 500.0, 500.0, 500.0, 500.0, 1000.0), 800)[6] == pytest.approx(
        -math.log1p(700)
    )


def test_request_inputs_never_reads_max_tokens():
    class Guarded:
        conversation_params = ConversationParams(conversation_id="c7")
        disaggregated_params = DisaggregatedParams(
            request_type="generation_only", ctx_usage=UsageInfo(prompt_tokens=321)
        )

        def __getattr__(self, name):
            raise AssertionError(f"{name} must not be read")

    assert olp.request_inputs(Guarded()) == ("c7", 321.0)
    assert olp.request_inputs(_gen_request(None, None)) == (None, None)


def test_running_median_matches_statistics():
    rng = random.Random(0)
    median = olp.RunningMedian()
    values = []
    assert median.value() != median.value()
    for _ in range(501):
        values.append(rng.choice([rng.random() * 1000, float(rng.randint(0, 9))]))
        median.add(values[-1])
        assert median.value() == statistics.median(values)


# ---------------------------------------------------------------------------------- ridge


def _ridge_closed_form(rows, ys, alpha):
    np = pytest.importorskip("numpy")
    x, y = np.asarray(rows), np.asarray(ys)
    xc, yc = x - x.mean(0), y - y.mean()
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return w, y.mean() - x.mean(0) @ w


def test_online_ridge_matches_closed_form_and_sklearn():
    rng = random.Random(1)
    rows, ys = [], []
    ridge = olp.OnlineRidge(2 * len(olp.FEATURES))
    for _ in range(400):
        snap = (
            None
            if rng.random() < 0.3
            else (
                float(rng.randint(1, 9)),
                rng.random() * 900,
                rng.random() * 900,
                rng.random() * 9000,
                rng.random() * 900,
                rng.random() * 30000,
            )
        )
        x = olp.ridge_row(olp.features(snap, rng.random() * 50000))
        y = rng.random() * 8
        rows.append(x)
        ys.append(y)
        ridge.add(x, y)
    ridge.fit()
    w, b = _ridge_closed_form(rows, ys, olp.RIDGE_ALPHA)
    assert ridge.coef == pytest.approx(list(w), abs=1e-8)
    assert ridge.intercept == pytest.approx(float(b), abs=1e-8)
    linear_model = pytest.importorskip("sklearn.linear_model")
    ref = linear_model.Ridge(alpha=olp.RIDGE_ALPHA).fit(rows, ys)
    assert ridge.coef == pytest.approx(list(ref.coef_), abs=1e-6)
    assert ridge.intercept == pytest.approx(float(ref.intercept_), abs=1e-6)


# ---------------------------------------------------------------------------------- the coordinator's predictor


def test_unknown_predictor_is_refused():
    with pytest.raises(ValueError, match="predictor"):
        olp.OnlineLengthPredictor("gbm_only")


def test_prior_then_median_then_ema():
    predictor = olp.OnlineLengthPredictor("ridge")
    tokens, key = predictor.predict("a", 1000, 0.0)
    assert tokens == olp.PRIOR_TOKENS
    assert predictor.record("a", 1000, 100, key, 1.0)
    predictor.record("b", 1000, 300, None, 2.0)
    assert predictor.unmatched == 1
    assert predictor.predict("new", 1000, 3.0)[0] == 200.0  # median of the learned replies
    assert predictor.predict("a", 1200, 3.0)[0] == 100.0  # the conversation's EMA
    assert not predictor.record("a", 1200, 1, None, 4.0)  # a primer is not learned
    assert predictor.median.n == 2


def _learn(predictor, n, t, conv=lambda i: f"c{i}"):
    for i in range(n):
        _, key = predictor.predict(conv(i), 1000 + i, t)
        predictor.record(conv(i), 1000 + i, 50 + i % 400, key, t)


def test_ridge_refits_on_boundaries_after_min_rows():
    predictor = olp.OnlineLengthPredictor("ridge")
    _learn(predictor, olp.RIDGE_MIN_ROWS - 1, 0.0)
    predictor.predict("x", 1000, olp.RIDGE_REFIT_S)  # boundary with too few rows: skipped
    assert predictor.ridge_fits == 0 and predictor.ridge.coef is None
    _learn(predictor, 1, olp.RIDGE_REFIT_S)
    predictor.predict("x", 1000, 2 * olp.RIDGE_REFIT_S - 1e-3)
    assert predictor.ridge_fits == 0
    predictor.predict("x", 1000, 2 * olp.RIDGE_REFIT_S)
    assert predictor.ridge_fits == 1 and predictor.ridge.fit_rows == olp.RIDGE_MIN_ROWS
    assert predictor.sources["ridge"] == 1
    # Several boundaries passed at once fit once per boundary.
    predictor.predict("x", 1000, 5 * olp.RIDGE_REFIT_S)
    assert predictor.ridge_fits == 4


def test_ridge_mode_never_fits_trees():
    predictor = olp.OnlineLengthPredictor("ridge", gbm_fitter=_FakeFitter(_good_fit()))
    _learn(predictor, olp.GBM_MIN_ROWS, 0.0)
    predictor.predict("x", 1000, 10 * olp.GBM_REFIT_S)
    assert predictor.gbm_fitter.started == [] and predictor.gbm is None


def test_tree_fit_goes_live_after_it_returns():
    fitter = _FakeFitter(_good_fit())
    predictor = olp.OnlineLengthPredictor("ridge_gbm", gbm_fitter=fitter)
    _learn(predictor, olp.GBM_MIN_ROWS, 0.0)
    predictor.predict("x", 1000, olp.GBM_REFIT_S)  # starts the fit
    assert [s[0] for s in fitter.started] == [olp.GBM_MIN_ROWS]
    assert fitter.started[0][1] == olp.GBM_MIN_ROWS * (len(olp.FEATURES) + 1)
    assert predictor.gbm_fits == 1  # the fake's fit has returned by the poll in the same event
    tokens, _ = predictor.predict("x", 1000, olp.GBM_REFIT_S + 1)
    assert tokens == pytest.approx(99.0) and predictor.sources["gbm"] >= 1


@pytest.mark.parametrize(
    "result", [{"error": "exit 1: boom"}, _good_fit(export_max_abs_diff=1e-3), {"rows": 3}]
)
def test_failed_or_bad_tree_fit_keeps_the_current_model(result):
    fitter = _FakeFitter(result)
    predictor = olp.OnlineLengthPredictor("ridge_gbm", gbm_fitter=fitter)
    _learn(predictor, olp.GBM_MIN_ROWS, 0.0)
    predictor.predict("x", 1000, olp.GBM_REFIT_S)
    assert predictor.gbm is None and predictor.gbm_errors == 1
    assert predictor.predict("x", 1000, olp.GBM_REFIT_S + 1)[0] != pytest.approx(99.0)


def test_busy_fitter_skips_the_boundary():
    fitter = _FakeFitter(None)
    fitter._busy = True
    predictor = olp.OnlineLengthPredictor("ridge_gbm", gbm_fitter=fitter)
    _learn(predictor, olp.GBM_MIN_ROWS, 0.0)
    predictor.predict("x", 1000, olp.GBM_REFIT_S)
    assert fitter.started == [] and predictor.gbm_busy_skips == 1


def test_fit_does_not_block_predictions():
    """A fit that takes 1 s in its thread does not delay the predictions made meanwhile."""

    class SlowFitter(_FakeFitter):
        def start(self, rows, n, meta):
            self.started.append((n, len(rows), dict(meta)))
            self._busy = True

            def run():
                time.sleep(1.0)
                self._ready = dict(_good_fit(), **meta)
                self._busy = False

            threading.Thread(target=run, daemon=True).start()

    fitter = SlowFitter()
    predictor = olp.OnlineLengthPredictor("ridge_gbm", gbm_fitter=fitter)
    _learn(predictor, olp.GBM_MIN_ROWS, 0.0)
    gc_was_enabled = gc.isenabled()
    gc.disable()  # a full collection of a large process takes longer than the bound below
    try:
        worst, start, sources = 0.0, time.monotonic(), set()
        while time.monotonic() - start < 2.0:
            t0 = time.perf_counter()
            tokens, _ = predictor.predict("x", 1000, olp.GBM_REFIT_S + time.monotonic() - start)
            worst = max(worst, time.perf_counter() - t0)
            sources.add("gbm" if tokens == pytest.approx(99.0) else "other")
            time.sleep(0.005)
    finally:
        if gc_was_enabled:
            gc.enable()
    assert len(fitter.started) == 1 and predictor.gbm_fits == 1
    assert sources == {"gbm", "other"}
    assert worst < 0.05


def test_real_tree_fit_matches_sklearn():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    olp.check_gbm_dependencies()
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=(n, len(olp.FEATURES)))
    x[rng.random(size=x.shape) < 0.2] = np.nan
    y = np.nan_to_num(x[:, 1]) * 0.5 + rng.normal(scale=0.1, size=n) + 5.0
    rows = olp.array("d", np.concatenate([x, y[:, None]], axis=1).ravel().tolist())
    fitter = olp.SubprocessGbmFitter(sync=True)
    fitter.start(rows, n, {"rows": n})
    result = fitter.poll()
    assert "error" not in result, result.get("error")
    assert result["export_max_abs_diff"] <= olp.GBM_EXPORT_TOL
    from sklearn.ensemble import HistGradientBoostingRegressor

    ref = HistGradientBoostingRegressor(**olp.GBM_PARAMS).fit(x, y)
    ensemble = olp.TreeEnsemble(result["baseline"], result["trees"])
    got = [ensemble.raw([float(v) for v in row]) for row in x[:200]]
    assert got == pytest.approx(list(ref.predict(x[:200])), abs=1e-5)


def test_missing_sklearn_fails_at_start(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\necho 'No module named sklearn' >&2\nexit 1\n")
    fake_python.chmod(0o755)
    with pytest.raises(ImportError, match="scikit-learn"):
        olp.check_gbm_dependencies(str(fake_python))


# ---------------------------------------------------------------------------------- frontend


class _FakeCoordinator:
    def __init__(self, tokens=100.0, delay=0.0, error=None):
        self.tokens, self.delay, self.error = tokens, delay, error
        self.predicted, self.recorded = [], []

    async def predict_output_length(self, conv_id, isl):
        self.predicted.append((conv_id, isl))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.tokens, 7

    async def record_output_lengths(self, records):
        self.recorded.extend(records)
        return len(records)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [CompletionRequest, ChatCompletionRequest])
async def test_assign_sets_priority_from_prediction(cls):
    coordinator = _FakeCoordinator(tokens=1024.0)
    frontend = olp.OutputLengthPriority(coordinator, "ridge")
    request = _gen_request("c1", 2000, cls)
    ticket = await frontend.assign(request)
    assert request.priority == 0.5 and ticket.key == 7
    assert coordinator.predicted == [("c1", 2000.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "coordinator", [_FakeCoordinator(delay=5.0), _FakeCoordinator(error=RuntimeError("down"))]
)
async def test_assign_without_prediction_uses_the_prior(coordinator, monkeypatch):
    monkeypatch.setattr(olp, "PREDICT_TIMEOUT_S", 0.05)
    frontend = olp.OutputLengthPriority(coordinator, "ridge")
    request = _gen_request("c1", 2000)
    ticket = await frontend.assign(request)
    assert request.priority == pytest.approx(olp.priority_of(olp.PRIOR_TOKENS))
    assert frontend.missing == 1 and ticket.key is None  # the reply is still recorded


@pytest.mark.asyncio
async def test_assign_without_prompt_tokens_records_nothing():
    frontend = olp.OutputLengthPriority(_FakeCoordinator(), "ridge")
    request = _gen_request("c1", None)
    assert await frontend.assign(request) is None
    assert request.priority == pytest.approx(frontend.prior_priority)


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


def test_completion_tokens_in():
    assert olp.completion_tokens_in(b"data: {}\n") == (None, False)
    usage = {"usage": {"completion_tokens": 42}}
    assert olp.completion_tokens_in(_sse(dict(usage, choices=[]))) == (42, True)
    assert olp.completion_tokens_in(_sse(dict(usage, choices=[{"x": 1}]))) == (42, False)
    assert olp.completion_tokens_in(b"data: [DONE]\n") == (None, False)


@pytest.mark.asyncio
async def test_track_records_the_final_usage_before_relaying_it():
    coordinator = _FakeCoordinator()
    frontend = olp.OutputLengthPriority(coordinator, "ridge")
    ticket = olp._Ticket("c1", 2000.0, 7)
    final = _sse({"choices": [], "usage": {"completion_tokens": 321}})
    head, tail = final[:10], final[10:]

    async def stream():
        yield _sse({"choices": [{"text": "a"}], "usage": {"completion_tokens": 1}})
        # The final chunk arrives split across two reads.
        yield head
        yield tail
        yield b"data: [DONE]\n\n"

    out = []
    async for chunk in frontend.track(stream(), ticket):
        out.append(chunk)
        assert ticket.recorded == (len(out) >= 3)
    await asyncio.sleep(0.01)
    assert b"".join(out).endswith(b"[DONE]\n\n")
    assert coordinator.recorded == [("c1", 2000.0, 321.0, 7)]


@pytest.mark.asyncio
async def test_track_non_streaming_response():
    coordinator = _FakeCoordinator()
    frontend = olp.OutputLengthPriority(coordinator, "ridge")

    class Response:
        usage = UsageInfo(prompt_tokens=10, completion_tokens=55)

    response = Response()
    assert frontend.track(response, olp._Ticket("c", 10.0, 3)) is response
    await asyncio.sleep(0.01)
    assert coordinator.recorded == [("c", 10.0, 55.0, 3)]
    assert frontend.track(response, None) is response


# ---------------------------------------------------------------------------------- review fixes


def test_non_finite_values_never_reach_the_models():
    predictor = olp.OnlineLengthPredictor("ridge")
    for bad in (float("nan"), float("inf"), -1.0):
        _, key = predictor.predict("c", 1000, 0.0)
        assert not predictor.record("c", 1000, bad, key, 0.0)
        assert not predictor.record("c", bad, 100, None, 0.0)
        with pytest.raises(ValueError, match="finite"):
            predictor.predict("c", bad, 0.0)
    assert predictor.invalid == 6 and predictor.learned == 0 and predictor.ridge.n == 0
    assert predictor.median.n == 0 and len(predictor.table) == 0
    _learn(predictor, olp.RIDGE_MIN_ROWS, 0.0)
    tokens, _ = predictor.predict("x", 1000, olp.RIDGE_REFIT_S)
    assert math.isfinite(tokens) and all(math.isfinite(w) for w in predictor.ridge.coef)
    assert olp.priority_of(float("nan")) == pytest.approx(olp.priority_of(olp.PRIOR_TOKENS))
    assert olp.priority_of(float("inf")) == 0.0


def test_out_of_range_lengths_never_reach_the_models():
    """Finite lengths past MAX_TOKENS are invalid, and a model's log prediction is clamped before expm1."""
    predictor = olp.OnlineLengthPredictor("ridge")
    for i in range(300):
        _, key = predictor.predict(f"c{i}", 1000 + i, 0.0)
        assert predictor.record(f"c{i}", 1000 + i, 200.0 + i, key, 0.0)
    for _ in range(2000):
        assert not predictor.record(None, 1000, 1e308, None, 0.0)
    assert not predictor.record(None, olp.MAX_TOKENS + 1, 100.0, None, 0.0)
    assert not predictor.record(None, 10**400, 100.0, None, 0.0)
    assert predictor.invalid == 2002 and predictor.learned == 300
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(None, 1e308, 0.0)
    for isl in (1000, 8192, 131072, olp.MAX_TOKENS):
        tokens, _ = predictor.predict(None, isl, olp.RIDGE_REFIT_S)
        assert 0.0 <= tokens <= olp.MAX_TOKENS * (1 + 1e-12)
    assert predictor.ridge_fits == 1
    # A model that extrapolates past expm1's range (log1p > 709.78) predicts MAX_TOKENS.
    predictor.ridge.intercept = 1e4
    tokens, _ = predictor.predict(None, 8192, olp.RIDGE_REFIT_S)
    assert tokens == pytest.approx(olp.MAX_TOKENS)
    predictor.ridge.intercept = -1e4
    assert predictor.predict(None, 8192, olp.RIDGE_REFIT_S)[0] == 0.0
    predictor.ridge.intercept = float("nan")
    assert predictor.predict(None, 8192, olp.RIDGE_REFIT_S)[0] == olp.PRIOR_TOKENS
    assert olp.is_token_count(olp.MAX_TOKENS) and olp.is_token_count(0)
    assert not olp.is_token_count(olp.MAX_TOKENS + 1) and not olp.is_token_count(float("nan"))


def test_conversationless_replies_train_the_global_models():
    predictor = olp.OnlineLengthPredictor("ridge")
    for i in range(olp.RIDGE_MIN_ROWS):
        _, key = predictor.predict(None, 1000 + i, 0.0)
        assert predictor.record(None, 1000 + i, 300.0, key, 0.0)
    assert predictor.learned == predictor.without_conversation == olp.RIDGE_MIN_ROWS
    assert len(predictor.table) == 0 and predictor.median.value() == 300.0
    assert predictor.predict(None, 1000, 1.0)[0] == 300.0  # the median, not the prior
    # A conversation-less reply without its key still has its (history-free) features.
    assert predictor.record(None, 1000, 300.0, None, 1.0) and predictor.unmatched == 0
    predictor.predict(None, 1000, olp.RIDGE_REFIT_S)
    assert predictor.ridge_fits == 1


def test_nothing_learned_is_reported(monkeypatch):
    warnings = []
    monkeypatch.setattr(olp.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    monkeypatch.setattr(olp, "NOTHING_LEARNED_WARN_AFTER", 5)
    predictor = olp.OnlineLengthPredictor("ridge")
    for _ in range(5):
        predictor.record("c", 1000, 1.0, None, 0.0)
    assert any("none of the first 5 replies was learned" in w for w in warnings)


def test_predictor_state_is_bounded():
    median = olp.RunningMedian(window=3)
    for v in (100, 1, 2, 3, 50):
        median.add(v)
    assert median.value() == 3 and median.n == 5 and len(median._sorted) == 3
    fitter = _FakeFitter(_good_fit())
    predictor = olp.OnlineLengthPredictor("ridge_gbm", gbm_fitter=fitter)
    predictor.gbm_max_rows = olp.GBM_MIN_ROWS
    width = len(olp.FEATURES) + 1
    _learn(predictor, 3 * olp.GBM_MIN_ROWS, 0.0)
    assert len(predictor.gbm_rows) == olp.GBM_MIN_ROWS * width
    assert predictor.gbm_n == 3 * olp.GBM_MIN_ROWS
    # The newest reply overwrote the oldest slot of the ring.
    newest = 3 * olp.GBM_MIN_ROWS - 1
    slot = (newest % olp.GBM_MIN_ROWS) * width
    assert predictor.gbm_rows[slot + width - 1] == pytest.approx(math.log1p(50 + newest % 400))
    predictor.predict("x", 1000, olp.GBM_REFIT_S)
    assert fitter.started[0][:2] == (olp.GBM_MIN_ROWS, olp.GBM_MIN_ROWS * width)
    assert olp.RunningMedian().window == olp.MEDIAN_WINDOW


def test_fit_children_get_the_server_import_path(tmp_path, monkeypatch):
    """-I drops PYTHONPATH and the user site; the children get the server's sys.path instead."""
    extra = tmp_path / "site"
    extra.mkdir()
    monkeypatch.syspath_prepend(str(extra))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    paths = olp.child_sys_path()
    assert str(extra) in paths and str(tmp_path) not in paths and "" not in paths
    argv_file = tmp_path / "argv.json"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump(sys.argv[1:], open({str(argv_file)!r}, 'w'))\n"
    )
    fake_python.chmod(0o755)
    olp.check_gbm_dependencies(str(fake_python))
    argv = json.loads(argv_file.read_text())
    assert argv[0] == "-I" and str(extra) in json.loads(argv[-1])
    fitter = olp.SubprocessGbmFitter(python=str(fake_python), sync=True)
    fitter.start(olp.array("d", [0.0] * len(olp.FEATURES) + [1.0]), 1, {"rows": 1})
    argv = json.loads(argv_file.read_text())
    header = json.loads(argv[-1])
    assert argv[0] == "-I" and header["n"] == 1 and str(extra) in header["sys_path"]
    # A real -I child imports a module that only the server's path has.
    (extra / "only_on_server_path.py").write_text("VALUE = 7\n")
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [
            _sys.executable,
            "-I",
            "-c",
            "import json, sys; sys.path[:0] = json.loads(sys.argv[1]); "
            "import only_on_server_path as m; print(m.VALUE)",
            json.dumps(olp.child_sys_path()),
        ],
        capture_output=True,
        check=False,
    )
    assert result.stdout.strip() == b"7", result.stderr


@pytest.mark.asyncio
async def test_streams_ask_for_usage_and_drop_it_when_the_client_did_not():
    coordinator = _FakeCoordinator()
    frontend = olp.OutputLengthPriority(coordinator, "ridge")
    request = _gen_request("c1", 2000)
    request.stream = True
    ticket = await frontend.assign(request)
    assert (
        request.stream_options.include_usage and not request.stream_options.continuous_usage_stats
    )
    assert ticket.strip_usage
    final = _sse({"choices": [], "usage": {"completion_tokens": 321}})

    async def stream():
        yield _sse({"choices": [{"text": "a"}]})[:7]
        yield _sse({"choices": [{"text": "a"}]})[7:] + final[:9]
        yield final[9:] + b"data: [DONE]\n\n"

    relayed = b"".join([chunk async for chunk in frontend.track(stream(), ticket)])
    assert relayed == _sse({"choices": [{"text": "a"}]}) + b"data: [DONE]\n\n"
    await asyncio.sleep(0.01)
    assert coordinator.recorded == [("c1", 2000.0, 321.0, 7)]
    # A client that asked for usage keeps the chunk.
    asked = _gen_request("c1", 2000)
    asked.stream = True
    asked.stream_options = olp_stream_options(include_usage=True)
    assert not (await frontend.assign(asked)).strip_usage


def olp_stream_options(**kwargs):
    from tensorrt_llm.serve.openai_protocol import StreamOptions

    return StreamOptions(**kwargs)


@pytest.mark.asyncio
async def test_interrupted_stream_is_not_learned():
    """Continuous usage of a stream that ends before its final usage chunk is not a reply length."""
    coordinator = _FakeCoordinator()
    frontend = olp.OutputLengthPriority(coordinator, "ridge")
    ticket = olp._Ticket("c1", 2000.0, 7)

    async def stream():
        yield _sse({"choices": [{"text": "a"}], "usage": {"completion_tokens": 5}})
        raise ConnectionResetError("client went away")

    with pytest.raises(ConnectionResetError):
        async for _ in frontend.track(stream(), ticket):
            pass
    await asyncio.sleep(0.01)
    assert coordinator.recorded == [] and frontend.no_usage == 1


@pytest.mark.asyncio
async def test_replies_are_reported_in_batches(monkeypatch):
    monkeypatch.setattr(olp, "RECORD_BATCH_MAX", 3)
    calls = []

    class Coordinator(_FakeCoordinator):
        async def record_output_lengths(self, records):
            calls.append(list(records))
            return len(records)

    frontend = olp.OutputLengthPriority(Coordinator(), "ridge", record_flush_s=0.05)
    for i in range(4):
        frontend._record(olp._Ticket("c", 10.0, i), 50 + i)
    await asyncio.sleep(0)
    assert [len(c) for c in calls] == [3]
    await asyncio.sleep(0.1)
    assert [len(c) for c in calls] == [3, 1]


@pytest.mark.asyncio
async def test_prediction_from_the_generation_placement():
    """In a fleet the prediction arrives with /select; no separate coordinator call."""
    coordinator = _FakeCoordinator()
    frontend = olp.OutputLengthPriority(coordinator, "ridge", via_select=True)
    request = _gen_request("c1", 2000)
    ticket = await frontend.assign(request, placed_later=True)
    assert coordinator.predicted == [] and ticket.via_select
    frontend.placed(ticket, request, {olp.OUTPUT_LENGTH_KEY: {"tokens": 9.0, "key": 11}})
    assert ticket.key == 11 and frontend.assigned == 1
    missing = _gen_request("c2", 2000)
    ticket = await frontend.assign(missing, placed_later=True)
    frontend.placed(ticket, missing, {})
    assert frontend.missing == 1 and missing.priority == pytest.approx(frontend.prior_priority)
    # A request placed before its context phase (conditional disaggregation) asks separately.
    early = _gen_request("c3", 2000)
    assert (await frontend.assign(early)).key == 7 and coordinator.predicted == [("c3", 2000.0)]
