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
"""Online output-length prediction for the generation admission order in disaggregated serving.

Enabled by ``output_length_prediction`` in the disaggregated server config. After a request's context
phase, the orchestrator predicts its output length and writes
``priority = 1 / (1 + tokens / PRIORITY_SCALE_TOKENS)`` (strictly decreasing in the prediction) into the
generation request's ``priority``. Generation servers that run ``waiting_queue_policy: priority`` then
admit the shortest predicted reply first, and a pooled generation router places in the same order.

The predictor learns from the replies of the running server only; nothing is loaded or saved. It lives
in the coordinator (one per orchestrator), so it sees the replies of every frontend. Its inputs are the
conversation's earlier replies (keyed by ``conversation_params.conversation_id``) and the context
phase's prompt length; the request's ``max_tokens`` is never read.

Per conversation it keeps the recorded replies' count, last / EMA(0.5) / mean / max output tokens and
the last prompt length. A request's 7 features: log1p of the history count, of its prompt tokens, and
of the last / EMA / mean / max output tokens (NaN without history), and the signed log1p of the tokens
appended since the last reply (NaN without history). Rules: a history of exactly one reply with a
prompt under 1,500 tokens is dropped when the next request appends over 10,000 tokens (a new prefix),
and replies of at most 2 tokens (cache primers) are not learned.

Predictors, each taking over at its first fit:
  * a conversation with history gets its EMA(0.5); otherwise the running median of all learned
    replies, or ``PRIOR_TOKENS`` before the first;
  * ``ridge``: a ridge regression of log1p(output tokens) on the features (NaN as -1 plus a missing
    indicator per feature), refit every ``RIDGE_REFIT_S`` on the replies learned so far;
  * ``ridge_gbm``: also a gradient-boosted tree ensemble on the same target (scikit-learn
    ``HistGradientBoostingRegressor``), refit every ``GBM_REFIT_S`` in a separate Python process off
    the request path; the coordinator evaluates its trees itself.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import math
import os
import subprocess  # nosec B404 - runs sys.executable -I with a fixed argv, no shell
import sys
import threading
import time
from array import array
from collections import OrderedDict
from typing import Any, AsyncIterator, Optional, Tuple

from tensorrt_llm.logger import logger

PREDICTORS = ("ridge", "ridge_gbm")

# priority = 1 / (1 + tokens / scale); the admission order does not depend on the scale.
PRIORITY_SCALE_TOKENS = 1024.0
# The prediction before any reply was learned, and for a request the coordinator could not predict.
PRIOR_TOKENS = 375.77
PREDICT_TIMEOUT_S = 1.0
RECORD_TIMEOUT_S = 5.0
STATS_LOG_PERIOD_S = 30.0

FEATURES = ("turn", "isl", "prev_out", "ema_out", "mean_out", "max_out", "appended")
RESET_MAX_PREV_ISL = 1500.0
RESET_MIN_APPENDED = 10000.0
SKIP_MAX_OUT = 2.0
MAX_CONVERSATIONS = 50000
# Features of predicted requests whose reply has not been learned yet (the oldest are dropped).
MAX_PENDING_ROWS = 200000

RIDGE_REFIT_S = 120.0
RIDGE_ALPHA = 1.0
RIDGE_MIN_ROWS = 200
GBM_REFIT_S = 300.0
GBM_MIN_ROWS = 1000
GBM_THREADS = 4
GBM_FIT_TIMEOUT_S = 600.0
# A fitted ensemble goes live only if the coordinator's tree walk matches scikit-learn's predict.
GBM_EXPORT_TOL = 1e-6
GBM_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 0,
}

NAN = float("nan")

# A conversation's history as passed to features(): (count, last out, EMA out, sum out, max out, last prompt).
Snapshot = Tuple[float, float, float, float, float, float]


def priority_of(tokens: float) -> float:
    """Map predicted output tokens into (0, 1], strictly decreasing: shorter replies go first."""
    return 1.0 / (1.0 + max(float(tokens), 0.0) / PRIORITY_SCALE_TOKENS)


class _History:
    __slots__ = ("n", "last_out", "ema", "sum_out", "max_out", "last_isl")

    def __init__(self) -> None:
        self.n = 0
        self.last_out = NAN
        self.ema = NAN
        self.sum_out = 0.0
        self.max_out = NAN
        self.last_isl = NAN

    def record(self, isl: float, out: float) -> None:
        self.ema = out if self.n == 0 else 0.5 * out + 0.5 * self.ema
        self.sum_out += out
        self.max_out = out if self.n == 0 else max(self.max_out, out)
        self.last_out = out
        self.last_isl = isl
        self.n += 1

    def snapshot(self) -> Snapshot:
        return (float(self.n), self.last_out, self.ema, self.sum_out, self.max_out, self.last_isl)


class ConversationLengthTable:
    """Per-conversation reply history (least recently used conversations evicted)."""

    def __init__(self, max_conversations: int = MAX_CONVERSATIONS) -> None:
        self.max_conversations = max_conversations
        self._histories: OrderedDict[str, _History] = OrderedDict()
        self.resets = 0
        self.recorded = 0
        self.skipped = 0

    def __len__(self) -> int:
        return len(self._histories)

    def snapshot_for(self, conv_id: Optional[str], isl: float) -> Optional[Snapshot]:
        """The conversation's history for a new request, after the reset rule; None without one."""
        if conv_id is None:
            return None
        history = self._histories.get(conv_id)
        if history is None:
            return None
        self._histories.move_to_end(conv_id)
        if (
            history.n == 1
            and history.last_isl < RESET_MAX_PREV_ISL
            and isl - history.last_isl - history.last_out > RESET_MIN_APPENDED
        ):
            del self._histories[conv_id]
            self.resets += 1
            return None
        return history.snapshot()

    def record(self, conv_id: Optional[str], isl: float, out: float) -> bool:
        """Record a finished reply of a conversation unless it has at most ``SKIP_MAX_OUT`` tokens.

        Returns whether it was recorded; only recorded replies are learned.
        """
        if conv_id is None:
            return False
        if out <= SKIP_MAX_OUT:
            self.skipped += 1
            return False
        history = self._histories.get(conv_id)
        if history is None:
            history = self._histories[conv_id] = _History()
            if len(self._histories) > self.max_conversations:
                self._histories.popitem(last=False)
        else:
            self._histories.move_to_end(conv_id)
        history.record(float(isl), float(out))
        self.recorded += 1
        return True


def _log1p_nonneg(value: float) -> float:
    return NAN if value != value else math.log1p(max(value, 0.0))


def _signed_log1p(value: float) -> float:
    if value != value:
        return NAN
    return math.copysign(math.log1p(abs(value)), value) if value != 0.0 else 0.0


def features(snapshot: Optional[Snapshot], isl: float) -> list:
    """The predictors' input row for a request with ``isl`` prompt tokens and a history (None: none)."""
    isl = float(isl)
    if snapshot is None or snapshot[0] <= 0:
        return [0.0, _log1p_nonneg(isl), NAN, NAN, NAN, NAN, NAN]
    n, last_out, ema, sum_out, max_out, last_isl = (float(v) for v in snapshot)
    return [
        math.log1p(n),
        _log1p_nonneg(isl),
        _log1p_nonneg(last_out),
        _log1p_nonneg(ema),
        _log1p_nonneg(sum_out / n),
        _log1p_nonneg(max_out),
        _signed_log1p(isl - last_isl - last_out),
    ]


class TreeEnsemble:
    """Raw prediction of a fitted ``HistGradientBoostingRegressor``: baseline + one leaf per tree.

    A node is ``(is_leaf, value, feature, threshold, missing_go_to_left, left, right)``.
    """

    def __init__(self, baseline: float, trees: list) -> None:
        self.baseline = float(baseline)
        self.trees = [[tuple(node) for node in tree] for tree in trees]

    def raw(self, x: list) -> float:
        acc = self.baseline
        for nodes in self.trees:
            i = 0
            while True:
                is_leaf, value, feature, threshold, missing_left, left, right = nodes[i]
                if is_leaf:
                    acc += value
                    break
                v = x[feature]
                if v != v:
                    i = left if missing_left else right
                else:
                    i = left if v <= threshold else right
        return acc


class RunningMedian:
    """Median of every value added so far (the mean of the two middle values for an even count)."""

    __slots__ = ("_lo", "_hi", "n")

    def __init__(self) -> None:
        self._lo: list = []  # max-heap of the lower half, negated
        self._hi: list = []  # min-heap of the upper half
        self.n = 0

    def add(self, value: float) -> None:
        value = float(value)
        if self._lo and value > -self._lo[0]:
            heapq.heappush(self._hi, value)
        else:
            heapq.heappush(self._lo, -value)
        if len(self._lo) > len(self._hi) + 1:
            heapq.heappush(self._hi, -heapq.heappop(self._lo))
        elif len(self._hi) > len(self._lo):
            heapq.heappush(self._lo, -heapq.heappop(self._hi))
        self.n += 1

    def value(self) -> float:
        if not self.n:
            return NAN
        if len(self._lo) > len(self._hi):
            return -self._lo[0]
        return (-self._lo[0] + self._hi[0]) / 2.0


def ridge_row(x: list) -> list:
    """The ridge's design row: the features with NaN as -1, then a 0/1 missing indicator per feature."""
    return [-1.0 if v != v else float(v) for v in x] + [1.0 if v != v else 0.0 for v in x]


def _solve(a: list, b: list) -> list:
    """Solve ``a @ w = b`` by Gaussian elimination with partial pivoting (``a`` and ``b`` are consumed)."""
    n = len(b)
    for k in range(n):
        p = max(range(k, n), key=lambda r: abs(a[r][k]))
        if a[p][k] == 0.0:
            raise ZeroDivisionError("singular ridge system")
        if p != k:
            a[k], a[p] = a[p], a[k]
            b[k], b[p] = b[p], b[k]
        row_k, pivot = a[k], a[k][k]
        for r in range(k + 1, n):
            f = a[r][k] / pivot
            if f != 0.0:
                row_r = a[r]
                for c in range(k, n):
                    row_r[c] -= f * row_k[c]
                b[r] -= f * b[k]
    w = [0.0] * n
    for k in range(n - 1, -1, -1):
        w[k] = (b[k] - sum(a[k][c] * w[c] for c in range(k + 1, n))) / a[k][k]
    return w


class OnlineRidge:
    """Ridge regression kept as running means and centered co-moments.

    ``fit`` solves ``(Cxx + alpha I) w = Cxy`` with ``intercept = mean(y) - mean(x) @ w``: scikit-learn's
    ``Ridge(alpha)`` on the same rows (the intercept is not penalized).
    """

    def __init__(self, dim: int, alpha: float = RIDGE_ALPHA) -> None:
        self.dim = dim
        self.alpha = float(alpha)
        self.n = 0
        self.mean_x = [0.0] * dim
        self.mean_y = 0.0
        self.cxx = [[0.0] * dim for _ in range(dim)]
        self.cxy = [0.0] * dim
        self.coef: Optional[list] = None
        self.intercept = NAN
        self.fit_rows = 0

    def add(self, z: list, y: float) -> None:
        """Welford update: ``C += (z - old mean) (z - new mean)^T``."""
        self.n += 1
        n = self.n
        dx = [zi - mi for zi, mi in zip(z, self.mean_x)]
        self.mean_x = [mi + di / n for mi, di in zip(self.mean_x, dx)]
        self.mean_y += (y - self.mean_y) / n
        ex = [zi - mi for zi, mi in zip(z, self.mean_x)]
        ey = y - self.mean_y
        for i in range(self.dim):
            di = dx[i]
            if di != 0.0:
                row = self.cxx[i]
                for j in range(self.dim):
                    row[j] += di * ex[j]
                self.cxy[i] += di * ey

    def fit(self) -> None:
        d = self.dim
        a = [
            [
                0.5 * (self.cxx[i][j] + self.cxx[j][i]) + (self.alpha if i == j else 0.0)
                for j in range(d)
            ]
            for i in range(d)
        ]
        w = _solve(a, list(self.cxy))
        self.coef = w
        self.intercept = self.mean_y - sum(wi * mi for wi, mi in zip(w, self.mean_x))
        self.fit_rows = self.n

    def raw(self, z: list) -> float:
        return self.intercept + sum(wi * zi for wi, zi in zip(self.coef, z))


# The fit process: numpy + scikit-learn only. Rows come in on stdin, the trees go out as JSON on stdout,
# with the largest difference between its tree walk and scikit-learn's predict on up to 2,000 rows.
_GBM_FIT_SCRIPT = r"""
import json, os, sys, time
try:
    os.nice(10)
except OSError:
    pass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
header = json.loads(sys.stdin.buffer.readline())
n, d = header["n"], header["d"]
data = np.frombuffer(sys.stdin.buffer.read(), dtype=np.float64).reshape(n, d + 1)
X, y = data[:, :d], data[:, d]
t0 = time.perf_counter()
model = HistGradientBoostingRegressor(**header["params"]).fit(X, y)
fit_s = time.perf_counter() - t0
trees = []
for predictors in model._predictors:
    nodes = predictors[0].nodes
    if len(predictors) != 1 or bool(np.any(nodes["is_categorical"])):
        raise SystemExit("unsupported tree")
    trees.append([[bool(r["is_leaf"]), float(r["value"]), int(r["feature_idx"]), float(r["num_threshold"]),
                   bool(r["missing_go_to_left"]), int(r["left"]), int(r["right"])] for r in nodes])
baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
def walk(x):
    acc = baseline
    for nodes in trees:
        i = 0
        while True:
            leaf, v, f, thr, ml, le, ri = nodes[i]
            if leaf:
                acc += v
                break
            xv = x[f]
            i = (le if ml else ri) if xv != xv else (le if xv <= thr else ri)
    return acc
k = min(n, 2000)
ref = model.predict(X[:k]) if k else []
diff = max((abs(walk([float(v) for v in X[j]]) - float(ref[j])) for j in range(k)), default=0.0)
json.dump({"baseline": baseline, "trees": trees, "n_iter": int(model.n_iter_), "fit_s": fit_s,
           "export_max_abs_diff": diff}, sys.stdout)
"""


def check_gbm_dependencies(python: Optional[str] = None) -> None:
    """Raise ImportError unless the fit process can import numpy and scikit-learn."""
    python = python or sys.executable
    result = subprocess.run(
        [python, "-I", "-c", "import numpy, sklearn.ensemble"],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise ImportError(
            "output_length_prediction.predictor='ridge_gbm' fits its trees with numpy and scikit-learn, "
            f"which {python} cannot import ({result.stderr.decode(errors='replace').strip()[-300:]}); "
            "install scikit-learn or use predictor='ridge'"
        )


class SubprocessGbmFitter:
    """Fits the tree ensemble off the request path, in a separate Python process.

    ``start`` hands the rows to a background thread that runs the process; ``poll`` returns its result
    once. ``sync=True`` fits inside ``start`` (tests).
    """

    def __init__(
        self,
        threads: int = GBM_THREADS,
        timeout_s: float = GBM_FIT_TIMEOUT_S,
        python: Optional[str] = None,
        sync: bool = False,
    ) -> None:
        self.threads = threads
        self.timeout_s = timeout_s
        self.python = python or sys.executable
        self.sync = sync
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[dict] = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, rows: array, n: int, meta: dict) -> None:
        header = {"n": n, "d": len(FEATURES), "params": GBM_PARAMS}
        payload = json.dumps(header).encode() + b"\n" + rows.tobytes()
        if self.sync:
            self._finish(self._run(payload), meta)
            return
        self._thread = threading.Thread(
            target=lambda: self._finish(self._run(payload), meta),
            name="output-length-gbm-fit",
            daemon=True,
        )
        self._thread.start()

    def _run(self, payload: bytes) -> dict:
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS=str(self.threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            result = subprocess.run(
                [self.python, "-I", "-c", _GBM_FIT_SCRIPT],
                input=payload,
                capture_output=True,
                timeout=self.timeout_s,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        if result.returncode != 0:
            return {
                "error": f"exit {result.returncode}: {result.stderr.decode(errors='replace')[-1500:]}"
            }
        try:
            return json.loads(result.stdout)
        except ValueError as e:
            return {"error": f"unreadable fit output: {e}"}

    def _finish(self, result: dict, meta: dict) -> None:
        result.update(meta)
        with self._lock:
            self._result = result

    def poll(self) -> Optional[dict]:
        with self._lock:
            result, self._result = self._result, None
        return result


class OnlineLengthPredictor:
    """The coordinator's predictor: the conversation table and the online models.

    Events carry their time ``t`` (``time.monotonic()`` in the coordinator), so the same events give the
    same predictions, except that a tree ensemble goes live at the first event after its fit returns.
    """

    def __init__(self, predictor: str, gbm_fitter: Any = None) -> None:
        if predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {PREDICTORS}, got {predictor!r}")
        self.predictor = predictor
        self.table = ConversationLengthTable()
        self.median = RunningMedian()
        self.ridge = OnlineRidge(2 * len(FEATURES))
        self.gbm_on = predictor == "ridge_gbm"
        self.gbm_fitter = gbm_fitter
        if self.gbm_on and self.gbm_fitter is None:
            check_gbm_dependencies()
            self.gbm_fitter = SubprocessGbmFitter()
        self.gbm_rows = array("d")
        self.gbm_n = 0
        self.gbm: Optional[TreeEnsemble] = None
        self.gbm_fits = 0
        self.gbm_errors = 0
        self.gbm_busy_skips = 0
        self.ridge_fits = 0
        self.t0: Optional[float] = None
        self.next_ridge_fit = math.inf
        self.next_gbm_fit = math.inf
        self._pending: OrderedDict[int, list] = OrderedDict()
        self._next_key = 0
        self.unmatched = 0
        self.sources = {"prior": 0, "median": 0, "ema": 0, "ridge": 0, "gbm": 0}
        self._last_stats = time.monotonic()

    def _tick(self, t: float) -> None:
        """Start the clock at the first event, then run every refit boundary at or before ``t``."""
        if self.t0 is None:
            self.t0 = t
            self.next_ridge_fit = t + RIDGE_REFIT_S
            if self.gbm_on:
                self.next_gbm_fit = t + GBM_REFIT_S
            return
        while t >= self.next_ridge_fit:
            if self.ridge.n >= RIDGE_MIN_ROWS:
                self.ridge.fit()
                self.ridge_fits += 1
            self.next_ridge_fit += RIDGE_REFIT_S
        while t >= self.next_gbm_fit:
            if self.gbm_n >= GBM_MIN_ROWS:
                if self.gbm_fitter.busy():
                    self.gbm_busy_skips += 1
                else:
                    meta = {"rows": self.gbm_n, "boundary_s": self.next_gbm_fit - self.t0}
                    self.gbm_fitter.start(
                        self.gbm_rows[: self.gbm_n * (len(FEATURES) + 1)], self.gbm_n, meta
                    )
            self.next_gbm_fit += GBM_REFIT_S
        if self.gbm_on:
            result = self.gbm_fitter.poll()
            if result is not None:
                self._install_gbm(result)

    def _install_gbm(self, result: dict) -> None:
        diff = result.get("export_max_abs_diff")
        if "error" in result or diff is None or not diff <= GBM_EXPORT_TOL:
            self.gbm_errors += 1
            logger.warning(
                f"[output_length] tree fit on {result.get('rows')} replies failed, the current model "
                f"stays: {result.get('error') or f'export check {diff}'}"
            )
            return
        self.gbm = TreeEnsemble(result["baseline"], result["trees"])
        self.gbm_fits += 1
        logger.info(
            f"[output_length] tree fit #{self.gbm_fits} live: {result['rows']} replies, {result['n_iter']} "
            f"iterations, fit {result['fit_s']:.2f} s, export check {diff:.1e}"
        )

    def predict(self, conv_id: Optional[str], isl: float, t: float) -> Tuple[float, int]:
        """(predicted output tokens, key): the key goes back with the reply, so its features can be learned."""
        self._tick(t)
        snapshot = self.table.snapshot_for(conv_id, isl)
        x = features(snapshot, isl)
        if self.gbm is not None:
            tokens, source = max(math.expm1(self.gbm.raw(x)), 0.0), "gbm"
        elif self.ridge.coef is not None:
            tokens, source = max(math.expm1(self.ridge.raw(ridge_row(x))), 0.0), "ridge"
        elif snapshot is not None:
            tokens, source = snapshot[2], "ema"
        elif self.median.n:
            tokens, source = self.median.value(), "median"
        else:
            tokens, source = PRIOR_TOKENS, "prior"
        self.sources[source] += 1
        key = self._next_key
        self._next_key += 1
        self._pending[key] = x
        if len(self._pending) > MAX_PENDING_ROWS:
            self._pending.popitem(last=False)
        self._maybe_log_stats()
        return tokens, key

    def record(
        self, conv_id: Optional[str], isl: float, out: float, key: Optional[int], t: float
    ) -> bool:
        """Learn a finished reply of ``out`` tokens; whether it was learned (see ConversationLengthTable.record)."""
        self._tick(t)
        x = self._pending.pop(key, None) if key is not None else None
        if not self.table.record(conv_id, isl, out):
            return False
        self.median.add(out)
        if x is None:
            self.unmatched += 1
        else:
            y = math.log1p(out)
            self.ridge.add(ridge_row(x), y)
            if self.gbm_on:
                self.gbm_rows.extend(x)
                self.gbm_rows.append(y)
                self.gbm_n += 1
        self._maybe_log_stats()
        return True

    def stats_line(self) -> str:
        sources = ", ".join(f"{k} {v}" for k, v in self.sources.items() if v) or "none"
        line = (
            f"[output_length] {self.predictor}: {len(self.table)} conversations, replies learned "
            f"{self.table.recorded} (skipped {self.table.skipped}, resets {self.table.resets}), median "
            f"{self.median.value():.1f} tokens; predictions by source: {sources}; ridge fits {self.ridge_fits}"
        )
        if self.gbm_on:
            line += f"; tree fits {self.gbm_fits}, errors {self.gbm_errors}, busy skips {self.gbm_busy_skips}"
        return (
            line
            + f"; pending {len(self._pending)}, replies without their features {self.unmatched}"
        )

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats >= STATS_LOG_PERIOD_S:
            self._last_stats = now
            logger.info(self.stats_line())


def request_inputs(request: Any) -> Tuple[Optional[str], Optional[float]]:
    """(conversation id, prompt tokens) of a generation request after its context phase.

    Reads ``conversation_params.conversation_id`` and ``disaggregated_params.ctx_usage.prompt_tokens``
    only; in particular never ``max_tokens``.
    """
    conversation = getattr(request, "conversation_params", None)
    conv_id = getattr(conversation, "conversation_id", None) if conversation is not None else None
    params = getattr(request, "disaggregated_params", None)
    usage = getattr(params, "ctx_usage", None) if params is not None else None
    isl = getattr(usage, "prompt_tokens", None) if usage is not None else None
    return (str(conv_id) if conv_id is not None else None, float(isl) if isl is not None else None)


_USAGE_MARK = b'"completion_tokens"'


def completion_tokens_in(data: bytes) -> Tuple[Optional[int], bool]:
    """Scan SSE ``data:`` lines for ``usage.completion_tokens``: (value, from a final usage-only chunk)."""
    if _USAGE_MARK not in data:
        return None, False
    found, final = None, False
    for line in data.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        usage = obj.get("usage") if isinstance(obj, dict) else None
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            found = int(usage["completion_tokens"])
            final = not obj.get("choices")
    return found, final


class _Ticket:
    __slots__ = ("conv_id", "isl", "key", "recorded")

    def __init__(self, conv_id: Optional[str], isl: float, key: Optional[int]) -> None:
        self.conv_id = conv_id
        self.isl = isl
        self.key = key
        self.recorded = False


class OutputLengthPriority:
    """Frontend side: set the generation request's priority, then report its reply's length.

    ``coordinator`` provides ``predict_output_length`` and ``record_output_length``
    (:class:`~tensorrt_llm.serve.disagg_coordinator.DisaggCoordinator`).
    """

    def __init__(self, coordinator: Any, predictor: str) -> None:
        self._coordinator = coordinator
        self.predictor = predictor
        self.prior_priority = priority_of(PRIOR_TOKENS)
        self._tasks: set = set()
        self.assigned = 0
        self.missing = 0
        self.recorded = 0
        self.no_usage = 0
        self.record_errors = 0
        self._call_ms: list = []
        self._last_stats = time.monotonic()

    def startup_line(self) -> str:
        return (
            f"[output_length] generation priority from the predicted output length ({self.predictor}, "
            f"learned online in the coordinator): priority = 1 / (1 + tokens / {PRIORITY_SCALE_TOKENS:g}); "
            f"without a prediction, {PRIOR_TOKENS:g} tokens (priority {self.prior_priority:.4f}); max_tokens "
            f"is never read"
        )

    async def assign(self, gen_req: Any) -> Optional[_Ticket]:
        """Set ``gen_req.priority`` from the prediction; the ticket (None: nothing to learn) records the reply.

        Never raises: without a prediction (no prompt tokens, a coordinator error or timeout) the request
        gets the prior's priority.
        """
        if "priority" not in getattr(type(gen_req), "model_fields", {}):
            return None
        conv_id, isl = request_inputs(gen_req)
        if isl is None:
            self._miss(gen_req, "no prompt tokens")
            return None
        t0 = time.perf_counter()
        try:
            tokens, key = await asyncio.wait_for(
                self._coordinator.predict_output_length(conv_id, isl), PREDICT_TIMEOUT_S
            )
        except Exception as e:  # noqa: BLE001  (an admission hint is never worth failing a request)
            self._miss(gen_req, f"{type(e).__name__}: {e}")
            return _Ticket(conv_id, isl, None)
        self._call_ms.append((time.perf_counter() - t0) * 1e3)
        gen_req.priority = priority_of(tokens)
        self.assigned += 1
        self._maybe_log_stats()
        return _Ticket(conv_id, isl, key)

    def _miss(self, gen_req: Any, reason: str) -> None:
        self.missing += 1
        gen_req.priority = self.prior_priority
        if self.missing <= 10 or self.missing % 1000 == 0:
            logger.warning(f"[output_length] no prediction #{self.missing}, prior used: {reason}")

    def _record(self, ticket: _Ticket, out: Optional[int]) -> None:
        if ticket.recorded:
            return
        ticket.recorded = True
        if out is None:
            self.no_usage += 1
            return
        self.recorded += 1

        async def _send() -> None:
            try:
                await asyncio.wait_for(
                    self._coordinator.record_output_length(
                        ticket.conv_id, ticket.isl, float(out), ticket.key
                    ),
                    RECORD_TIMEOUT_S,
                )
            except Exception as e:  # noqa: BLE001  (learning must not fail the request)
                self.record_errors += 1
                logger.warning(f"[output_length] recording a reply failed: {type(e).__name__}: {e}")

        task = asyncio.get_running_loop().create_task(_send())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def track(self, response: Any, ticket: Optional[_Ticket]) -> Any:
        """Report the reply's ``usage.completion_tokens``: wraps a stream, reads a complete response."""
        if ticket is None:
            return response
        if hasattr(response, "__aiter__"):
            return self._track_stream(response, ticket)
        usage = getattr(response, "usage", None)
        out = getattr(usage, "completion_tokens", None) if usage is not None else None
        self._record(ticket, int(out) if out is not None else None)
        return response

    async def _track_stream(self, stream: AsyncIterator, ticket: _Ticket):
        carry = b""
        last_seen: Optional[int] = None
        try:
            async for chunk in stream:
                if not ticket.recorded:
                    data = chunk if isinstance(chunk, (bytes, bytearray)) else str(chunk).encode()
                    scan = carry + bytes(data)
                    newline = scan.rfind(b"\n")
                    carry = scan[newline + 1 :][-4096:] if newline >= 0 else scan[-4096:]
                    if newline >= 0:
                        out, final = completion_tokens_in(scan[: newline + 1])
                        if out is not None:
                            last_seen = out
                            if final:
                                self._record(ticket, out)
                yield chunk
        finally:
            if not ticket.recorded:
                self._record(ticket, last_seen)

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats < STATS_LOG_PERIOD_S:
            return
        self._last_stats = now
        ms = sorted(self._call_ms)
        self._call_ms = []
        p50 = ms[len(ms) // 2] if ms else NAN
        logger.info(
            f"[output_length] assigned {self.assigned}, missing {self.missing}, replies recorded {self.recorded} / "
            f"without usage {self.no_usage} / record errors {self.record_errors}; coordinator call p50 "
            f"{p50:.2f} ms, max {ms[-1] if ms else NAN:.2f} ms"
        )
