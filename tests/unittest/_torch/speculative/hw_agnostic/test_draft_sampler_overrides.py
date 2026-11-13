# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft samplers' contract between SpecWorkerBase and its overrides.

``sample_draft_tokens`` hands ``tokens_out`` to ``greedy_sample_draft_with_tp_gather`` and the
draft loops hand ``out`` to ``sample_draft_tokens``; an override with an older parameter list
raises TypeError at the first draft step. The signature check reads the source of every
override in ``tensorrt_llm/`` (modeling_v2 targets included, whose GPU-only modules it does not
import).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.cpu_only

_PKG = Path(__file__).resolve().parents[5] / "tensorrt_llm"
_BASE = _PKG / "_torch/speculative/interface.py"
_METHODS = ("greedy_sample_draft_with_tp_gather", "sample_draft_tokens")


def _params(fn: ast.FunctionDef) -> list[tuple[str, str]]:
    a = fn.args
    params = [(p.arg, "positional") for p in a.posonlyargs + a.args]
    params += [(p.arg, "keyword") for p in a.kwonlyargs]
    if a.vararg is not None:
        params.append((a.vararg.arg, "*"))
    if a.kwarg is not None:
        params.append((a.kwarg.arg, "**"))
    return params


def _definitions(path: Path) -> list[tuple[str, str, ast.FunctionDef]]:
    """(class, method, def) for every draft-sampler method defined in a class of ``path``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in _METHODS:
                    found.append((node.name, item.name, item))
    return found


def test_every_draft_sampler_override_takes_the_base_parameters() -> None:
    base = {
        method: _params(fn) for cls, method, fn in _definitions(_BASE) if cls == "SpecWorkerBase"
    }
    assert set(base) == set(_METHODS)
    overrides = []
    for path in sorted(_PKG.rglob("*.py")):
        for cls, method, fn in _definitions(path):
            if path == _BASE and cls == "SpecWorkerBase":
                continue
            overrides.append((path.relative_to(_PKG).as_posix(), cls))
            assert _params(fn) == base[method], (path, cls, method)
    # Guard against a scan that finds nothing: the targets known to override.
    names = {path.split("/")[-2] for path, _ in overrides}
    assert {
        "glm_5_2_nvfp4__sm_103__tep4",
        "minimax_m3_nvfp4__sm_107__tp2",
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1",
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4",
    } <= names


def _worker(cls: type) -> SpecWorkerBase:
    w = cls.__new__(cls)
    nn.Module.__init__(w)  # attribute storage only; the worker's own init needs a model config
    w.mapping = SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False)
    w._d2t = None
    return w


class _Worker(SpecWorkerBase):
    """The base samplers on a concrete worker."""

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def max_draft_len(self) -> int:
        return 1


class _OwnTensorWorker(_Worker):
    """An override that ignores ``tokens_out`` and returns a tensor of its own."""

    def greedy_sample_draft_with_tp_gather(
        self, logits, spec_metadata=None, mapping_lm_head_tp=None, tokens_out=None
    ):
        return logits.argmax(dim=-1).to(torch.int32)


class _ThreeArgWorker(_Worker):
    """An override with main's three-argument signature (out-of-tree subclasses)."""

    def greedy_sample_draft_with_tp_gather(
        self, logits, spec_metadata=None, mapping_lm_head_tp=None
    ):
        return logits.argmax(dim=-1).to(torch.int32)


_META = SimpleNamespace(wants_advanced_draft_sampling=False, is_cuda_graph=False)


@pytest.mark.parametrize("cls", [_Worker, _OwnTensorWorker, _ThreeArgWorker])
def test_the_greedy_step_fills_out_on_every_override(cls) -> None:
    w = _worker(cls)
    logits = torch.randn(5, 64)
    want = logits.argmax(dim=-1).to(torch.int32)
    assert torch.equal(w.sample_draft_tokens(logits, _META, 5, draft_step=0), want)
    if cls is _ThreeArgWorker:
        return  # takes no tokens_out: only calls without ``out`` reach it unchanged
    out = torch.full((5,), -1, dtype=torch.int32)
    tokens = w.sample_draft_tokens(logits, _META, 5, draft_step=0, out=out)
    assert tokens.data_ptr() == out.data_ptr()  # the tokens are out's storage
    assert torch.equal(out, want)
