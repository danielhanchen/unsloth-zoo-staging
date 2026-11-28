# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""CPU chunked CE with bf16 weights (bf16 full finetuning): the LM head gradient comes from bf16 GEMMs, not an fp32
GEMM into an fp32 [V, H] buffer, and stays as close to fp64 as the fp32 path."""

import pytest
import torch

from unsloth_zoo.cpu import ops


@pytest.fixture(autouse = True)
def _native_bf16(monkeypatch):
    # The bf16 GEMM path is for CPUs with native bf16 off Windows; pretend this host is one, whatever runs the test.
    monkeypatch.setattr(ops, "_native_bf16_gemm", lambda: True)


def _inputs(rows, hd = 64, vocab = 512, dtype = torch.bfloat16, seed = 0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, rows, hd, generator = g).to(dtype)
    weight = (torch.randn(vocab, hd, generator = g) * hd ** -0.5).to(dtype)
    labels = torch.randint(0, vocab, (1, rows), generator = g)
    labels[0, ::7] = -100
    return hidden, weight, labels


def _grads(hidden, weight, labels, chunk_rows, autocast = True):
    h = hidden.clone().requires_grad_()
    w = weight.clone().requires_grad_()
    with torch.autocast("cpu", dtype = torch.bfloat16, enabled = autocast):
        loss = ops.chunked_linear_cross_entropy(h, w, labels, chunk_rows = chunk_rows)
    loss.backward()
    return loss.detach(), h.grad, w.grad


def _oracle(hidden, weight, labels):
    h = hidden.double().requires_grad_()
    w = weight.double().requires_grad_()
    logits = (h @ w.t())[:, :-1].reshape(-1, w.shape[0])
    loss = torch.nn.functional.cross_entropy(logits, labels[:, 1:].reshape(-1), ignore_index = -100)
    loss.backward()
    return loss.detach(), h.grad, w.grad


def _rel(a, ref):
    return float((a.double() - ref).norm() / ref.norm())


def _stock_autograd(hidden, weight, labels):
    """What a plain bf16 nn.Linear head under autocast gives: bf16 GEMMs on the bf16 logit gradient."""
    h = hidden.clone().requires_grad_()
    w = weight.clone().requires_grad_()
    with torch.autocast("cpu", dtype = torch.bfloat16):
        logits = torch.nn.functional.linear(h, w)
    logits = logits.float()[:, :-1].reshape(-1, w.shape[0])
    torch.nn.functional.cross_entropy(logits, labels[:, 1:].reshape(-1), ignore_index = -100).backward()
    return w.grad


def _forward_head_grad_dtype(hidden, weight, labels, chunk_rows):
    flat = hidden.reshape(-1, hidden.shape[-1])
    shift = torch.nn.functional.pad(labels, (0, 1), value = -100)[..., 1:].reshape(-1)
    _, _, grad_weight = ops._chunked_ce_forward_impl(
        flat, weight, shift, chunk_rows, -100, True, True, False, True, torch.bfloat16)
    return grad_weight.dtype


@pytest.mark.parametrize("rows, hd, chunk_rows, dtype", [
    (64, 64, 128, torch.bfloat16),    # one chunk
    (300, 512, 64, torch.bfloat16),   # five chunks, one GEMM over all rows
    (300, 64, 64, torch.float32),     # more rows than the hidden size: the fp32 path
])
def test_bf16_weights_take_one_bf16_head_gradient_gemm(rows, hd, chunk_rows, dtype):
    hidden, weight, labels = _inputs(rows, hd = hd)
    assert _forward_head_grad_dtype(hidden, weight, labels, chunk_rows) == dtype


@pytest.mark.parametrize("rows, hd, chunk_rows", [(64, 64, 128), (300, 512, 64), (300, 512, 300)])
def test_bf16_head_gradient_matches_stock_autograd_precision(rows, hd, chunk_rows, monkeypatch):
    hidden, weight, labels = _inputs(rows, hd = hd)
    _, _, ref_dw = _oracle(hidden, weight, labels)
    loss, dh, dw = _grads(hidden, weight, labels, chunk_rows)
    monkeypatch.setenv("UNSLOTH_CPU_HEAD_GRAD_FP32", "1")
    loss32, dh32, dw32 = _grads(hidden, weight, labels, chunk_rows)
    assert dw.dtype == dw32.dtype == torch.bfloat16
    # Loss and hidden gradient do not depend on how the head gradient is formed.
    assert torch.equal(loss, loss32) and torch.equal(dh, dh32)
    # The bf16 GEMM rounds the logit gradient to bf16, as autograd does for a bf16 nn.Linear: the same distance from
    # fp64 as stock autograd, and at most twice the fp32 path's (both at bf16 resolution, 2^-9).
    err, err32, stock = _rel(dw, ref_dw), _rel(dw32, ref_dw), _rel(_stock_autograd(hidden, weight, labels), ref_dw)
    assert err <= 1.02 * stock + 1e-5, (err, stock)
    assert err <= 2 * err32, (err, err32)


def test_kill_switch_more_rows_and_fp32_weights_keep_the_fp32_gemm(monkeypatch):
    hidden, weight, labels = _inputs(300, hd = 64)
    _, _, dw = _grads(hidden, weight, labels, 64)
    monkeypatch.setenv("UNSLOTH_CPU_HEAD_GRAD_FP32", "1")
    assert _forward_head_grad_dtype(hidden, weight, labels, 64) == torch.float32
    _, _, dw32 = _grads(hidden, weight, labels, 64)
    assert torch.equal(dw, dw32)  # more rows than the hidden size: bitwise the fp32 path
    monkeypatch.delenv("UNSLOTH_CPU_HEAD_GRAD_FP32")
    # fp32 master weights under bf16 autocast: the fp32 path, so the fp32 gradient keeps its precision.
    hidden, weight, labels = _inputs(64)
    assert _forward_head_grad_dtype(hidden, weight.float(), labels, 128) == torch.float32
    _, _, dw = _grads(hidden, weight.float(), labels, 128)
    assert dw.dtype == torch.float32


def test_all_labels_ignored_gives_a_zero_bf16_head_grad():
    hidden, weight, labels = _inputs(16)
    labels[:] = -100
    _, _, dw = _grads(hidden, weight, labels, 128)
    assert dw is not None and dw.dtype == torch.bfloat16 and not bool(dw.any())


def test_gradient_accumulation_and_a_non_unit_grad_output(monkeypatch):
    hidden, weight, labels = _inputs(64)
    w = weight.clone().requires_grad_()
    for factor in (1.0, 0.5):
        h = hidden.clone().requires_grad_()
        with torch.autocast("cpu", dtype = torch.bfloat16):
            loss = ops.chunked_linear_cross_entropy(h, w, labels, chunk_rows = 128)
        (loss * factor).backward()
    monkeypatch.setenv("UNSLOTH_CPU_HEAD_GRAD_FP32", "1")
    w32 = weight.clone().requires_grad_()
    for factor in (1.0, 0.5):
        h = hidden.clone().requires_grad_()
        with torch.autocast("cpu", dtype = torch.bfloat16):
            loss = ops.chunked_linear_cross_entropy(h, w32, labels, chunk_rows = 128)
        (loss * factor).backward()
    assert w.grad.dtype == torch.bfloat16
    assert _rel(w.grad, w32.grad.double()) < 1e-2


def test_cpus_without_native_bf16_and_windows_keep_the_fp32_gemm(monkeypatch):
    hidden, weight, labels = _inputs(64)
    monkeypatch.setattr(ops, "_native_bf16_gemm", lambda: False)
    assert _forward_head_grad_dtype(hidden, weight, labels, 128) == torch.float32
    _, _, dw = _grads(hidden, weight, labels, 128)
    monkeypatch.setenv("UNSLOTH_CPU_HEAD_GRAD_FP32", "1")
    _, _, dw32 = _grads(hidden, weight, labels, 128)
    assert torch.equal(dw, dw32)
    monkeypatch.undo()
    monkeypatch.setattr(ops.sys, "platform", "win32")
    assert not ops._native_bf16_gemm()
