# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's l2norm_fwd_kernel1 Triton kernel.

The l2norm_fwd wrapper dispatches to l2norm_fwd_kernel1 only when
USE_DEFAULT_FLA_NORM is set and the feature dim D > 512 (the one-row-per-program
path). These tests force that dispatch and compare against a float32 PyTorch
reference: y = x / sqrt(sum(x^2) + eps), computed per row.

Source: vllm/model_executor/layers/fla/ops/l2norm.py
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops import l2norm as l2norm_mod
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


@pytest.fixture(autouse=True)
def force_default_fla_norm(monkeypatch):
    """Route l2norm_fwd through the D > 512 kernel1 path.

    USE_DEFAULT_FLA_NORM is read from the environment at import time; patch the
    module attribute directly so the dispatch is deterministic regardless of
    import order.
    """
    monkeypatch.setattr(l2norm_mod, "USE_DEFAULT_FLA_NORM", 1)


def l2norm_ref(x, eps=1e-6):
    """Pure PyTorch L2 norm over the last dim: y = x / sqrt(sum(x^2) + eps)."""
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# D > 512 targets l2norm_fwd_kernel1.
CONFIGS = [
    (16, 1024),
    (32, 1024),
    (1, 2048),
    (64, 768),
    (128, 1024),
    (16, 4096),
]


@pytest.mark.parametrize("T,D", CONFIGS, ids=[f"T{t}_D{d}" for t, d in CONFIGS])
@torch.inference_mode()
def test_l2norm_kernel1(T, D):
    """l2norm_fwd (D > 512 path) must match the PyTorch reference (fp32)."""
    torch.manual_seed(0)
    x = torch.randn(T, D, device=DEVICE, dtype=torch.float32)

    y = l2norm_fwd(x, eps=1e-6)
    y_ref = l2norm_ref(x, eps=1e-6)

    assert y.shape == y_ref.shape
    assert not torch.isnan(y).any()
    torch.testing.assert_close(y.float(), y_ref, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_l2norm_kernel1_3d():
    """3D input is flattened to 2D rows by the wrapper."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, 1024, device=DEVICE, dtype=torch.float32)

    y = l2norm_fwd(x, eps=1e-6)
    y_ref = l2norm_ref(x, eps=1e-6)

    assert y.shape == x.shape
    torch.testing.assert_close(y.float(), y_ref, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_l2norm_kernel1_bfloat16():
    """bfloat16 input matches the fp32 reference within a loose tolerance."""
    torch.manual_seed(0)
    x = torch.randn(32, 1024, device=DEVICE, dtype=torch.bfloat16)

    y = l2norm_fwd(x, eps=1e-6)
    y_ref = l2norm_ref(x.float(), eps=1e-6)

    torch.testing.assert_close(y.float(), y_ref, rtol=5e-3, atol=5e-3)
