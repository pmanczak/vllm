# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_gated_delta_rule_fwd_h Triton operator.

Exercises chunk_gated_delta_rule_fwd_kernel_h_blockdim64 via its Python wrapper.
The kernel maintains a recurrent hidden state h of shape (V, K) and, per chunk
of BT=64 timesteps: stores the current state, computes v = u - w @ h^T, saves v
(pre-gating), applies scalar (g) and/or key-wise (gk) gating, then updates
h += v^T @ k. Compared against a naive float32 PyTorch reference.

Source: vllm/model_executor/layers/fla/ops/chunk_delta_h.py
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def chunk_gated_delta_rule_fwd_ref(
    k, w, u, g=None, gk=None, initial_state=None,
    output_final_state=False, chunk_size=64,
):
    """Naive PyTorch reference for the chunked gated delta rule forward pass.

    Args:
        k: (B, T, Hg, K) — key tensor (Hg may be < H for GQA).
        w: (B, T, H, K) — weight/decay tensor.
        u: (B, T, H, V) — value/input tensor.
        g:  (B, T, H) — optional scalar (per-head) gating.
        gk: (B, T, H, K) — optional key-wise gating.
        initial_state: (B, H, V, K) — optional initial hidden state.

    Returns:
        h_out: (B, NT, H, V, K) — hidden state at each chunk boundary.
        v_new: (B, T, H, V) — residual-corrected values (pre-gating).
        final_state: (B, H, V, K) or None.
    """
    B, T, Hg, K = k.shape
    H, V = u.shape[2], u.shape[3]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    rep = H // Hg  # GQA replication factor (1 when Hg == H)

    h_out = torch.zeros(B, NT, H, V, K, dtype=torch.float32, device=k.device)
    v_new = torch.zeros_like(u)
    final_state = (
        torch.zeros(B, H, V, K, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    for b in range(B):
        for h_idx in range(H):
            hg_idx = h_idx // rep
            h_state = torch.zeros(V, K, dtype=torch.float32, device=k.device)
            if initial_state is not None:
                h_state = initial_state[b, h_idx].float().clone()

            for ci in range(NT):
                t0 = ci * BT
                t1 = min(t0 + BT, T)
                bt = t1 - t0

                h_out[b, ci, h_idx] = h_state

                k_c = k[b, t0:t1, hg_idx].float()  # (bt, K)
                w_c = w[b, t0:t1, h_idx].float()  # (bt, K)
                u_c = u[b, t0:t1, h_idx].float()  # (bt, V)

                v_c = u_c - w_c @ h_state.t()  # (bt, V)
                v_new[b, t0:t1, h_idx] = v_c.to(u.dtype)  # saved pre-gating

                last = bt - 1
                if g is not None:
                    g_c = g[b, t0:t1, h_idx].float()
                    g_last = g_c[last]
                    v_c = v_c * torch.exp(g_last - g_c).unsqueeze(-1)
                    h_state = h_state * torch.exp(g_last)
                if gk is not None:
                    gk_last = gk[b, t0 + last, h_idx].float()  # (K,)
                    h_state = h_state * torch.exp(gk_last).unsqueeze(0)

                h_state = h_state + v_c.t() @ k_c

            if output_final_state:
                final_state[b, h_idx] = h_state

    return h_out, v_new, final_state


def _make_inputs(B, T, H, K, V=None, dtype=torch.float32,
                 use_g=True, use_gk=False, use_initial_state=False):
    if V is None:
        V = K
    Hg = H  # no GQA in these tests
    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.1
    w = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.1
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.1
    g = torch.randn(B, T, H, device=DEVICE, dtype=dtype) * 0.1 if use_g else None
    gk = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.1 if use_gk else None
    h0 = (
        torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.1
        if use_initial_state
        else None
    )
    return k, w, u, g, gk, h0


# (B, T, H, K, V, use_g, use_gk, use_init) — T is always a multiple of 64.
CONFIGS = [
    (1, 64, 2, 64, 64, True, False, False),
    (1, 64, 4, 64, 64, True, False, False),
    (2, 128, 2, 64, 64, True, False, False),
    (1, 64, 2, 64, 32, True, False, False),  # V != K
    (1, 128, 2, 128, 64, True, False, False),  # K = 128 (two blocks)
    (1, 64, 2, 64, 64, False, True, False),  # gk only
    (1, 64, 2, 64, 64, True, False, True),  # with initial state
    (1, 64, 2, 64, 64, True, True, False),  # both g and gk
    (1, 64, 2, 64, 64, False, False, False),  # no gating
    (1, 192, 2, 64, 64, True, False, False),  # three chunks
]


@pytest.mark.parametrize(
    "B,T,H,K,V,use_g,use_gk,use_init",
    CONFIGS,
    ids=[
        f"B{b}_T{t}_H{h}_K{kk}_V{v}_g{int(ug)}_gk{int(ugk)}_h0{int(ui)}"
        for b, t, h, kk, v, ug, ugk, ui in CONFIGS
    ],
)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd(B, T, H, K, V, use_g, use_gk, use_init):
    """chunk_gated_delta_rule_fwd_h must match the naive reference (fp32)."""
    torch.manual_seed(0)
    k, w, u, g, gk, h0 = _make_inputs(
        B, T, H, K, V, torch.float32,
        use_g=use_g, use_gk=use_gk, use_initial_state=use_init,
    )

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, gk=gk, initial_state=h0,
        output_final_state=False, chunk_size=64, save_new_value=True,
    )
    h_ref, v_new_ref, _ = chunk_gated_delta_rule_fwd_ref(
        k, w, u, g=g, gk=gk, initial_state=h0, chunk_size=64,
    )

    torch.testing.assert_close(h.float(), h_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_new.float(), v_new_ref.float(), atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_chunk_gated_delta_rule_final_state():
    """output_final_state=True must produce the correct final hidden state."""
    torch.manual_seed(0)
    B, T, H, K, V = 1, 128, 2, 64, 64
    k, w, u, g, _, _ = _make_inputs(B, T, H, K, V, use_g=True)

    _, _, ht = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, output_final_state=True, chunk_size=64,
    )
    _, _, ht_ref = chunk_gated_delta_rule_fwd_ref(
        k, w, u, g=g, output_final_state=True, chunk_size=64,
    )

    assert ht is not None and ht_ref is not None
    torch.testing.assert_close(ht.float(), ht_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_low_precision(dtype):
    """bf16/fp16 with scalar gating must match the fp32 reference."""
    torch.manual_seed(0)
    B, T, H, K, V = 1, 64, 2, 64, 64
    k, w, u, g, _, _ = _make_inputs(B, T, H, K, V, dtype, use_g=True)

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, chunk_size=64, save_new_value=True,
    )
    h_ref, v_new_ref, _ = chunk_gated_delta_rule_fwd_ref(k, w, u, g=g, chunk_size=64)

    torch.testing.assert_close(h.float(), h_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_new.float(), v_new_ref.float(), atol=5e-2, rtol=5e-2)
