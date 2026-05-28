"""Tests for Protocol-MoE predictor bias head."""

import pytest
import torch

from nichejepa.models.protocol_moe import (
    ProtocolBias,
    extract_protocol_index,
)


# ---------------------------------------------------------------------------
# ProtocolBias
# ---------------------------------------------------------------------------

def test_protocol_bias_zero_init_returns_zero():
    """At construction, the per-protocol bias is exactly zero so
    Protocol-MoE-at-step-0 is a no-op."""
    torch.manual_seed(0)
    pb = ProtocolBias(n_experts=5, embed_dim=8)
    idx = torch.tensor([0, 1, 2, 3, 4])
    out = pb(idx)
    assert out.shape == (5, 8)
    assert torch.allclose(out, torch.zeros_like(out))


def test_protocol_bias_diverges_after_training():
    torch.manual_seed(0)
    pb = ProtocolBias(n_experts=3, embed_dim=4)
    idx = torch.tensor([0, 1, 2])
    target = torch.randn(3, 4)
    optim = torch.optim.SGD(pb.parameters(), lr=1e-1)
    for _ in range(5):
        optim.zero_grad()
        out = pb(idx)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optim.step()
    out_after = pb(idx)
    assert not torch.allclose(out_after, torch.zeros_like(out_after))


def test_protocol_bias_independent_per_expert():
    """Different protocol IDs return their own row of the
    embedding table."""
    torch.manual_seed(0)
    pb = ProtocolBias(n_experts=4, embed_dim=6)
    with torch.no_grad():
        pb.bias.weight[0].fill_(1.0)
        pb.bias.weight[1].fill_(2.0)
    out = pb(torch.tensor([0, 1, 0, 1]))
    assert torch.allclose(out[0], torch.full((6,), 1.0))
    assert torch.allclose(out[1], torch.full((6,), 2.0))
    assert torch.allclose(out[2], torch.full((6,), 1.0))


def test_protocol_bias_no_weight_decay_flag():
    """The embedding parameters must be marked so the optimizer's
    weight-decay group can skip them."""
    pb = ProtocolBias(n_experts=3, embed_dim=4)
    for p in pb.bias.parameters():
        assert getattr(p, "_no_weight_decay", False) is True


def test_protocol_bias_oob_raises():
    pb = ProtocolBias(n_experts=4, embed_dim=4)
    with pytest.raises(RuntimeError, match="out of range"):
        pb(torch.tensor([0, 1, 4]))  # 4 is out of range
    with pytest.raises(RuntimeError, match="out of range"):
        pb(torch.tensor([-1, 0]))


def test_protocol_bias_requires_1d_indices():
    pb = ProtocolBias(n_experts=3, embed_dim=4)
    with pytest.raises(RuntimeError, match=r"\(B,\)"):
        pb(torch.tensor([[0, 1], [1, 0]]))


# ---------------------------------------------------------------------------
# extract_protocol_index
# ---------------------------------------------------------------------------

def test_extract_protocol_index_pulls_correct_column():
    values = torch.tensor([
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
    ])
    out = extract_protocol_index({'values': values}, routing_index=2)
    assert out.tolist() == [30, 31]
    assert out.dtype == torch.long


def test_extract_protocol_index_raises_on_oob():
    values = torch.zeros(2, 3)
    with pytest.raises(RuntimeError, match="out of bounds"):
        extract_protocol_index({'values': values}, routing_index=5)


def test_extract_protocol_index_raises_on_1d():
    with pytest.raises(RuntimeError, match="at least 2-D"):
        extract_protocol_index({'values': torch.zeros(4)}, routing_index=0)
