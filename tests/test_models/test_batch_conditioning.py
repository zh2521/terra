"""Tests for the batch-correction features:
   1. Adversarial batch classifier with gradient reversal.
   2. AdaLN-style per-batch conditioning of the transformer norms.

The key correctness invariant for both features is **backward
compatibility**: with the features disabled, the encoder/predictor
produce byte-identical outputs to the pre-feature implementation,
and with AdaLN enabled at construction time the zero-init of the
modulation hypernetwork makes the AdaLN-on output exactly equal to
the AdaLN-off output at step 0.
"""

import math

import pytest
import torch
import torch.nn as nn

from nichejepa.models.adaln import AdaLN
from nichejepa.models.batch_classifier import (
    BatchClassifierHead,
    GradReverseFn,
    GradReverseLayer,
    grad_reverse,
    mean_pool_cell_embedding,
)
from nichejepa.models.gene_transformers import GeneTransformerRankEncoder
from nichejepa.models.modules import Block


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

def test_grad_reverse_forward_is_identity():
    x = torch.randn(4, 8)
    y = GradReverseFn.apply(x, 1.0)
    torch.testing.assert_close(y, x)


def test_grad_reverse_backward_negates_scaled():
    x = torch.randn(4, 8, requires_grad=True)
    alpha = 2.5
    y = GradReverseFn.apply(x, alpha)
    # dL/dy = ones; gradient flowing back should be -alpha * ones.
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(
        x.grad, torch.full_like(x.grad, -alpha), atol=1e-6, rtol=0)


def test_grad_reverse_layer_module_form():
    layer = GradReverseLayer(alpha=0.3)
    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.full_like(x.grad, -0.3), atol=1e-6, rtol=0)


def test_grad_reverse_alpha_zero_blocks_grad():
    """Alpha=0 effectively detaches the classifier from the encoder
    via gradient reversal. Useful for a curriculum that delays the
    adversarial signal."""
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, alpha=0.0)
    y.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.zeros_like(x.grad), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Batch classifier head
# ---------------------------------------------------------------------------

def test_batch_classifier_head_shape():
    head = BatchClassifierHead(embed_dim=32, n_batches=10, hidden_dim=16)
    out = head(torch.randn(7, 32))
    assert out.shape == (7, 10)


def test_batch_classifier_n_batches_validated():
    with pytest.raises(ValueError, match="n_batches"):
        BatchClassifierHead(embed_dim=32, n_batches=1)


def test_mean_pool_cell_embedding_excludes_special_tokens():
    z = torch.zeros(2, 8, 4)
    # Set special-token positions to 100, non-special to 1. Mean
    # pool with n_special=2 should give all 1s.
    z[:, :2, :] = 100.0
    z[:, 2:, :] = 1.0
    pooled = mean_pool_cell_embedding(z, n_special_tokens=2)
    torch.testing.assert_close(pooled, torch.ones(2, 4))


# ---------------------------------------------------------------------------
# AdaLN module
# ---------------------------------------------------------------------------

def test_adaln_zero_init_matches_plain_layernorm():
    """The single most important invariant: at construction time
    (before any training), AdaLN(x, cond) must equal LayerNorm(x).
    This is what makes ``adaln.enabled = True`` a zero-cost
    addition that doesn't disturb the rest of the model.
    """
    torch.manual_seed(0)
    embed_dim, cond_dim = 16, 8
    adaln = AdaLN(embed_dim, cond_dim)
    plain_ln = nn.LayerNorm(embed_dim, eps=1e-6)
    # plain_ln is initialized with weight=1, bias=0 by default.
    x = torch.randn(3, 5, embed_dim)
    cond = torch.randn(3, cond_dim)  # any cond -> hypernet outputs zeros
    out_adaln = adaln(x, cond)
    out_plain = plain_ln(x)
    torch.testing.assert_close(
        out_adaln, out_plain, atol=1e-5, rtol=1e-5)


def test_adaln_invariant_under_cond_at_step_zero():
    """At init, the AdaLN output does NOT depend on cond (because
    gamma=1, beta=0 regardless of the cond value). Two different
    cond vectors must produce the same output for the same x."""
    embed_dim, cond_dim = 16, 8
    adaln = AdaLN(embed_dim, cond_dim)
    x = torch.randn(3, 5, embed_dim)
    cond_a = torch.randn(3, cond_dim)
    cond_b = torch.randn(3, cond_dim) * 10.0
    out_a = adaln(x, cond_a)
    out_b = adaln(x, cond_b)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_adaln_diverges_from_layernorm_after_training_step():
    """After a backward pass that updates the modulation MLP, the
    AdaLN output should DIVERGE from plain LayerNorm. Confirms the
    parameters are reachable by gradient descent (i.e. AdaLN
    actually does something after training)."""
    embed_dim, cond_dim = 16, 8
    adaln = AdaLN(embed_dim, cond_dim)
    plain_ln = nn.LayerNorm(embed_dim, eps=1e-6)
    x = torch.randn(3, 5, embed_dim)
    cond = torch.randn(3, cond_dim)
    # Take one fake gradient step on the modulation params.
    out = adaln(x, cond)
    loss = (out ** 2).sum()
    loss.backward()
    # Use a large LR so the divergence is unmistakable.
    with torch.no_grad():
        for p in adaln.modulation.parameters():
            p.copy_(p - 0.5 * p.grad)
        out_after = adaln(x, cond)
    out_plain = plain_ln(x)
    # After update, AdaLN should NOT match plain LN anymore.
    assert not torch.allclose(out_after, out_plain, atol=1e-3)


# ---------------------------------------------------------------------------
# Block integration with AdaLN
# ---------------------------------------------------------------------------

def test_block_cond_dim_none_means_plain_layernorm():
    """Default (cond_dim=None) Block must be exactly equivalent to
    the pre-AdaLN Block. Verified by checking that `uses_adaln` is
    False and that forward doesn't require a cond argument."""
    block = Block(
        dim=32, num_heads=4, use_flash_attention=False, cond_dim=None)
    assert block.uses_adaln is False
    # Plain forward works without cond.
    x = torch.randn(2, 6, 32)
    out = block(x)
    assert out.shape == (2, 6, 32)


def test_block_cond_dim_set_uses_adaln_and_requires_cond():
    block = Block(
        dim=32, num_heads=4, use_flash_attention=False, cond_dim=8)
    assert block.uses_adaln is True
    assert isinstance(block.norm1, AdaLN)
    assert isinstance(block.norm2, AdaLN)
    x = torch.randn(2, 6, 32)
    cond = torch.randn(2, 8)
    out = block(x, cond=cond)
    assert out.shape == (2, 6, 32)
    # Forgetting cond must raise loudly rather than silently failing.
    with pytest.raises(RuntimeError, match="AdaLN"):
        block(x)


def test_block_adaln_zero_init_matches_plain_at_step_zero():
    """Block(cond_dim=8) at step 0 produces the same output as
    Block(cond_dim=None) when all other weights are shared. This is
    the structural backward-compat guarantee for AdaLN.
    """
    torch.manual_seed(0)
    dim, num_heads = 32, 4
    # Build the AdaLN Block first; copy its non-norm weights into the
    # plain Block so the only difference is the norm.
    blk_adaln = Block(
        dim=dim, num_heads=num_heads,
        use_flash_attention=False, cond_dim=8)
    blk_plain = Block(
        dim=dim, num_heads=num_heads,
        use_flash_attention=False, cond_dim=None)
    # Copy state. The norm modules are different types; copy
    # everything else.
    sd_a = blk_adaln.state_dict()
    sd_p = blk_plain.state_dict()
    for k in sd_p:
        if k in sd_a and sd_a[k].shape == sd_p[k].shape:
            sd_p[k] = sd_a[k].clone()
    blk_plain.load_state_dict(sd_p)
    blk_adaln.eval()
    blk_plain.eval()
    x = torch.randn(2, 6, dim)
    cond = torch.randn(2, 8)
    out_a = blk_adaln(x, cond=cond)
    out_p = blk_plain(x)
    torch.testing.assert_close(out_a, out_p, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Encoder backward-compat
# ---------------------------------------------------------------------------

def _make_encoder(adaln_kwargs=None):
    return GeneTransformerRankEncoder(
        vocab_size=16, seq_len=12, n_special_tokens=2, n_segments=2,
        cell_pos_enc='segment', embed_dim=32, depth=2, num_heads=4,
        use_flash_attention=False,
        adaln_kwargs=adaln_kwargs,
    )


def test_encoder_default_no_adaln_attrs():
    """When adaln_kwargs is None, none of the AdaLN infrastructure
    is created. Backward-compat invariant for existing configs."""
    enc = _make_encoder(adaln_kwargs=None)
    assert enc.adaln_enabled is False
    assert enc.batch_emb_table is None
    for blk in enc.blocks:
        assert blk.uses_adaln is False


def test_encoder_with_adaln_creates_batch_emb_table():
    enc = _make_encoder(adaln_kwargs={
        'enabled': True,
        'n_batches': 5,
        'batch_embed_dim': 16,
    })
    assert enc.adaln_enabled is True
    assert isinstance(enc.batch_emb_table, nn.Embedding)
    assert enc.batch_emb_table.num_embeddings == 5
    assert enc.batch_emb_table.embedding_dim == 16
    for blk in enc.blocks:
        assert blk.uses_adaln is True


def test_encoder_disabled_adaln_block_doesnt_get_cond_dim():
    enc = _make_encoder(adaln_kwargs={'enabled': False, 'n_batches': 5})
    assert enc.adaln_enabled is False
    for blk in enc.blocks:
        assert blk.uses_adaln is False


def test_encoder_compute_cond_returns_none_when_disabled():
    enc = _make_encoder(adaln_kwargs=None)
    batch = {'values': torch.zeros(2, 12)}
    assert enc._compute_cond(batch) is None


def test_encoder_compute_cond_shape_when_enabled():
    enc = _make_encoder(adaln_kwargs={
        'enabled': True, 'n_batches': 5, 'batch_embed_dim': 16,
    })
    batch = {'values': torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
    cond = enc._compute_cond(batch)
    assert cond is not None
    assert cond.shape == (2, 16)


def test_encoder_compute_cond_clamps_out_of_range_labels():
    """A batch label outside [0, n_batches) must be clamped to
    n_batches-1 rather than raising or OOB-indexing the table."""
    enc = _make_encoder(adaln_kwargs={
        'enabled': True, 'n_batches': 3, 'batch_embed_dim': 16,
    })
    # Label 99 should be clamped to 2.
    batch = {'values': torch.tensor([[99.0] + [0]*11])}
    cond = enc._compute_cond(batch)
    expected = enc.batch_emb_table(torch.tensor([2]))
    torch.testing.assert_close(cond, expected, atol=0, rtol=0)
