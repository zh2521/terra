"""Tests for ``terra.models.modules.ClassificationModel``.

``ClassificationModel`` is the supervised head used by the finetuning
pipeline (``terra.training.finetune``): it wraps a pretrained base model,
mean-pools the per-token embeddings over a selection mask, layer-norms the
pooled embedding, and projects it to class logits.

These tests exercise the head in isolation with a tiny *stub* base model
(so they need only ``torch`` — no pretrained weights, no GPU). They guard
the integration seam between ``finetune.py`` and the head:

* ``embed_dim`` is taken from ``base_model.backbone.embed_dim``;
* ``forward(udata, masks_attention, selection_mask)`` returns
  ``(batch_size, num_classes)`` for both the linear and MLP heads;
* gradients flow back through both the head and the base model.
"""

import torch
import torch.nn as nn

from terra.models.modules import ClassificationModel


class _StubBackbone:
    """Minimal stand-in exposing the single attribute the head reads."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim


class _StubBaseModel(nn.Module):
    """Stub pretrained model.

    Mirrors the call contract ``ClassificationModel.forward`` relies on:
    it is called as ``base_model(batch=udata, masks_attention=...)`` and
    returns a ``(token embeddings, *)`` tuple of shape ``(B, S, E)``.
    A real ``nn.Linear`` is included so gradients have somewhere to flow.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.backbone = _StubBackbone(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, batch, masks_attention=None):
        # `udata` carries the per-token embeddings under "emb".
        return self.proj(batch["emb"]), None


def _make_inputs(batch_size=4, seq_len=6, embed_dim=8):
    udata = {"emb": torch.randn(batch_size, seq_len, embed_dim)}
    masks_attention = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Selection mask: at least one selected position per row (the head
    # divides by the per-row count, so an all-False row would be degenerate).
    selection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    selection_mask[:, 0] = True
    selection_mask[0, 1] = True
    return udata, masks_attention, selection_mask


def test_embed_dim_is_taken_from_backbone():
    base = _StubBaseModel(embed_dim=8)
    model = ClassificationModel(base_model=base, num_classes=3)
    assert model.embed_dim == 8


def test_linear_head_output_shape():
    base = _StubBaseModel(embed_dim=8)
    model = ClassificationModel(base_model=base, num_classes=5, use_mlp=False)
    udata, masks_attention, selection_mask = _make_inputs(batch_size=4, embed_dim=8)

    logits = model(udata, masks_attention=masks_attention,
                   selection_mask=selection_mask)

    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_mlp_head_output_shape():
    base = _StubBaseModel(embed_dim=8)
    model = ClassificationModel(base_model=base, num_classes=2,
                                use_mlp=True, hidden_dim=16)
    udata, masks_attention, selection_mask = _make_inputs(batch_size=3, embed_dim=8)

    logits = model(udata, masks_attention=masks_attention,
                   selection_mask=selection_mask)

    assert logits.shape == (3, 2)
    # The MLP head must actually introduce the hidden layer.
    assert isinstance(model.classification_head, nn.Sequential)
    assert model.classification_head[0].out_features == 16


def test_gradients_flow_to_base_model_and_head():
    base = _StubBaseModel(embed_dim=8)
    model = ClassificationModel(base_model=base, num_classes=4)
    udata, masks_attention, selection_mask = _make_inputs(batch_size=4, embed_dim=8)

    logits = model(udata, masks_attention=masks_attention,
                   selection_mask=selection_mask)
    logits.sum().backward()

    # Head receives gradient.
    assert model.classification_head.weight.grad is not None
    assert model.classification_head.weight.grad.abs().sum() > 0
    # Gradient propagates through the (unfrozen) base model too.
    assert model.base_model.proj.weight.grad is not None
    assert model.base_model.proj.weight.grad.abs().sum() > 0
