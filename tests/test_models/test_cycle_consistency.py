"""Tests for batch-swap cycle-consistency loss."""

import pytest
import torch

from nichejepa.models.cycle_consistency import (
    cycle_consistency_loss,
    make_swapped_batch,
)


# ---------------------------------------------------------------------------
# make_swapped_batch
# ---------------------------------------------------------------------------

def test_make_swapped_batch_replaces_first_column():
    torch.manual_seed(0)
    values = torch.zeros(8, 4)
    values[:, 0] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    batch = {'values': values}
    swapped, changed = make_swapped_batch(batch, n_batches=8)

    # Other columns must be untouched.
    assert torch.allclose(swapped['values'][:, 1:], values[:, 1:])
    # First column must be in [0, 8).
    new_labels = swapped['values'][:, 0].long()
    assert (new_labels >= 0).all()
    assert (new_labels < 8).all()
    # The original batch must not have been mutated.
    assert torch.equal(batch['values'][:, 0],
                       torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).float())
    # changed_mask shape sanity.
    assert changed.shape == (8,)
    assert changed.dtype == torch.bool


def test_make_swapped_batch_changed_mask_matches_diff():
    torch.manual_seed(123)
    values = torch.zeros(50, 2)
    values[:, 0] = torch.randint(0, 5, (50,))
    batch = {'values': values}
    swapped, changed = make_swapped_batch(batch, n_batches=5)
    expected = swapped['values'][:, 0].long() != values[:, 0].long()
    assert torch.equal(changed, expected)


def test_make_swapped_batch_respects_n_batches():
    torch.manual_seed(0)
    values = torch.zeros(100, 2)
    batch = {'values': values}
    swapped, _ = make_swapped_batch(batch, n_batches=3)
    new_labels = swapped['values'][:, 0].long()
    assert int(new_labels.max().item()) < 3
    assert int(new_labels.min().item()) >= 0


def test_make_swapped_batch_raises_on_1d_values():
    with pytest.raises(RuntimeError, match="at least 2-D"):
        make_swapped_batch({'values': torch.zeros(8)}, n_batches=4)


# ---------------------------------------------------------------------------
# cycle_consistency_loss
# ---------------------------------------------------------------------------

def test_cycle_loss_zero_when_outputs_identical():
    z = torch.randn(4, 10, 8)
    out = cycle_consistency_loss(z, z.clone())
    assert float(out.item()) == 0.0


def test_cycle_loss_positive_when_outputs_differ():
    torch.manual_seed(0)
    z_a = torch.randn(4, 10, 8)
    z_b = z_a + 1.0
    out = cycle_consistency_loss(z_a, z_b)
    assert float(out.item()) > 0.0


def test_cycle_loss_changed_mask_filters_unchanged():
    """Cells outside changed_mask must not contribute."""
    torch.manual_seed(0)
    z_a = torch.randn(4, 10, 8)
    z_b = z_a.clone()
    z_b[2] += 1.0  # cell 2 differs
    # Only cell 2 is "changed"
    mask = torch.tensor([False, False, True, False])
    out = cycle_consistency_loss(z_a, z_b, changed_mask=mask)
    # Should reduce to mean over cell 2 only -> 1.0 (sq diff is 1.0)
    assert pytest.approx(float(out.item()), rel=1e-5) == 1.0


def test_cycle_loss_changed_mask_all_false_returns_zero():
    z_a = torch.randn(4, 10, 8)
    z_b = torch.randn(4, 10, 8)
    mask = torch.zeros(4, dtype=torch.bool)
    out = cycle_consistency_loss(z_a, z_b, changed_mask=mask)
    assert float(out.item()) == 0.0


def test_cycle_loss_accepts_list_of_tensors():
    """JEPA multi-mask layout: outputs are a list of (B, L, D)."""
    z_a = [torch.randn(4, 10, 8), torch.randn(4, 12, 8)]
    z_b = [z_a[0].clone(), z_a[1].clone()]
    out = cycle_consistency_loss(z_a, z_b)
    assert float(out.item()) == 0.0


def test_cycle_loss_gradient_flows():
    torch.manual_seed(0)
    z_a = torch.randn(4, 10, 8, requires_grad=True)
    z_b = torch.randn(4, 10, 8)
    out = cycle_consistency_loss(z_a, z_b)
    out.backward()
    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()
