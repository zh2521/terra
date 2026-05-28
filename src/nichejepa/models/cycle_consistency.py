"""
Batch-swap cycle-consistency loss for batch-invariant encoder
representations.

The idea (FADVI-style, Chen et al. 2024): if the encoder has truly
learned batch-invariant features, then swapping a cell's batch label
to a random other batch should leave the encoded representation
roughly unchanged. We enforce this by re-encoding the same cells
with random batch labels and minimizing the MSE between the original
and re-encoded outputs.

This is a strictly auxiliary loss applied after the main encoder
forward; it does NOT replace the JEPA objective. The double-forward
cost is gated by ``swap_fraction``: with probability ``swap_fraction``
per step, the swapped forward pass runs and contributes to the loss.
Otherwise the step is identical to a normal training step.

Implementation notes
--------------------
- The batch label lives in ``batch['values'][:, 0]`` by the
  tokenizer convention. Swapping it changes both:
  (a) the spv_batch special-value embedding at position 0, and
  (b) the AdaLN conditioning (if AdaLN is enabled), since AdaLN
  reads its cond from the same slot.
  Both are desired -- we want to test "what would the encoder
  output if you told it this cell was from a different batch."
- We swap *uniformly at random* over ``[0, n_batches)`` so the swap
  is symmetric and unbiased. Cells whose original label happens to
  equal the swap target are left as-is for that step (no MSE
  contribution, since the forward would be identical).
"""

from __future__ import annotations

import torch


def make_swapped_batch(
        batch: dict,
        n_batches: int,
        generator: torch.Generator | None = None,
        ) -> tuple[dict, torch.Tensor]:
    """Return a shallow-copied batch dict with ``values[:, 0]``
    replaced by random batch labels in ``[0, n_batches)``.

    Args
    ----
    batch:
        Original batch dict. Not mutated.
    n_batches:
        Upper bound (exclusive) on the swapped labels. Must match
        what AdaLN / batch-token embedding expect; values outside
        this range would otherwise crash the AdaLN range check.
    generator:
        Optional torch.Generator for reproducibility.

    Returns
    -------
    swapped_batch:
        Shallow copy of ``batch`` whose ``'values'`` tensor has been
        cloned and the first column replaced.
    changed_mask:
        Bool tensor ``(B,)``. ``True`` for cells whose label
        actually changed (i.e. the random draw differed from the
        original). The loss should be averaged only over these
        cells so identical-label cycles don't dilute the signal.
    """
    values = batch['values']
    if values.dim() < 2:
        raise RuntimeError(
            "make_swapped_batch expected `values` to be at least 2-D "
            f"(B, L); got shape {tuple(values.shape)}.")
    B = values.size(0)
    device = values.device
    original = values[:, 0].long()
    swap = torch.randint(
        low=0, high=int(n_batches), size=(B,),
        device=device, dtype=torch.long, generator=generator)
    changed_mask = swap != original

    new_values = values.clone()
    new_values[:, 0] = swap.to(values.dtype)
    swapped_batch = dict(batch)
    swapped_batch['values'] = new_values
    return swapped_batch, changed_mask


def cycle_consistency_loss(
        z_original: list[torch.Tensor] | torch.Tensor,
        z_swapped: list[torch.Tensor] | torch.Tensor,
        changed_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
    """MSE between original and batch-swapped encoder outputs.

    Accepts either a single tensor or a list of tensors (one per
    context mask, matching the JEPA multi-mask layout used by
    ``EncoderMultiMaskWrapper``).

    If ``changed_mask`` is provided, only cells whose label actually
    changed contribute to the loss (returns zero when none changed).
    """
    if isinstance(z_original, list):
        return torch.stack([
            cycle_consistency_loss(zo, zs, changed_mask)
            for zo, zs in zip(z_original, z_swapped)
        ]).mean()

    diff_sq = (z_original - z_swapped) ** 2          # (B, L, D)
    if changed_mask is not None:
        if changed_mask.sum() == 0:
            return torch.zeros((), device=diff_sq.device, dtype=diff_sq.dtype)
        # Reduce over (L, D) per cell, then mean over changed cells.
        per_cell = diff_sq.mean(dim=tuple(range(1, diff_sq.dim())))
        per_cell = per_cell[changed_mask]
        return per_cell.mean()
    return diff_sq.mean()
