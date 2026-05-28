"""
Distribution-alignment losses for batch effect removal without an
adversarial arms race.

Two methods are exposed via a shared API:

- **CORAL** (Sun & Saenko 2016): Frobenius-distance between the sample
  covariance matrices of two batches. Cheap (O(D^2) per pair), no
  kernel choice.
- **MMD** (Gretton et al. 2007, with multi-bandwidth RBF kernel as in
  Long et al. 2015): Maximum Mean Discrepancy between two batches'
  embedding distributions. Slightly more expressive but O(N*M) per
  pair.

Both are *non-adversarial*: there's no classifier to defeat, no
game-theoretic instability. They sit alongside the JEPA loss and
pull batch-conditional embedding distributions toward each other.
Combined with VICReg's variance loss (to prevent the trivial
all-batches-collapse-to-the-same-point solution), they provide a
stable batch-correction signal.

Used by ``train.py``: per minibatch, sample distinct batches present
in the data, compute the alignment loss between each pair, average,
and add to the total loss with a config-controlled weight.
"""

from __future__ import annotations

from typing import Sequence

import torch


def coral_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Frobenius distance between unbiased sample covariances of two
    embedding sets, normalized by ``4 * D^2`` so the magnitude is
    insensitive to embedding dimension.

    z_a:  (n_a, D)
    z_b:  (n_b, D)
    """
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        return torch.zeros((), device=z_a.device, dtype=z_a.dtype)
    d = z_a.size(-1)
    z_a_c = z_a - z_a.mean(dim=0, keepdim=True)
    z_b_c = z_b - z_b.mean(dim=0, keepdim=True)
    cov_a = (z_a_c.T @ z_a_c) / (z_a.size(0) - 1)
    cov_b = (z_b_c.T @ z_b_c) / (z_b.size(0) - 1)
    return ((cov_a - cov_b) ** 2).sum() / (4.0 * d * d)


def _rbf_kernel_matrix(x: torch.Tensor,
                       y: torch.Tensor,
                       sigmas: Sequence[float],
                       ) -> torch.Tensor:
    """Multi-bandwidth RBF Gram matrix:
        K[i, j] = (1 / |sigmas|) * sum_s exp(-||x_i - y_j||^2 / (2 sigma_s^2))
    """
    sq = torch.cdist(x, y, p=2).pow(2)  # (n, m), float
    out = torch.zeros_like(sq)
    for sigma in sigmas:
        out = out + torch.exp(-sq / (2.0 * float(sigma) ** 2))
    return out / len(sigmas)


def mmd_loss(z_a: torch.Tensor,
             z_b: torch.Tensor,
             sigmas: Sequence[float] = (0.1, 1.0, 10.0),
             ) -> torch.Tensor:
    """Maximum Mean Discrepancy with a multi-bandwidth RBF kernel.

    Returns the V-statistic estimator:
        MMD^2 = E[K(x, x')] + E[K(y, y')] - 2 E[K(x, y)]

    z_a:  (n_a, D)
    z_b:  (n_b, D)
    """
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        return torch.zeros((), device=z_a.device, dtype=z_a.dtype)
    k_aa = _rbf_kernel_matrix(z_a, z_a, sigmas).mean()
    k_bb = _rbf_kernel_matrix(z_b, z_b, sigmas).mean()
    k_ab = _rbf_kernel_matrix(z_a, z_b, sigmas).mean()
    return k_aa + k_bb - 2.0 * k_ab


def compute_distribution_alignment_loss(
        cell_emb: torch.Tensor,
        batch_label: torch.Tensor,
        method: str = "coral",
        mmd_sigmas: Sequence[float] | None = None,
        max_pairs: int | None = None,
        ) -> tuple[torch.Tensor, dict]:
    """Average the pairwise alignment loss across all (or a sample of)
    batch pairs present in the minibatch.

    cell_emb:    (N, D) per-cell embedding (e.g. ``mean_pool_cell_embedding``).
    batch_label: (N,)   long, batch id per cell.
    method:      ``'coral'`` or ``'mmd'``.
    mmd_sigmas:  RBF bandwidths if ``method == 'mmd'``. Default
                 ``(0.1, 1.0, 10.0)``.
    max_pairs:   If set, randomly sample at most this many batch pairs
                 instead of all C(n_batches, 2) pairs. Useful when the
                 minibatch contains many distinct batches.

    Returns ``(loss, info)`` where ``info`` is a small dict for
    diagnostic logging. ``loss`` is a 0-D tensor on the same
    device / dtype as ``cell_emb``.
    """
    device, dtype = cell_emb.device, cell_emb.dtype
    unique_batches = torch.unique(batch_label)
    n_batches = int(unique_batches.numel())
    if n_batches < 2:
        return (
            torch.zeros((), device=device, dtype=dtype),
            {"n_batches_in_minibatch": n_batches, "n_pairs": 0},
        )

    pairs = [(i, j) for i in range(n_batches) for j in range(i + 1, n_batches)]
    if max_pairs is not None and len(pairs) > max_pairs:
        # Deterministic-ish sampling per call so the loss is smooth
        # across iterations even with stochastic pair sampling.
        idx = torch.randperm(len(pairs), device="cpu")[:max_pairs].tolist()
        pairs = [pairs[k] for k in idx]

    if method == "coral":
        pair_loss_fn = coral_loss
    elif method == "mmd":
        sigmas = tuple(mmd_sigmas) if mmd_sigmas else (0.1, 1.0, 10.0)

        def pair_loss_fn(a, b):
            return mmd_loss(a, b, sigmas=sigmas)
    else:
        raise ValueError(
            f"Unknown distribution-alignment method: {method!r}. "
            "Expected 'coral' or 'mmd'.")

    losses = []
    for i, j in pairs:
        mask_a = batch_label == unique_batches[i]
        mask_b = batch_label == unique_batches[j]
        z_a = cell_emb[mask_a]
        z_b = cell_emb[mask_b]
        losses.append(pair_loss_fn(z_a, z_b))

    if not losses:
        return (
            torch.zeros((), device=device, dtype=dtype),
            {"n_batches_in_minibatch": n_batches, "n_pairs": 0},
        )

    loss = torch.stack(losses).mean()
    info = {
        "n_batches_in_minibatch": n_batches,
        "n_pairs": len(losses),
    }
    return loss, info
