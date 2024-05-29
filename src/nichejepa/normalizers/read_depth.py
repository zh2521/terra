from __future__ import annotations

import scipy


def read_depth(x: scipy.sparse.csr_matrix, target_size: int = 10_000) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per cell by read depth.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    target_size: int
        The target read depth per observation (i.e. the sum of features across an observation).

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y
