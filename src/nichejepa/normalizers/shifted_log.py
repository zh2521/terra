from __future__ import annotations

import scanpy as sc
import scipy


def shifted_log(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Normalize by shifted log.

    Implements normalization using `sc.pp.log1p`. This function should be applied following normalization per cell by
    cell area or read depth.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = sc.pp.log1p(x)

    return y
