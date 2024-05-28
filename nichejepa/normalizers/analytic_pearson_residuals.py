from __future__ import annotations

import numpy as np
import scipy


def analytic_pearson_residuals(x: scipy.sparse.csr_matrix, theta: float = 100) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per gene and cell using analytic pearson residuals.

    Implements normalization as described in "Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for
    normalization of single-cell RNA-seq UMI data. Genome Biol. 22, 258 (2021)". Residuals are based on a negative
    binomial offset model with overdispersion shared across genes. Residuals are clipped to 'sqrt(n_obs)'. Negative
    residuals for a cell and gene indicate that fewer counts are observed than expected, compared to the gene’s average
    expression and cell read depth. Positive residuals indicate more counts than expected.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/experimental/pp/_normalization.py#L36
    (03.05.2024).

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    theta: float
        The overdispersion parameter.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    sum_counts_cells = np.sum(x, axis=1).reshape(-1, 1)
    sum_counts_genes = np.sum(x, axis=0).reshape(1, -1)
    sum_counts_total = np.sum(sum_counts_genes)
    mu_counts = np.array(sum_counts_cells @ sum_counts_genes / sum_counts_total)
    diff_counts = np.array(x - mu_counts)
    residuals_counts = diff_counts / np.sqrt(mu_counts + mu_counts ** 2 / theta)
    y = np.clip(residuals_counts, a_min=-np.sqrt(x.shape[0]), a_max=np.sqrt(x.shape[0]))

    return y
