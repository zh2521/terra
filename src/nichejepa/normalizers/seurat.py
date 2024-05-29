from __future__ import annotations

import pickle
from pathlib import Path
from skmisc.loess import loess

import numpy as np
import scipy


def seurat_v3(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per gene using seurat v3 `FindVariableFeatures`.

    Implements normalization as described in "Stuart, T. et al. Comprehensive Integration of Single-Cell Data. Cell 177,
    1888–1902.e21 (2019)". This function implicitly applies normalization per cell by read depth, prior to fitting a
    model to counts for each gene. Counts are normalized by centering around the expected mean and scaling by the
    expected standard deviation, as learned from the global mean-variance relationships. This normalisation should be applied
    independently for each dataset in the training corpus. Note, we do not implement clipping as described in the seurat
    publication.

    " Feature selection for individual datasets In each dataset, we next aimed to identify a subset of features
    (i.e. genes) exhibiting high variability across cells, and therefore represent heterogeneous features to prioritize
    for downstream analysis. Choosing genes solely based on their log-normalized single-cell variance fails to account
    for the mean-variance relationship that is inherent to single-cell RNA-seq. Therefore, we first applied a
    variance-stabilizing transformation to correct for this, as first outlined by Mayer, Hafemeister & Bandler et al.
    [Mayer et al., 2018, Hafemeister and Satija, 2019].

    " To learn the mean-variance relationship from the data, we computed the mean and variance of each gene using the
    unnormalized data (i.e. UMI or counts matrix), and applied log10-transformation to both. We then fit a curve to
    predict the variance of each gene as a function of its mean, by calculating a local fitting of polynomials of degree
    2 (R function loess, span = 0.3). This global fit provided us with a regularized estimator of variance given the
    mean of a feature. As such, we could use it to standardize feature counts without removing higher-than-expected
    variation.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature,
        containing raw counts as features (i.e. not scaled or normalized).

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    gene_means = np.mean(x.toarray(), axis=0)
    gene_vars = np.var(x.toarray(), axis=0)

    not_const = gene_vars > 0

    gene_log_means = np.log10(gene_means[not_const])
    gene_log_variances = np.log10(gene_vars[not_const])

    model = loess(gene_log_means, gene_log_variances, options={"span": 0.3, "degree": 2})
    model.fit()

    expected_log_variances = np.zeros(x.shape[1], dtype=np.float64)
    expected_log_variances[not_const] = model.outputs.fitted_values
    expected_variances = 10 ** expected_log_variances
    expected_standard_deviation = np.sqrt(expected_variances)

    y = (x - gene_means) / expected_standard_deviation

    return y
