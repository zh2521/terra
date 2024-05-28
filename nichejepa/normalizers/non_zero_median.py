from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy


def non_zero_median(x: scipy.sparse.csr_matrix,
                    gene_nzmedians_file: Path | str,
                    probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize by non-zero median expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_nzmedians_file: Path | str
        Path to pickle file containing dictionary of non-zero median gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene non-zero medians
    with open(gene_nzmedians_file, "rb") as f:
        gene_nzmedians_dict = pickle.load(f)

    # Retrieve gene non-zero medians
    gene_nzmedians = np.array([gene_nzmedians_dict[gene_id] for gene_id in probed_genes])

    # Normalize counts
    y = x / gene_nzmedians

    return y
