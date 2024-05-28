from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy


def seurat_v3(x: scipy.sparse.csr_matrix,
              gene_means_file: Path | str,
              gene_reg_stds_file: Path | str,
              probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per gene using seurat v3.

    Implements normalization as described in "Stuart, T. et al. Comprehensive Integration of Single-Cell Data. Cell 177,
    1888–1902.e21 (2019)". This function should be applied following normalization per cell by read depth or cell area.
    Subtraction of means and division by expected standard deviations derived from learned global mean-variance
    relationships.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26
    (29.04.2024).

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_means_file: Path | str
        Path to pickle file containing dictionary of mean gene expression values.
    gene_reg_stds_file: Path | str
        Path to pickle file containing dictionary of regularizing standard deviation values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionaries of gene means and reg stds
    with open(gene_means_file, "rb") as f:
        gene_means_dict = pickle.load(f)
    with open(gene_reg_stds_file, "rb") as f:
        gene_reg_stds_dict = pickle.load(f)

    # Retrieve gene means and reg stds for probed genes
    gene_means = np.array([gene_means_dict[gene_id] for gene_id in probed_genes])
    gene_reg_stds = np.array([gene_reg_stds_dict[gene_id] for gene_id in probed_genes])

    # Normalize counts
    y = x - gene_means / gene_reg_stds

    return y
