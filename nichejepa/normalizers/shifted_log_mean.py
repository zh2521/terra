from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy


def shifted_log_mean(x: scipy.sparse.csr_matrix,
                     gene_logmeans_file: Path | str,
                     probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize by shifted log and log mean gene expression.

    Implements normalization using `sc.pp.log1p`. This function should be applied following normalization per cell by
    cell area or read depth.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_logmeans_file: Path | str
        Path to pickle file containing dictionary of log mean gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes that will be normalized.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = sc.pp.log1p(x)

    # Load dictionary of gene logmeans
    with open(gene_logmeans_file, "rb") as f:
        gene_logmeans_dict = pickle.load(f)

    gene_logmeans = np.array([gene_logmeans_dict[gene_id] for gene_id in probed_genes])

    y = y / gene_logmeans

    return y
