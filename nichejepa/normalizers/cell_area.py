from __future__ import annotations

import numpy as np
import scipy


def cell_area(x: scipy.sparse.csr_matrix,
              cell_areas: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per cell by cell area.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    cell_areas: np.ndarray
        Numpy array with the cell areas.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = x / cell_areas.values.reshape(-1, 1) * np.mean(cell_areas)

    return y
