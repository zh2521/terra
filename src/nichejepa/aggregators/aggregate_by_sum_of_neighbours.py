import anndata
import squidpy as sq
import scipy
import numpy as np
import anndata as ad


def aggregate_by_sum_of_neighbours(
    x: scipy.sparse.csr_matrix,
    coordinates: np.ndarray,
    radius: float,
) -> scipy.sparse.csr_matrix:
    """
    Aggregate neighborhood gene expression by radius.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        Features for each cell.
    coordinates: np.ndarray
        An array of lists, arrays or tuples containing the x and y coordinates of each cell in um.
    radius: float
        Radius within which neighboring cells will be aggregated, in um. Use 27.5 um for a
        radius equivalent to the 10x Visium spot size.

    Returns
    ----------
    scipy.sparse.csr_matrix
        A feature matrix with aggregated counts.
    """

    if x.shape[0] != coordinates.shape[0]:
        raise ValueError("x and coordinates should be the same length")

    adata = ad.AnnData(x.toarray())
    adata.obsm["spatial"] = coordinates

    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key="spatial",
                            radius=radius,
                            set_diag=True)

    y = adata.obsp["spatial_connectivities"].T @ adata.X
    y = scipy.sparse.csr_matrix(y)

    return y
