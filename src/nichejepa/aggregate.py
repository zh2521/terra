import anndata
import squidpy as sq


def aggregate_by_radius(adata: anndata.AnnData, radius: float = 27.5) -> anndata.AnnData:
    """
    Aggregate neighbourhood gene expression by radius

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with spatial coordinates available in `adata.obsm["spatial"]`.
    radius : float
        Radius within which neighbouring cells will be aggregated, in um. Defaults to 27.5 um, which corresponds
        to the 10x Visium spot size of 55 um.

    Returns
    ----------
    adata : anndata.AnnData
        AnnData object with aggregated counts available in `adata.layers["X_neighborhood"]`.
    """

    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key="spatial",
                            radius=radius,
                            set_diag=True
                            )

    adata.layers["X_neighborhood"] = adata.obsp["spatial_connectivities"].T @ adata.X

    return adata
