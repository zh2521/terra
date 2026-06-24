import numpy as np
import anndata as ad

from terra.preprocessors import filter_cells


def test_filter_poor_quality_cells():
    """Check filter_poor_quality_cells works with a simple example"""

    x = np.array([
        [0, 3, 0, 4, 0, 0],
        [0, 7, 9, 8, 3, 2],
        [4, 0, 4, 6, 8, 0],
        [0, 0, 3, 0, 6, 4],
        [8, 0, 0, 4, 6, 0],
    ])

    filter_pass = [1, 1, 0, 1, 0]

    expected_x = np.array([
        [0, 3, 0, 4, 0, 0],
        [0, 7, 9, 8, 3, 2],
        [0, 0, 3, 0, 6, 4],
    ])

    adata = ad.AnnData(x)
    adata.obs["filter_pass"] = filter_pass

    adata_filtered = filter_cells(adata)
    X = adata_filtered.X
    filtered_x = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    np.testing.assert_array_equal(filtered_x, expected_x)
