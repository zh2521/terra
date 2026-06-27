"""Opt-in integration smoke test for TERRA downstream-analysis functions.

This complements ``test_pipeline_smoke`` (which covers the embedding pipeline)
by exercising the analyses built on top of the embeddings against the *real*
published Hugging Face model: average and per-cell gene embeddings, the spatial
gene-pair score (and its top-gene/top-pair rankings), the EMD spatial-structure
score, and in-silico perturbation (including the ``perturbed_cell_id="all"`` /
``"all"``-gene fast path).

These functions previously had no coverage, which let three regressions ship
undetected (``get_spatial_score``/``get_emd_distance`` missing a helper import,
and the all-cells perturbation fast path referencing an unbound variable). This
test guards that whole surface.

Like ``test_pipeline_smoke`` it is **opt-in** and self-skips when the inference
stack, the Hub dependency, or network access is missing.

How to run it
-------------
1. Install the inference stack plus the Hub extra::

       pip install -e '.[hub]'

2. Enable and run the opt-in test::

       TERRA_MODEL_SMOKE=1 pytest tests/test_inference/test_downstream_smoke.py -v

The TERRA model repos are public, so no Hugging Face authentication is required.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# --- Opt-in / dependency gating -------------------------------------------
torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")
pytest.importorskip("huggingface_hub")
pytest.importorskip("squidpy")
pytest.importorskip("scanpy")
pytest.importorskip("pyensembl")
pytest.importorskip("datasets")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TERRA_MODEL_SMOKE"),
    reason="Set TERRA_MODEL_SMOKE=1 to run the published-model downstream "
    "smoke test.",
)

# Imported only after the importorskip gate above.
import pandas as pd  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from terra import (  # noqa: E402
    download_pretrained,
    embed_dataset,
    get_average_gene_embed,
    get_emd_distance,
    get_gene_embed,
    get_spatial_score,
    harmonize_tokenize_embed_pipeline,
    perturb_dataset,
)
from terra.utils.evaluation import (  # noqa: E402
    get_top_gene_pairs,
    get_top_gene_score,
)

# One repo is enough for a regression smoke test (test_pipeline_smoke covers
# both); use the model the tutorials use.
REPO_ID = "Lotfollahi-lab/TERRA-112M"

# ~40 human gene SYMBOLS (housekeeping / lineage markers) -- broad enough that
# plenty survive symbol->Ensembl mapping and the pretraining-occurrence filter.
GENE_SYMBOLS = [
    "ACTB", "GAPDH", "B2M", "TUBB", "VIM", "PTPRC", "CD3D", "CD3E", "CD8A",
    "CD4", "CD19", "MS4A1", "CD14", "FCGR3A", "NKG7", "GNLY", "EPCAM", "KRT8",
    "KRT18", "KRT19", "COL1A1", "COL1A2", "COL3A1", "PECAM1", "VWF", "CDH5",
    "ACTA2", "PDGFRB", "MKI67", "PCNA", "CD68", "CD163", "LYZ", "HLA-DRA",
    "IL7R", "FOXP3", "GZMB", "CD79A", "ITGAM", "ENG", "THY1", "NCAM1",
]

N_CELLS = 64
COUNTS_SEED = 1234
COORDS_SEED = 5678


def _make_synthetic_spatial_adata() -> "ad.AnnData":
    """Build a small synthetic spatial AnnData that satisfies the pipeline.

    Mirrors ``test_pipeline_smoke``: integer counts in ``X``, gene SYMBOLS in
    ``var_names``, 2D coordinates in ``obsm['spatial']``, and a ``sample`` obs
    column so the per-sample tokenization path is used.
    """
    rng = np.random.default_rng(COUNTS_SEED)
    counts = rng.poisson(
        lam=3.0, size=(N_CELLS, len(GENE_SYMBOLS))
    ).astype("float32")

    adata = ad.AnnData(X=counts)
    adata.var_names = GENE_SYMBOLS
    adata.obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    adata.obs["sample"] = "smoke_sample"

    coord_rng = np.random.default_rng(COORDS_SEED)
    adata.obsm["spatial"] = coord_rng.uniform(
        low=0.0, high=1000.0, size=(N_CELLS, 2)
    ).astype("float32")
    return adata


@pytest.fixture(scope="module")
def embedded(tmp_path_factory):
    """Download the model, run the pipeline once, and save the tokenized dataset.

    Returns the model dir, the harmonized AnnData, the tokenized dataset (loaded
    from disk -- what the downstream functions consume), and the panel's Ensembl
    IDs. Module-scoped so the (slow) download + embed happens a single time.
    """
    try:
        model_dir = download_pretrained(REPO_ID)
    except Exception as exc:  # noqa: BLE001 - any download failure -> skip
        pytest.skip(
            f"Could not download '{REPO_ID}' ({type(exc).__name__}: {exc}). "
            "Ensure network access to Hugging Face (TERRA repos are public)."
        )

    tmp = tmp_path_factory.mktemp("downstream")
    dataset_path = str(tmp / "smoke.dataset")

    adata = harmonize_tokenize_embed_pipeline(
        adata=_make_synthetic_spatial_adata(),
        sample_key="sample",
        batch_key="batch",
        model_folder_path=model_dir,
        cache_directory_path=str(tmp),
        save_dataset_path=dataset_path,
        nproc=1,
        processing_mode="sequential",
        num_workers=0,
        batch_size=32,
    )

    dataset = load_from_disk(dataset_path)
    gene_ids = list(adata.var["ensembl_id"])
    assert len(gene_ids) > 1, "need at least two harmonized genes for downstream"

    return {
        "model_dir": model_dir,
        "adata": adata,
        "dataset": dataset,
        "gene_ids": gene_ids,
    }


def test_average_gene_embed(embedded):
    """Both the intrinsic (cell) and spatial (neighborhood) views are returned."""
    gene_ids = embedded["gene_ids"]
    avg = get_average_gene_embed(
        dataset=embedded["dataset"],
        model_folder_path=embedded["model_dir"],
        cell_gene_ensembl_id=gene_ids,
        neighborhood_gene_ensembl_id=gene_ids,
    )
    for key in ("cell_gene_emb_average_per_data",
                "neighborhood_gene_emb_average_per_data"):
        emb = np.asarray(avg[key])
        assert emb.ndim == 2, f"'{key}' should be 2D, got {emb.shape}"
        assert emb.shape[0] == len(gene_ids), f"'{key}' row count != n_genes"
        assert emb.shape[1] > 0, f"'{key}' has non-positive embed_dim"


def test_gene_embed(embedded):
    """Per-cell, per-gene embeddings are returned for both contexts."""
    goi = embedded["gene_ids"][:2]
    out = get_gene_embed(
        dataset=embedded["dataset"],
        model_folder_path=embedded["model_dir"],
        cell_gene_ensembl_id=goi,
        neighborhood_gene_ensembl_id=goi,
    )
    assert any(k.startswith("cell_emb_gene") for k in out)
    assert any(k.startswith("neighborhood_emb_gene") for k in out)
    n_cells = embedded["adata"].n_obs
    for key, values in out.items():
        arr = np.asarray(values)
        assert arr.shape[0] == n_cells, f"'{key}' has {arr.shape[0]} rows != n_cells"


def test_spatial_score_and_rankings(embedded):
    """get_spatial_score feeds get_top_gene_score / get_top_gene_pairs."""
    gene_ids = embedded["gene_ids"]
    score = get_spatial_score(
        dataset=embedded["dataset"],
        model_folder_path=embedded["model_dir"],
        cell_gene_ensembl_id=gene_ids,
        neighborhood_gene_ensembl_id=gene_ids,
    )
    for key in ("cos_sim_cell", "cos_sim_neighborhood",
                "cell_count_cell", "cell_count_neighborhood"):
        assert key in score, f"missing '{key}' in spatial-score output"

    gene_pair_score = score["cos_sim_neighborhood"] / score["cos_sim_cell"]

    # min_count=0 / permissive sim threshold so the tiny synthetic panel yields
    # rows (the point is to exercise the code path, not the biology).
    df_scores = get_top_gene_score(
        gene_pair_score,
        cell_gene_ensembl_id=gene_ids,
        gene_df=embedded["adata"].var,
        gene_counts=score["cell_count_neighborhood"],
        min_count=0,
    )
    assert "gene_score" in df_scores.columns

    df_pairs = get_top_gene_pairs(
        torch.from_numpy(gene_pair_score),
        count_cell=torch.from_numpy(score["cell_count_cell"]),
        count_neb=torch.from_numpy(score["cell_count_neighborhood"]),
        cos_sim_cell=torch.from_numpy(score["cos_sim_cell"]),
        cos_sim_neb=torch.from_numpy(score["cos_sim_neighborhood"]),
        gene_df=embedded["adata"].var,
        cell_gene_ids=gene_ids,
        neighborhood_gene_ids=gene_ids,
        min_count=0,
        sim_thresh=-1.0,
        k=10,
    )
    assert isinstance(df_pairs, pd.DataFrame)


def test_emd_distance(embedded):
    """get_emd_distance returns a per-cell distance array (first of the tuple)."""
    emd_dist, emd_matrix = get_emd_distance(
        dataset=embedded["dataset"],
        model_folder_path=embedded["model_dir"],
        cell_gene_ensembl_id=embedded["gene_ids"],
        neighborhood_gene_ensembl_id=embedded["gene_ids"],
    )
    n_cells = embedded["adata"].n_obs
    emd_dist = np.asarray(emd_dist)
    assert emd_dist.shape[0] == n_cells, "EMD distance is not per-cell"
    assert np.isfinite(emd_dist).all(), "EMD distance has non-finite values"
    assert np.asarray(emd_matrix).shape[0] == n_cells


def test_perturb_all_cells_all_genes(embedded):
    """The perturbed_cell_id='all' / 'all'-gene fast path runs and re-embeds.

    This is the exact path the regression broke: fold-change all genes and knock
    out a few specific genes, on both the cell and neighborhood, for every cell.
    """
    gene_ids = embedded["gene_ids"]
    rows = []
    for target in ("cell", "neighborhood"):
        rows.append({"perturbed_ensembl_id": "all",
                     "perturbation_target": target,
                     "perturbation_type": "foldchange", "foldchange": 0.5})
        for gene in gene_ids[:2]:
            rows.append({"perturbed_ensembl_id": gene,
                         "perturbation_target": target,
                         "perturbation_type": "knockout", "foldchange": np.nan})
    perturb_df = pd.DataFrame(rows)
    perturb_df["perturbed_cell_id"] = "all"

    perturbed = perturb_dataset(
        dataset=embedded["dataset"],
        perturb_df=perturb_df,
        model_folder_path=embedded["model_dir"],
        nproc=1,
    )
    emb = embed_dataset(
        dataset=perturbed,
        model_folder_path=embedded["model_dir"],
        num_workers=0,
    )

    n_cells = embedded["adata"].n_obs
    for key in ("cell_emb", "neighborhood_emb"):
        out = np.asarray(emb[key])
        assert out.shape[0] == n_cells, f"perturbed '{key}' row count != n_cells"
        assert np.isfinite(out).all(), f"perturbed '{key}' has non-finite values"
