"""Opt-in integration smoke test for the TERRA inference pipeline.

This test exercises the *real* published Hugging Face models end-to-end:
``download_pretrained`` -> ``harmonize_tokenize_embed_pipeline`` on a small
synthetic spatial AnnData. It is deliberately **opt-in** so it never runs (and
never breaks) the default CI, which has neither the GPU/inference stack, the
``huggingface_hub`` dependency, nor network access.

How to run it
-------------
1. Install the inference stack plus the Hub extra::

       pip install -e '.[hub]'

   (the pipeline also needs torch, anndata, scanpy, squidpy, pyensembl and
   datasets, which are core ``terra-st`` dependencies.)

   The TERRA model repos are **public**, so no Hugging Face authentication is
   required -- only network access to download the bundle.

2. Enable and run the opt-in test::

       TERRA_MODEL_SMOKE=1 pytest tests/test_inference -v

The test self-skips (rather than fails) when any precondition is missing: the
inference/Hub libraries are not importable, ``TERRA_MODEL_SMOKE`` is unset, or
the model download fails (offline / network error). It downloads the
real model bundle, builds a tiny synthetic spatial dataset, runs the full
harmonize+tokenize+embed pipeline and asserts -- at the smoke level -- that the
returned AnnData carries finite, non-trivial embeddings of the expected shape in
``adata.obsm``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# --- Opt-in / dependency gating -------------------------------------------
# Skip the whole module unless every heavy dependency the pipeline imports is
# present. ``importorskip`` turns a missing dependency into a clean skip rather
# than a collection error.
torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")
pytest.importorskip("huggingface_hub")
pytest.importorskip("squidpy")
pytest.importorskip("scanpy")
pytest.importorskip("pyensembl")
pytest.importorskip("datasets")

# Even with all deps installed, only run when the user explicitly opts in. This
# is the master switch that keeps the slow, network-dependent test out of
# the default suite.
pytestmark = pytest.mark.skipif(
    not os.environ.get("TERRA_MODEL_SMOKE"),
    reason="Set TERRA_MODEL_SMOKE=1 to run the published-model inference "
    "smoke test.",
)

# Imported only after the importorskip gate above, so the module still collects
# cleanly when the inference stack is absent.
from terra import download_pretrained, harmonize_tokenize_embed_pipeline  # noqa: E402

# Both published TERRA model repos. Each is one HF repo per named model.
MODEL_REPOS = [
    "Lotfollahi-lab/TERRA-96M",
    "Lotfollahi-lab/TERRA-112M",
]

# ~40 well-known HUMAN gene SYMBOLS. harmonize_adata maps SYMBOL -> Ensembl ID
# via the model bundle's ensembl_dictionary.pkl (keyed by gene_name), so var
# names must be symbols, not Ensembl IDs. This is an intentionally broad,
# housekeeping/lineage-marker mix so that plenty survive the symbol->Ensembl
# mapping AND the pretraining-occurrence filter (gene_count_dictionary.pkl,
# default cutoff 10) inside the bundle.
GENE_SYMBOLS = [
    "ACTB", "GAPDH", "B2M", "TUBB", "VIM", "PTPRC", "CD3D", "CD3E", "CD8A",
    "CD4", "CD19", "MS4A1", "CD14", "FCGR3A", "NKG7", "GNLY", "EPCAM", "KRT8",
    "KRT18", "KRT19", "COL1A1", "COL1A2", "COL3A1", "PECAM1", "VWF", "CDH5",
    "ACTA2", "PDGFRB", "MKI67", "PCNA", "CD68", "CD163", "LYZ", "HLA-DRA",
    "IL7R", "FOXP3", "GZMB", "CD79A", "ITGAM", "ENG", "THY1", "NCAM1",
]

N_CELLS = 64
# Fixed seeds so the synthetic data (and thus the test) is deterministic.
COUNTS_SEED = 1234
COORDS_SEED = 5678


def _make_synthetic_spatial_adata() -> "ad.AnnData":
    """Build a small synthetic spatial AnnData that satisfies the pipeline.

    Requirements encoded here, derived from the source:

    * ``adata.X`` holds raw INTEGER counts. ``harmonize_adata`` validates that
      X is all-integer (otherwise it errors unless an integer ``counts`` layer
      exists). We use Poisson draws cast to a float matrix whose values are
      whole numbers, which passes the ``np.allclose(data, data.astype(int))``
      check.
    * ``adata.var_names`` are human gene SYMBOLS (mapped to Ensembl IDs during
      harmonization via the bundle's ensembl_dictionary.pkl).
    * ``adata.obsm['spatial']`` holds 2D coordinates -- the tokenizer builds a
      squidpy spatial neighborhood graph from this exact key
      (cell_tokenizers reads ``adata.obsm['spatial']``).
    * an obs column (``sample``) for ``sample_key`` so the pipeline runs its
      per-sample path, which tokenizes with ``include_special_tokens=False``
      and therefore does NOT require ``adata.uns['dataset_id']``/``['batch']``.
    """
    rng = np.random.default_rng(COUNTS_SEED)
    # Poisson counts -> guaranteed non-negative integers. Use a float dtype but
    # whole-number values so the integer-count validation passes. lam>1 keeps
    # most cells well above any min_genes_per_cell filter.
    counts = rng.poisson(lam=3.0, size=(N_CELLS, len(GENE_SYMBOLS))).astype("float32")

    adata = ad.AnnData(X=counts)
    adata.var_names = GENE_SYMBOLS
    adata.obs_names = [f"cell_{i}" for i in range(N_CELLS)]

    # Single sample: the pipeline still exercises its sample_key branch, which
    # avoids the adata.uns['batch'] requirement of the no-sample-key branch.
    adata.obs["sample"] = "smoke_sample"

    # Random 2D spatial coordinates (fixed seed). squidpy's KNN graph needs
    # only coordinates; the absolute scale is irrelevant for a smoke test.
    coord_rng = np.random.default_rng(COORDS_SEED)
    adata.obsm["spatial"] = coord_rng.uniform(
        low=0.0, high=1000.0, size=(N_CELLS, 2)
    ).astype("float32")

    return adata


@pytest.mark.parametrize("repo_id", MODEL_REPOS)
def test_pipeline_smoke(repo_id, tmp_path):
    """Download a published model and embed a tiny synthetic dataset."""
    # 1. Download the model bundle. Skip (don't fail) on any download error --
    #    offline, network error, etc. (the repos are public, no token needed).
    try:
        model_dir = download_pretrained(repo_id)
    except Exception as exc:  # noqa: BLE001 - any download failure -> skip
        pytest.skip(
            f"Could not download '{repo_id}' ({type(exc).__name__}: {exc}). "
            "Ensure you have network access to Hugging Face (the TERRA model "
            "repos are public, so no token is required)."
        )

    # 2. Build synthetic input.
    adata = _make_synthetic_spatial_adata()

    # 3. Run the full pipeline with conservative params:
    #    - sequential processing_mode + nproc=1 avoid multiprocessing fan-out.
    #    - num_workers=0 keeps the dataloader in the main process.
    #    - gene_mapping_dict_file_path / gene_occurrence_count_file_path are
    #      left as None: the pipeline auto-resolves them from the model bundle
    #      (ensembl_dictionary.pkl / gene_count_dictionary.pkl).
    result = harmonize_tokenize_embed_pipeline(
        adata=adata,
        sample_key="sample",
        batch_key="batch",
        model_folder_path=model_dir,
        cache_directory_path=str(tmp_path),
        nproc=1,
        processing_mode="sequential",
        num_workers=0,
        batch_size=32,
    )

    # 4. Smoke-level assertions.
    assert isinstance(result, ad.AnnData), "pipeline must return an AnnData"

    # Embedding keys come from embed_dataset() in src/terra/inference/embed.py:
    # output_embed["cell_emb"] (line ~337), output_embed["neighborhood_emb"]
    # (line ~340), and -- because include_spatial_cell_emb defaults to True --
    # output_embed["spatial_cell_emb"] (line ~344). harmonize_tokenize_embed_
    # pipeline copies each of these into adata.obsm (line ~570-571).
    expected_keys = ["cell_emb", "neighborhood_emb", "spatial_cell_emb"]
    n_cells_kept = result.n_obs
    assert n_cells_kept > 0, "no cells survived harmonization/QC"

    for key in expected_keys:
        assert key in result.obsm, f"missing embedding '{key}' in adata.obsm"
        emb = np.asarray(result.obsm[key])

        # Shape: (n_cells_kept, embed_dim) with a positive embedding dim.
        assert emb.ndim == 2, f"'{key}' should be 2D, got shape {emb.shape}"
        assert emb.shape[0] == n_cells_kept, (
            f"'{key}' has {emb.shape[0]} rows but adata has {n_cells_kept} cells"
        )
        assert emb.shape[1] > 0, f"'{key}' has non-positive embed_dim {emb.shape[1]}"

        # Values are finite and not a degenerate all-zero array.
        assert np.isfinite(emb).all(), f"'{key}' contains non-finite values"
        assert np.any(emb != 0), f"'{key}' is all zeros"
