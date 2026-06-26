<p align="center">
  <img src="https://raw.githubusercontent.com/Lotfollahi-lab/terra/main/docs/_static/terra_logo.png" width="300" alt="TERRA logo">
</p>

<p align="center">
  <a href="https://pypi.org/project/terra-st/"><img src="https://img.shields.io/pypi/v/terra-st.svg" alt="PyPI"></a>
  <a href="https://terra-st.readthedocs.io/"><img src="https://readthedocs.org/projects/terra-st/badge/?version=latest" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
</p>

**TERRA** is a foundation model for spatial transcriptomics. It uses a
Joint-Embedding Predictive Architecture (JEPA): cells are tokenized together with
their spatial neighbors, parts of the input are masked, and the model learns by
predicting the *latent* representations of the masked cell and neighborhood
tokens. The resulting embeddings capture both a cell's own expression and its
tissue microenvironment.

Pretrained on **HST-Corpus-112M** (>100M cells at single-cell resolution spanning
human spatial-transcriptomics datasets), TERRA produces cell- and neighborhood-level embeddings that transfer to
downstream tasks such as niche and cell-type identification, batch-integrated
atlasing, spatial gene-pair scoring, and in-silico perturbation — without
task-specific retraining.

- 📖 **Documentation:** <https://terra-st.readthedocs.io>
- 🤗 **Pretrained models:** <https://huggingface.co/Lotfollahi-lab> (`TERRA-96M`, `TERRA-112M`)
- 📓 **Tutorial:** [end-to-end walkthrough](https://terra-st.readthedocs.io/en/latest/tutorials.html)

## Key features

- **Spatially-aware embeddings** — cell and neighborhood representations learned in latent space via JEPA.
- **Pretrained and ready to use** — download a model from the Hugging Face Hub and embed your own `AnnData` in a few lines.
- **Self-contained model bundles** — each release ships the checkpoint, tokenizer, and gene-reference files needed to reproduce its training-time harmonization.
- **Downstream analyses** — niche/cell-type clustering, gene-pair spatial scoring, EMD-based spatial structure, and perturbation.

## Installation

TERRA is published on PyPI as `terra-st` (the import name is `terra`). We
recommend installing with [uv](https://docs.astral.sh/uv/):

```shell
uv pip install terra-st
```

Plain `pip install terra-st` works too.

For a development install from a clone of this repository:

```shell
uv pip install -e ".[dev,test,doc]"
```

### PyTorch / GPU note

TERRA requires an NVIDIA GPU. Install the [PyTorch](https://pytorch.org) build
that matches your GPU driver **before** installing TERRA: run `nvidia-smi` and
read the "CUDA Version" shown in the top-right, then install the matching CUDA
build (see the [official PyTorch install guide](https://pytorch.org/get-started/locally/)),
e.g.:

```shell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install TERRA:

```shell
uv pip install terra-st
```

Verify the install with:

```shell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Quickstart

Download a pretrained model and embed your own spatial `AnnData` with the
end-to-end pipeline. Each downloaded bundle contains the gene-reference files
needed for harmonization, so no external paths are required:

```python
import anndata as ad
from terra import download_pretrained, harmonize_tokenize_embed_pipeline

adata = ad.read_h5ad("my_spatial_data.h5ad")   # raw counts in adata.X

model_dir = download_pretrained("Lotfollahi-lab/TERRA-96M")

adata = harmonize_tokenize_embed_pipeline(
    adata=adata,
    sample_key="sample",            # column in adata.obs identifying samples
    batch_key="batch",              # column to store the batch identifier
    model_folder_path=model_dir,
    cache_directory_path="./terra_cache",
)

# Cell- and neighborhood-level embeddings are now in adata.obsm.
```

See the [documentation](https://terra-st.readthedocs.io) for the step-by-step
pipeline, downstream analyses (niche identification, gene-pair scoring,
perturbation), and the full [tutorial](https://terra-st.readthedocs.io/en/latest/tutorials.html).

## Citation

If you use TERRA in your research, please cite the manuscript (in preparation).
A BibTeX entry and DOI will be added here on publication.

## License

The TERRA **code** is released under the [BSD 3-Clause License](LICENSE).
Pretrained **model weights** distributed on the Hugging Face Hub are released
under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use).
