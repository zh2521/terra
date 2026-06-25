# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0] - 2026-06-25

First public release of **TERRA**, a JEPA-based foundation model for spatial
transcriptomics, pretrained on HST-Corpus-112M (>100 million cells at
single-cell resolution).

### Added

-   **Pretrained models** distributed as self-contained bundles on the Hugging
    Face Hub — `TERRA-96M` and `TERRA-112M` — fetched with `download_pretrained`
    and published with the `terra-hub` CLI / `push_model_to_hub`. Each bundle
    ships the checkpoint, tokenizer, and gene-reference files needed to
    reproduce training-time harmonization.
-   **Zero-shot inference pipeline** — `harmonize_tokenize_embed_pipeline`, plus
    the individual `harmonize_adata` / `tokenize_adata` / `embed_dataset` steps —
    producing cell- and neighborhood-level embeddings for single- and
    multi-sample data.
-   **Downstream analyses**: gene-level embeddings (`get_gene_embed`,
    `get_average_gene_embed`), spatial gene-pair scoring (`get_spatial_score`),
    EMD-based spatial structure (`get_emd_distance`), and in-silico perturbation
    (`perturb_dataset`).
-   **Finetuning** of the pretrained encoder with LoRA/PEFT (`terra.training`).
-   **Data reader** for public 10x Genomics Xenium samples
    (`terra.datasets.read_xenium_10x`), used by the tutorials.
-   **Documentation** on Read the Docs: an installation guide, three end-to-end
    tutorials (single-sample quickstart, multi-sample quickstart, and downstream
    analysis), a user guide, and a full API reference.

[0.1.0]: https://github.com/Lotfollahi-lab/terra/releases/tag/v0.1.0
