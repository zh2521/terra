# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0] - 2026-06-26

First public release of **TERRA**, a JEPA-based foundation model for spatial
transcriptomics, pretrained on HST-Corpus-112M (>100 million cells at
single-cell resolution).

### Added

-   Pretrained models (`TERRA-96M`, `TERRA-112M`) distributed as self-contained
    bundles on the Hugging Face Hub, with download and publishing utilities
    (`download_pretrained`, `terra-hub`).
-   Zero-shot inference pipeline producing cell- and neighborhood-level
    embeddings for single- and multi-sample spatial data.
-   Downstream analyses: gene-level embeddings, spatial gene-pair scoring,
    EMD-based spatial structure, and in-silico perturbation.
-   Finetuning of the pretrained encoder with LoRA/PEFT.
-   Documentation, tutorials, and API reference.

[0.1.0]: https://github.com/Lotfollahi-lab/terra/releases/tag/v0.1.0
