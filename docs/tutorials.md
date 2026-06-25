# Tutorials

End-to-end notebooks for applying a pretrained TERRA model to your own spatial
data. They require an NVIDIA GPU and ship with outputs cleared — run them on a
GPU to reproduce the figures.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart: Single Sample
:link: notebooks/zero_shot_quickstart
:link-type: doc

Embed one spatial sample with a pretrained model and identify cell types and
spatial niches.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Quickstart: Multiple Samples
:link: notebooks/zero_shot_quickstart_multisample
:link-type: doc

Embed several sections/donors together and integrate them into a shared space.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Downstream Analysis
:link: notebooks/downstream_analysis
:link-type: doc

Gene-level embeddings, spatial gene-pair scoring, EMD spatial structure,
subsetting, and in-silico perturbation.
:::

::::

```{toctree}
:hidden: true
:maxdepth: 1

notebooks/zero_shot_quickstart
notebooks/zero_shot_quickstart_multisample
notebooks/downstream_analysis
```
