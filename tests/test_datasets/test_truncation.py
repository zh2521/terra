"""Tests for the load-time truncation logic in ``CellGraphDataset``.

``CellGraphDataset.__getitem__`` re-slices a larger-tokenized dataset to the
configured layout along two independent axes, *before* any segment
re-assembly happens (see ``cell_datasets.py``):

  * neighbours -- keep the first ``n_segments`` segments (cell + nearest
    ``n_neighs`` neighbours); enabled by ``truncate_neighbors=True``.
  * genes/cell -- keep the first ``seq_len_cell`` (top-ranked) gene tokens of
    each segment; enabled by setting ``tokenized_seq_len_cell`` (the stored
    per-segment stride) larger than ``seq_len_cell``.

The gather builds an index

    idx = cat([arange(s*stride, s*stride + seq_len_cell) for s in range(keep)])

and applies it to ``item['gene_tokens']`` and (when present) ``gene_expr`` /
``seg_tokens`` / ``rel_x_coord`` / ``rel_y_coord``.

These tests construct a minimal ``CellGraphDataset`` over a tiny in-memory
``datasets.Dataset`` and drive ``__getitem__`` so the REAL gather code runs.
To assert on the gather *output* directly (and avoid coupling to the
downstream per-segment re-assembly / padding / masking), we capture the
mutated ``item`` row the moment it is first handed to ``_get_segment_seq`` --
which in ``__getitem__`` happens immediately after the gather block, after
coord expansion and ``seg_tokens`` reconstruction. The capture wrapper calls
through to the real ``_get_segment_seq`` so the rest of ``__getitem__`` still
runs to completion.

The data layout uses hand-checkable token values: for a dataset with
``n_seg`` segments of stride ``stride`` genes each, position
``seg*stride + rank`` holds ``100*seg + (rank + 1)`` (segments 0-indexed,
genes rank/count-descending within a segment). So segment 0 = [1, 2, 3, 4],
segment 1 = [101, 102, 103, 104], segment 2 = [201, 202, 203, 204] for
stride 4.

The full GPU stack (and ``datasets``) may be absent locally; these tests are
intended to run in CI where torch + datasets are installed. ``datasets`` is
imported lazily inside a fixture so collection does not hard-fail when it is
missing -- the tests skip instead.
"""

import functools

import pytest
import torch

# datasets is an optional/heavy dep; skip the whole module if unavailable so
# local `py_compile` / collection without the GPU stack stays green.
datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentinel recognizable token value: position (seg, rank) -> 100*seg + rank + 1.
def _token_value(seg: int, rank: int) -> int:
    return 100 * seg + (rank + 1)


def _gene_tokens(n_seg: int, stride: int) -> list[int]:
    """Dense per-segment layout, genes count-descending within a segment."""
    return [_token_value(seg, rank)
            for seg in range(n_seg)
            for rank in range(stride)]


def _make_hf_dataset(n_seg: int, stride: int,
                     with_gene_expr: bool = True,
                     with_seg_tokens: bool = False,
                     gene_tokens: list[int] | None = None,
                     ) -> "datasets.Dataset":
    """Build a tiny single-row HF dataset with exactly the columns
    ``CellGraphDataset.__getitem__`` reads. Tensors come back from
    ``self.dataset[idx]`` via ``.with_format('torch')`` so the size/count
    ops in the gather + ``_get_segment_seq`` work."""
    gt = gene_tokens if gene_tokens is not None else _gene_tokens(n_seg, stride)
    cols: dict = {"gene_tokens": [gt]}
    if with_gene_expr:
        # gene_expr made distinct from gene_tokens so we can prove the SAME
        # index is applied (not e.g. accidentally re-derived from tokens):
        # offset by 1000.
        cols["gene_expr"] = [[float(v + 1000) for v in gt]]
    if with_seg_tokens:
        # One segment id per token, 1-based: stride copies of segment k+1.
        n = len(gt)
        seg = [(i // stride) + 1 for i in range(n)]
        cols["seg_tokens"] = [seg]
    ds = datasets.Dataset.from_dict(cols)
    return ds.with_format("torch")


def _make_dataset(n_seg: int,
                  stride: int,
                  seq_len_cell: int,
                  truncate_neighbors: bool,
                  n_neighbors_kept: int | None = None,
                  with_gene_expr: bool = True,
                  with_seg_tokens: bool = False,
                  gene_tokens: list[int] | None = None,
                  ) -> CellGraphDataset:
    """Construct a CellGraphDataset with the heavy/optional paths disabled.

    Config choices (all read from ``CellBaseDataset.__init__`` defaults):
      * ``gt_type='rank'`` -- no ``gene_expr`` is required by the segment
        re-assembly path (it's deleted), keeping the wiring minimal; the
        gather still re-slices ``gene_expr`` whenever the column is present.
      * ``cell_pos_enc='segment'`` -- simplest pos-enc, NOT in
        ``_COORD_BASED_POS_ENCS`` so no rel-coord columns are required.
      * ``special_tokens=[]`` -- ``n_special_tokens == 0``; no special-token
        prepend, no ``nz_spc`` segment shift gating issues.
      * ``sampling_strategy=None`` -- deterministic "first seq_len_cell tokens
        per segment" behaviour, no RNG.
      * ``n_nonzero_tokens_list`` supplied so no ``n_nonzero_tokens`` column
        is needed.

    ``seq_len_neighborhood`` is derived so that
    ``n_segments = (seq_len_cell + seq_len_neighborhood)/seq_len_cell`` equals
    the number of segments we want to KEEP (cell + neighbours).
    """
    if truncate_neighbors:
        assert n_neighbors_kept is not None
        keep_seg = n_neighbors_kept  # total segments to keep (cell + neighbours)
    else:
        keep_seg = n_seg  # keep all data segments
    # n_segments = (seq_len_cell + seq_len_neighborhood) / seq_len_cell = keep_seg
    seq_len_neighborhood = seq_len_cell * (keep_seg - 1)

    ds = _make_hf_dataset(
        n_seg=n_seg,
        stride=stride,
        with_gene_expr=with_gene_expr,
        with_seg_tokens=with_seg_tokens,
        gene_tokens=gene_tokens,
    )
    return CellGraphDataset(
        gt_type="rank",
        cell_pos_enc="segment",
        dataset=ds,
        vocab_size=16,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        special_tokens=[],
        sampling_strategy=None,
        n_nonzero_tokens_list=[stride * n_seg],  # avoid needing the column
        truncate_neighbors=truncate_neighbors,
        tokenized_seq_len_cell=stride,
    )


def _capture_gathered_item(ds: CellGraphDataset, idx: int = 0) -> dict:
    """Run the REAL ``__getitem__`` but intercept the mutated HF row the
    moment it is first passed to ``_get_segment_seq`` -- i.e. right after the
    gather block has re-sliced ``gene_tokens`` / ``gene_expr`` / ``seg_tokens``.

    Returns the captured ``item`` dict (post-gather). ``__getitem__`` still
    runs to completion so we exercise the whole code path.
    """
    captured = {}
    real = ds._get_segment_seq

    @functools.wraps(real)
    def wrapper(item, segment, segment_seq_len):
        if not captured:  # first (segment == 1) call only
            captured["item"] = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in item.items()
            }
        return real(item=item, segment=segment,
                    segment_seq_len=segment_seq_len)

    ds._get_segment_seq = wrapper  # instance attr shadows the bound method
    try:
        ds[idx]
    finally:
        del ds._get_segment_seq
    assert "item" in captured, "_get_segment_seq was never called"
    return captured["item"]


# ---------------------------------------------------------------------------
# (a) sequence-length-only truncation
# ---------------------------------------------------------------------------

def test_seq_len_only_truncation_keeps_top_genes_per_segment():
    """tokenized_seq_len_cell=4, seq_len_cell=2, keep ALL 3 segments,
    truncate_neighbors=False.

    Gather index = [0,1, 4,5, 8,9] -> keeps the first 2 (top-ranked) genes of
    each of the 3 segments. Expected gene_tokens:
    [1, 2, 101, 102, 201, 202].
    """
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=2,
                       truncate_neighbors=False, with_gene_expr=True)
    item = _capture_gathered_item(ds)

    expected = [1, 2, 101, 102, 201, 202]
    assert item["gene_tokens"].tolist() == expected

    # gene_expr re-sliced with the SAME index (values are tokens + 1000).
    assert item["gene_expr"].tolist() == [float(v + 1000) for v in expected]


# ---------------------------------------------------------------------------
# (b) neighbor-only truncation
# ---------------------------------------------------------------------------

def test_neighbor_only_truncation_keeps_first_n_segments():
    """stride == seq_len_cell == 4, truncate_neighbors=True, keep 2 segments
    (cell + 1 nearest neighbour) out of 3 data segments.

    Because stride == seq_len_cell, no per-gene truncation happens; the gather
    just keeps segments contiguously: index = arange(0, 8) ->
    [1,2,3,4, 101,102,103,104]. This is the old neighbor-truncation behaviour.
    """
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=4,
                       truncate_neighbors=True, n_neighbors_kept=2,
                       with_gene_expr=True, with_seg_tokens=True)
    item = _capture_gathered_item(ds)

    expected = [1, 2, 3, 4, 101, 102, 103, 104]
    assert item["gene_tokens"].tolist() == expected
    assert item["gene_expr"].tolist() == [float(v + 1000) for v in expected]

    # seg_tokens (supplied as a column) re-sliced with the SAME index:
    # first 8 of [1,1,1,1, 2,2,2,2, 3,3,3,3].
    assert item["seg_tokens"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


# ---------------------------------------------------------------------------
# (c) combined neighbor + sequence-length truncation
# ---------------------------------------------------------------------------

def test_combined_truncation_top_genes_of_first_n_segments():
    """tokenized_seq_len_cell=4, seq_len_cell=2, truncate_neighbors=True,
    keep 2 segments (cell + 1 neighbour) out of 3.

    Gather index = [0,1, 4,5] -> top 2 genes of the first 2 segments:
    [1, 2, 101, 102].
    """
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=2,
                       truncate_neighbors=True, n_neighbors_kept=2,
                       with_gene_expr=True, with_seg_tokens=True)
    item = _capture_gathered_item(ds)

    expected = [1, 2, 101, 102]
    assert item["gene_tokens"].tolist() == expected
    assert item["gene_expr"].tolist() == [float(v + 1000) for v in expected]

    # seg_tokens re-sliced with the SAME index [0,1,4,5] of
    # [1,1,1,1, 2,2,2,2, 3,3,3,3] -> [1,1,2,2].
    assert item["seg_tokens"].tolist() == [1, 1, 2, 2]


# ---------------------------------------------------------------------------
# (d) disabled / matching -> identity (no-op)
# ---------------------------------------------------------------------------

def test_no_truncation_is_identity():
    """tokenized_seq_len_cell == seq_len_cell == 4 and
    truncate_neighbors=False.

    The guard ``truncate_neighbors or stride != seq_len_cell`` is False, so the
    whole gather block is skipped and gene_tokens / gene_expr are returned
    unchanged: the full [1,2,3,4, 101,102,103,104, 201,202,203,204].
    """
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=4,
                       truncate_neighbors=False, with_gene_expr=True)
    item = _capture_gathered_item(ds)

    expected = _gene_tokens(n_seg=3, stride=4)
    assert item["gene_tokens"].tolist() == expected
    assert item["gene_expr"].tolist() == [float(v + 1000) for v in expected]


# ---------------------------------------------------------------------------
# (e) error cases
# ---------------------------------------------------------------------------

def test_raises_when_seq_len_cell_exceeds_stride():
    """seq_len_cell (5) > tokenized_seq_len_cell (4) must raise ValueError.

    n_tok = 12 is a multiple of stride 4, so the first guard passes and the
    seq_len_cell-vs-stride guard fires. The block runs because
    ``stride != seq_len_cell``.
    """
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=5,
                       truncate_neighbors=True, n_neighbors_kept=2,
                       with_gene_expr=True)
    with pytest.raises(ValueError):
        ds[0]


def test_raises_when_length_not_multiple_of_stride():
    """Stored gene_tokens length not divisible by tokenized_seq_len_cell must
    raise ValueError. gene_tokens length 10, stride 4 -> 10 % 4 != 0.

    We pass an explicit length-10 token list; the gather runs because
    ``truncate_neighbors`` is True.
    """
    bad_tokens = list(range(1, 11))  # length 10
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=4,
                       truncate_neighbors=True, n_neighbors_kept=2,
                       with_gene_expr=True, gene_tokens=bad_tokens)
    with pytest.raises(ValueError):
        ds[0]


# ---------------------------------------------------------------------------
# gene_expr / seg_tokens consistency (same index as gene_tokens)
# ---------------------------------------------------------------------------

def test_gene_expr_and_seg_tokens_share_gather_index():
    """The gather must apply the IDENTICAL index to gene_tokens, gene_expr and
    seg_tokens. With gene_expr = gene_tokens + 1000, the re-sliced gene_expr
    must equal the re-sliced gene_tokens + 1000 elementwise, and seg_tokens
    must follow the same positions. Covers the combined-truncation path.
    """
    ds = _make_dataset(n_seg=3, stride=4, seq_len_cell=2,
                       truncate_neighbors=True, n_neighbors_kept=2,
                       with_gene_expr=True, with_seg_tokens=True)
    item = _capture_gathered_item(ds)

    gt = item["gene_tokens"]
    ge = item["gene_expr"]
    st = item["seg_tokens"]

    assert gt.shape == ge.shape == st.shape
    # gene_expr tracks gene_tokens (offset 1000) => same index applied.
    assert torch.allclose(ge, gt.to(ge.dtype) + 1000.0)
    # seg_tokens are the segment ids at the gathered positions [0,1,4,5].
    assert st.tolist() == [1, 1, 2, 2]
