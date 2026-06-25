"""Tests for ``terra.utils.NestedStratifiedGroupKFold``.

The splitter produces nested (outer test / inner train-val) cross-validation
folds in which whole *groups* (e.g. patients) are kept together. The single
most important correctness property for a grouped CV splitter is the absence
of **group leakage**: no group may appear on both sides of any split, because
that silently inflates evaluation metrics.

These tests drive ``design_splits`` (the pure, no-I/O planning phase) on a
small synthetic AnnData and assert the structural invariants directly on the
returned indices:

* within each outer fold, ``trainval``/``test`` partition every cell and share
  no group;
* the outer test folds partition the whole dataset, and each group lands in
  exactly one outer test fold;
* within each inner fold, ``train``/``val`` share no group.

``group_balance_tolerance`` is set high so the splitter's separate
group-balance heuristic never rejects a fold — these tests target leakage and
partition correctness, not that heuristic.
"""

import contextlib
import io

import numpy as np

from terra.utils import NestedStratifiedGroupKFold


def _make_adata(n_groups=12, cells_per_group=5):
    """Synthetic AnnData with homogeneous-label groups.

    ``n_groups`` patients, alternating labels A/B, ``cells_per_group`` cells
    each. Labels are constant within a group so stratified *group* splitting
    is well defined.
    """
    import anndata as ad
    import pandas as pd

    groups, labels = [], []
    for g in range(n_groups):
        label = "A" if g % 2 == 0 else "B"
        groups.extend([f"patient_{g}"] * cells_per_group)
        labels.extend([label] * cells_per_group)

    n = len(groups)
    obs = pd.DataFrame(
        {"group": groups, "label": labels},
        index=[f"cell_{i}" for i in range(n)],
    )
    X = np.zeros((n, 3), dtype="float32")
    return ad.AnnData(X=X, obs=obs)


def _design(adata, **overrides):
    kwargs = dict(
        stratify_group="group",
        label_column="label",
        K_outer=3,
        K_inner=2,
        shuffle=True,
        seed=42,
        require_all_labels=False,
        group_balance_tolerance=99,  # isolate leakage/partition from balance
    )
    kwargs.update(overrides)
    splitter = NestedStratifiedGroupKFold(**kwargs)
    # design_splits prints a verbose report; silence it for the test.
    with contextlib.redirect_stdout(io.StringIO()):
        return splitter.design_splits(adata)


def test_outer_folds_partition_and_no_group_leakage():
    adata = _make_adata()
    info = _design(adata)
    vd = info["_validation_data"]
    groups = np.asarray(vd["groups"])
    n = len(groups)
    outer = vd["outer_fold_validations"]

    assert len(outer) == 3

    test_sets = []
    for fold in outer:
        tv = np.asarray(fold["trainval_pos"])
        te = np.asarray(fold["test_pos"])

        # trainval + test partition every cell exactly once.
        assert sorted(tv.tolist() + te.tolist()) == list(range(n))
        assert set(tv.tolist()).isdisjoint(set(te.tolist()))

        # No group spans both sides of the outer split (group leakage).
        assert set(groups[tv]).isdisjoint(set(groups[te]))

        test_sets.append(set(te.tolist()))

    # Outer test folds partition the dataset (disjoint + cover everything).
    union = set().union(*test_sets)
    assert union == set(range(n))
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            assert test_sets[i].isdisjoint(test_sets[j])


def test_each_group_in_exactly_one_outer_test_fold():
    adata = _make_adata()
    info = _design(adata)
    vd = info["_validation_data"]
    groups = np.asarray(vd["groups"])
    outer = vd["outer_fold_validations"]

    group_to_folds = {}
    for k, fold in enumerate(outer):
        te = np.asarray(fold["test_pos"])
        for grp in set(groups[te]):
            group_to_folds.setdefault(grp, set()).add(k)

    # Every group is held out for testing in exactly one outer fold.
    assert set(group_to_folds) == set(np.unique(groups))
    for grp, folds in group_to_folds.items():
        assert len(folds) == 1, f"group {grp} appears in outer test folds {folds}"


def test_inner_folds_no_group_leakage():
    adata = _make_adata()
    info = _design(adata)
    vd = info["_validation_data"]
    groups = np.asarray(vd["groups"])
    outer = vd["outer_fold_validations"]
    inner = vd["inner_fold_validations"]

    for fold in outer:
        of = fold["outer_fold"]
        trainval_pos = np.asarray(fold["trainval_pos"])
        trainval_groups = groups[trainval_pos]

        inner_folds = inner[of]
        assert len(inner_folds) >= 1
        for inv in inner_folds:
            # train_pos / val_pos are positional indices INTO trainval_pos.
            tr = np.asarray(inv["train_pos"])
            va = np.asarray(inv["val_pos"])

            assert set(tr.tolist()).isdisjoint(set(va.tolist()))
            assert set(trainval_groups[tr]).isdisjoint(set(trainval_groups[va]))


def test_reproducible_with_same_seed():
    adata = _make_adata()
    a = _design(adata)["_validation_data"]["outer_fold_validations"]
    b = _design(adata)["_validation_data"]["outer_fold_validations"]
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(np.asarray(fa["test_pos"]),
                                      np.asarray(fb["test_pos"]))
