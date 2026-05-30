"""Tests for the unified batch-label extraction helper."""

import pytest
import torch

from nichejepa.models.batch_labels import extract_batch_label


def test_metadata_path_when_key_present():
    batch = {
        'values': torch.zeros(3, 5),
        'batch_value': torch.tensor([7, 8, 9]),
    }
    out = extract_batch_label(batch, key='batch_value')
    assert out.tolist() == [7, 8, 9]
    assert out.dtype == torch.long


def test_metadata_path_with_offset():
    batch = {
        'values': torch.zeros(2, 3),
        'batch_value': torch.tensor([10, 12]),
    }
    out = extract_batch_label(batch, key='batch_value', offset=10)
    assert out.tolist() == [0, 2]


def test_falls_back_to_values_when_key_missing():
    batch = {'values': torch.tensor([[5.0, 0.0], [9.0, 0.0]])}
    out = extract_batch_label(batch, key='batch_value')   # not in batch
    assert out.tolist() == [5, 9]


def test_falls_back_when_key_none():
    batch = {'values': torch.tensor([[3.0], [4.0]])}
    out = extract_batch_label(batch, key=None)
    assert out.tolist() == [3, 4]


def test_values_path_with_offset():
    batch = {'values': torch.tensor([[10.0, 0.0], [12.0, 0.0]])}
    out = extract_batch_label(batch, key=None, offset=10)
    assert out.tolist() == [0, 2]


def test_values_fallback_raises_on_1d():
    with pytest.raises(RuntimeError, match="at least 2-D"):
        extract_batch_label({'values': torch.zeros(5)}, key=None)
