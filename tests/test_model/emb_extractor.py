import torch
import sys
import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from nichejepa.utils.emb_utils import mean_nonpadding_embs
def test_mean_nonpadding_embs_3d():
    # Test case for 3D tensor
    embs_3d = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], 
                            [[7.0, 8.0, 9.0], [1.0, 2.0, 1.0]]], dtype=torch.float32)
    mask_3d = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    expected_mean_3d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 5.0]], dtype=torch.float32)
    result_3d = mean_nonpadding_embs(embs_3d, mask_3d, dim=1)
    assert torch.allclose(result_3d, expected_mean_3d), f"Expected {expected_mean_3d}, but got {result_3d}"
    embs_3d_large = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [0.0, -10.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        [[7.0, 8.0, 9.0, 12.0], [1.0, 2.0, 1.0, 2.0], [-12.0, 12.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        [[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 2.0, 23.0], [4.0, 12.0, 2.0, 31.0], [1.0, 1.0, 1.0, 1.0]],
        [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [12.0, 0.3, 2.0, 0.0]]
    ], dtype=torch.float32)
    mask_3d_large = torch.tensor([
        [1, 0, 1, 1, 1],  # 4 valid rows
        [1, 1, 0, 1, 1],  # 4 valid rows
        [1, 1, 0, 0, 1],  # 2 valid rows
        [1, 1, 1, 1, 0]   # 4 valid rows
    ], dtype=torch.bool)
    expected_mean_3d_large = torch.tensor([
        [1.0, 1.25, 1.5, 1.5],
        [2.5, 3.0, 3.0, 4.0],
        [3.67, 3.67, 3.67, 3.67],
        [3.0, 3.0, 3.0, 3.0]
    ], dtype=torch.float32)
    result_3d_large = mean_nonpadding_embs(embs_3d_large, mask_3d_large, dim=1)
    assert torch.allclose(result_3d_large, expected_mean_3d_large, atol=1e-2), f"Expected {expected_mean_3d_large}, but got {result_3d_large}"

if __name__ == "__main__":
    test_mean_nonpadding_embs_3d()
