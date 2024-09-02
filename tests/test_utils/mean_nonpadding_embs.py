import unittest
import torch

def mean_nonpadding_embs(embs, mask, dim=1):
    """
    Compute the mean of non-padding embeddings.
    
    Parameters:
    embs (torch.Tensor): The input embeddings tensor (can be 2D or 3D).
    mask (torch.Tensor): A boolean mask tensor indicating the non-padding or cls positions (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the mean.
    
    Returns:
    torch.Tensor: The mean embeddings tensor.
    """
    # Use broadcasting to sum across non-padding positions
    if embs.dim() == 3:
        masked_embs = embs * mask.unsqueeze(2)  # Broadcasting mask to match embs dimensions
        sum_embs = masked_embs.sum(dim)
        mean_embs = sum_embs / mask.sum(dim).view(-1, 1).float()
    else:
        raise ValueError('Expected a 3D tensor for embs, but got a tensor with {} dimensions.'.format(items.dim()))

    return mean_embs

class TestMeanNonPaddingEmbs(unittest.TestCase):

    def test_mean_nonpadding_embs_3d(self):
        # Test case for 3D tensor
        embs_3d = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                                [[7.0, 8.0, 9.0], [1.0, 2.0, 1.0]]], dtype=torch.float32)
        mask_3d = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        expected_mean_3d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 5.0]], dtype=torch.float32)
        result_3d = mean_nonpadding_embs(embs_3d, mask_3d, dim=1)
        self.assertTrue(torch.allclose(result_3d, expected_mean_3d),
                        f"Expected {expected_mean_3d}, but got {result_3d}")

        # Additional test case for a larger 3D tensor
        embs_3d_large = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [0.0, -10.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[7.0, 8.0, 9.0, 12.0], [1.0, 2.0, 1.0, 2.0], [-12.0, 12.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 2.0, 23.0], [4.0, 12.0, 2.0, 31.0], [1.0, 1.0, 1.0, 1.0]],
            [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [12.0, 0.3, 2.0, 0.0]]
        ], dtype=torch.float32)
        mask_3d_large = torch.tensor([
            [1, 0, 1, 1, 1],  # 4 valid rows
            [1, 1, 0, 1, 1],  # 4 valid rows
            [1, 1, 0, 0, 1],  # 3 valid rows
            [1, 1, 1, 1, 0]   # 4 valid rows
        ], dtype=torch.bool)
        expected_mean_3d_large = torch.tensor([
            [1.0, 1.25, 1.5, 1.5],
            [2.5, 3.0, 3.0, 4.0],
            [3.67, 3.67, 3.67, 3.67],
            [3.0, 3.0, 3.0, 3.0]
        ], dtype=torch.float32)
        result_3d_large = mean_nonpadding_embs(embs_3d_large, mask_3d_large, dim=1)
        self.assertTrue(torch.allclose(result_3d_large, expected_mean_3d_large, atol=1e-2),
                        f"Expected {expected_mean_3d_large}, but got {result_3d_large}")

if __name__ == "__main__":
    unittest.main()

