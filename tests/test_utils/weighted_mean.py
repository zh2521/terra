import torch
import unittest

# Define the weighted_mean function as provided
def weighted_mean(embs, weights, dim=1):
    """
    Compute the weighted mean of embs.

    Parameters:
    embs (torch.Tensor): The input embs tensor (3D).
    weights (torch.Tensor): A tensor of weights (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the weighted mean.

    Returns:
    torch.Tensor: The weighted mean tensor.
    
    Raises:
    ValueError: If the items tensor is not 3D.

    """
    # Use broadcasting to multiply items by weights
    if embs.dim() == 3:
        weighted_embs = embs * weights.unsqueeze(2)  # Broadcasting weights to match embs dimensions
        weighted_sum = weighted_embs.sum(dim)
        weights_sum = weights.sum(dim).unsqueeze(1)  # Sum weights along the specified dimension and keep the dimensions consistent
        weighted_mean = weighted_sum / weights_sum
    else:
        raise ValueError('Expected a 3D tensor for items, but got a tensor with {} dimensions.'.format(embs.dim()))

    return weighted_mean

class TestWeightedMean(unittest.TestCase):

    def test_weighted_mean_basic(self):
        embs = torch.tensor([[[1.0, 2.0, 4.0], [3.0, 4.0, 5.0]], [[5.0, 6.0, 18], [7.0, 8.0, -9]]])
        weights = torch.tensor([[0.2, 0.8],[1.0,0.0]])
        expected_output = torch.tensor([[2.6, 3.6, 4.8], [5.0, 6.0, 18.0]])
        result = weighted_mean(embs, weights, dim=1)
        self.assertTrue(torch.allclose(result, expected_output), f"Expected {expected_output} but got {result}")
    def test_weighted_invalid_dimention(self):
        embs = torch.tensor([[1.0, 2.0], [3.0, 4.0],[5.0, 6.0], [7.0, 8.0]])
        weights = torch.tensor([[0.2, 0.8],[1.0,0.0]])
        with self.assertRaises(ValueError):
            weighted_mean(embs, weights, dim=2)  # 2 is out of range for a 3D tensor with dim 1
    def test_all_zeros(self):
        embs = torch.tensor([[[0.0, 0.0], [0.0, 0.0],[0.0, 0.0], [0.0, 0.0]]])
        weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        expected_output = torch.tensor([[0.0, 0.0]])
        result = weighted_mean(embs, weights, dim=1)
        self.assertTrue(torch.allclose(result, expected_output), f"Expected {expected_output} but got {result}")

if __name__ == "__main__":
    unittest.main()

