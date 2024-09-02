import unittest
import torch

def compute_weight_based_ranks(tokens):
    """
    Compute rank-based weights for a 2D tensor of tokens.

    Parameters:
    tokens (torch.Tensor): A 2D tensor where each row represents a sequence of tokens. The tokens are gene_id of cell or neighborhood

    Returns:
    torch.Tensor: A 2D tensor of the same shape as `tokens` containing the computed weights.
    """
    # Create a mask for non-zero tokens
    mask = tokens != 0

    # Compute the ranks based on the mask
    ranks = mask.cumsum(dim=1).float() * mask.float()

    rank_max = ranks.max(dim=1, keepdim=True)[0]
    rank_sum = ranks.sum(dim=1, keepdim=True)
    weights = (rank_max - ranks + 1) / (rank_sum + 1e-9)
    # Mask rank of padding tokens 
    weights = weights * mask.float()

    return weights
class TestComputeWeightBasedRanks(unittest.TestCase):
    def test_basic_case(self):
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
        expected_weights = torch.tensor([[0.5, 0.3333, 0.1667], [0.6667, 0.3333, 0.0000]])
        computed_weights = compute_weight_based_ranks(tokens)
        print(computed_weights)
        torch.testing.assert_close(computed_weights, expected_weights, atol=1e-4, rtol=1e-5)

    def test_all_zero_tokens(self):
        tokens = torch.tensor([[0, 0, 0], [0, 0, 0]])
        expected_weights = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        computed_weights = compute_weight_based_ranks(tokens)
        print(computed_weights)
        torch.testing.assert_close(computed_weights, expected_weights, atol=1e-4, rtol=1e-5)

    def test_single_non_zero_token(self):
        tokens = torch.tensor([[1, 0, 0], [0, 0, 0]])
        expected_weights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        computed_weights = compute_weight_based_ranks(tokens)
        print(computed_weights)
        torch.testing.assert_close(computed_weights, expected_weights, atol=1e-4, rtol=1e-5)

    def test_harder(self):
        tokens = torch.tensor([[1, 5, 0, 0], [0, 0, 0, 0], [1, 5, 6, 7], [1, 3, 111, 0]])
        expected_weights = torch.tensor([[0.6667, 0.3333, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000, 0.0000], [0.4000, 0.3000, 0.2000, 0.1000], [0.5000, 0.3333, 0.1667, 0.0000]])
        computed_weights = compute_weight_based_ranks(tokens)
        print(computed_weights)
        torch.testing.assert_close(computed_weights, expected_weights, atol=1e-4, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()


