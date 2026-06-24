import unittest

import torch

from terra.utils import retrieve_gene_emb


class TestRetrieveGeneEmb(unittest.TestCase):
    def test_basic_cell_case(self):
        # Cell path (aggregate_multiple=False) returns
        # (gene_presence, gene_indices): whether the gene is present in the cell
        # segment, and the index of its (first) occurrence within the sequence.
        tokens = torch.tensor([[1, 3, 0, 5, 4, 3],
                               [3, 5, 0, 2, 3, 0]])
        gene_presence, gene_indices = retrieve_gene_emb(
            ns_tokens=tokens,
            seq_len_cell=3,
            gene_type="cell",
            gene_id=3)
        torch.testing.assert_close(gene_presence,
                                   torch.tensor([True, True]))
        torch.testing.assert_close(gene_indices,
                                   torch.tensor([1, 0]))

    def test_sparsegene_neighborhood_case(self):
        # Neighborhood path (aggregate_multiple=True) returns
        # (gene_occ, occ_mask, gene_presence). Masking the first occurrence
        # embedding by occ_mask zeroes out sequences where the gene is absent.
        tokens = torch.tensor([[1, 3, 0, 5, 4, 3],
                               [3, 5, 0, 2, 3, 0],
                               [8, 4, 2, 4, 1, 0]])
        emb = torch.tensor([
            [[0.2, 0.4], [0.3, 0.1], [0.0, 0.0], [0.1, 0.3], [0.5, 0.4], [0.2, 0.6]],
            [[0.4, 0.1], [0.2, 0.2], [0.0, 0.0], [0.2, 0.5], [0.3, 0.1], [0.0, 0.0]],
            [[0.1, 0.5], [0.4, 0.6], [0.2, 0.3], [0.1, 0.1], [0.2, 0.5], [0.0, 0.0]]])
        gene_occ, occ_mask, gene_presence = retrieve_gene_emb(
            ns_tokens=tokens,
            seq_len_cell=3,
            gene_type="neighborhood",
            gene_id=4,
            emb=emb,
            aggregate_multiple=True)
        torch.testing.assert_close(gene_presence,
                                   torch.tensor([True, False, True]))
        torch.testing.assert_close(occ_mask[:, 0],
                                   torch.tensor([1.0, 0.0, 1.0]))
        # First-occurrence embedding, masked so absent genes are zeroed.
        first_occ = gene_occ[:, 0, :] * occ_mask[:, 0:1]
        torch.testing.assert_close(first_occ,
                                   torch.tensor([[0.5, 0.4],
                                                 [0.0, 0.0],
                                                 [0.1, 0.1]]))


if __name__ == '__main__':
    unittest.main()
