"""
Tests for the pure gene-tokenization helpers in
``terra.tokenizers.tokenize``.

The functions under test are deterministic and operate on plain
NumPy/Python inputs, so every expected value below is hand-computed from
the documented contract:

* ``rank_gene_tokens`` sorts ``gene_tokens`` by descending ``gene_scores``
  (highest score -> rank 1) via ``np.argsort(-gene_scores)`` and returns
  the first ``n_tokens`` of the reordered tokens.
* ``process_gene_expr`` right-pads a 1D expression list with ``0.`` up to
  ``length`` or truncates it to ``length``.
* ``process_gene_tokens`` (small sibling helper) right-pads a token list
  with the ``<pad>`` id from ``token_dict`` up to ``length`` or truncates
  it to ``length``; it also reports the number of non-pad tokens.
"""

import unittest

import numpy as np

from terra.tokenizers.tokenize import (
    process_gene_expr,
    process_gene_tokens,
    rank_gene_tokens,
)


class TestRankGeneTokens(unittest.TestCase):
    def test_orders_tokens_by_descending_score(self):
        # Scores deliberately out of order so the correct ranking is obvious:
        # token 50 has the highest score (0.9), then 10 (0.5), then 30 (0.3),
        # then 20 (0.1).
        gene_scores = np.array([0.5, 0.1, 0.3, 0.9])
        gene_tokens = np.array([10, 20, 30, 50])

        ranked = rank_gene_tokens(gene_scores, gene_tokens)

        # argsort(-scores) -> [3, 0, 2, 1] -> tokens [50, 10, 30, 20]
        np.testing.assert_array_equal(ranked, np.array([50, 10, 30, 20]))

    def test_n_tokens_truncates_to_top_k(self):
        gene_scores = np.array([0.5, 0.1, 0.3, 0.9])
        gene_tokens = np.array([10, 20, 30, 50])

        # Only the top-2 scoring tokens should be returned, in rank order.
        ranked = rank_gene_tokens(gene_scores, gene_tokens, n_tokens=2)

        np.testing.assert_array_equal(ranked, np.array([50, 10]))
        self.assertEqual(ranked.shape, (2,))

    def test_zero_scores_rank_last(self):
        # A gene with zero (or lowest) score must be pushed to the back.
        gene_scores = np.array([0.0, 0.8, 0.4])
        gene_tokens = np.array([7, 8, 9])

        ranked = rank_gene_tokens(gene_scores, gene_tokens)

        # argsort(-[0.0, 0.8, 0.4]) -> [1, 2, 0] -> tokens [8, 9, 7]
        np.testing.assert_array_equal(ranked, np.array([8, 9, 7]))

    def test_all_zero_scores_keeps_all_tokens(self):
        # Edge case: every score is identical (all zero). np.argsort is
        # stable, so the original token order is preserved.
        gene_scores = np.zeros(3)
        gene_tokens = np.array([4, 5, 6])

        ranked = rank_gene_tokens(gene_scores, gene_tokens)

        np.testing.assert_array_equal(ranked, np.array([4, 5, 6]))
        self.assertEqual(ranked.shape, (3,))

    def test_ties_resolved_by_stable_sort(self):
        # Tokens 10 and 30 tie at 0.5; numpy's default argsort is stable,
        # so the lower original index (token 10) comes first.
        gene_scores = np.array([0.5, 0.9, 0.5])
        gene_tokens = np.array([10, 20, 30])

        ranked = rank_gene_tokens(gene_scores, gene_tokens)

        # argsort(-[0.5, 0.9, 0.5]) -> [1, 0, 2] -> tokens [20, 10, 30]
        np.testing.assert_array_equal(ranked, np.array([20, 10, 30]))


class TestProcessGeneExpr(unittest.TestCase):
    def test_pads_with_zeros_to_length(self):
        gene_expr = [3.0, 2.0, 1.0]

        processed = process_gene_expr(gene_expr, length=5)

        # Two trailing zeros appended to reach length 5.
        np.testing.assert_allclose(
            processed, np.array([3.0, 2.0, 1.0, 0.0, 0.0])
        )
        self.assertEqual(processed.shape, (5,))

    def test_truncates_to_length(self):
        gene_expr = [3.0, 2.0, 1.0, 0.5]

        processed = process_gene_expr(gene_expr, length=2)

        # Keeps only the first two entries (ranking order preserved).
        np.testing.assert_allclose(processed, np.array([3.0, 2.0]))
        self.assertEqual(processed.shape, (2,))

    def test_exact_length_is_unchanged(self):
        gene_expr = [1.5, 2.5]

        processed = process_gene_expr(gene_expr, length=2)

        np.testing.assert_allclose(processed, np.array([1.5, 2.5]))

    def test_all_zero_row_pads_to_all_zeros(self):
        # Edge case: an all-zero expression row stays all zeros after padding.
        gene_expr = [0.0, 0.0]

        processed = process_gene_expr(gene_expr, length=4)

        np.testing.assert_allclose(processed, np.zeros(4))
        self.assertEqual(processed.shape, (4,))

    def test_expr_aligns_with_ranked_tokens(self):
        # Integration-style check: rank tokens, then ensure the matching
        # expression vector lines up with the ranking before padding.
        gene_scores = np.array([0.2, 0.9, 0.5])
        gene_tokens = np.array([11, 22, 33])

        order = np.argsort(-gene_scores)  # [1, 2, 0]
        ranked_tokens = rank_gene_tokens(gene_scores, gene_tokens)
        ranked_expr = gene_scores[order]  # scores double as expression here

        np.testing.assert_array_equal(ranked_tokens, np.array([22, 33, 11]))

        processed = process_gene_expr(list(ranked_expr), length=4)
        # Ranked expression [0.9, 0.5, 0.2] then padded with one zero.
        np.testing.assert_allclose(
            processed, np.array([0.9, 0.5, 0.2, 0.0])
        )


class TestProcessGeneTokens(unittest.TestCase):
    TOKEN_DICT = {"<pad>": 0}

    def test_pads_with_pad_token_to_length(self):
        gene_tokens = [5, 6, 7]

        processed, num_nonzero = process_gene_tokens(
            gene_tokens, length=5, token_dict=self.TOKEN_DICT
        )

        # Two <pad> (=0) tokens appended; num_nonzero is the original count.
        np.testing.assert_array_equal(
            processed, np.array([5, 6, 7, 0, 0], dtype=np.int64)
        )
        self.assertEqual(processed.dtype, np.int64)
        self.assertEqual(num_nonzero, 3)

    def test_truncates_and_reports_full_length(self):
        gene_tokens = [5, 6, 7, 8]

        processed, num_nonzero = process_gene_tokens(
            gene_tokens, length=2, token_dict=self.TOKEN_DICT
        )

        np.testing.assert_array_equal(
            processed, np.array([5, 6], dtype=np.int64)
        )
        # On truncation the helper reports num_nonzero == length.
        self.assertEqual(num_nonzero, 2)

    def test_exact_length_no_padding(self):
        gene_tokens = [9, 8]

        processed, num_nonzero = process_gene_tokens(
            gene_tokens, length=2, token_dict=self.TOKEN_DICT
        )

        np.testing.assert_array_equal(
            processed, np.array([9, 8], dtype=np.int64)
        )
        self.assertEqual(num_nonzero, 2)

    def test_fewer_tokens_than_seq_length_is_all_padded(self):
        # Edge case: a single non-zero token padded out to the sequence length.
        gene_tokens = [42]

        processed, num_nonzero = process_gene_tokens(
            gene_tokens, length=4, token_dict=self.TOKEN_DICT
        )

        np.testing.assert_array_equal(
            processed, np.array([42, 0, 0, 0], dtype=np.int64)
        )
        self.assertEqual(num_nonzero, 1)


if __name__ == "__main__":
    unittest.main()
