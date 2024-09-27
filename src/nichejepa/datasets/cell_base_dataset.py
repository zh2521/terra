import os
import subprocess
import time
from logging import getLogger
from typing import Optional, Tuple, List, Literal

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.distributed import CustomDistributedLengthGroupedSampler


_GLOBAL_SEED = 0
logger = getLogger()


class CellBaseDataset(Dataset):
    def __init__(self,
                 dataset: datasets.Dataset,
                 vocab_size: int,
                 special_tokens: list=[
                    'cls', 'assay', 'species', 'tissue', 'gene_panel', 'batch'],
                 sampling_strategy: Optional[
                    Literal['norm_count_rank_sampling',
                            'norm_count_rank_sampling_rep',
                            'rand_sampling',
                            'rand_sampling_rep']]=None,
                 ):
        """
        Torch CellBaseDataset class.

        Parameters
        -----------
        dataset:
            Huggingface dataset with gene and special tokens.
        vocab_size:
            Size of the vocabulary.
        special_tokens:
            Special tokens to be included in the sequence.
        sampling_strategy:
            Token sampling strategy.
        """
        self.dataset = dataset
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.sampling_strategy = sampling_strategy
        
    def __len__(self):
        return self.len
         
    def _get_gene_tokens_for_segment(self, 
                                     item: int,
                                     segment_idx: int,
                                     segment_seq_len: int,
                                     ) -> List:
        """
        Get gene tokens for a given segment based on sampling strategy.

        Parameters
        -----------
        item:
            Index of the cell in the huggingface dataset.
        segment_idx:
            Index of the segment for which tokens are retrieved.
        segment_seq_len:
            Desired length of the segment token sequence.

        Returns
        --------
        seg_gene_tokens:
            List of tokens for a given segment.
        """
        # Only keep gene tokens in specified segment
        seg_gene_tokens = self.dataset[item]["gene_tokens"][
            self.dataset[item]["seg_tokens"] == segment_idx]

        # Validate that segment sequence length is specified correctly
        if 'rep' in self.sampling_strategy:
            pass
        else:
            if segment_seq_len > len(seg_gene_tokens):
                raise ValueError(
                    'Sequence length for a given segment cannot be larger than'
                    'segment size when not sampling with replacement.')

        # If no sampling strategy is specified, use all tokens up to specified
        # length
        if self.sampling_strategy is None:
            seg_gene_tokens = seg_gene_tokens[:segment_seq_len]
        # Otherwise, sample a subset of tokens based on the sampling strategy
        else:
            n_nonzero_tokens = sum(1 for token in seg_gene_tokens if token != 0)

            seg_gene_tokens = self._create_sampled_token_sequence(
                tokens=seg_gene_tokens,
                n_nonzero_tokens=n_nonzero_tokens,
                size=segment_seq_len,
                self.sampling_strategy)
                
        return seg_gene_tokens
            
    def _create_sampled_token_sequence(self,
                                       tokens: List,
                                       n_nonzero_tokens: int,
                                       size: int,
                                       sampling_strategy: Literal[
                                         'norm_count_rank_sampling',
                                         'norm_count_rank_sampling_rep',
                                         'rand_sampling',
                                         'rand_sampling_rep']
                                       ) -> List:
        """
        Sample a subset of tokens based on a sampling strategy.

        Parameters
        -----------
        tokens:
            List of tokens.
        n_nonzero_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
        sampling_strategy:
            Sampling strategy for the token selection.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        """
        if 'norm_count_rank_sampling' in sampling_strategy:
            # Calculate weights based on rank and number of nonzero tokens
            # Higher the rank, higher the weight
            # a = [4, 1, 3, 2, 5, 0, 0, 0]
            # n_nonzero_tokens = 5  
            # sum_rank = 5 * (5 + 1) / 2.0 = 15.0
            # weights = [(n_nonzero_tokens - i)/sum_rank for i in range(n_nonzero_tokens)] 
            # = [0.3333333333333333, 0.26666666666666666, 0.2, 0.13333333333333333, 0.06666666666666667]
            # np.sum(weights) = 1.0
            sum_rank = (n_nonzero_tokens * (n_nonzero_tokens + 1) / 2.0) + 1e-9
            weights = [(n_nonzero_tokens - i)/sum_rank for i in range(n_nonzero_tokens)]
            assert np.isclose(np.sum(weights), 1.0)
        elif 'rand_sampling' in sampling_strategy:
            weights = np.ones(n_nonzero_tokens) / n_nonzero_tokens
        else:
            raise ValueError(
                f"'{sampling_strategy}' is an invalid sampling strategy.")
            
        # Sample token indices based on weights
        sampled_indices = np.random.choice(
            np.arange(n_nonzero_tokens),
            size=min(size, n_nonzero_tokens),
            p=weights,
            replace=(True if 'rep' in sampling_strategy else False))
            
        # Sort sampled indices to preserve rank order
        sampled_indices = np.sort(sampled_indices)
        sampled_tokens = [tokens[i] for i in sampled_indices]

        if size > n_nonzero_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
        
        return sampled_tokens
