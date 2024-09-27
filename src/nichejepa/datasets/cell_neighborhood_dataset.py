import os
import subprocess
import time
from logging import getLogger
from typing import Optional, Tuple, List, Literal

import datasets
import numpy as np
import torch
from .cell_base_dataset import CellBaseDataset
from ..utils.distributed import CustomDistributedLengthGroupedSampler


_GLOBAL_SEED = 0
logger = getLogger()


class CellNeighborhoodDataset(CellBaseDataset):
    def __init__(self,
                 dataset: datasets.Dataset,
                 vocab_size: int,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 special_tokens: list=[
                    'cls', 'assay', 'species', 'tissue', 'gene_panel', 'batch'],
                 sampling_strategy: Optional[str]=None,
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        dataset:
            Huggingface dataset with gene and special tokens.
        vocab_size:
            Size of the vocabulary.
        seq_len_cell:
            Sequence length of the cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        special_tokens:
            Special tokens to be included in the sequence.
        sampling_strategy:
            Token sampling strategy.
        """
        self.dataset = dataset
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len = seq_len_cell + seq_len_neighborhood
        self.special_tokens = special_tokens
        self.sampling_strategy = sampling_strategy
         
    def __getitem__(self, item):
        # Get (sampled) gene tokens
        gene_tokens_cell = self._get_gene_tokens_for_segment(
            item=item,
            segment_idx=1, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neighborhood = self._get_gene_tokens_for_segment(
            item=item,
            segment_idx=2, # neighborhood seg
            segment_seq_len=self.seq_len_neighborhood)
        seq_tokens = gene_tokens_cell + gene_tokens_neighborhood

        # Add special tokens to sequence and update segment tokens
        if 'batch' in special_tokens:
            seq_tokens = self.dataset[item]["batch_token"] + tokens
        if 'gene_panel' in special_tokens:
            seq_tokens = self.dataset[item]["gene_panel_token"] + tokens
        if 'tissue' in special_tokens:
            seq_tokens = self.dataset[item]["tissue_token"] + tokens
        if 'species' in special_tokens:
            seq_tokens = self.dataset[item]["species_token"] + tokens
        if 'assay' in special_tokens:
            seq_tokens = self.dataset[item]["assay_token"] + tokens
        if 'cls' in special_tokens:
            seq_tokens = self.dataset[item]["cls_tokens"] + tokens
        seq_tokens = torch.tensor(seq_tokens)
        n_special_tokens = len(special_tokens)
        seg_tokens = torch.cat(
            (torch.zeros(self.n_special_tokens),
             torch.tensor(self.dataset[item]["seg_tokens"])))

        return seq_tokens, seg_tokens


def make_cell_neighborhood_dataset(batch_size: int,
                                   data: datasets.Dataset,
                                   vocab_size: int,
                                   collator=None,
                                   pin_mem: bool=True,
                                   num_workers: int=8,
                                   world_size: int=1,
                                   rank: int=0,
                                   drop_last: bool=True,
                                   seq_len_cell: int=0,
                                   seq_len_neighborhood: int=0,
                                   special_tokens: list
                                   distributed: bool=True,
                                   sampling_strategy: Optional[Literal[
                                       'norm_count_rank_sampling',
                                       'norm_count_rank_sampling_rep',
                                       'rand_sampling',
                                       'rand_sampling_rep']]=None,
    ) -> Tuple[CellNeighborhoodDataset,
               torch.utils.data.DataLoader,
               Optional[torch.utils.data.distributed.DistributedSampler]]:
    """
    Convert huggingface Dataset into a torch CellNeighborhoodDataset object and
    create corresponding data loader.

    Parameters
    -----------
    batch_size:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    data:
        Huggingface dataset with cell and neighborhood tokens and cell-level
        labels.
    vocab_size:
        Size of the vocabulary.
    collator:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    pin_mem:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    num_workers:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    world_size:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.
    rank:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.
    drop_last:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    seq_len_cell:
        Sequence length of the cell tokens.
    seq_len_neighborhood:
        Sequence length of the neighborhood tokens.
    special_tokens:
    distributed:
        If 'True', use distributed mode.

    Returns
    --------
    dataset:
        Torch CellNeighborhoodDataset.
    data_loader:
        Torch data loader based on CellNeighborhoodDataset.
    dist_sampler:
        Torch distributed sampler based on CellNeighborhoodDataset.
    """
    dataset = CellNeighborhoodDataset(data,
                                      vocab_size,
                                      seq_len_cell=seq_len_cell,
                                      seq_len_neighborhood=seq_len_neighborhood,
                                      special_tokens=special_tokens,
                                      sampling_strategy=sampling_strategy)
    
    if distributed:
        dist_sampler = CustomDistributedLengthGroupedSampler(
            dataset,
            batch_size,
            hugging_face_dataset=data,
            num_replicas=world_size,
            rank=rank,
            seed=_GLOBAL_SEED)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=collator,
                                                  sampler=dist_sampler,
                                                  batch_size=batch_size,
                                                  drop_last=drop_last,
                                                  pin_memory=pin_mem,
                                                  num_workers=num_workers,
                                                  persistent_workers=False)
        logger.info('Data loader created.')

        return dataset, data_loader, dist_sampler
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=collator,
                                                  batch_size=batch_size,
                                                  drop_last=drop_last,
                                                  pin_memory=pin_mem,
                                                  num_workers=num_workers,
                                                  persistent_workers=False)
        logger.info('Data loader created.')
        
        return dataset, data_loader
