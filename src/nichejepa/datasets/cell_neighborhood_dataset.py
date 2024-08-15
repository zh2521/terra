import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
from torch.utils.data import Dataset

_GLOBAL_SEED = 0
logger = getLogger()

def make_cell_neighborhood_dataset(
        batch_size,
        data,
        vocab_size,
        seq_len,
        mask_index=1,
        collator=None,
        pin_mem=True,
        num_workers=8,
        world_size=1,
        rank=0,
        root_path=None,
        gene_folder=None,
        training=True, 
        copy_data=False,
        drop_last=True,
        subset_file=None,
        just_cell = True,
        just_neighborhood = False,
        seq_len_cell=0,
        seq_len_neighborhood=0,
        has_cls = True,
        distributed= True):
       
      dataset = CellNeighborhoodDataset(data,
                                        vocab_size,
                                        mask_index,
                                        seq_len=seq_len,
                                        just_cell=just_cell,
                                        just_neighborhood=just_neighborhood,
                                        seq_len_cell = seq_len_cell,
                                        seq_len_neighborhood = seq_len_neighborhood,
                                        has_cls = has_cls)
      
      if distributed:
          dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset,
                                                                     num_replicas=world_size,
                                                                     rank=rank)

          data_loader = torch.utils.data.DataLoader(dataset,
                                                collate_fn=collator,
                                                sampler=dist_sampler,
                                                batch_size=batch_size,
                                                drop_last=drop_last,
                                                pin_memory=pin_mem,
                                                num_workers=num_workers,
                                                persistent_workers=False)
      
          logger.info('Gene unsupervised data loader created')
      
          return dataset, data_loader, dist_sampler
      else:
          data_loader = torch.utils.data.DataLoader(dataset,
                                                collate_fn=collator,
                                                batch_size=batch_size,
                                                pin_memory=pin_mem,
                                                num_workers=num_workers,
                                                persistent_workers=False)
          return dataset, data_loader
class CellNeighborhoodDataset(Dataset):
    def __init__(self,
                 data,
                 vocab_size,
                 mask_index,
                 seq_len,
                 just_cell=True,
                 just_neighborhood=False,
                 seq_len_cell=0,
                 seq_len_neighborhood=0,
                 has_cls = True):
        """
        CellNeighborhoodDataset.
        """
        self.dataset = data
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.seq_len = seq_len
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.just_cell = just_cell
        self.just_neighborhood = just_neighborhood
        self.has_cls = has_cls
    def __len__(self):
      
        return self.len
    def __getitem__(self, item):
      gene_tokens_cell = self.dataset[item]["gene_tokens_cell"][:self.seq_len_cell]
      gene_tokens_neighborhood = self.dataset[item]["gene_tokens_neighborhood"][:self.seq_len_neighborhood]

      if self.just_cell and self.just_neighborhood:
        tokens = (gene_tokens_cell + gene_tokens_neighborhood)
        niche_types = self.dataset[item]['niche_types']
        cell_types = self.dataset[item]['cell_types']
        
        if self.has_cls:
            tokens = [self.vocab_size] + tokens + [self.vocab_size + 1]
            labels = torch.cat((torch.ones(self.seq_len_cell + 1), torch.ones(self.seq_len_neighborhood + 1) * 2)).int()
        else:
            labels = torch.cat((torch.ones(self.seq_len_cell), torch.ones(self.seq_len_neighborhood) * 2)).int()
        return torch.tensor(tokens), labels, niche_types, cell_types

      if self.just_cell:
        tokens = gene_tokens_cell
        cell_types = self.dataset[item]['cell_types']
        
        if self.has_cls:
            tokens = [self.vocab_size] + tokens
            labels = torch.ones(self.seq_len_cell + 1).int()
        else:
            labels = torch.ones(self.seq_len_cell).int()
        return torch.tensor(tokens), labels, cell_types

      if self.just_neighborhood:
        tokens = gene_tokens_neighborhood
        niche_types = self.dataset[item]['niche_types']
        
        if self.has_cls:
            tokens += [self.vocab_size + 1]
            labels = (torch.ones(self.seq_len_neighborhood + 1) * 2).int()
        else: 
            labels = (torch.ones(self.seq_len_neighborhood) * 2).int()
        return torch.tensor(tokens), labels, niche_types

        

