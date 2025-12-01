from typing import List, Literal, Optional, Tuple, Union

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


class CellBaseDataset(Dataset):
    def __init__(self,
                 dataset: datasets.Dataset,
                 vocab_size: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 max_cls_tokens: int,
                 max_special_tokens: int,
                 tokenizer_type: Literal['cell_neighborhood', 'cell_graph'],
                 gt_type: Literal['rank', 'counts'],
                 special_tokens: List=[
                    'cls_cell',
                    'cls_neighborhood',
                    'assay',
                    'species',
                    'tissue',
                    'gene_panel',
                    'batch'],
                 sampling_strategy: Optional[
                    Literal['norm_count_rank_sampling',
                            'norm_count_rank_sampling_rep',
                            'rand_sampling',
                            'rand_sampling_rep']]=None,
                 n_nonzero_tokens_list: list[int] | None = None,
                 include_cell_id: bool = False,
                 ):
        """
        Torch CellBaseDataset class.

        Parameters
        -----------
        dataset:
            Hugging Face dataset with sequences of gene tokens and special
            tokens.
        vocab_size:
            Size of the vocabulary.
        seq_len_cell:
            Sequence length of the (index) cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        max_cls_tokens:
        max_special_tokens:
        tokenizer_type;
        gt_type:
            Gene transformer type.
        special_tokens:
            Special tokens to be included in the sequence processed by the
            model.
        sampling_strategy:
            Token sampling strategy.
        """
        if gt_type not in ['rank', 'counts']:
            raise ValueError(f'Invalid "gt_type": {gt_type}.')

        # Format dataset
        cols = [
            c for c in dataset.column_names if c != 'cell_id' or include_cell_id]
        dataset.set_format(
            type="torch", columns=cols, output_all_columns=False)
        self.dataset = dataset

        self.len = len(self.dataset)
        if n_nonzero_tokens_list:
            self.n_nonzero_tokens = n_nonzero_tokens_list
        else:
            self.n_nonzero_tokens = list(self.dataset['n_nonzero_tokens'])
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.max_cls_tokens = max_cls_tokens
        self.max_special_tokens = max_special_tokens
        self.gt_type = gt_type
        self.n_special_tokens = len(special_tokens)
        self.seq_len = (seq_len_cell +
                        seq_len_neighborhood +
                        self.n_special_tokens)
        self.n_segments = (seq_len_cell + seq_len_neighborhood) / seq_len_cell
        self.special_tokens = special_tokens
        self.sampling_strategy = sampling_strategy
        self.include_cell_id = include_cell_id

    def __len__(self) -> int:
        return self.len

    def _add_special_tokens_to_seq(self,
                                   item: int,
                                   item_dict: dict,
                                   ) -> Tuple[List[int], List[int]]:
        """
        Add special tokens to sequence and update segment and positions tokens.

        Parameters
        -----------
        tokens:
            Token sequence including all segments.
        segments:
            Segment tokens including all segments.
        positions:
            Position tokens including all segments.
        values:
            Gene expression values including all segments.
        item:
            Index of the cell in the Hugging Face dataset.

        Returns
        -----------
        tokens:
            Sequence of tokens with special tokens included at sequence start.
        segments:
            Segment labels with 0s for special tokens at sequence start.
        """
        for spc_tk in self.special_tokens:
            if 'cls' not in spc_tk:
                if self.gt_type == 'rank':
                    item_dict['tokens'] = torch.cat(
                        [item[f'{spc_tk}_value_token'],
                        item_dict['tokens']])
                elif self.gt_type == 'counts':
                    item_dict['tokens'] = torch.cat(
                        [item[f'{spc_tk}_token'],
                        item_dict['tokens']])
                item_dict['values'] = torch.cat(
                    [item[f'{spc_tk}_value'],
                    item_dict['values']])
                
        n_cls_tokens = item["cls_tokens"].shape[0]
        item_dict['tokens'] = torch.cat(
            [item['cls_tokens'], item_dict['tokens']])
        item_dict['values'] = torch.cat([
            torch.arange(2, 2 + n_cls_tokens), item_dict['values']])
        item_dict['segments'] = torch.cat([
            torch.arange(1, 1 + self.n_special_tokens), item_dict['segments']])
        if self.gt_type != 'counts':
            item_dict['positions'] = torch.cat([
                torch.arange(1, 1 + self.n_special_tokens),
                item_dict['positions']
            ])

        return item_dict

    def _sample_seq(self,
                    tokens: List,
                    counts: List,
                    n_nonzero_tokens: int,
                    size: int,
                    ) -> List[int]:
        """
        Sample a subset of tokens based on a sampling strategy.

        Parameters
        -----------
        tokens:
            List of tokens.
        counts:
            List of counts.
        n_nonzero_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        sampled_counts:
            List of (corresponding) sampled counts.
        """
        if 'norm_count_rank_sampling' in self.sampling_strategy:
            # Calculate weights based on rank and number of nonzero tokens:
            # the higher the rank, the higher the weight
            # seq = [4, 1, 3, 2, 5, 0, 0, 0]
            # n_nonzero_tokens = 5  
            # sum_rank = 5 * (5 + 1) / 2.0 = 15.0
            # weights = [(n_nonzero_tokens - i)/sum_rank for i in range(
            #     n_nonzero_tokens)] 
            # = [0.333, 0.266, 0.2, 0.133, 0.066]
            # np.sum(weights) = 1.0
            sum_rank = (
                n_nonzero_tokens * (n_nonzero_tokens + 1) / 2.0) + 1e-9
            weights = [(n_nonzero_tokens - i)/sum_rank for i in range(
                n_nonzero_tokens)]
            assert np.isclose(np.sum(weights), 1.0)
        elif 'rand_sampling' in self.sampling_strategy:
            weights = np.ones(n_nonzero_tokens) / n_nonzero_tokens
        else:
            raise ValueError(f"'{self.sampling_strategy}' is invalid.")
            
        # Sample token indices based on weights
        sampled_indices = np.random.choice(
            np.arange(n_nonzero_tokens),
            size=min(size, n_nonzero_tokens),
            p=weights,
            replace=(True if 'rep' in self.sampling_strategy else False))
            
        # Sort sampled indices to preserve rank order
        sampled_indices = np.sort(sampled_indices)
        sampled_tokens = [tokens[i] for i in sampled_indices]
        sampled_counts = [counts[i] for i in sampled_indices]

        if size > n_nonzero_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            sampled_counts.extend([0.0] * (size - len(sampled_counts)))

        return sampled_tokens, sampled_counts
         
    def _get_segment_seq(self, 
                         item: int,
                         segment: int,
                         segment_seq_len: int,
                         ) -> tuple[list[int], list[float]]:
            """
            Get gene tokens and counts for a given segment based on a sampling
            strategy.

            Parameters
            -----------
            item:
                Index of the cell in the Hugging Face dataset.
            segment:
                Index of the segment for which tokens are retrieved.
            segment_seq_len:
                Desired length of the segment token sequence.

            Returns
            --------
            segment_tokens:
                List of tokens for a given segment.
            """
            # Only keep gene tokens and gene expr in specified segment
            segment_indices = torch.where(item["seg_tokens"] == segment)[0]
            segment_start_idx = segment_indices[0].item()
            next_segment_indices = torch.where(
                item["seg_tokens"] == segment + 1)[0]
            if len(next_segment_indices) > 0:
                segment_end_idx = next_segment_indices[0].item()
            else:
                segment_end_idx = item["seg_tokens"].shape[0]
            
            segment_tokens = item["gene_tokens"][
                segment_start_idx: segment_end_idx]
            segment_values = item["gene_expr"][
                segment_start_idx: segment_end_idx]

            # Validate that segment sequence length is specified correctly
            if (self.sampling_strategy is not None and 'rep' in
            self.sampling_strategy):
                pass
            else:
                if segment_seq_len > len(segment_tokens):
                    raise ValueError(
                        'Sequence length for a given segment cannot be larger '
                        'than segment size when not sampling with replacement.')

            # If no sampling strategy is specified, use all tokens up to
            # specified length
            if self.sampling_strategy is None:
                segment_tokens = segment_tokens[:segment_seq_len]
                segment_values = segment_values[:segment_seq_len]
            # Otherwise, sample a subset of tokens based on the sampling
            # strategy
            else:
                segment_n_nonzero_tokens = sum(
                    1 for token in segment_tokens if token != 0)

                segment_tokens, segment_values = self._sample_seq(
                    tokens=segment_tokens,
                    counts=segment_values,
                    n_nonzero_tokens=segment_n_nonzero_tokens,
                    size=segment_seq_len)       
                    
            return segment_tokens, segment_values


class CellGraphDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs,
                 ):
        """
        Torch CellGraphDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of the CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)
         
    def __getitem__(self,
                    item: int
                    ) -> Tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               List[int]]:
        item_dict = {}

        # Retrieve Huggingface item once
        item = self.dataset[item]

        # Add <cls> and special tokens
        item["cls_tokens"] = torch.arange(2, 2 + self.max_cls_tokens)
        #item["tissue_token"] = torch.tensor([103])
        #item["assay_token"] = torch.tensor([104])
        #item["gene_panel_token"] = torch.tensor([105])
        item["batch_token"] = torch.tensor([106])

        """
        item['rel_x_coord'] = torch.repeat_interleave(
            item['rel_x_coord'], self.seq_len_cell)
        item['rel_y_coord'] = torch.repeat_interleave(
            item['rel_y_coord'], self.seq_len_cell)
        """

        seg_tokens = torch.arange(105, 105 + self.n_segments)
        seg_tokens = seg_tokens.repeat_interleave(self.seq_len_cell)
        mask = (item["gene_tokens"] != 0)
        item["seg_tokens"] = seg_tokens * mask

        # Get (sampled) gene tokens and counts
        gene_tokens_cell, values_cell = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens, # index cell seg
            segment_seq_len=self.seq_len_cell)
        item_dict['segments'] = (
            gene_tokens_cell != 0).long() * self.max_special_tokens
        if self.gt_type != 'counts':
            item_dict['positions'] = torch.arange(
                1, gene_tokens_cell.shape[0]) + 1)
        gene_tokens_neighborhood = []
        values_neighborhood = []
        for segment in torch.unique(item['seg_tokens']):
            if segment.item() > self.max_special_tokens: # neighbor cell segments
                segment_tokens, segment_values = self._get_segment_seq(
                    item=item,
                    segment=segment, # neighbor cell segs
                    segment_seq_len=self.seq_len_cell)
                gene_tokens_neighborhood.extend(segment_tokens)
                values_neighborhood.extend(segment_values)
                item_dict['segments'].extend(
                    [segment if gene_token != 0 else 0 for
                    gene_token in segment_tokens])
                if self.gt_type != 'counts':
                    item_dict['positions'].extend(
                        list(range(1, len(segment_tokens) + 1)))
        item_dict['tokens'] = gene_tokens_cell + gene_tokens_neighborhood
        if self.gt_type != 'counts':
            item_dict['positions'] = [
                position if item_dict['tokens'][i] != 0 else 0 for i, position in
                enumerate(item_dict['positions'])]
        item_dict['values'] = values_cell + values_neighborhood

        current_len = len(item_dict['tokens'])
        target_len = self.seq_len_cell + self.seq_len_neighborhood

        if current_len > target_len:
            # Truncate
            item_dict['tokens'] = item_dict['tokens'][:target_len]
            item_dict['segments'] = item_dict['segments'][:target_len]
            if self.gt_type != 'counts':
                item_dict['positions'] = item_dict['positions'][:target_len]
            item_dict['values'] = item_dict['values'][:target_len]
        elif current_len < target_len:
            # Add padding
            item_dict['tokens'] += [0] * (target_len - current_len)
            item_dict['segments'] += [0] * (target_len - current_len)
            if self.gt_type != 'counts':
                item_dict['positions'] += [0] * (target_len - current_len)
            item_dict['values'] += [0.0] * (target_len - current_len)

        # Add special tokens
        item_dict = self._add_special_tokens_to_seq(
            item=item,
            item_dict=item_dict)

        item_dict['tokens'] = torch.tensor(item_dict['tokens'])
        item_dict['segments'] = torch.tensor(item_dict['segments']).long()
        if self.gt_type != 'counts':
            item_dict['positions'] = torch.tensor(item_dict['positions'])
        item_dict['values'] = torch.tensor(item_dict['values'])

        # Add cell ID
        if self.include_cell_id:
            item_dict['cell_id'] = item['cell_id']

        return item_dict


class CellNeighborhoodDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of the CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)

    def __getitem__(self,
                    item: int
                    ) -> Tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               List[int]]:
        # Retrieve Huggingface item once
        item = self.dataset[item]

        # Get (sampled) gene tokens and counts
        gene_tokens_cell, values_cell = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neighborhood, values_neighborhood = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens + 1, # neighborhood seg
            segment_seq_len=self.seq_len_neighborhood)

        tokens = gene_tokens_cell + gene_tokens_neighborhood
        values = values_cell + values_neighborhood
        segments = [
            self.max_special_tokens if gene_token != 0 else 0 for gene_token
            in gene_tokens_cell
            ] + [
            self.max_special_tokens + 1 if gene_token != 0 else 0 for
            gene_token in gene_tokens_neighborhood]
        if self.gt_type != 'counts':
            positions = list(range(1, len(gene_tokens_cell) + 1)) + list(
                range(1, len(gene_tokens_neighborhood) + 1))
            positions = [position if tokens[i] != 0 else 0 for i, position in 
                        enumerate(positions)]

        # Add special tokens
        tokens, segments, positions, values = self._add_special_tokens_to_seq(
            tokens=tokens,
            segments=segments,
            positions=positions,
            values=values,
            item=item)

        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        if self.gt_type != 'counts':
            positions = torch.tensor(positions)
        else:
            positions = None
        values = torch.tensor(values)

        return tokens, segments, positions, values, item["cell_id"]


def make_cell_dataset(tokenizer_type: Literal['cell_graph',
                                              'cell_neighborhood'],
                      **cell_dataset_kwargs
                      ) -> Union[CellGraphDataset,
                                 CellNeighborhoodDataset]:
    """
    Based on tokenizer type, return CellGraphDataset or CellNeighborhoodDataset.
    """
    if tokenizer_type == 'cell_graph':
        cell_dataset = CellGraphDataset(tokenizer_type=tokenizer_type,
                                        **cell_dataset_kwargs)
    elif tokenizer_type  == 'cell_neighborhood':
        cell_dataset = CellNeighborhoodDataset(tokenizer_type=tokenizer_type,
                                               **cell_dataset_kwargs)

    return cell_dataset