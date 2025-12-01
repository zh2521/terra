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
        exclude_cols = [
            'rel_x_coord',
            'rel_y_coord',
            'gene_panel_value',
            'assay_value',
            'species_value',
            'tissue_value']
        cols = [
            c for c in dataset.column_names if c not in exclude_cols and (
                c != 'cell_id' or include_cell_id)]
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

    def _add_special_seq(self,
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
        # Add special tokens other than <cls> token
        for spc_tk in self.special_tokens:
            if 'cls' not in spc_tk:
                if self.gt_type == 'rank':
                    item_dict['tokens'] = torch.cat(
                        [item[f'{spc_tk}_value_token'],
                        item_dict['tokens']])
                else:
                    item_dict['tokens'] = torch.cat(
                        [item[f'{spc_tk}_token'],
                        item_dict['tokens']])
                    item_dict['values'] = torch.cat(
                        [item[f'{spc_tk}_value'],
                        item_dict['values']])

        # Add <cls> token 
        n_cls_tokens = item["cls_tokens"].shape[0]
        item_dict['tokens'] = torch.cat(
            [item['cls_tokens'], item_dict['tokens']])
        item_dict['segments'] = torch.cat([
            torch.arange(1, 1 + self.n_special_tokens), item_dict['segments']])
        if self.gt_type != 'counts':
            item_dict['positions'] = torch.cat([
                torch.arange(1, 1 + self.n_special_tokens),
                item_dict['positions']
            ])
        if self.gt_type != 'rank':
            item_dict['values'] = torch.cat([
                torch.arange(2, 2 + n_cls_tokens), item_dict['values']])

        #item_dict['rel_x_coords'] = torch.cat(
        #    [torch.full((self.n_special_tokens,),
        #        float('-inf'), dtype=torch.float),
        #        item_dict['rel_x_coords']])   
        #item_dict['rel_y_coords'] = torch.cat(
        #    [torch.full((self.n_special_tokens,),
        #        float('-inf'), dtype=torch.float),
        #        item_dict['rel_y_coords']])

        return item_dict

    def _sample_seq(self,
                    tokens: List,
                    values: List,
                    #rel_x_coords: list[float] | None,
                    #rel_y_coords: list[float] | None,
                    n_nonzero_tokens: int,
                    size: int,
                    ) -> tuple[list[int]]: # TODO update with tensor logic
        """
        Sample a subset of gene tokens and corresponding values based on
        a sampling strategy.

        Parameters
        -----------
        tokens:
            List of tokens.
        values:
            List of values.
        rel_x_coords:
            List of relative x coordinates.
        rel_y_coords:
            List of relative y coordinates.
        n_nz_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        sampled_values:
            List of (corresponding) sampled values.
        """
        if 'norm_value_rank_sampling' in self.sampling_strategy:
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
        sampled_values = [values[i] for i in sampled_indices]

        if size > n_nonzero_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            sampled_values.extend([0.0] * (size - len(sampled_values)))

        return sampled_tokens, sampled_values
         
    def _get_segment_seq(self, 
                         item: int,
                         segment: int,
                         segment_seq_len: int,
                         ) -> tuple[list[int], list[float]]:
            """
            Get gene tokens and values for a given segment based on a
            sampling strategy.

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
            segment_values:
                List of values for a given segment.
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
            if self.gt_type != 'rank':
                segment_values = item["gene_expr"][
                    segment_start_idx: segment_end_idx]
            else:
                segment_values = None
            #segment_rel_x_coords = item['rel_x_coord'][
            #    segment_start_idx: segment_end_idx]
            #segment_rel_y_coords = item['rel_y_coord'][
            #    segment_start_idx: segment_end_idx]
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
                if self.gt_type != 'rank':
                    segment_values = segment_values[:segment_seq_len]
            # Otherwise, sample a subset of tokens based on the sampling
            # strategy
            else:
                segment_n_nonzero_tokens = int(
                    torch.count_nonzero(segment_tokens))

                segment_tokens, segment_values = self._sample_seq(
                    tokens=segment_tokens,
                    values=segment_values,
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

        # Get (sampled) gene tokens, positions, segments, and values for
        # index cell segment
        item_dict['tokens'], item_dict['values'] = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens, # index cell seg
            segment_seq_len=self.seq_len_cell)

        if self.gt_type == 'rank':
            del(item_dict['values'])

        segment_token_zero_mask = item_dict['tokens'].eq(0)

        item_dict['segments'] = torch.ones_like(
            item_dict['tokens']) * self.max_special_tokens
        item_dict['segments'][segment_token_zero_mask] = torch.tensor(
            0, dtype=torch.long)
        if self.gt_type != 'counts':
            item_dict['positions'] = torch.arange(
                1, item_dict['tokens'].shape[0] + 1, dtype=torch.long)
            item_dict['positions'][segment_token_zero_mask] = torch.tensor(
                0, dtype=torch.long)
        #item_dict['rel_x_coords'][segment_token_zero_mask] = torch.tensor(
        #    float('-inf'), dtype=torch.float)
        #item_dict['rel_y_coords'][segment_token_zero_mask] = torch.tensor(
        #    float('-inf'), dtype=torch.float)

        # Get (sampled) gene tokens, positions, segments and values for
        # neighbor cell segments
        for segment in torch.unique(item['seg_tokens']):
            if segment.item() > self.max_special_tokens: # neighbor cell segments
                segment_tokens, segment_values = self._get_segment_seq(
                    item=item,
                    segment=segment, # neighbor cell segs
                    segment_seq_len=self.seq_len_cell)

                segment_zero_mask = segment_tokens.eq(0)

                segment_tensor = torch.where(
                    segment_tokens != 0,
                    segment,
                    torch.tensor(0, dtype=torch.long)).to(dtype=torch.long)
                item_dict['segments'] = torch.cat(
                    [item_dict['segments'], segment_tensor], dim=0)
                item_dict['tokens'] = torch.cat(
                    [item_dict['tokens'], segment_tokens], dim=0)
                if self.gt_type != 'counts':
                    segment_pos = torch.arange(
                        1, segment_tokens.size(0) + 1, dtype=torch.long)
                    segment_pos[segment_zero_mask] = torch.tensor(
                        0, dtype=torch.long)
                    item_dict['positions'] = torch.cat(
                        [item_dict['positions'], segment_pos], dim=0)
                if self.gt_type != 'rank':
                    item_dict['values'] = torch.cat(
                        [item_dict['values'], segment_values], dim=0)
                #segment_rel_x_coords[segment_zero_mask] = torch.tensor(
                #    float('-inf'), dtype=torch.float)
                #segment_rel_y_coords[segment_zero_mask] = torch.tensor(
                #    float('-inf'), dtype=torch.float)
                #item_dict['rel_x_coords'] = torch.cat(
                #[item_dict['rel_x_coords'], segment_rel_x_coords], dim=0)
                #item_dict['rel_y_coords'] = torch.cat(
                #[item_dict['rel_y_coords'], segment_rel_y_coords], dim=0)

        current_len = item_dict['tokens'].shape[0]
        target_len = self.seq_len_cell + self.seq_len_neighborhood

        if current_len > target_len:
            # Truncate
            item_dict['tokens'] = item_dict['tokens'][:target_len]
            item_dict['segments'] = item_dict['segments'][:target_len]
            if self.gt_type != 'counts':
                item_dict['positions'] = item_dict['positions'][:target_len]
            if self.gt_type != 'rank':
                item_dict['values'] = item_dict['values'][:target_len]
            #item_dict['rel_x_coords'] = item_dict['rel_x_coords'][
            #    :target_len]
            #item_dict['rel_y_coords'] = item_dict['rel_y_coords'][
            #    :target_len]
        elif current_len < target_len:
            # Add padding
            pad_len = target_len - current_len
            item_dict['tokens'] = torch.cat([
                    item_dict['tokens'],
                    torch.zeros(pad_len, dtype=item_dict['tokens'].dtype)
                ])
            item_dict['segments'] = torch.cat([
                item_dict['segments'],
                torch.zeros(pad_len, dtype=item_dict['segments'].dtype)
            ])
            if self.gt_type != 'counts':
                item_dict['positions'] = torch.cat([
                    item_dict['positions'],
                    torch.zeros(pad_len, dtype=item_dict['positions'].dtype)
                ])
            if self.gt_type != 'rank':
                item_dict['values'] = torch.cat([
                    item_dict['values'],
                    torch.zeros(pad_len, dtype=item_dict['values'].dtype)
                ])

            #item_dict['rel_x_coords'] = torch.cat([
            #    item_dict['rel_x_coords'],
            #    torch.zeros(pad_len, dtype=item_dict['rel_x_coords'].dtype)
            #])
            #item_dict['rel_y_coords'] = torch.cat([
            #    item_dict['rel_y_coords'],
            #    torch.zeros(pad_len, dtype=item_dict['rel_y_coords'].dtype)
            #])

        # Add special tokens
        item_dict = self._add_special_seq(
            item=item,
            item_dict=item_dict)

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

        # Get (sampled) gene tokens and values
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
        tokens, segments, positions, values = self._add_special_seq(
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