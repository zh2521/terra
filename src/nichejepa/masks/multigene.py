"""
MaskCollator.

Adapted from Assran, M. et al. Self-supervised learning from images with a Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py (05.06.2024).
"""

from logging import getLogger
from multiprocessing import Value
from typing import Optional, Tuple

import torch

_GLOBAL_SEED = 0  # Global seed for reproducibility across processes
logger = getLogger()

class MaskCollator:
    """
    MaskCollator class for sampling target and context masks from cell and neighborhood segments.

    Parameters
    ----------
    n_targets: int, optional, default=2
        Number of target masks to sample for each input sequence.
    n_contexts: int, optional, default=1
        Number of context masks to sample for each input sequence.
    target_mask_size: int, optional, default=2
        The size (in number of tokens) of each target mask.
    context_mask_size: int, optional, default=10
        The size (in number of tokens) of each context mask.
    seq_len_cell: int, optional, default=0
        The length of the token sequence representing the cell segment.
    seq_len_neighborhood: int, optional, default=0
        The length of the token sequence representing the neighborhood segment.
    has_cls: bool, optional, default=False
        If True, the sequence contains a <cls> token at the 0th index, which will be included in the masks.
    just_cell: bool, optional, default=True
        If True, the sequence contains cell segment
    just_neighborhood: bool, optional, default=False
        If True, the sequence contains neighborhood segment
    """
    def __init__(self,
                 n_targets: int = 2,
                 n_contexts: int = 1,
                 target_mask_size: int = 2,
                 context_mask_size: int = 10,
                 seq_len_cell: int = 0,
                 seq_len_neighborhood: int = 0,
                 just_cell: bool = True,
                 just_neighborhood: bool = False,
                 has_cls: bool = False):
        self.seq_len_cell = seq_len_cell
        if just_neighborhood:
           self.seq_len_neighborhood = seq_len_neighborhood
        else:
           self.seq_len_neighborhood = 0
        self.seq_len = self.seq_len_cell + self.seq_len_neighborhood
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.target_mask_size = target_mask_size
        self.context_mask_size = context_mask_size
        self.has_cls = has_cls
        # Shared counter to manage iterations across multiple processes
        self._itr_counter = Value('i', -1)

    def step(self):
        """ 
        Increment the iteration counter. 
        
        This is used to ensure that each worker process generates a unique seed for random sampling.
        """
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_gene_mask(self,
                          non_zero_seq_len_cell: int,
                          non_zero_seq_len_neighborhood: int,
                          generator,
                          mask_size: int,
                          mask_type =None,
                          valid_token_masks: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample context or target gene masks, considering both cell and neighborhood segments.

        Parameters
        ----------
        non_zero_seq_len_cell: int
            Number of non-zero tokens in the cell segment.
        non_zero_seq_len_neighborhood: int
            Number of non-zero tokens in the neighborhood segment.
        generator: torch.Generator
            Pseudorandom number generator to ensure reproducibility.
        mask_size: int
            Length of the masked token sequence.
        valid_token_masks: Optional[List[torch.Tensor]]
            A list of binary masks that constrain the valid token positions.

        Returns
        ----------
        mask: torch.Tensor
            Binary tensor with 1s for sampled tokens and 0s otherwise.
        mask_complement: torch.Tensor
            Binary tensor complementing the mask, with 0s for sampled tokens and 1s otherwise.
        """

        # Determine the valid minimum start position based on the presence of a CLS token
        valid_min_start = 1 if self.has_cls else 0

        if mask_type == 'target':
            mask_size = min(non_zero_seq_len_cell + non_zero_seq_len_neighborhood, mask_size)
            # Sample the start position for the mask within the valid range using the provided generator for reproducibility
            start = torch.randint(valid_min_start,
                                  valid_min_start + non_zero_seq_len_cell + non_zero_seq_len_neighborhood - mask_size + 1,
                                  generator=generator,
                                  size=(1,))

            # Initialize the mask and its complement with zeros and ones, respectively
            mask = torch.zeros(self.seq_len_cell + self.seq_len_neighborhood, dtype=torch.int32)
            mask_complement = torch.ones_like(mask)

            # Apply the mask within the cell segment if the start is within the cell
            if start < non_zero_seq_len_cell:
                # Determine the end of the mask within the cell
                cell_end = min(start + mask_size, non_zero_seq_len_cell)
                mask[start:cell_end] = 1
                mask_complement[start:cell_end] = 0

                # Handle overflow into the neighborhood segment if the mask exceeds the cell length
                if cell_end < start + mask_size:
                    overflow = start + mask_size - non_zero_seq_len_cell
                    mask[self.seq_len_cell + valid_min_start :self.seq_len_cell + valid_min_start + overflow] = 1
                    mask_complement[self.seq_len_cell + valid_min_start:self.seq_len_cell + valid_min_start + overflow] = 0
            else:
                # Apply the mask entirely within the neighborhood segment
                neighborhood_start = start - non_zero_seq_len_cell
                neighborhood_end = neighborhood_start + mask_size
                mask[self.seq_len_cell + valid_min_start + neighborhood_start:self.seq_len_cell + valid_min_start + neighborhood_end] = 1
                mask_complement[self.seq_len_cell + valid_min_start + neighborhood_start:self.seq_len_cell + valid_min_start + neighborhood_end] = 0

        elif mask_type == 'context':
            # Sample the start position for the context mask
            start = torch.randint(valid_min_start,
                                  valid_min_start + self.seq_len_neighborhood + self.seq_len_cell - mask_size + 1,
                                  generator=generator,
                                  size=(1,))

            # Initialize the mask and its complement with zeros and ones, respectively
            mask = torch.zeros(self.seq_len_cell + self.seq_len_neighborhood, dtype=torch.int32)
            mask_complement = torch.ones_like(mask)

            # Apply the mask starting from the sampled position
            mask[start:mask_size + start] = 1
            mask_complement[start:mask_size + start] = 0

        # Include the CLS token in the mask if applicable
        if self.has_cls:
            mask[0] = 1

        # Constrain the mask to valid token positions if provided
        if valid_token_masks is not None:
            for valid_mask in valid_token_masks:
                mask *= valid_mask

        # Convert the mask to a tensor of indices where the mask is applied
        mask = torch.nonzero(mask).squeeze()

        return mask, mask_complement

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create context and target masks when collating cell neighborhoods into a batch.
        # 1. Sample several target masks.
        # 2. Sample non-overlapping context masks.
        # 3. Add <cls> token to both context and target masks if applicable.
        # 4. Return context and target masks.

        Parameters
        ----------
        batch:
            Tuple containing the input sequence tokens, segment labels, and cell/niche-level labels for all observations in the batch.

        Returns
        ----------
        collated_batch:
            Input gene tokens, segment labels, and cell/niche-level labels collated by batch.
        collated_masks_context:
            Sampled context masks collated by batch.
        collated_masks_target:
            Sampled target masks collated by batch.
        """

        # Number of observations in the batch
        B = len(batch)
        # Collate the batch using the default PyTorch collate function
        collated_batch = torch.utils.data.default_collate(batch)

        # Initialize lists to hold the masks for each observation in the batch
        collated_masks_target, collated_masks_context = [], []

        # Initialize variables to track the minimum length of masks across the batch
        keep_tokens_target = self.seq_len
        keep_tokens_context = self.seq_len

        # Create a pseudorandom number generator for sampling
        seed = self.step()  # Get a unique seed for this iteration
        g = torch.Generator()
        g.manual_seed(seed)  # Ensure reproducibility by setting the seed

        # Iterate over each observation in the batch
        for i in range(B):
            # Initialize lists to store target masks and their complements for the current observation
            masks_target_complement = []
            masks_target = []
            masks_context = []

            # Calculate the number of non-zero tokens in the cell segment
            non_zero_seq_len_cell = torch.nonzero(batch[i][0][:self.seq_len_cell]).size(0)

            # Calculate the number of non-zero tokens in the neighborhood segment (if any)
            if self.seq_len_neighborhood != 0:
                non_zero_seq_len_neighborhood = torch.nonzero(batch[i][0][self.seq_len_cell:]).size(0)
            else:
                non_zero_seq_len_neighborhood = 0
            # Sample target masks for the current observation
            for _ in range(self.n_targets):
                mask_target, mask_target_complement = self._sample_gene_mask(
                    non_zero_seq_len_cell=non_zero_seq_len_cell,
                    non_zero_seq_len_neighborhood=non_zero_seq_len_neighborhood,
                    generator=g,  # Use the generator for reproducibility
                    mask_size=self.target_mask_size,
                    mask_type ='target'
                )
                masks_target.append(mask_target)
                masks_target_complement.append(mask_target_complement)
                keep_tokens_target = min(keep_tokens_target, len(mask_target))

            # Sample context masks for the current observation, ensuring they do not overlap with target masks
            for _ in range(self.n_contexts):
                mask_context, _ = self._sample_gene_mask(
                    non_zero_seq_len_cell=non_zero_seq_len_cell,
                    non_zero_seq_len_neighborhood=non_zero_seq_len_neighborhood,
                    mask_size=self.context_mask_size,
                    generator=g,  # Use the same generator to ensure reproducibility
                    valid_token_masks=masks_target_complement,
                    mask_type='context'
                )
                masks_context.append(mask_context)
                keep_tokens_context = min(keep_tokens_context, len(mask_context))

            # Append the masks for the current observation to the collated lists
            collated_masks_target.append(masks_target)
            collated_masks_context.append(masks_context)

        # Trim the masks to the minimum size across the batch and collate them
        collated_masks_target = [[cm[:keep_tokens_target] for cm in cm_list] for cm_list in collated_masks_target]
        collated_masks_target = torch.utils.data.default_collate(collated_masks_target)
        collated_masks_context = [[cm[:keep_tokens_context] for cm in cm_list] for cm_list in collated_masks_context]
        collated_masks_context = torch.utils.data.default_collate(collated_masks_context)

        # Return the collated batch, context masks, and target masks
        return collated_batch, collated_masks_context, collated_masks_target

