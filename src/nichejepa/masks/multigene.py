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


_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator():
    """
    MaskCollator.
    """
    def __init__(self,
                 seq_len: int,
                 n_targets: int=2,
                 target_mask_size: int=2,
                 context_mask_size: int=10,
                 n_contexts: int=1,
                 has_cls: bool=False):
        self.seq_len = seq_len
        self.n_targets = n_targets
        self.target_mask_size = target_mask_size
        self.context_mask_size = context_mask_size
        self.n_contexts = n_contexts
        self.has_cls = has_cls
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def _sample_gene_mask(self,
                          non_zero_seq_len: int,
                          mask_size: int,
                          valid_token_masks: Optional[list]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample context or target gene masks.

        Parameters
        ----------
        non_zero_seq_len:
            Length of token sequence without padding tokens.
        mask_size:
            Size of mask in tokens.
        valid_token_masks:
            List of token masks that indicate which tokens are valid for masking.
            Used to only keep tokens in sampled context mask if they are not part of target masks.
    
        Returns
        ----------
        mask:
            Binary tensor with 1s for sampled tokens and 0s otherwise.
        mask_complement:
            Binary tensor with 0s for sampled tokens and 1s otherwise.
        """
        # Sample mask start token
        if self.has_cls:
            # Do not mask cls
            valid_min_start = 1
        else:
            valid_min_start = 0
        if mask_size < non_zero_seq_len:
            start = torch.randint(valid_min_start,
                                  non_zero_seq_len - (mask_size - 1),
                                  size=(1,))
        else:
            start = valid_min_start

        # Create mask
        mask = torch.zeros(self.seq_len, dtype=torch.int32)
        mask[start:start+mask_size] = 1
        if valid_token_masks is not None:
            # Constrain mask to a set of valid tokens
            for k in range(len(valid_token_masks)):
                mask *= valid_token_masks[k]
        mask = torch.nonzero(mask.flatten())
        mask = mask.squeeze()
        
        # Create complement mask (considers original sampled masked without invalid tokens)
        mask_complement = torch.ones(self.seq_len, dtype=torch.int32)
        mask_complement[start:start+mask_size] = 0

        return mask, mask_complement

    def __call__(self,
                 batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create context and target masks when collating cell neighborhoods into a batch
        # 1. sample context block (size + location) using seed
        # 2. sample target block (size) using seed
        # 3. sample several context block locations for each cell neighborhood (w/o seed)
        # 4. sample several target block locations for each cell neighborhood (w/o seed)
        # 5. return context mask and target mask

        Parameters
        ----------
        batch:
            Tuple of tensors containing the sequence tokens (dim: n_batch x seq_len).
    
        Returns
        ----------
        collated_batch:
        collated_masks_context:
        collated_masks_target:
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_target, collated_masks_context = [], []
        keep_tokens_target = self.seq_len
        keep_tokens_context = self.seq_len

        for i in range(B):
            masks_target_complement = []
            masks_target = []
            masks_context = []

            non_zero_seq_len = torch.nonzero(batch[i]).size(0)

            for _ in range(self.n_targets):
                mask_target, mask_target_complement = self._sample_gene_mask(
                    non_zero_seq_len=non_zero_seq_len,
                    mask_size=self.target_mask_size)
                masks_target.append(mask_target)
                masks_target_complement.append(mask_target_complement)
                keep_tokens_target = min(keep_tokens_target, len(mask_target))
            
            for _ in range(self.n_contexts):
                mask_context, _ = self._sample_gene_mask(
                    non_zero_seq_len=non_zero_seq_len,
                    mask_size=self.context_mask_size,
                    valid_token_masks=masks_target_complement)
                masks_context.append(mask_context)
                keep_tokens_context = min(keep_tokens_context, len(mask_context))

            collated_masks_target.append(masks_target)
            collated_masks_context.append(masks_context)

        collated_masks_target = [[cm[:keep_tokens_target] for cm in cm_list] for cm_list in collated_masks_target]
        collated_masks_target = torch.utils.data.default_collate(collated_masks_target)
        collated_masks_context = [[cm[:keep_tokens_context] for cm in cm_list] for cm_list in collated_masks_context]
        collated_masks_context = torch.utils.data.default_collate(collated_masks_context)
        return collated_batch, collated_masks_context, collated_masks_target
