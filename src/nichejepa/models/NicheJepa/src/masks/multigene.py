# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        ratio=0.7,
    ):
        super(MaskCollator, self).__init__()
        self.ratio = ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        '''
        Create context and target masks when collating genes into a batch
        # 1. sample context and target block
        # 2. sample pred block (size) using seed
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        ratio = self.ratio
        batch_size  = len(batch)
        

        context_mask = torch.zeros((batch_size,batch[0].shape[0]), dtype=torch.int)
        target_mask = torch.zeros((batch_size,batch[0].shape[0]), dtype=torch.int)

        #collated_masks_context, collated_masks_target = [], []

        for i in range(batch_size):
             row = batch[i]
             nonzero_indices = torch.nonzero(row, as_tuple=False).squeeze()

             for idx in nonzero_indices:
                if torch.rand(1, generator=g).item() < self.ratio:
                    context_mask[i, idx] = 1
                else:
                    target_mask[i, idx] = 1

        #collated_masks_context = torch.utils.data.default_collate(context_mask)
        #collated_masks_target = torch.utils.data.default_collate(target_mask)

        return collated_batch, [context_mask], [target_mask]
