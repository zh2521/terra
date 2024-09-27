import random
from typing import Tuple, Union

import datasets
from datasets import load_from_disk
from sklearn.model_selection import train_test_split


def init_dataloader_and_sampler():
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


def prepare_dataset(args: dict,
                    split_dataset: bool=True
                    ) -> Union[Tuple[datasets.arrow_dataset.Dataset,
                                     datasets.arrow_dataset.Dataset],
                               datasets.arrow_dataset.Dataset]:
    """
    Prepare the dataset by loading it, determining sample size, and splitting it
    into training and test sets based on the provided configuration parameters.

    Parameters
    -----------
    args:
        A dictionary containing the configuration parameters, including:
            - data_path: The path to the dataset.
            - sample_size: The size of the dataset to sample.
            - sample_subset: Whether to sample a subset of the dataset.
            - split: The train-test split ratio.
            - stratify: Whether to stratify the dataset during the split.
            - random_state: The random seed for reproducibility.
    split_dataset:
        If 'True', split the huggingface dataset into train and test datasets.

    Returns
    -----------
    1)
    train_dataset:
        The training portion of the dataset.
    test_dataset:
        The test portion of the dataset.
    
    2)
    dataset:
        The combined training and test portion of the dataset with a 'split'
        label.
    """
    # Load dataset from the specified path
    data_path = args['data']['data_path']
    dataset = load_from_disk(data_path)

    # Sample subset if specified
    if args['data']['sample_subset']:
        total_size = len(dataset)
        sample_size = min(args['data']['sample_size'], total_size)
        rng = random.Random(args['data']['random_state'])
        sampled_indices = rng.sample(range(total_size), sample_size)
        dataset = dataset.select(sampled_indices)

    # Prepare for dataset split
    indices = list(range(len(dataset)))

    # Prepare train-test split parameters
    split_params = {
        'test_size': args['data']['split'],
        'random_state': args['data']['random_state']
    }
    if args['data']['stratify']:
        split_params['stratify'] = dataset['cell_types']

    # Perform train test split
    if args['data']['split'] > 0:
        train_indices, test_indices = train_test_split(indices, **split_params)

        if split_dataset:
            train_dataset = dataset.select(train_indices)
            test_dataset = dataset.select(test_indices)

            return train_dataset, test_dataset

        else:
            split_labels = {i: 'train' for i in train_indices}
            split_labels.update({i: 'test' for i in test_indices})
            def add_split_label(example, idx):
                return {'split': split_labels[idx]}
            dataset = dataset.map(add_split_label, with_indices=True)

            return dataset
            