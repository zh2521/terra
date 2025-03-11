"""
Adapted from Bardes, A. et al. Revisiting feature prediction for learning visual
representations from video. arXiv [cs.CV] (2024).; 
https://github.com/facebookresearch/jepa/blob/main/app/main.py (10.03.2025).
"""


import argparse
import multiprocessing as mp
import os
import logging

import pprint
import wandb
import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file_name',
    type=str,
    help='Name of config file to load.',
    default='configs.yaml')
parser.add_argument(
    '--devices',
    type=str,
    nargs='+',
    default=['cuda:0'],
    help='Devices to use on local machine.')


def process_main(rank, config_file_name, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    # Set up logger
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Load config
    params = None
    with open(config_file_name, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info(f'Loaded params from config file: {config_file_name}.')

    # Initialize wandb on main process and log config
    if rank == 0:
        wandb.init(project="NEMO", config=params)
        wandb.config.update(params)
        pprint.PrettyPrinter(indent=4).pprint(params)
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the app with loaded config
    app_main(params['app'], args=params)

    # Finish wandb run
    if rank == 0:
        wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.config_file_name, num_gpus, args.devices)
        ).start()