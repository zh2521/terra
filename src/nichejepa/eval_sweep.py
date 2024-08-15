import sys
import yaml
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
import os
import copy
import logging

from .masks.multigene import MaskCollator
from .utils.distributed import init_distributed
from .utils.logging import CSVLogger
from .datasets.cell_neighborhood_dataset import make_cell_neighborhood_dataset
from .helper import load_checkpoint, init_model
from tqdm import tqdm
import anndata
from .utils.eval_utils  import process_loader

# Set global seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def eval_main(args, resume_preempt=False):
    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    seq_len = args['data']['seq_len']
    vocab_size = args['data']['vocab_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_set_name = args['data']['data_set_name']
    num_epochs = args['optimization']['epochs']

    # -- MASK
    n_targets = args['mask']['n_targets']
    n_contexts = args['mask']['n_contexts']
    target_mask_size = args['mask']['target_mask_size']
    context_mask_size = args['mask']['context_mask_size']

    # -- LOGGING

    # Set the folder for saving extracted features
    folder = (f"logs/{data_set_name}_"
               f"pred_depth_{pred_depth}_pred_emb_dim_{pred_emb_dim}_"
               f"enc_depth_{enc_depth}_n_targets_{n_targets}_"
               f"n_contexts_{n_contexts}_target_mask_size_{target_mask_size}_"
               f"context_mask_size_{context_mask_size}_num_epochs_{num_epochs}")
    save_folder = folder + '/extracted_features'
    
    os.makedirs(save_folder, exist_ok=True)
    tag = args['logging']['write_tag']
    dump = os.path.join(folder, f'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # -- log/checkpointing paths
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # Initialize encoder, predictor, and target encoder
    encoder, predictor = init_model(
        device=device,
        seq_len=seq_len,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        vocab_size=vocab_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # Initialize mask collator
    mask_collator = MaskCollator(seq_len=seq_len,
                                 target_mask_size=target_mask_size,
                                 context_mask_size=context_mask_size,
                                 n_targets=n_targets,
                                 n_contexts=n_contexts)

    # Initialize dataloader and -sampler
    data_path = args['data']['data_path']
    dataset = load_from_disk(data_path, keep_in_memory=True)
    dataset = dataset.train_test_split(test_size=args['data']['split'], seed=0)

    _, train_loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=dataset["train"],
        vocab_size=vocab_size,
        seq_len=seq_len,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        distributed=False)
    _, test_loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=dataset["test"],
        vocab_size=vocab_size,
        seq_len=seq_len,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=False,
        num_workers=num_workers,
        distributed=False)

    # Load training checkpoint
    ipe = len(train_loader)
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)
        for _ in range(start_epoch * ipe):
            mask_collator.step()

    target_encoder.eval()
    all_features = []
    all_obs = []
    process_loader(target_encoder, train_loader, args, 'train', gene_id=592, all_features=all_features, all_obs=all_obs)
    process_loader(target_encoder, test_loader, args, 'test', gene_id=592, all_features=all_features, all_obs=all_obs)

    merged_features = np.vstack(all_features)
    final_obs = pd.concat(all_obs, axis=0).reset_index(drop=True)
    final_obs.index = final_obs.index.astype(str)
    final_adata = anndata.AnnData(obs=final_obs)
    final_adata.obsm['jepa_emb'] = merged_features
    final_adata.write('final_result.h5ad')

