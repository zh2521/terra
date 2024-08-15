import yaml
import logging

def setup_batch_size(enc_pred_depth, is_training):
    """
    Determine and set the appropriate batch size based on the encoder depth and whether the model is training.

    Parameters:
    - enc_pred_depth (int): The depth of the encoder and prediction. This influences the batch size selection.
    - is_training (bool): Indicates if the model is in training mode (True) or evaluation mode (False).

    Returns:
    - int: The computed batch size based on the provided encoder depth and training status.
    """
    # Adjust batch size if we are in evaluation mode
    if not is_training:
        return 1000
    
    # Default batch size assignment based on the encoder prediction depth
    if enc_pred_depth < 41:
        return 80
    elif 41 <= enc_pred_depth < 51:
        return 40
    else:
        return 70

def create_params_from_YAML_wandb_config(config, args, logger, is_training=True, has_same_dimention=True):
    """
    Updates the `params` dictionary with values from the YAML config file and the object loaded from the wandb configuration file.
    This can be useful when using wandb sweeps for hyperparameter optimization. Also sets the seed in `params` from `args.seed`.

    Parameters:
    - config (object): A configuration object (such as one from wandb) containing the parameters to update in `params`.
    - args (object): An object that contains the filename of the YAML configuration and the seed value.
    - logger (object): Logger object to log the loaded parameters.
    - is_training (bool): Indicates if the model is in training mode (True) or evaluation mode (False).
    - has_same_dimention (bool) Indicates if the pred_emb_dim and enc_emb_dim be same
    Returns:
    - dict: The updated `params` dictionary.
    """

    try:
        # Load parameters from YAML configuration file
        with open(args.fname, 'r') as y_file:
            params = yaml.safe_load(y_file)
            logger.info('Loaded parameters from YAML file.')
    except FileNotFoundError:
        logger.error(f"YAML configuration file '{args.fname}' not found.")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        raise

    # Update 'meta' section with values from wandb config
    params['meta']['enc_pred_depth'] = int(config.enc_pred_depth)
    params['meta']['pred_depth'] = int(config.enc_pred_depth % 10)
    params['meta']['enc_depth'] = int(config.enc_pred_depth // 10)
    params['meta']['enc_emb_dim'] = config.enc_emb_dim
    if has_same_dimention:
        params['meta']['pred_emb_dim'] = config.enc_emb_dim
    params['meta']['top_layer'] = config.top_layer
    params['meta']['top_k'] = config.top_k

    # Update 'mask' section wandb config
    params['mask']['n_targets'] = config.n_targets
    params['mask']['context_mask_size'] = config.context_mask_size
    params['mask']['target_mask_size'] = config.target_mask_size

    # Update 'optimization' section wandb config
    params['optimization']['ema'] = config.ema
    params['optimization']['epochs'] = config.epochs
    params['optimization']['learnable'] = config.learnable

    # Set seed 
    params['seed'] = args.seed

    # Set Batch
    params['data']['batch_size'] = setup_batch_size(params['meta']['enc_pred_depth'], is_training)

    # Return the updated params dictionary
    return params

