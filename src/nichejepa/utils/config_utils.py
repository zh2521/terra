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
        return 200

    # Default batch size assignment based on the encoder prediction depth
    if enc_pred_depth < 41:
        return 20
    elif 41 <= enc_pred_depth < 51:
        return 40
    else:
        return 70


def create_params_from_YAML_wandb_config(YAML_file,
                                         logger,
                                         sweep_config=None,
                                         is_training=True,
                                         has_same_dimention=True,
                                         update_from_sweep=False):
    """
    Updates the `params` dictionary with values from the YAML config file and optionally from the wandb configuration file.
    This can be useful when using wandb sweeps for hyperparameter optimization. Also sets the seed in `params` from `args.seed`.

    Parameters:
    - YAML_file (object): contains the filename of the YAML configuration for static params.
    - logger (object): Logger object to log the loaded parameters.
    - sweep_config (object): A configuration object (such as one from wandb) containing the parameters to update in `params` for dynamic change.
    - is_training (bool): Indicates if the model is in training mode (True) or evaluation mode (False).
    - has_same_dimention (bool): Indicates if the pred_emb_dim and enc_emb_dim should be the same.
    - update_from_sweep (bool): Flag to determine whether to update parameters from the sweep_config (wandb).

    Returns:
    - dict: The updated `params` dictionary.
    """

    try:
        # Load parameters from YAML configuration file
        with open(YAML_file, 'r') as y_file:
            params = yaml.safe_load(y_file)
            logger.info('Loaded parameters from YAML file.')
    except FileNotFoundError:
        logger.error(f"YAML configuration file '{YAML_file}' not found.")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        raise

    if update_from_sweep:
        # Update 'meta' section with values from wandb config
        params['meta']['enc_pred_depth'] = int(sweep_config.enc_pred_depth)
        params['meta']['pred_depth'] = int(sweep_config.enc_pred_depth % 10)
        params['meta']['enc_depth'] = int(sweep_config.enc_pred_depth // 10)
        params['meta']['enc_emb_dim'] = sweep_config.enc_emb_dim
        if has_same_dimention:
            params['meta']['pred_emb_dim'] = sweep_config.enc_emb_dim
        params['meta']['top_layer'] = sweep_config.top_layer
        params['meta']['top_k'] = sweep_config.top_k

        # Update 'mask' section with values from wandb config
        params['mask']['n_targets'] = sweep_config.n_targets
        params['mask']['context_mask_size'] = sweep_config.context_mask_size
        params['mask']['target_mask_size'] = sweep_config.target_mask_size

        # Update 'optimization' section with values from wandb config
        params['optimization']['ema'] = sweep_config.ema
        params['optimization']['epochs'] = sweep_config.epochs
        params['optimization']['learnable'] = sweep_config.learnable


    # Set Batch
    params['data']['batch_size'] = setup_batch_size(params['meta']['enc_pred_depth'], is_training)

    # Return the updated params dictionary
    return params


def generate_output_name(args):
    """Generates a descriptive file name based on input arguments.

    Args:
        args (dict): A dictionary containing the configuration options.
            Expected keys include 'meta', with possible sub-keys:
            - 'just_cell' (bool): Whether to include 'cell_embedding' in the name.
            - 'just_neighborhood' (bool): Whether to include 'niche_embedding' in the name.
            - 'weighted_average' (bool): Whether to include 'weighted_average' in the name.

    Returns:
        str: A generated output file name with relevant parts joined by underscores and a '.h5ad' extension.
    """
    name_parts = []

    # Add specific components to the name based on provided flags
    if args['emb']['retrieve_cell']:
        name_parts.append("cell_embedding")

    if args['emb']['retrieve_niche']:
        name_parts.append("niche_embedding")

    if args['emb']['retrieve_gene']:
        name_parts.append("gene_embedding")
        name_parts.append("gene_id")
        name_parts.append(str(args['emb']['gene_id']))

    if args['emb']['weighted_average']:
        name_parts.append("weighted_average")

    elif args['emb']['cls']:
        name_parts.append("cls")

    else:
        name_parts.append("average")

    # Construct the final name and add file extension
    name = "_".join(name_parts) + '.h5ad'

    return name


