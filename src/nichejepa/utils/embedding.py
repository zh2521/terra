from typing import List, Literal, Optional

import anndata
import numpy as np
import pandas as pd
import torch


def compute_rank_based_weights(tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute rank-based weights for a 2D tensor of tokens.

    Parameters
    -----------
    tokens:
        A 2D tensor where each row represents a sequence of tokens. The tokens
        are gene_id of cell or neighborhood

    Returns
    -----------
    A 2D tensor of the same shape as `tokens` containing the computed weights (
    weights sum up to 1).
    """
    # Create a mask where each element is True (1) if the corresponding token is
    # non-zero, and False (0) if it is zero (padding token)
    mask = tokens != 0

    # Compute cumulative sum along the sequence dimension (dim=1), which gives
    # ranks for non-zero tokens. Each token's rank is incremented based on its
    # position in the sequence, with padding tokens maintaining a rank of 0
    ranks = mask.cumsum(dim=1).float() * mask.float()

    # Find the maximum rank in each sequence, keeping the dimension for
    # broadcasting
    rank_max = ranks.max(dim=1, keepdim=True)[0]

    # Compute the sum of ranks for each sequence, keeping the dimension for
    # broadcasting
    rank_sum = ranks.sum(dim=1, keepdim=True)

    # Calculate the weights for each token: the weight is inversely proportional
    # to the rank within the sequence (higher ranks have lower weights). The
    # 1e-9 is added to avoid division by zero.
    weights = (rank_max - ranks + 1) / (rank_sum + 1e-9)

    # Apply the mask to ensure that padding tokens receive a weight of 0
    weights = weights * mask.float()

    # Return the computed weights
    return weights


def compute_mean_nonpadding_emb(embs: torch.Tensor,
                                mask: torch.Tensor,
                                dim=1):
    """
    Compute the mean of non-padding embeddings.
    
    Parameters
    -----------
    embs (torch.Tensor): The input embeddings tensor (3D).
    mask (torch.Tensor): A boolean mask tensor indicating the positions that mean should computed (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the mean.
    
    Returns
    -----------
    torch.Tensor: The mean embeddings tensor.
    torch.Tensor: Number of genes that used in computing average

    Raises:
    ValueError: If the items tensor is not 3D.
    """
    # Use broadcasting to sum embeddings across non-padding positions
    if embs.dim() == 3:
        # If the embeddings tensor has 3 dimensions (batch_size, sequence_length, embedding_dim),
        # broadcast the mask to match the dimensions of embs.
        # The mask tensor is initially (batch_size, sequence_length), so we unsqueeze to (batch_size, sequence_length, 1).
        masked_embs = embs * mask.unsqueeze(2)  # Broadcasting the mask along the embedding dimension

        # Sum the masked embeddings along the specified dimension
        sum_embs = masked_embs.sum(dim)

        # Calculate the mean by dividing the summed embeddings by the number of non-padding positions
        # The mask is summed along the same dimension to count non-padding tokens, and view(-1, 1) ensures
        # The resulting tensor has the correct dimensions for broadcasting during division.
        # The + 1e-9 will handle the case that we are reterriving gene that may mask.sum(dim).view(-1, 1).float() be zero
        # then we won't have INF value
        mean_embs = sum_embs / (mask.sum(dim).view(-1, 1).float() + 1e-9)

    else:
        # Raise an error if the input embeddings tensor is not 3D, as this function expects a 3D tensor
        raise ValueError('Expected a 3D tensor for embs, but got a tensor with {} dimensions.'.format(embs.dim()))

    # Return the calculated mean embeddings and the number of unmasked values for each row when the mask is computed for retrieve_gene.
    # This value serves as a gene_id identifier, indicating the cells where the gene is present.
    # A value of zero means that the gene is not present, while a value of one means that the sample contains the gene.
    # This value in upper function could be used as gene_count

    return mean_embs, mask.sum(dim)


def create_binary_selection_mask(
    tokens: torch.Tensor,
    seq_len_cell: int,
    has_cls: bool,
    selection_type: Literal['cls',
                            'agg_cell',
                            'agg_neighborhood',
                            'gene_cell',
                            'gene_neighborhood'],
    top_k: Optional[int]=None,
    gene_id: Optional[int]=None,
    ) -> torch.Tensor:
    """
    Create a selection mask for cell and neighborhood tokens based on
    specificiations.

    Parameters
    -----------
    tokens:
        2D input tokens to be masked.
    seq_len_cell:
    has_cls:
    selection_type:
    top_k:
    gene_id:

    Returns
    -----------
    selection_mask:
        The resulting 2D selection mask tensor.
    """
    if selection_type == 'cls':
        # Select only the first token in each sequence
        selection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        selection_mask[:, 0] = True
        return selection_mask
    elif selection_type == 'agg_cell':
        selection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        # Select non-padding tokens in the cell segment
        selection_mask[:, (1 if has_cls else 0):
                          (1 if has_cls else 0) + seq_len_cell] = True
        selection_mask[tokens == 0] = False # exclude padding tokens
        if top_k:
            # Exclude tokens beyond the top_k positions in the cell segment
            selection_mask[:, (1 if has_cls else 0) + top_k:] = False
    elif selection_type == 'agg_neighborhood':
        # Select non-padding tokens in the neighborhood segment
        selection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        selection_mask[:, (1 if has_cls else 0) + seq_len_cell:] = True
        selection_mask[tokens == 0] = False # exclude padding tokens
        if top_k:
            # Exclude tokens beyond the top_k positions in the neighborhood
            # segment
            selection_mask[
                :, (1 if has_cls else 0) + seq_len_cell + top_k:] = False    
    elif selection_type == 'gene_cell':
        # Select only positions corresponding to the specified gene_id in the
        # cell segment
        selection_mask = tokens == gene_id
        selection_mask[:, (1 if has_cls else 0) + seq_len_cell:] = False
    elif selection_type == 'gene_neighborhood':
        # Select only positions corresponding to the specified gene_id in the
        # neighborhood segment
        selection_mask = tokens == gene_id
        selection_mask[:, (1 if has_cls else 0):
                          (1 if has_cls else 0) + seq_len_cell] = False
    else:
        raise ValueError('The "selection_type" is not valid.')

    return selection_mask


def retrieve_gene_emb_from_cell_emb(cell_neighborhood_tokens: torch.Tensor,
                                    cell_emb: torch.Tensor,
                                    gene_id: int,
                                    gene_type: Literal["cell", "neighborhood"],
                                    has_cls: bool,
                                    seq_len_cell: int
                                    ) -> torch.Tensor:
    """
    Retrieve contextual gene embeddings from contextual cell embeddings based
    on specified gene IDs and gene types.

    Parameters
    -----------
    cell_neighborhood_tokens:
    cell_emb:
    gene_id:
    gene_type:
    has_cls:
    seq_len_cell:

    Returns
    --------
    gene_emb:
    """
    gene_mask = create_binary_selection_mask(
        cell_neighborhood_tokens,
        selection_type=f"gene_{gene_type}",
        gene_id=gene_id,
        seq_len_cell=seq_len_cell,
        has_cls=has_cls)

    gene_indices = gene_mask.argmax(dim=1) 
    gene_emb_selection = cell_emb.gather(
        1,
        gene_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, cell_emb.size(2)))
    gene_emb = gene_emb_selection.squeeze(1)

    return gene_emb