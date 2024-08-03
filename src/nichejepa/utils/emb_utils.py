import torch

def mean_nonpadding_embs(embs, mask, dim=1):
    """
    Compute the mean of non-padding embeddings.
    
    Parameters:
    embs (torch.Tensor): The input embeddings tensor (can be 2D or 3D).
    mask (torch.Tensor): A boolean mask tensor indicating the non-padding or cls positions (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the mean.
    
    Returns:
    torch.Tensor: The mean embeddings tensor.
    """
    # Use broadcasting to sum across non-padding positions
    if embs.dim() == 3:
        masked_embs = embs * mask.unsqueeze(2)  # Broadcasting mask to match embs dimensions
        sum_embs = masked_embs.sum(dim)
        mean_embs = sum_embs / mask.sum(dim).view(-1, 1).float()

    elif embs.dim() == 2:
        masked_embs = embs * mask  # Broadcasting mask to match embs dimensions
        sum_embs = masked_embs.sum(dim)
        mean_embs = sum_embs / mask.sum(dim).float()
        
    return mean_embs

def create_selection_mask(cell_neighborhood_tokens, label_name, seq_len_cell, top_k, 
                          just_cell=False, just_neighborhood=False, 
                          gene_id=None, mask_large_than_k=False):
    """
    Create a selection mask for cell index tokens or neighborhood tokens based on various conditions.

    Parameters:
    cell_neighborhood_tokens (torch.Tensor): Tensor containing cell or neighborhood tokens.
    label_name (str): Label name to determine selection rules.
    seq_len_cell (int): Sequence length of the cell tokens.
    top_k (int): Top k value for selection masking.
    just_cell (bool): Whether to select only the cell index token.
    just_neighborhood (bool): Whether to select only the neighborhood index token.
    gene_id (int, optional): Gene ID or CLS ID to be used for specific selection.
    mask_large_than_k (bool): Whether to mask position larger than k in cell or neighborhood.

    Returns:
    torch.Tensor: The resulting selection mask tensor.

    Raises:
    AssertionError: If more than one of mask_large_than_k or specific_gene_mask is True.
    """

    # Ensure at most one of the conditions is true
    assert (int(mask_large_than_k) + int(gene_id is not None)) <= 1, \
        "At most one of mask_large_than_k or specific_gene_mask must be True"

    # Initialize selection mask based on specific gene or non-zero tokens
    if gene_id is not None:
        select = (cell_neighborhood_tokens == gene_id).int()
    else:
        select = (cell_neighborhood_tokens != 0).int()

    # Apply just_cell and just_neighborhood conditions
    if just_cell and just_neighborhood:
      if label_name == "niche_type":
          select[:, :seq_len_cell] = 0
          if mask_large_than_k:
            select[:, seq_len_cell + top_k:] = 0
            return select
      elif label_name == "cell_type":
           select[:, seq_len_cell:] = 0
    
    # Apply masking for elements larger than k
    if mask_large_than_k:
        select[:, top_k:] = 0

    return select

