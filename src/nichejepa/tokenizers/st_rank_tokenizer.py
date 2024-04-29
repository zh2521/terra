"""
NicheJEPA tokenizer.

Adapted from Theodoris, C. V. et al. Transfer learning enables predictions in network biology. Nature 618, 616–624
(2023); https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py (12.04.2024).

Input Data
----------
Required format:
    Raw counts spatial transcriptomics (ST) data with all genes (no feature selection) as '.h5ad' (anndata) files.
    Neighborhood graph of cells is stored in adata.obsp['spatial_connectivities'] or spatial coordinates are stored in
    adata.obsm["spatial"].
Required gene attributes:
    Ensembl ID for each gene ('ensembl_id').
Optional cell attributes:
    Binary indicator of whether cell should be tokenized based on user-defined filtering criteria ('filter_pass').
    Any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary.

Usage
----------
.. code-block :: python
    >>> from nichejepa import STRankTokenizer
    >>> tk = STRankTokenizer({"cell_type": "cell_types"}, nproc=4)
    >>> tk.tokenize_data("input_directory", "output_directory", "output_file_prefix")

Description
----------
Input data is a directory with '.h5ad' files containing raw counts from ST data, including all genes detected without
feature selection. The input file type is specified by the argument 'file_format' in the tokenize_data function. Genes
should be labeled with Ensembl IDs (adata.var['ensembl_id']), which provide a unique identifer for conversion to tokens.
Gene names can be converted to Ensembl IDs via the helper function nichejepa.utils.genes.get_ensembl_ids(). No cell
metadata is required, but custom cell attributes may be passed onto the tokenized dataset by providing a dictionary of
custom attributes, which is formatted as {original_attr_name: desired_dataset_attr_name}. For example, if the original
'.h5ad' file has cell attributes in adata.obs["cell_type"] and one would like to retain these attributes as labels in
the tokenized dataset with the new names "cell_types", the following custom attribute dictionary should be provided:
{"cell_type": "cell_types"}. Additionally, if the original '.h5ad' file contains a cell attribute called
adata.obs["filter_pass"], this will be used as a binary indicator of whether to include these cells in the tokenized
data. All cells with "1" in this attribute will be tokenized, whereas the others will be excluded. One may use
this column to indicate QC filtering or other criteria for selection for inclusion in the final tokenized dataset. If
one's data is in other formats besides '.h5ad', one can use the relevant tools (such as Anndata tools) to convert the
file to '.h5ad' format prior to initializing the STRankTokenizer.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
from datasets import Dataset
from skmisc.loess import loess


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*") # noqa

logger = logging.getLogger(__name__)

CELL_GENE_MEANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_means_dictionary.pkl"
CELL_GENE_REG_STDS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_reg_stds_dictionary.pkl"
NEIGHBORHOOD_GENE_MEANS_FILE = Path(__file__).parent.parent.parent.parent / "neighborhood_gene_means_dictionary.pkl"
NEIGHBORHOOD_GENE_REG_STDS_FILE = Path(__file__).parent.parent.parent.parent / "neighborhood_gene_reg_stds_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent.parent.parent.parent / "token_dictionary.pkl"


def process_gene_tokens(
    gene_tokens: list,
    length: int,
    token_dict: dict,
    special_tokens: Optional[list],
    special_tokens_idx: Optional[list],
    ) -> list:
    """
    Add pad tokens or truncate gene token list based on length and add special tokens if defined.

    Parameters
    ----------
    gene_tokens:
       List containing (ranked) gene tokens.
    length:
        Length to which to pad or truncate the gene token list to.
    token_dict:
        Token dictionary.
    special_tokens:
        List of special tokens to be added to the gene token list.
    special_tokens_idx:
        List with indices where special tokens are added to the gene token list.

    Returns
    ----------
    processed_gene_tokens:
       List containing padded or truncated (ranked) gene tokens, including special tokens if defined.       
        
    """
    if special_tokens:
        # Make space for special tokens
        processed_gene_tokens = gene_tokens[:(length-len(special_tokens))]

        # Add special tokens
        for special_token, special_token_idx in zip(special_tokens, special_tokens_idx):
            processed_gene_tokens = np.insert(
                processed_gene_tokens, special_token_idx, token_dict.get(special_token)
                )
    else:
        processed_gene_tokens = gene_tokens

    pad_size = int(length - len(processed_gene_tokens))
    if pad_size < 0:
        # Truncate
        processed_gene_tokens = processed_gene_tokens[:length]
    else:
        # Add pad tokens
        processed_gene_tokens = np.pad(
            processed_gene_tokens, (0, pad_size), 'constant', constant_values=token_dict.get("<pad>")
            )
    return processed_gene_tokens
    

def rank_gene_tokens(
    gene_scores: np.array,
    gene_tokens: np.array
    ) -> np.array:
    """
    Rank gene tokens based on matching gene scores (highest gene score -> rank 1 gene).

    Parameters
    ----------
    gene_scores:
        1D vector containing gene scores (read depth normalized gene expression scaled by means and regularizing
        standard deviations).
    gene_tokens:
        1D vector containing gene tokens.

    Returns
    ----------
    ranked_gene_tokens:
        1D vector containing gene tokens ranked by gene scores.       
        
    """
    # Sort gene tokens by gene scores
    sorted_indices = np.argsort(-gene_scores)
    ranked_gene_tokens = gene_tokens[sorted_indices]
    return ranked_gene_tokens


def tokenize_cell(
    norm_counts_cell: np.array,
    norm_counts_neighborhood: np.array,
    coding_miRNA_tokens_cell: np.array,
    coding_miRNA_tokens_neighborhood: np.array) -> Tuple[np.array, np.array]:
    """
    Convert read depth normalized and scaled gene expression counts to tokenized rank value encoding.

    Parameters
    ----------
    norm_counts_cell:
        Read-depth normalized and scaled gene expression counts of the cell.
    norm_counts_neighborhood:
        Read-depth normalized and scaled gene expression counts of the neighborhood.
    coding_miRNA_tokens_cell:
        Protein-coding and micro RNA gene tokens of the cell.
    coding_miRNA_tokens_neighborhood:
        Protein-coding and micro RNA gene tokens of the neighborhood.

    Returns
    ----------
    gene_tokens_cell:
        Ranked gene tokens of the cell.
    gene_tokens_neighborhood:
        Ranked gene tokens of the neighborhood.
    """
    # Mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    
    # Rank gene tokens based on norm counts
    gene_tokens_cell = rank_gene_tokens(
        norm_counts_cell[nonzero_mask], coding_miRNA_tokens_cell[nonzero_mask]
        )
    gene_tokens_neighborhood = rank_gene_tokens(
        norm_counts_neighborhood[nonzero_mask], coding_miRNA_tokens_neighborhood[nonzero_mask]
        )
    return gene_tokens_cell, gene_tokens_neighborhood


class STRankTokenizer:
    def __init__(
        self,
        custom_attr_name_dict: Optional[dict] = None,
        nproc: int = 1,
        chunk_size: int = 512,
        model_input_size: int = 2048,
        cell_gene_means_file: Path | str = CELL_GENE_MEANS_FILE,
        cell_gene_reg_stds_file: Path | str = CELL_GENE_REG_STDS_FILE,
        neighborhood_gene_means_file: Path | str = NEIGHBORHOOD_GENE_MEANS_FILE,
        neighborhood_gene_reg_stds_file: Path | str = NEIGHBORHOOD_GENE_REG_STDS_FILE,
        token_dictionary_file: Path | str = TOKEN_DICTIONARY_FILE,
        cell_special_tokens: Optional[list[str]] = None, # = ["<cls>"],
        cell_special_tokens_idx: Optional[list[int]] = None, # = [0],
        neighborhood_special_tokens: Optional[list[str]] = None, # = ["<sep>"],
        neighborhood_special_tokens_idx: Optional[list[int]] = None, # = [2048],
        ):
        """
        Initialize spatial transcriptomics rank tokenizer.

        Parameters
        ----------
        custom_attr_name_dict:
            Dictionary of custom attributes to be added to the Hugging Face dataset. Keys are the names of the
            attributes in the '.h5ad' files. Values are the names of the attributes in the Hugging Face dataset.
        nproc
            Number of processes to use for dataset mapping.
        chunk_size:
            Chunk size for adata tokenizer.
        model_input_size:
            Max input size of the model to truncate input to.
        cell_gene_means_file:
            Path to pickle file containing dictionary of mean gene expression of cells across STcorpus (for each gene).
        cell_gene_reg_stds_file:
            Path to pickle file containing dictionary of regularizing standard deviations of gene expression of cells
            across STcorpus (for each gene). Regularizing standard deviations are expected standard deviations based
            on means and are used for normalization to stabilize variances and only keep 'unexpected' variation.
        neighborhood_gene_means_file:
            Path to pickle file containing dictionary of mean gene expression of neighborhoods across STcorpus (for each
            gene).
        neighborhood_gene_reg_stds_file:
            Path to pickle file containing dictionary of regularizing standard deviations of gene expression of
            neighborhoods across STcorpus (for each gene). Regularizing standard deviations are expected standard
            deviations based on means and are used for normalization to stabilize variances and only keep 'unexpected'
            variation.
        token_dictionary_file:
            Path to pickle file containing token dictionary (gene tokens are Ensembl IDs + '_cell' or '_neighborhood').
        cell_special_tokens:
            List with special tokens inserted into the gene token vector containing cell gene tokens.
        cell_special_tokens_idx:
            Index where special tokens are to be inserted into the cell gene token vector.
        neighborhood_special_tokens:
            List with special tokens inserted into the gene token vector containing neighborhood gene tokens.
        neighborhood_special_tokens_idx:
            Index where special tokens are to be inserted into the neighborhood gene token vector.
        """
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.cell_special_tokens = cell_special_tokens
        self.cell_special_tokens_idx = cell_special_tokens_idx
        self.neighborhood_special_tokens = neighborhood_special_tokens
        self.neighborhood_special_tokens_idx = neighborhood_special_tokens_idx

        # Load dictionaries of cell gene means and reg stds
        with open(cell_gene_means_file, "rb") as f:
            self.cell_gene_means_dict = pickle.load(f)
        with open(cell_gene_reg_stds_file, "rb") as f:
            self.cell_gene_reg_stds_dict = pickle.load(f)

        # Load dictionaries of neighborhood gene means and reg stds
        with open(neighborhood_gene_means_file, "rb") as f:
            self.neighborhood_gene_means_dict = pickle.load(f)
        with open(neighborhood_gene_reg_stds_file, "rb") as f:
            self.neighborhood_gene_reg_stds_dict = pickle.load(f)

        # Load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.token_dict = pickle.load(f)

        # Get vocabulary and gene Ensembl IDs (protein-coding and miRNA genes)
        self.vocab = list(self.token_dict.keys())
        self.coding_miRNA_ids = [
            key.split("_")[0] for key in list(self.vocab) if "_cell" in key
            ]
        self.coding_miRNA_dict = dict(zip(self.coding_miRNA_ids, [True] * len(self.vocab)))

    def tokenize_data(
        self,
        input_directory: Path | str,
        output_directory: Path | str,
        output_file_prefix: str,
        file_format: Literal["h5ad"] = "h5ad",
        use_generator: bool = False,
        ):
        """
        Tokenize files in 'input_directory' and save as tokenized '.dataset' file in 'output_directory'.

        Parameters
        ----------
        input_directory:
            Path to directory containing '.h5ad' (anndata) files.
        output_directory:
            Path to directory where tokenized data will be saved as '.dataset' file.
        output_file_prefix:
            Prefix for output file.
        file_format:
            Format of input files. Can be '.h5ad'.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        """
        gene_tokens_cell, gene_tokens_neighborhood, cell_metadata = self.tokenize_files(
            Path(input_directory), file_format
            )

        tokenized_dataset = self.create_dataset(
            gene_tokens_cell,
            gene_tokens_neighborhood,
            cell_metadata,
            use_generator=use_generator
            )

        output_path = str((Path(output_directory) / output_file_prefix).with_suffix(".dataset"))
        tokenized_dataset.save_to_disk(output_path)

    def tokenize_files(
        self,
        data_directory: Path | str,
        file_format: Literal["h5ad"] = "h5ad"
        ) -> Tuple[np.array, np.array, dict]:
        """
        Tokenize multiple files.

        Parameters
        ----------
        data_directory:
            Path to the directory containing the files to be tokenized.
        file_format:
            Format of the files to be tokenized.

        Returns 
        ----------
        gene_tokens_cell:
            Cell-wise vector of ranked cell gene tokens.
        gene_tokens_neighborhood:
            Cell-wise vector of ranked neighborhood gene tokens.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """
        gene_tokens_cell = []
        gene_tokens_neighborhood = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        file_found = 0

        tokenize_file_fn = self.tokenize_adata

        # Loop through data directory to tokenize '.h5ad' files    
        for file_path in data_directory.glob(f"*.{file_format}"):
            file_found = 1
            print(f"Tokenizing '{file_path}'...")
            file_gene_tokens_cell, file_gene_tokens_neighborhood, file_cell_metadata = tokenize_file_fn(file_path)
            gene_tokens_cell += file_gene_tokens_cell
            gene_tokens_neighborhood += file_gene_tokens_neighborhood
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

        if file_found == 0:
            logger.error(
                f"No '.{file_format}' files found in directory '{data_directory}'."
                )
            raise
        return gene_tokens_cell, gene_tokens_neighborhood, cell_metadata

    def tokenize_adata(
        self,
        adata_file_path: Path | str,
        target_sum: int=10_000
        ) -> Tuple[np.array, np.array, dict]:
        """
        Tokenize cells from an '.h5ad' (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.
        target_sum:
            Target sum for counts after read depth normalization.

        Returns 
        ----------
        gene_tokens_cell:
            Cell-wise vector of ranked cell gene tokens.
        gene_tokens_neighborhood:
            Cell-wise vector of ranked neighborhood gene tokens.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """
        adata = ad.read_h5ad(adata_file_path)

        if "spatial_connectivities" not in adata.obsp.keys():
            # Compute spatial neighborhood graph based on Visium spot diameter of 55 microns
            sq.gr.spatial_neighbors(adata,
                                    coord_type="generic",
                                    spatial_key="spatial",
                                    radius=55
                                    )

        if "counts_neighborhood" not in adata.layers.keys():
            # Compute sum of raw counts across each cell's neighborhood to get neighborhood counts per cell
            adata.layers["counts_neighborhood"] = np.array(
                adata.obsp["spatial_connectivities"].T @ adata.X
                )

        # Normalize counts and neighborhood counts by read depth
        adata.X = adata.X / adata.X.sum(1).reshape(-1, 1) * target_sum
        adata.layers["counts_neighborhood"] = (
            adata.layers["counts_neighborhood"] /
            adata.layers["counts_neighborhood"].sum(1).reshape(-1, 1) * target_sum
            )

        # Store cell metadata
        if self.custom_attr_name_dict is not None:
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        # Tokenize only protein-coding and miRNA genes
        coding_miRNA_idx = np.where(
            [self.coding_miRNA_dict.get(gene_id, False) for gene_id in adata.var["ensembl_id"]]
            )[0]
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_idx]
        coding_miRNA_tokens_cell = np.array([self.token_dict[gene_id + "_cell"] for gene_id in coding_miRNA_ids])
        coding_miRNA_tokens_neighborhood = np.array(
            [self.token_dict[gene_id + "_neighborhood"] for gene_id in coding_miRNA_ids])

        # Retrieve cell and neighborhood gene means and reg stds
        cell_gene_means = np.array(
            [self.cell_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"][coding_miRNA_idx]]
            )
        cell_gene_reg_stds = np.array(
            [self.cell_gene_reg_stds_dict[gene_id] for gene_id in adata.var["ensembl_id"][coding_miRNA_idx]]
            )
        neighborhood_gene_means = np.array(
            [self.neighborhood_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"][coding_miRNA_idx]]
            )
        neighborhood_gene_reg_stds = np.array(
            [self.neighborhood_gene_reg_stds_dict[gene_id] for gene_id in adata.var["ensembl_id"][coding_miRNA_idx]]
            )

        # Filter cells that did not pass QC
        if "filter_pass" in adata.obs.columns:
            filter_pass_idx = np.where(
                [filter_pass == 1 for filter_pass in adata.obs["filter_pass"]]
                )[0]
        else:
            print(f"'{adata_file_path}' has no column 'filter_pass'; tokenizing all cells.")
            filter_pass_idx = np.array([i for i in range(adata.shape[0])])

        gene_tokens_cell = []
        gene_tokens_neighborhood = []

        # Divide cells into chunks and loop through chunks
        for i in range(0, len(filter_pass_idx), self.chunk_size):
            chunk_idx = filter_pass_idx[i : i + self.chunk_size]

            # Normalize counts and neighborhood counts by normalization factor from corpus
            norm_counts_cell = sp.csr_matrix(
                (adata[chunk_idx, coding_miRNA_idx].X - cell_gene_means) / cell_gene_reg_stds
                )
            norm_counts_neighborhood = sp.csr_matrix(
                (adata[chunk_idx, coding_miRNA_idx].layers["counts_neighborhood"] - neighborhood_gene_means) /
                neighborhood_gene_reg_stds
                )

            # Rank cell gene tokens and append across chunks
            gene_tokens_cell += [
                rank_gene_tokens(norm_counts_cell[i].data, coding_miRNA_tokens_cell[norm_counts_cell[i].indices])
                for i in range(norm_counts_cell.shape[0])
                ]
            
            # Rank neighborhood gene tokens and append across chunks
            gene_tokens_neighborhood += [
                rank_gene_tokens(norm_counts_neighborhood[i].data,
                                 coding_miRNA_tokens_neighborhood[norm_counts_neighborhood[i].indices])
                for i in range(norm_counts_neighborhood.shape[0])
                ]

            # Addd values to cell metadata
            if self.custom_attr_name_dict is not None:
                for k in cell_metadata.keys():
                    cell_metadata[k] += adata[chunk_idx].obs[k].tolist()
            else:
                cell_metadata = None

        return gene_tokens_cell, gene_tokens_neighborhood, cell_metadata


    def create_dataset(
        self,
        gene_tokens_cell: np.array,
        gene_tokens_neighborhood: np.array,
        cell_metadata: dict,
        use_generator: bool = False,
        keep_original_gene_tokens: bool = False
        ) -> Dataset:
        """
        Create a Hugging Face dataset based on tokenized cells.


        Parameters
        ----------
        gene_tokens_cell:
            Cell-wise vector of ranked cell gene tokens.
        gene_tokens_neighborhood:
            Cell-wise vector of ranked neighborhood gene tokens.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        keep_original_gene_tokens:
            If 'True', keep original gene tokens in Hugging Face dataset (before padding/truncation and addition of
            special tokens).

        Returns 
        ----------
        dataset:
            Hugging Face dataset containing the tokenized cells.        
        """
        print("Creating Hugging Face dataset...")
        # Create dict for Hugging Face dataset creation
        dataset_dict = {"gene_tokens_cell": gene_tokens_cell,
                        "gene_tokens_neighborhood": gene_tokens_neighborhood}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # Create Hugging Face dataset
        if use_generator:
            def dict_generator():
                for i in range(len(gene_tokens_cell)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            dataset = Dataset.from_dict(dataset_dict)

        def format_gene_tokens(example):
            if keep_original_gene_tokens:
                # Store original gene tokens in separate features
                example["gene_tokens_cell_original"] = example["gene_tokens_cell"]
                example["gene_tokens_cell_original_length"] = len(example["gene_tokens_cell"])
                example["gene_tokens_neighborhood_original"] = example["gene_tokens_neighborhood"]
                example["gene_tokens_neighborhood_original_length"] = len(example["gene_tokens_neighborhood"])

            example["gene_tokens_cell"] = process_gene_tokens(
                    example["gene_tokens_cell"],
                    int(self.model_input_size / 2),
                    self.token_dict,
                    self.cell_special_tokens,
                    self.cell_special_tokens_idx
                    )

            example["gene_tokens_neighborhood"] = process_gene_tokens(
                    example["gene_tokens_neighborhood"],
                    int(self.model_input_size / 2),
                    self.token_dict,
                    self.neighborhood_special_tokens,
                    self.neighborhood_special_tokens_idx
                    )

            example["input_ids"] = np.concatenate((
                example["gene_tokens_cell"],
                example["gene_tokens_neighborhood"]
                ))

            return example

        formatted_dataset = dataset.map(
            format_gene_tokens, num_proc=self.nproc)
        return formatted_dataset