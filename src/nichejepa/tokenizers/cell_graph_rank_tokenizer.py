"""
NicheJEPA tokenizer.

Adapted from Theodoris, C. V. et al. Transfer learning enables predictions in network biology. Nature 618, 616–624
(2023); https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py (12.04.2024).

Input Data
----------
Required format:
    Raw counts spatial transcriptomics (ST) data with all genes (no feature selection) as '.h5ad' (anndata) files.
    Spatial coordinates are stored in adata.obsm["spatial"].
Required gene attributes:
    Ensembl ID for each gene ('ensembl_id').
Optional cell attributes:
    Binary indicator of whether cell should be used for tokenization based on user-defined filtering criteria
    ('filter_pass').
    Any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary.

Usage
----------
.. code-block :: python
    >>> from nichejepa import CellGraphRankTokenizer
    >>> tk = CellGraphRankTokenizer(custom_attr_name_dict={"cell_type": "cell_types"}, nproc=4)
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
adata.obs["filter_pass"], this will be used as a binary indicator of whether to include these cells in the tokenization.
All cells with "1" in this attribute will be tokenized, whereas the others will be excluded. One may use this column to
indicate QC filtering or other criteria for selection for inclusion in the final tokenized dataset. If one's data is in
other formats besides '.h5ad', one can use the relevant tools (such as Anndata tools) to convert the file to '.h5ad'
format prior to initializing the CellGraphRankTokenizer.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
from datasets import Dataset
from skmisc.loess import loess


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*") # noqa

logger = logging.getLogger(__name__)


GENE_MEANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_means_dictionary.pkl"
GENE_REG_STDS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_reg_stds_dictionary.pkl"
GENE_NZMEDIANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_nzmedians_dictionary.pkl"
GENE_LOGMEANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_logmeans_dictionary.pkl"
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
    gene_tokens: np.array,
    n_tokens: int,
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
    n_tokens:
        Number of tokens to be returned.

    Returns
    ----------
    ranked_gene_tokens:
        1D vector containing gene tokens ranked by gene scores.       
        
    """
    # Sort gene tokens by gene scores
    sorted_indices = np.argsort(-gene_scores)
    ranked_gene_tokens = gene_tokens[sorted_indices][:n_tokens]
    return ranked_gene_tokens


def tokenize_cell(
    norm_counts: np.array,
    coding_miRNA_tokens: np.array,
    n_tokens: int) -> Tuple[np.array, np.array]:
    """
    Convert normalized gene expression counts to tokenized rank value encoding.

    Parameters
    ----------
    norm_counts:
        Normalized gene expression counts of the cell.
    coding_miRNA_tokens:
        Protein-coding and micro RNA gene tokens of the cell.
    n_tokens:
        Number of tokens.

    Returns
    ----------
    gene_tokens:
        Ranked gene tokens.
    """
    # Rank gene tokens based on norm counts
    gene_tokens = rank_gene_tokens(norm_counts.data,
                                   coding_miRNA_tokens[norm_counts.indices],
                                   n_tokens)
    return gene_tokens


class CellGraphRankTokenizer:
    def __init__(
        self,
        custom_attr_name_dict: Optional[dict] = None,
        nproc: int = 1,
        chunk_size: int = 512,
        model_input_size: int = 2048,
        tokens_per_cell: int = 64, 
        norm_method: Literal["analytic_pearson_residuals",
                             "seurat_v3",
                             "mean",
                             "nzmedian",
                             "shifted_log"]="analytic_pearson_residuals",
        norm_factor: Optional[Literal["read_depth", "cell_area"]]=None,
        gene_means_file: Path | str = GENE_MEANS_FILE,
        gene_reg_stds_file: Path | str = GENE_REG_STDS_FILE,
        gene_nzmedians_file: Path | str = GENE_NZMEDIANS_FILE,
        gene_logmeans_file: Path | str = GENE_LOGMEANS_FILE, 
        token_dictionary_file: Path | str = TOKEN_DICTIONARY_FILE,
        special_tokens: Optional[list[str]] = ["<cls>"],
        special_tokens_idx: Optional[list[int]] = [0],
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
        norm_method:
            Normalization method used for count normalization before ranking.
            'analytic_pearson_residuals': Normalization as per Lause, J., Berens, P. & Kobak, D. Analytic Pearson
            residuals for normalization of single-cell RNA-seq UMI data. Genome Biol. 22, 258 (2021). The residuals are
            based on a negative binomial offset model with overdispersion 'theta' shared across genes. Residuals are
            clipped to 'sqrt(n_obs)' and overdispersion 'theta=100' is used. Negative residuals for a cell and gene
            indicate that less counts are observed than expected compared to the gene’s average expression and cellular
            sequencing depth. Positive residuals indicate more counts than expected. The implementation is based on
            https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/experimental/pp/_normalization.py#L36
            (03.05.2024).
            'seurat_v3': Normalization as per Stuart, T. et al. Comprehensive Integration of Single-Cell Data. Cell 177,
            1888–1902.e21 (2019). Feature counts are normalized by 'norm_factor', subsequent subtraction of means and
            division by expected standard deviations derived from learned global mean-variance relationships. The
            implementation is based on
            https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26
            (29.04.2024).
            'mean': Normalization by 'norm_factor' followed by normalization by corpus gene means.
            'nzmedian': Normalization by 'norm_factor' followed by normalization by corpus gene non-zero medians.
            'shifted_log': Normalization by 'norm_factor' followed by shifted log transformation.
        norm_factor:
            Norm factor for cellular normalization to adjust for cell size differences. Has to match norm factor used
            for computation of means and regularized stds. Is not used if 'norm_method' is 'analytic_pearson_residuals'. 
        gene_means_file:
            Path to pickle file containing dictionary of mean gene expression of cells across STcorpus (for each gene).
            Only relevant if 'norm_method' in ['seurat_v3', 'mean'].
        gene_reg_stds_file:
            Path to pickle file containing dictionary of regularizing standard deviations of gene expression of cells
            across STcorpus (for each gene). Regularizing standard deviations are expected standard deviations based
            on means and are used for normalization to stabilize variances and only keep 'unexpected' variation.
            Only relevant if 'norm_method' in ['seurat_v3'].
        gene_nzmedians_file:
            Path to pickle file containing dictionary of non-zero median gene expression of cells across STcorpus (for
            each gene).
            Only relevant if 'norm_method' in ['nzmean'].
        gene_logmeans_file:
            Path to pickle file containing dictionary of log mean gene expression of cells across STcorpus (for each
            gene).
            Only relevant if 'norm_method' in ['log_shifted'].      
        token_dictionary_file:
            Path to pickle file containing token dictionary (gene tokens are Ensembl IDs).
        special_tokens:
            List with special tokens inserted into the gene token vector containing cell gene tokens.
        special_tokens_idx:
            Index where special tokens are to be inserted into the cell gene token vector.
        """
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.tokens_per_cell = tokens_per_cell
        self.norm_method = norm_method
        self.norm_factor = norm_factor
        self.special_tokens = special_tokens
        self.special_tokens_idx = special_tokens_idx

        # Load normalization factors
        if self.norm_method == "seurat_v3":
            # Load dictionaries of gene means and reg stds
            with open(gene_means_file, "rb") as f:
                self.cell_gene_means_dict = pickle.load(f)
            with open(gene_reg_stds_file, "rb") as f:
                self.cell_gene_reg_stds_dict = pickle.load(f)
        elif self.norm_method == "mean":
            # Load dictionaries of gene means
            with open(gene_means_file, "rb") as f:
                self.cell_gene_means_dict = pickle.load(f)
        elif self.norm_method == "nzmedian":
            # Load dictionaries of gene non-zero medians
            with open(gene_nzmedians_file, "rb") as f:
                self.cell_gene_nzmedians_dict = pickle.load(f)
        elif self.norm_method == "shifted_log":
            # Load dictionaries of gene logmeans
            with open(gene_logmeans_file, "rb") as f:
                self.cell_gene_logmeans_dict = pickle.load(f)

        # Load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.token_dict = pickle.load(f)

        # Get vocabulary and gene Ensembl IDs (protein-coding and miRNA genes)
        self.vocab = list(self.token_dict.keys())
        self.coding_miRNA_ids = [
            key.split("_")[0] for key in list(self.vocab) if "ENS" in key
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
        gene_tokens, cell_pos_tokens, gene_pos_tokens, cell_metadata = self.tokenize_files(
            Path(input_directory), file_format
            )

        tokenized_dataset = self.create_dataset(
            gene_tokens,
            cell_pos_tokens,
            gene_pos_tokens,
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
        gene_tokens:
            Cell-wise vector of ranked gene tokens.
        cell_pos_tokens:
            Cell-wise vector of positional tokens for cells.
        gene_pos_tokens:
            Cell-wise vector of positional tokens for genes.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """
        gene_tokens = []
        cell_pos_tokens = []
        gene_pos_tokens = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        file_found = 0

        tokenize_file_fn = self.tokenize_adata

        # Loop through data directory to tokenize '.h5ad' files    
        for file_path in data_directory.glob(f"*.{file_format}"):
            file_found = 1
            print(f"Tokenizing '{file_path}'...")
            file_gene_tokens, file_cell_pos_tokens, file_gene_pos_tokens, file_cell_metadata = tokenize_file_fn(file_path)
            gene_tokens += file_gene_tokens
            cell_pos_tokens += file_cell_pos_tokens
            gene_pos_tokens += file_gene_pos_tokens
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
        return gene_tokens, cell_pos_tokens, gene_pos_tokens, cell_metadata

    def tokenize_adata(
        self,
        adata_file_path: Path | str,
        ) -> Tuple[np.array, np.array, dict]:
        """
        Tokenize cells from an '.h5ad' (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.

        Returns 
        ----------
        gene_tokens:
            Cell-wise vector of ranked cell gene tokens.
        cell_pos_tokens:
            Cell-wise vector of positional tokens for cells.
        gene_pos_tokens:
            Cell-wise vector of positional tokens for genes.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """
        adata = ad.read_h5ad(adata_file_path)

        print("Filtering cells.")
        # Filter cells that did not pass QC
        if "filter_pass" in adata.obs.columns:
            filter_pass_idx = np.where([filter_pass == 1 for filter_pass in adata.obs["filter_pass"]])[0]
        else:
            print(f"'{adata_file_path}' has no column 'filter_pass'; utilizing all cells.")
            filter_pass_idx = np.array([i for i in range(adata.shape[0])])
        adata = adata[filter_pass_idx]

        print("Computing spatial neighborhood graph.")
        # Compute spatial neighborhood graph based on Visium spot diameter of 55 microns
        sq.gr.spatial_neighbors(adata,
                                coord_type="generic",
                                spatial_key="spatial",
                                radius=27.5,
                                )
        
        print("Normalizing counts.")
        # Normalize counts before ranking of genes
        if self.norm_method == "analytic_pearson_residuals":
            # Define negative binomial overdispersion parameter
            theta = 100

            # Normalize counts
            sum_counts_cells = np.sum(adata.X, axis=1).reshape(-1, 1)
            sum_counts_genes = np.sum(adata.X, axis=0).reshape(1, -1)
            sum_counts_total = np.sum(sum_counts_genes)
            mu_counts = np.array(sum_counts_cells @ sum_counts_genes / sum_counts_total)
            diff_counts = np.array(adata.X - mu_counts)
            residuals_counts = diff_counts / np.sqrt(mu_counts + mu_counts**2 / theta)
            adata.X = np.clip(residuals_counts, a_min=-np.sqrt(adata.shape[0]), a_max=np.sqrt(adata.shape[0]))
        elif self.norm_method in ["seurat_v3", "mean", "nzmedian", "shifted_log"]:
            if self.norm_factor == "read_depth":
                # Normalize cell counts
                target_sum = 10_000
                adata.X = adata.X / adata.X.sum(1).reshape(-1, 1) * target_sum
            elif self.norm_factor == "cell_area":
                # Normalize cell counts
                adata.X = adata.X / adata.obs["cell_area"].values.reshape(-1, 1) * np.mean(adata.obs["cell_area"])
            if self.norm_method == "seurat_v3":
                # Retrieve gene means and reg stds
                cell_gene_means = np.array([self.cell_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"]])
                cell_gene_reg_stds = np.array(
                    [self.cell_gene_reg_stds_dict[gene_id] for gene_id in adata.var["ensembl_id"]]
                    )
            
                # Normalize cell counts
                adata.X  = adata.X - cell_gene_means / cell_gene_reg_stds
            elif self.norm_method == "mean":
                # Retrieve gene means
                cell_gene_means = np.array([self.cell_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"]])

                # Normalize counts
                adata.X = adata.X / cell_gene_means
            elif self.norm_method == "nzmedian":
                # Retrieve gene non-zero medians
                cell_gene_nzmedians = np.array([self.cell_gene_nzmedians_dict[gene_id] for gene_id in adata.var["ensembl_id"]])

                # Normalize counts
                adata.X = adata.X / cell_gene_nzmedians
            elif self.norm_method == "shifted_log":
                sc.pp.log1p(adata)

                # Retrieve gene logmeans
                cell_gene_logmeans = np.array(
                    [self.cell_gene_logmeans_dict[gene_id] for gene_id in adata.var["ensembl_id"]])

                # Normalize counts
                adata.X = adata.X / cell_gene_logmeans
        else:
            raise ValueError(f"'norm_method' {self.norm_method} is not valid.")

        # Initialize cell metadata
        if self.custom_attr_name_dict is not None:
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        # Retrieve gene tokens for genes contained in dataset and vocab, i.e. protein-coding and miRNA genes
        coding_miRNA_idx = np.where(
            [self.coding_miRNA_dict.get(gene_id, False) for gene_id in adata.var["ensembl_id"]]
            )[0]
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_idx]
        coding_miRNA_tokens = np.array([self.token_dict[gene_id] for gene_id in coding_miRNA_ids])

        gene_tokens = []
        cell_pos_tokens = []
        gene_pos_tokens = []

        print("Retrieving tokens for index cell and adding cell metadata.")
        # Divide cells into chunks and loop through chunks
        for i in range(0, len(adata), self.chunk_size):
            # Normalize counts by normalization factor from corpus
            norm_counts = sp.csr_matrix(adata[i : i + self.chunk_size, coding_miRNA_idx].X)

            # Rank gene tokens of index cell and append across chunks
            gene_tokens += [
                rank_gene_tokens(norm_counts[j].data,
                                 coding_miRNA_tokens[norm_counts[j].indices],
                                 self.tokens_per_cell
                                 )
                for j in range(norm_counts.shape[0])
                ]
            cell_pos_tokens += [[0] * self.tokens_per_cell for j in range(norm_counts.shape[0])]
            gene_pos_tokens += [[k for k in range(self.tokens_per_cell)] for j in range(norm_counts.shape[0])]

            # Add values to cell metadata
            if self.custom_attr_name_dict is not None:
                for k in cell_metadata.keys():
                    cell_metadata[k] += adata[i : i + self.chunk_size].obs[k].tolist()
            else:
                cell_metadata = None

        print("Retrieving tokens for neighborhood cells.")
        gene_tokens_copy = gene_tokens.copy()
        # Loop through all cells to add neighbor cell gene tokens based on position of cell compared to index cell.
        # Gene tokens of cells that are closer to the index cell will be added first.
        for i in range(0, len(adata)):
            # Get sorted indices of neighbor cells based on distance to index cell
            row_start = adata.obsp["spatial_distances"].indptr[i]
            row_end = adata.obsp["spatial_distances"].indptr[i+1]
            row_data = adata.obsp["spatial_distances"].data[row_start:row_end]
            sorted_indices = np.argsort(row_data)
            # Loop through distance-sorted neighbor cells and add gene, cell pos and gene pos tokens
            for j, k in enumerate(adata.obsp["spatial_connectivities"][i].nonzero()[1][sorted_indices]):
               gene_tokens[i] = np.hstack((gene_tokens[i], gene_tokens_copy[k]))
               cell_pos_tokens[i] = np.hstack((cell_pos_tokens[i],
                                               [j+1] * len(gene_tokens_copy[k])))
               gene_pos_tokens[i] = np.hstack((gene_pos_tokens[i],
                                               [l for l in range(len(gene_tokens_copy[k]))]))

        return gene_tokens, cell_pos_tokens, gene_pos_tokens, cell_metadata


    def create_dataset(
        self,
        gene_tokens: np.array,
        cell_pos_tokens: np.array,
        gene_pos_tokens: np.array,
        cell_metadata: dict,
        use_generator: bool = False,
        add_pos_tokens: bool = True,
        keep_original_gene_tokens: bool = False
        ) -> Dataset:
        """
        Create a Hugging Face dataset based on tokenized cells.

        Parameters
        ----------
        gene_tokens:
            Cell-wise vector of ranked gene tokens.
        cell_pos_tokens:
            Cell-wise vector of positional tokens for cells.
        gene_pos_tokens:
            Cell-wise vector of positional tokens for genes.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        add_pos_tokens:
            If 'True', add positional cell and gene tokens.
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
        dataset_dict = {"gene_tokens": gene_tokens}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        if add_pos_tokens:
            dataset_dict["cell_pos_tokens"] = cell_pos_tokens
            dataset_dict["gene_pos_tokens"] = gene_pos_tokens

        # Create Hugging Face dataset
        if use_generator:
            def dict_generator():
                for i in range(len(gene_tokens)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            dataset = Dataset.from_dict(dataset_dict)

        def format_gene_tokens(example):
            if keep_original_gene_tokens:
                # Store original gene tokens in separate features
                example["gene_tokens_original"] = example["gene_tokens"]
                example["gene_tokens_original_length"] = len(example["gene_tokens"])

            example["input_ids"] = process_gene_tokens(
                    example["gene_tokens"],
                    self.model_input_size,
                    self.token_dict,
                    self.special_tokens,
                    self.special_tokens_idx
                    )

            return example

        formatted_dataset = dataset.map(
            format_gene_tokens, num_proc=self.nproc)
        return formatted_dataset