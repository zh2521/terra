"""
Niche-JEPA tokenizer.

Adapted from Theodoris, C. V. et al. Transfer learning enables predictions in network biology. Nature 618, 616–624
(2023); https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py (12.04.2024).

Input Data
----------
Required format:
    Raw counts spatial transcriptomics (ST) data without feature selection as '.loom' or '.h5ad' (anndata) files.
    Neighborhood graph of cells is stored in adata.obsp['spatial_connectivities'].
Required gene attributes:
    Ensembl ID for each gene ('ensembl_id')
Optional cell attributes:
    Binary indicator of whether cell should be tokenized based on user-defined filtering criteria ('filter_pass').
    Any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

Usage
----------
.. code-block :: python
    >>> from nichejepa import STRankTokenizer
    >>> tk = STRankTokenizer({"cell_type": "cell_type", "organ_major": "organ"}, nproc=4)
    >>> tk.tokenize_data("data_directory", "output_directory", "output_prefix")

Description
----------
Input data is a directory with .loom or .h5ad files containing raw counts from single cell RNAseq data, including all
genes detected in the transcriptome without feature selection. The input file type is specified by the argument
file_format in the tokenize_data function. The discussion below references the .loom file format, but the analagous
labels are required for .h5ad files, just that they will be column instead of row attributes and vice versa due to the
transposed format of the two file types. Genes should be labeled with Ensembl IDs (loom row attribute "ensembl_id"),
which provide a unique identifer for conversion to tokens. Other forms of gene annotations (e.g. gene names) can be
converted to Ensembl IDs via Ensembl Biomart. Cells should be labeled with the total read count in the cell (loom column
attribute "n_counts") to be used for normalization. No cell metadata is required, but custom cell attributes may be
passed onto the tokenized dataset by providing a dictionary of custom attributes to be added, which is formatted as
loom_col_attr_name : desired_dataset_col_attr_name. For example, if the original .loom dataset has column attributes
"cell_type" and "organ_major" and one would like to retain these attributes as labels in the tokenized dataset with
the new names "cell_type" and "organ", respectively, the following custom attribute dictionary should be provided:
{"cell_type": "cell_type", "organ_major": "organ"}. Additionally, if the original .loom file contains a cell column
attribute called "filter_pass", this column will be used as a binary indicator of whether to include these cells in the
tokenized data. All cells with "1" in this attribute will be tokenized, whereas the others will be excluded. One may use
this column to indicate QC filtering or other criteria for selection for inclusion in the final tokenized dataset. If
one's data is in other formats besides .loom or .h5ad, one can use the relevant tools (such as Anndata tools) to convert
the file to a .loom or .h5ad format prior to initializing the STRankTokenizer.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
# from datasets import Dataset

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa
#import loompy as lp  # noqa

logger = logging.getLogger(__name__)

CELL_GENE_MEDIAN_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_median_dictionary.pkl"
NICHE_GENE_MEDIAN_FILE = Path(__file__).parent.parent.parent.parent / "niche_gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent.parent.parent.parent / "token_dictionary.pkl"


def rank_gene_tokens(
    gene_scores: np.array,
    gene_tokens: np.array
    ) -> np.array:
    """
    Rank gene tokens based on matching gene scores (highest gene score -> rank 1 gene)

    Parameters
    ----------
    gene_scores:
        1D vector containing gene scores (normalized gene expression scaled by corpus median).
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


def add_pad_tokens(arr, length, pad_with):
    # Calculate the number of items needed to reach the desired length
    pad_size = int(max(0, length - arr.size))
    # Use numpy.pad to extend the array
    padded_array = np.pad(arr, (0, pad_size), 'constant', constant_values=pad_with)
    return padded_array


def tokenize_cell(gene_vector, gene_tokens):
    """
    ADAPT
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    return rank_gene_tokens(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


class STRankTokenizer:
    def __init__(
        self,
        custom_attr_name_dict: Optional[dict] = None,
        nproc: int = 1,
        chunk_size: int = 512,
        model_input_size: int = 2048,
        special_token: bool = False,
        cell_gene_median_file: Path | str = CELL_GENE_MEDIAN_FILE,
        niche_gene_median_file: Path | str = NICHE_GENE_MEDIAN_FILE,
        token_dictionary_file: Path | str = TOKEN_DICTIONARY_FILE,
        ):
        """
        Initialize spatial transcriptomics rank tokenizer.

        Parameters
        ----------
        custom_attr_name_dict:
            Dictionary of custom attributes to be added to the dataset. Keys are the names of the
            attributes in the loom file. Values are the names of the attributes in the dataset.
        nproc
            Number of processes to use for dataset mapping.
        chunk_size:
            Chunk size for anndata tokenizer.
        model_input_size:
            Max input size of the model to truncate input to.
        special_token:
            If 'True', adds CLS token before and SEP token after rank value encoding.
        cell_gene_median_file:
            Path to pickle file containing dictionary of non-zero median gene expression values of
            cells across STcorpus.
        niche_gene_median_file:
            Path to pickle file containing dictionary of non-zero median gene expression values of
            niches across STcorpus.
        token_dictionary_file:
            Path to pickle file containing token dictionary (Tokens are Ensembl IDs + '_cell' or
            '_niche').
        """
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.special_token = special_token

        # Load dictionary of cell gene normalization factors
        with open(cell_gene_median_file, "rb") as f:
            self.cell_gene_median_dict = pickle.load(f)

        # Load dictionary of niche gene normalization factors
        with open(niche_gene_median_file, "rb") as f:
            self.niche_gene_median_dict = pickle.load(f)

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
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["h5ad"] = "h5ad",
        use_generator: bool = False,
        ):
        """
        Tokenize files in data_directory and save as tokenized '.dataset' file in output_directory.

        Parameters
        ----------
        data_directory:
            Path to directory containing '.h5ad' (anndata) files.
        output_directory:
            Path to directory where tokenized data will be saved as '.dataset' file.
        output_prefix:
            Prefix for output file.
        file_format:
            Format of input files. Can be '.h5ad'.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        """
        tokenized_cells, cell_metadata = self.tokenize_files(
            Path(data_directory), file_format
            )
        tokenized_dataset = self.create_dataset(
            tokenized_cells,
            cell_metadata,
            use_generator=use_generator,
            )

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)

    def tokenize_files(
        self,
        data_directory: Path | str,
        file_format: Literal["loom", "h5ad"] = "h5ad"
        ):
        """
        """
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        file_found = 0

        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom"
            else self.tokenize_anndata
            )

        # Loop through directories to tokenize .loom or .h5ad files    
        for file_path in data_directory.glob(f"*.{file_format}"):
            file_found = 1
            print(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}."
                )
            raise
        return tokenized_cells, cell_metadata

    def tokenize_adata(
        self,
        adata_file_path: Path | str,
        target_sum: int=10_000
        ) -> Tuple[np.array, np.array, dict]:
        """
        Tokenize cells from an anndata ('.h5ad') file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.
        target_sum:
            Target sum for counts after read depth normalization.

        Returns 
        ----------
        gene_tokens_cell:
            Cell-wise tokens for the genes of the cell.
        gene_tokens_niche:
            Cell-wise tokens for the genes of the niche.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """
        adata = ad.read_h5ad(adata_file_path)

        # Compute mean raw counts across each cell's niche to get niche counts per cell
        adata.layers["counts_niche"] = np.array(
            (adata.obsp["spatial_connectivities"].T @ adata.X) /
            adata.obsp["spatial_connectivities"].sum(0).T
            )

        # Get cell-wise counts for read depth normalization
        adata.obs["total_counts_cell"] = adata.X.sum(1)
        adata.obs["total_counts_niche"] = adata.layers["counts_niche"].sum(1)

        # Store cell metadata
        if self.custom_attr_name_dict is not None:
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        # Tokenize only protein-coding and miRNA genes
        coding_miRNA_idx = np.where(
            [self.coding_miRNA_dict.get(gene_id, False) for gene_id in adata.var["ensembl_id"]]
            )[0]
        norm_factors_cell = np.array(
            [self.cell_gene_median_dict[gene_id] for gene_id in adata.var["ensembl_id"][coding_miRNA_idx]]
            )
        norm_factors_niche = np.array(
            [self.niche_gene_median_dict[gene_id] for gene_id in adata.var["ensembl_id"][coding_miRNA_idx]]
            )
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_idx]
        coding_miRNA_tokens_cell = np.array([self.token_dict[gene_id + "_cell"] for gene_id in coding_miRNA_ids])
        coding_miRNA_tokens_niche = np.array([self.token_dict[gene_id + "_niche"] for gene_id in coding_miRNA_ids])

        # Filter cells that did not pass QC
        if "filter_pass" in adata.obs.columns:
            filter_pass_idx = np.where(
                [filter_pass == 1 for filter_pass in adata.obs["filter_pass"]]
                )[0]
        else:
            print(f"'{adata_file_path}' has no column 'filter_pass'; tokenizing all cells.")
            filter_pass_idx = np.array([i for i in range(adata.shape[0])])

        gene_tokens_cell = []
        gene_tokens_niche = []

        # Divide cells into chunks and loop through chunks
        for i in range(0, len(filter_pass_idx), self.chunk_size):
            chunk_idx = filter_pass_idx[i : i + self.chunk_size]

            # Perform read depth normalization of counts and scale by median values from corpus
            total_counts_cell = adata[chunk_idx].obs["total_counts_cell"].values[:, None]
            total_counts_niche = adata[chunk_idx].obs["total_counts_niche"].values[:, None]
            counts_cell = adata[chunk_idx, coding_miRNA_idx].X
            counts_niche = adata[chunk_idx, coding_miRNA_idx].layers["counts_niche"]
            norm_counts_cell = counts_cell / total_counts_cell * target_sum / norm_factors_cell
            norm_counts_niche = counts_niche / total_counts_niche * target_sum / norm_factors_niche
            norm_counts_cell = sp.csr_matrix(norm_counts_cell)
            norm_counts_niche = sp.csr_matrix(norm_counts_niche)

            # Get ranked cell gene tokens for genes with non-zero counts
            gene_tokens_cell += [add_pad_tokens(
                rank_gene_tokens(norm_counts_cell[i].data, coding_miRNA_tokens_cell[norm_counts_cell[i].indices]),
                self.model_input_size / 2,
                self.token_dict["<pad>"])
                for i in range(norm_counts_cell.shape[0])
                ]

            # Get ranked niche gene tokens for genes with non-zero counts
            gene_tokens_niche += [add_pad_tokens(
                rank_gene_tokens(norm_counts_niche[i].data, coding_miRNA_tokens_niche[norm_counts_niche[i].indices]),
                self.model_input_size / 2,
                self.token_dict["<pad>"]
                ) for i in range(norm_counts_niche.shape[0])
                ]

            # Addd values to cell metadata
            if self.custom_attr_name_dict is not None:
                for k in cell_metadata.keys():
                    cell_metadata[k] += adata[chunk_idx].obs[k].tolist()
            else:
                cell_metadata = None

        return gene_tokens_cell, gene_tokens_niche, cell_metadata


    def create_dataset(
        self,
        gene_tokens_cell: np.array,
        gene_tokens_niche: np.array,
        cell_metadata: dict,
        use_generator: bool = False,
        keep_uncropped_input_ids: bool = False,
        ) -> Dataset:
        print("Creating Hugging Face dataset...")
        # Create dict for Hugging Face dataset creation
        dataset_dict = {"gene_tokens_cell": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # Create dataset
        if use_generator:

            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        def format_cell_features(example):
            # Store original uncropped input_ids in separate feature
            if keep_uncropped_input_ids:
                example["input_ids_uncropped"] = example["input_ids"]
                example["length_uncropped"] = len(example["input_ids"])

            # Truncate input_ids to input size
            if self.special_token:
                # Leave space for CLS and SEP token
                example["input_ids"] = example["input_ids"][
                    0 : self.model_input_size - 2
                    ]
                example["input_ids"] = np.insert(
                    example["input_ids"], 0, self.token_dict.get("<cls>")
                    )
                example["input_ids"] = np.insert(
                    example["input_ids"],
                    len(example["input_ids"]),
                    self.token_dict.get("<sep>")
                    )
            else:
                example["input_ids"] = example["input_ids"][0 : self.model_input_size]
            example["length"] = len(example["input_ids"])

            return example

        output_dataset_truncated = output_dataset.map(
            format_cell_features, num_proc=self.nproc)
        return output_dataset_truncated