# PARSE
Protein Annotation by Residue-Specific Enrichment: a site-based statistical method for interpretable protein function annotation

## Install dependencies

We recommend installing all dependencies in a conda environment:

```conda create -n parse python=3.10```

To install, run the following script on a GPU-enabled machine. To install for CPU only, replace `cu117` with `cpu`.

```./install_dependencies.sh```

## Download data

Download CSA reference data and precomputed embeddings for AlphaFoldDB datasets (human and dark proteomes) from [Zenodo](https://zenodo.org/record/).

CSA data (required):
* `csa_function_sets_nn.pkl`: function sets dictionary
* `csa_site_db_nn.pkl`: embeddings database
* `function_score_dists.pkl`: function-specific background distributions

AlphaFoldDB precomputed embeddings (optional):
* `af2_human_lmdb.zip`: AF2 human proteome embeddings in LMDB format
* `af2_dark_hernandez_lmdb.zip`: AF2 dark proteome embeddings from [Barrio-Hernandez et al.](http://dx.doi.org/10.1038/s41586-023-06510-w) in LMDB format
* `af2_dark_durairaj_lmdb.zip`: AF2 human proteome embeddings from [Durairaj et al.](http://dx.doi.org/10.1038/s41586-023-06622-3) in LMDB format


## Run PARSE on a single PDB

Run the following to annotate a protein using standard CSA reference database.

From PDB file:

```python predict.py --pdb PDB_PATH```

From pre-computed embedding database, given a PDB id:

```python predict.py --precomputed_id PDB_ID --precomputed_lmdb LMDB_PATH```

Optional arguments:

```
--chain CHAIN: annotate only a specified chain
--db DB_PATH: reference database embeddings, in pickle format
--function_sets FN_PATH: reference database function sets, in pickle format
--background BKG_PATH: function-specific background distributions, in pickle format
--cutoff CUTOFF: FDR cutoff for reporting results (default 0.001)
--use_gpu : flag for running with GPU
```

## Create embedding database from directory of PDB files

To run PARSE on a large number of PDB files, you can create a pre-computed embedding database in LMDB format for all pdb files in a directory (including subdirectories). 
Valid filetypes include pdb, pdb.gz, ent, ent.gz, cif

For small datasets, run without optional split arguments:

```python embed_pdb_dataset.py PDB_DIR OUT_LMDB_DIR --filetype=pdb```

For large datasets (e.g. Swissprot, AlphaFoldDB), we recommend processing in parallel using the num_splits argument.

```python embed_pdb_dataset.py PDB_DIR OUT_LMDB_DIR --split_id=$i --num_splits=NUM_SPLITS --filetype=pdb```

This produces NUM_SPLITS (e.g. 20) tmp files in OUT_LMDB_DIR. To combine all into the full dataset, run the following:

```python -m atom3d.datasets.scripts.combine_lmdb OUT_LMDB_DIR/tmp_* OUT_LMDB_DIR/full```

To run PARSE on each protein in the embedding database:

```python run_parse_lmdb.py --dataset=LMDB_DIR```

Optional arguments are the same as `predict.py`, with additional split arguments for parallel processing:
--db DB_PATH: reference database embeddings, in pickle format
--function_sets FN_PATH: reference database function sets, in pickle format
--background BKG_PATH: function-specific background distributions, in pickle format
--cutoff CUTOFF: FDR cutoff for reporting results (default 0.001)
--split_id SPLIT: split id (int) representing index from 0 to NUM_SPLITS-1
--num_splits NUM_SPLITS: number of splits for parallel processing

## Create new reference database

To create a new reference database of functional sites, first generate a csv file with the following columns (see `data/csa_functional_sites.csv` for example):
```
site: function ID from source database (e.g. M-CSA 993)
pdb: pdb and chain (e.g. 2j9hA)
locs: list of residue ids which are functionally important
source: database source (e.g. M-CSA; used in case of more than one source database)
description: text description of function (e.g. Glutathione S-transferase A)
```

Then, create embedding database using the following, where EMBEDDING_OUT is the generated .pkl file containing DB embeddings and FUNCSET_OUT is the generated .pkl file containing function sets:

```python create_reference_database.py DATABASE.csv EMBEDDING_OUT FUNCSET_OUT```

