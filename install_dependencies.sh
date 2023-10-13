#!/bin/bash

# install the following in conda env created by `conda create -n parse python=3.10`
# if installing GPU version, make sure to run on GPU-enabled machine

pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117

pip install torch_geometric

pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

pip install -r requirements.txt

git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .