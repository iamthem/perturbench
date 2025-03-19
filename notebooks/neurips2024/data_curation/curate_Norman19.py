# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: perturbench-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2023-04-17-Curation: Norman19 Combo Screen

# %% [markdown]
# Add cross validation splits to Norman19

# %%
import scanpy as sc
import os
import subprocess as sp
from perturbench.analysis.preprocess import preprocess
from scipy.sparse import csr_matrix

#%reload_ext autoreload
#%autoreload 2

# %% [markdown]
# Download from: https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1

# %%
data_url = 'https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1'
data_cache_dir = '/home/ubuntu/perturbench_data' ## Change this to your local data directory

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/norman19_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget {data_url} -O {tmp_data_dir}', shell=True)

# %%
adata = sc.read_h5ad(tmp_data_dir)
#adata

# %%
adata.obs.rename(columns = {
    'nCount_RNA': 'ncounts',
    'nFeature_RNA': 'ngenes',
    'percent.mt': 'percent_mito',
    'cell_line': 'cell_type',
}, inplace=True)

# %%
#adata.obs.perturbation.unique()

# %%
adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('_', '+')
adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
#adata.obs.perturbation.value_counts()

# %%
adata.obs['condition'] = adata.obs.perturbation.copy()

# %%
adata.X = csr_matrix(adata.X)

# %%
adata = preprocess(
    adata,
    perturbation_key='condition',
    covariate_keys=['cell_type'],
)

# %%
adata = adata.copy()
output_data_path = f'{data_cache_dir}/norman19_processed.h5ad'
adata.write_h5ad(output_data_path)
