# pip install --user scikit-misc
# RunDocker evelyns2000/foltz_tools

import os
os.chdir('../')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import scanpy as sc
import anndata
import torch
import scarches as sca
import matplotlib.pyplot as plt
import numpy as np
import scvi as scv
import pandas as pd
import time
import datetime



sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

condition_key = 'orig.ident'
cell_type_key = 'seurat_clusters'
target_conditions = [451]

adata_all = sc.read('/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/data/fig4_data/adata/TotalVi_final_adata_reference.h5ad')
adata = adata_all.raw.to_adata()

# split into three data sets and assign a batch Key to each one
adata_3228 = adata[adata.obs['orig.ident'].isin([3228])].copy()
adata_3228.obs["batch"] = "3228"
adata_730 = adata[adata.obs['orig.ident'].isin([730])].copy()
adata_730.obs["batch"] = "730"
adata_451 = adata[adata.obs['orig.ident'].isin([451])].copy()
adata_451.obs["batch"] = "451"


# create the reference
adata_ref = anndata.concat([adata_730,adata_451])

# separate the query 
adata_query = adata_3228
# put matrix of zeros for protein expression (considered missing)
pro_exp = adata_ref.obsm["protein_expression"]
data = np.zeros((adata_query.n_obs, pro_exp.shape[1]))
adata_query.obsm["protein_expression"] = pd.DataFrame(columns=pro_exp.columns, index=adata_query.obs_names, data = data)

# concatenate the objects
adata_full = anndata.concat([adata_ref, adata_query])

sc.pp.highly_variable_genes(
    adata_full,
    n_top_genes=4000,
    flavor="seurat_v3",
    batch_key="batch",
    subset=True,
)

adata_ref = adata_full[np.logical_or(adata_full.obs.batch == "730", adata_full.obs.batch == "451")].copy()
adata_query = adata_full[adata_full.obs.batch == "3228"].copy()
adata_query = adata_query[:, adata_ref.var_names].copy()


TF_CPP_MIN_LOG_LEVEL=0
sca.models.TOTALVI.setup_anndata(
    adata_ref,
    batch_key = 'batch',
    protein_expression_obsm_key="protein_expression"
)

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
)

start = time.time()
vae_ref = sca.models.TOTALVI(
    adata_ref,
    **arches_params
)
vae_ref.train()
end = time.time()
print("\n Total reference train time: {}".format(end-start))

record = open("record.txt", "a")  # append mode
record.write("3228 ref written at ")
record.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
record.write("\n Total 3228 ref train time: {}".format(end-start))
record.write("\n")
record.close()


dir_path = "reference_model_3228/"
vae_ref.save(dir_path, overwrite=True)

# Perform surgery on reference model and train on query dataset without protein data
start = time.time()
vae_q = sca.models.TOTALVI.load_query_data(
    adata_query,
    dir_path,
    freeze_expression=True
)
vae_q.train(200, plan_kwargs=dict(weight_decay=0.0))
end = time.time()
print("\n Total query train time: {}".format(end-start))

dir_path = "query_model2_3228/"
vae_q.save(dir_path, overwrite=True)

record = open("record.txt", "a")  # append mode
record.write("730,451 query written at ")
record.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
record.write("\n Total 730,451 query train time: {}".format(end-start))
record.write("\n")
record.close()
