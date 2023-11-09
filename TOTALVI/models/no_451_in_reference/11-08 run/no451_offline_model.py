# RunDocker evelyns2000/foltz_tools

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import umap
import scanpy as sc
import anndata
import torch
# import scarches as sca
import matplotlib.pyplot as plt
import numpy as np
import scvi as scv
import pandas as pd
from scvi.model import TOTALVI
from sklearn.ensemble import RandomForestClassifier
import time
import datetime


sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

adata = sc.read('../../data/fig4_data/adata/TotalVi_final_adata_reference.h5ad')
adata = adata[np.logical_or(adata.obs.batch == "730", adata.obs.batch == "3228")].copy()


TOTALVI.setup_anndata(
    adata,
    layer="counts",
    batch_key = 'batch',
    protein_expression_obsm_key="protein_expression", 
)

start = time.time()


arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    n_layers_decoder=2,
    n_layers_encoder=2,
)

vae = TOTALVI(adata, **arches_params)
vae.train(max_epochs=250)
end = time.time()


vae.save("fig4_model_no451", overwrite=True)

record = open("record.txt", "a")  # append mode
record.write("NO 451, fig4_model_official written at ")
record.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
record.write("\n NO 451, Total fig4_model_officialtrain time: {}".format(end-start))
record.write("\n")
record.close()



query = sc.read('../../data/fig7_data/adata/TotalVi_final_adata_query.h5ad')

vae_q = TOTALVI.load_query_data(
    query,
    vae,
)

start = time.time()
vae_q.train(
    max_epochs=150,
    plan_kwargs=dict(weight_decay=0.0, scale_adversarial_loss=0.0),
)
end = time.time()

vae_q.save("fig7_query_no451", overwrite=True)

record = open("record.txt", "a")  # append mode
record.write("NO 451, fig7_query_official written at ")
record.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
record.write("\n NO 451, Total fig7_query_official train time: {}".format(end-start))
record.write("\n")
record.close()