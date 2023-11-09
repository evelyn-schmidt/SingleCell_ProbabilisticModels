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

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

condition_key = 'orig.ident'
cell_type_key = 'seurat_clusters'
target_conditions = [3228]

adata_all = sc.read('/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/conversion/Protein_folzconversion_fig5.h5ad')
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

adata_ref = adata_full[np.logical_or(adata_full.obs.batch == "451", adata_full.obs.batch == "730")].copy()
adata_query = adata_full[adata_full.obs.batch == "3228"].copy()
adata_query = adata_query[:, adata_ref.var_names].copy()

# did not give me the same INFO as the example
TF_CPP_MIN_LOG_LEVEL=0
sca.models.TOTALVI.setup_anndata(
    adata_full,
    batch_key="batch",
    protein_expression_obsm_key="protein_expression"
)


N_EPOCHS=250
start = time.time()

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    n_layers_decoder=2,
    n_layers_encoder=2,
)

vae = sca.models.TOTALVI(
    adata_full,
    **arches_params
)
vae.train(max_epochs=N_EPOCHS, batch_size=256, lr=4e-3)
end = time.time()
print("\n Total default train time: {}".format(end-start))


dir_path = "yosef_model/"
vae.save(dir_path, overwrite=True)

adata_full.obsm["X_totalVI"] = vae.get_latent_representation()

plt.plot(vae.history["elbo_validation"][10:], label="validation")
plt.title("Negative ELBO over training epochs")
plt.legend()

adata_ref.obsm["X_totalvi"] = vae.get_latent_representation(adata_ref)
adaata_query.obsm["X_totalvi_default"] = vae.get_latent_representation(adtat_query)

# predict cell types of query
query.obs["predicted_l2_default"] = classify_from_latent(ref, query, ref_obsm_key="X_totalvi")
query.obs["celltype.l2"] = query.obs["predicted_l2_default"]

print("Computing full umap")
sc.pp.neighbors(adata_full_new, use_rep="X_totalvi_default", metric="cosine")
sc.tl.umap(adata_full_new, min_dist=0.3)








sc.pp.neighbors(adata_ref, use_rep="X_totalVI")
sc.tl.umap(adata_ref, min_dist=0.4)

sc.pl.umap(
    adata_ref,
    color=["batch"],
    frameon=False,
    ncols=1,
    title="Reference",
    save='_reference.png'
)

dir_path = "refererence_model/"
vae_ref.save(dir_path, overwrite=True)

# Perform surgery on reference model and train on query dataset without protein data
vae_q = sca.models.TOTALVI.load_query_data(
    adata_query,
    dir_path,
    freeze_expression=True
)
vae_q.train(200, plan_kwargs=dict(weight_decay=0.0))

adata_query.obsm["X_totalVI"] = vae_q.get_latent_representation()
sc.pp.neighbors(adata_query, use_rep="X_totalVI")
sc.tl.umap(adata_query, min_dist=0.4)


dir_path = "query_model/"
vae_q.save(dir_path, overwrite=True)



