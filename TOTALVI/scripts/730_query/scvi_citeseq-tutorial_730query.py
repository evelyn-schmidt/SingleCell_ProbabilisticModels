# RunDocker evelyns2000/foltz_tools


import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import umap
from scvi.model import TOTALVI
from sklearn.ensemble import RandomForestClassifier

sc.set_figure_params(figsize=(4, 4))


adata_full = sc.read('/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/data/fig4_data/adata/fig4_Protein_folzconversion_prepped.h5ad')

adata = adata_full[np.logical_or(adata_full.obs.batch == "451", adata_full.obs.batch == "3228")].copy()
query = adata_full[adata_full.obs.batch == "730"].copy()


adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=4000,
    flavor="seurat_v3",
    batch_key="orig.ident",
    subset=True,
    layer="counts",
)

print(adata)

TOTALVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="orig.ident",
    protein_expression_obsm_key="protein_expression",
)

# # training code here
arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    n_layers_decoder=2,
    n_layers_encoder=2,
)

vae = TOTALVI(adata, **arches_params)
vae.train(max_epochs=250)
vae.save("/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/11-09_run/scvi-tools-cite_reference_model", overwrite=True)

# vae = TOTALVI.load("/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/full_model/11-09_run/scvi-tools-cite_reference_model", adata=adata)

plt.plot(vae.history["elbo_train"].iloc[10:], label="train")
plt.plot(vae.history["elbo_validation"].iloc[10:], label="validation")
plt.title("Negative ELBO over training epochs")
plt.legend()
plt.savefig("/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/11-09_run/figures/elbo.jpg")

adata.obsm["X_totalvi_scarches"] = vae.get_latent_representation()

y_train = adata.obs["celltype.l2"].astype("category").cat.codes.to_numpy()
X_train = adata.obsm["X_totalvi_scarches"]
clf = RandomForestClassifier(
    random_state=1,
    class_weight="balanced_subsample",
    verbose=1,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

vae.latent_space_classifer_ = clf

X = adata.obsm["X_totalvi_scarches"]
trans = umap.UMAP(
    n_neighbors=10,
    random_state=42,
    min_dist=0.4,
)
adata.obsm["X_umap"] = trans.fit_transform(X)

vae.umap_op_ = trans

sc.pl.umap(
    adata,
    color=["celltype.l2", "batch"],
    frameon=False,
    ncols=1,
    save="_inspect_reference"
)



# Store the counts in a layer, perform standard preprocessing
query.layers["counts"] = query.X.copy()
sc.pp.normalize_total(query, target_sum=1e4)
sc.pp.log1p(query)
query.raw = query
# subset to reference vars
query = query[:, adata.var_names].copy() 

# Add blank metadata that we will later fill in with predicted labels
query.obsm["protein_counts"] = query.obsm["protein_expression"].copy()
query.obs["celltype.l2"] = "Unknown"
# query.obs["orig.ident"] = query.obs["set"] # Rename the batch key to correspond to the reference data, 
# I don't think we need to do this becasue the batch key already exsists in the same slot as the reference data
# query.obsm["X_umap"] = query.obs[["UMAP1", "UMAP2"]].values
# there is no umap info in my data set


# reorganize query proteins, missing proteins become all 0
for p in adata.obsm["protein_expression"].columns:
    if p not in query.obsm["protein_counts"].columns:
        query.obsm["protein_counts"][p] = 0.0
# ensure columns are in same order
query.obsm["protein_counts"] = query.obsm["protein_counts"].loc[:, adata.obsm["protein_expression"].columns]

adata.obs["dataset_name"] = "Reference"
query.obs["dataset_name"] = "Query"

vae_q = TOTALVI.load_query_data(
    query,
    vae,
)
vae_q.train(
    max_epochs=150,
    plan_kwargs=dict(weight_decay=0.0, scale_adversarial_loss=0.0),
)
vae_q.save("/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/11-09_run/scvi-tools-cite_query_model", overwrite=True)

# vae_q = TOTALVI.load("/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/11-09_run/scvi-tools-cite_query_model", adata=query)


query.obsm["X_totalvi_scarches"] = vae_q.get_latent_representation(query)

# predict cell types of query
predictions = vae.latent_space_classifer_.predict(query.obsm["X_totalvi_scarches"])
categories = adata.obs["celltype.l2"].astype("category").cat.categories
cat_preds = [categories[i] for i in predictions]
query.obs["celltype.l2"] = cat_preds
query.obs["predicted_l2_scarches"] = cat_preds

# I had to add query umap generation
sc.pp.neighbors(query, use_rep="X_totalvi_scarches", metric="cosine")
sc.tl.umap(query, min_dist=0.3)

sc.pl.umap(
    query,
    color=["celltype.l2", "seurat_clusters"],
    frameon=False,
    ncols=1,
    save="_evaluate_predictions"
)

query.obsm["X_umap_project"] = vae_q.umap_op_.transform(
    query.obsm["X_totalvi_scarches"]
)

sc.pl.embedding(
    query,
    "X_umap_project",
    color=["celltype.l2", "seurat_clusters"],
    frameon=False,
    ncols=1,
    save="_ref_visualization"
)

umap_adata = sc.AnnData(
    np.concatenate(
        [
            query.obsm["X_umap_project"],
            adata.obsm["X_umap"],
        ],
        axis=0,
    )
)
umap_adata.obs["celltype"] = np.concatenate(
    [query.obs["celltype.l2"].values, adata.obs["celltype.l2"].values]
)
umap_adata.obs["dataset"] = np.concatenate(
    [query.shape[0] * ["query"], adata.shape[0] * ["reference"]]
)
umap_adata.obsm["X_umap"] = umap_adata.X

inds = np.random.permutation(np.arange(umap_adata.shape[0]))
sc.pl.umap(
    umap_adata[inds],
    color=["celltype", "dataset"],
    frameon=False,
    ncols=1,
    save="_combined"
)

