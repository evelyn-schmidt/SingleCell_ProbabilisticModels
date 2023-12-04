import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
#import scrublet as scr
import scvi
import umap
from scvi.model import TOTALVI
from sklearn.ensemble import RandomForestClassifier
import anndata

# VARiABLES TO CHANGE FOR EACH RUN ============================================

# bsub -n 1 -Is -G compute/ -g /evelyn/default -q general-interactive -M 16G  -R "select[hname='compute1-exec-120.ris.wustl.edu']rusage[mem=16G]" -a 'docker(evelyns2000/foltz_tools)' /bin/bash
# mkdir /storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/12-01/
# cd  /storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/12-01/
# mkdir figures
# python3 ../../../scripts/no451/scarches_citseq_tutoral_no451.py 

ref_model = "/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/11-09_run/scvi-tools-cite_reference_model"
plots="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/12-01/figures/"
query_model = "/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/11-09_run/scvi-tools-cite_query_model"

# ref_model = "/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/451_query/11-09_run/scvi-tools-cite_reference_model"
# plots="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/451_query/11-09_run/figures/"
# query_model = "/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/451_query/11-09_run/scvi-tools-cite_query_model"

# ref_model = "/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/11-09_run/scvi-tools-cite_reference_model"
# plots="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/12-04/figures/"
# query_model = "/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/11-09_run/scvi-tools-cite_query_model"

# =============================================================================



# Building a reference model ==================================================
sc.set_figure_params(figsize=(4, 4))

#%config InlineBackend.print_figure_kwargs={'facecolor' : "w"}
#%config InlineBackend.figure_format='retina'

adata_full = sc.read('/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/data/fig4_data/adata/fig4_Protein_folzconversion_prepped.h5ad')

#adata_full = sc.read('/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/data/fig4_data/adata/fig4_Protein_folzconversion_prepped.h5ad')
#adata = adata_full[np.logical_or(adata_full.obs.batch == "730", adata_full.obs.batch == "451")].copy()
#query = adata_full[adata_full.obs.batch == "3228"].copy()


# no 451
# adata = adata[np.logical_or(adata.obs.batch == "730", adata.obs.batch == "3228")].copy()
# 3228 query
# adata = adata[np.logical_or(adata.obs.batch == "730", adata.obs.batch == "451")].copy()
# 730 query
adata = adata_full[np.logical_or(adata_full.obs.batch == "451", adata_full.obs.batch == "730")].copy()

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
# arches_params = dict(
#     use_layer_norm="both",
#     use_batch_norm="none",
#     n_layers_decoder=2,
#     n_layers_encoder=2,
# )

# vae = TOTALVI(adata, **arches_params)
# vae.train(max_epochs=250)
# vae.save("seurat_reference_model", overwrite=True)

vae = TOTALVI.load(ref_model, adata=adata)

print(vae.view_anndata_setup())

plt.plot(vae.history["elbo_train"].iloc[10:], label="train")
plt.plot(vae.history["elbo_validation"].iloc[10:], label="validation")
plt.title("Negative ELBO over training epochs")
plt.legend()
plt.savefig(plots + "verify-elbo.jpg")

adata.obsm["X_totalvi_scarches"] = vae.get_latent_representation()

# Train classifier on latenet space ###########################################
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



# Inspect reference model #####################################################
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
    save="_verify_inspect_reference.png"
)

# Map Query Date ##############################################################

# 3228 query
# query = adata[np.logical_or(adata.obs.batch == "3228"].copy()
# 730 query
query = adata_full[adata_full.obs['batch'].isin(['3228'])].copy()

# No need to preprocess 

# query.obs["doublet_scores"] = 0
# query.obs["predicted_doublets"] = True
# for s in np.unique(query.obs["set"]):
   # mask = query.obs["set"] == s
   # counts_matrix = query[mask].X.copy()
   # scrub = scr.Scrublet(counts_matrix)
   # doublet_scores, predicted_doublets = scrub.scrub_doublets()
   # query.obs["doublet_scores"].iloc[mask] = doublet_scores
   # query.obs["predicted_doublets"].iloc[mask] = predicted_doublets


query.layers["counts"] = query.X.copy()
sc.pp.normalize_total(query, target_sum=1e4)
sc.pp.log1p(query)
query.raw = query
# subset to reference vars
query = query[:, adata.var_names].copy()

#query.obsm["protein_counts"] = query.obsm["protein_expression"].copy()
query.obs["celltype.l2"] = "Unknown"
#query.obs["orig.ident"] = query.obs["set"] # Just use "batch"
#query.obsm["X_umap"] = query.obs[["UMAP1", "UMAP2"]].values

# reorganize query proteins, missing proteins become all 0
for p in adata.obsm["protein_expression"].columns:
    if p not in query.obsm["protein_expression"].columns:
        query.obsm["protein_expression"][p] = 0.0

# ensure columns are in same order
query.obsm["protein_expression"] = query.obsm["protein_expression"].loc[
    :, adata.obsm["protein_expression"].columns]

adata.obs["dataset_name"] = "Reference"
query.obs["dataset_name"] = "Query"

# Query Model Training ########################################################

#vae_q = TOTALVI.load_query_data(
    #query,
    #vae,
#)
#vae_q.train(
    #max_epochs=150,
    #plan_kwargs=dict(weight_decay=0.0, scale_adversarial_loss=0.0),
#)
# save model and reload for faster training
vae_q = TOTALVI.load(query_model, adata=query)

query.obsm["X_totalvi_scarches"] = vae_q.get_latent_representation(query)

# Query Cell Type Predictions #################################################

# predict cell types of query
predictions = vae_q.latent_space_classifer_.predict(query.obsm["X_totalvi_scarches"])
categories = adata.obs["celltype.l2"].astype("category").cat.categories
cat_preds = [categories[i] for i in predictions]
query.obs["celltype.l2"] = cat_preds
query.obs["predicted_l2_scarches"] = cat_preds

# Evaluate Label Transfer

# Added query umap generation (not in tutoral)
sc.pp.neighbors(query, use_rep="X_totalvi_scarches", metric="cosine")
sc.tl.umap(query, min_dist=0.3)

sc.pl.umap(
    query,
    color=["celltype.l2", "seurat_clusters"],
    frameon=False,
    ncols=1,
    save="_verify_evaluate_predictions.png"
)

# Use Reference UMAP
query.obsm["X_umap_project"] = vae_q.umap_op_.transform(
    query.obsm["X_totalvi_scarches"]
)

sc.pl.embedding(
    query,
    "X_umap_project",
    color=["celltype.l2", "seurat_clusters"],
    frameon=False,
    ncols=1,
    save="_verify_ref_visualization.png"
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
    save="_verify_combined.png"
)


# Analysis ====================================================================

# Should always be 1.0
print("Acc: {}".format(np.mean(adata.obs["seurat_clusters"] == adata.obs["celltype.l2"])))

# Get Accuarcy for Query Only

query.obs["ref_celltypes_unified"] = query.obs["seurat_clusters"].replace({
    })
query.obs["celltypes_unified_predicted"] = query.obs["celltype.l2"].replace({
    })


ref_categories = set(query.obs["ref_celltypes_unified"].unique())
predicted_categories = set(query.obs["celltypes_unified_predicted"].unique())

common_categories = ref_categories.intersection(predicted_categories)

if common_categories:
    ref_matches = query.obs["ref_celltypes_unified"].isin(common_categories)
    predicted_matches = query.obs["celltypes_unified_predicted"].isin(common_categories)

    total_matches = np.sum(ref_matches & predicted_matches)
    total_samples = len(query.obs)

    accuracy = total_matches / total_samples
    print("Acc: {}".format(accuracy))
else:
    print("NA")

print("Acc: {}".format(np.mean(query.obs["ref_celltypes_unified"] == query.obs["celltypes_unified_predicted"])))

# Get Accuaracy For Full Model
adata.obs["ref_celltypes_unified"] = adata.obs["seurat_clusters"].replace({     
    })
adata.obs["celltypes_unified_predicted"] = adata.obs["celltype.l2"].replace({
    })

full_adata = anndata.concat([adata, query])

print("Acc: {}".format(np.mean(full_adata.obs["ref_celltypes_unified"] == full_adata.obs["celltypes_unified_predicted"])))
# Acc: 0.8958766670745136 for Full Model

#full_adata.obs['orig.ident'] = full_adata.obs['orig.ident'].astype("string")
#full_adata.obs['seurat_clusters'] = full_adata.obs['seurat_clusters'].astype("string")
#full_adata.obs['celltypes_unified_predicted'] = full_adata.obs['celltypes_unified_predicted'].astype("string")
#full_adata.obs['ref_celltypes_unified'] = full_adata.obs['ref_celltypes_unified'].astype("string")
#full_adata.obs['celltype.l2'] = full_adata.obs['celltype.l2'].astype("string")

full_adata.write_h5ad("full_object_annotated.h5ad")
