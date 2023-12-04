import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
#import scrublet as scr
import scvi
import umap
from scvi.model import TOTALVI
from sklearn.ensemble import RandomForestClassifier
import anndata
# Evaluate the Accuarcy by Celltype
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#object="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/no_451_in_reference/11-27/"
#object="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/451_query/12-01/"
#object="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/730_query/12-01/"
#object="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/3228_query/12-01/"

object="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/TOTALVI/models/full_model/11-18/"

adata_full = sc.read(object+"full_object_annotated.h5ad")
adata = adata_full[adata_full.obs['dataset_name'].isin(['Reference'])].copy()
query = adata_full[adata_full.obs['dataset_name'].isin(['Query'])].copy()

# Bar Plot of Acc by cell type =====================================================================================================
obs_df = pd.DataFrame({
    'ref_celltypes_unified': adata_full.obs['ref_celltypes_unified'],
    'celltypes_unified_predicted': adata_full.obs['celltypes_unified_predicted']
})

# Calculate the mean for each category
mean_accuracy_by_category = obs_df.groupby("ref_celltypes_unified") \
    .apply(lambda x: np.mean(x["ref_celltypes_unified"] == x["celltypes_unified_predicted"]))

# Plot the results
plt.figure(figsize=(10, 8))
mean_accuracy_by_category.plot(kind='bar', color='skyblue')
plt.title('Mean Accuracy by Cell Type Category')
plt.xlabel('Cell Type Category')
plt.ylabel('Mean Accuracy')
plt.xticks(rotation=45, ha='right')
plt.savefig(object+"figures/acc.jpg")

'''
# add the orginal numbers back and graph
full_query = sc.read('/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/fig4_foltz/SingleCell_ProbabilisticModels/data/fig7_data/adata/fig7_k562_pagaobject3223withleiden31423pagaonleidenclusters_RAW.h5ad')
query.obs['leiden_original'] = full_query.obs['leiden']

sc.pl.umap(
    query,
    color=['celltypes_unified',"celltypes_unified_predicted"],
    frameon=False,
    ncols=1,
    palette={"CD56dim":"#F8766D", "CD56bright":"#0CB702", "ML1":"#CD9600", "ML2":"#ABA300", "Proliferating":"#8494FF", "nonNK":"#FF61CC", "Other":"#00BFC4"},
    save="_pallet_test"
)

sc.pl.umap(
    query,
    color=['celltypes_unified','celltypes_unified_predicted', 'leiden_original'],
    frameon=False,
    ncols=1,
    save="_compare_fig7orginal_clusters"
)


# umap by Cell_Types for comparison
sc.pl.umap(
    query[query.obs['Cell_Types'] == "ML_unstim"].copy(),
    color=['celltypes_unified','celltypes_unified_predicted', 'Cell_Types'],
    frameon=False,
    ncols=1,
    save="_compare_fig7_ML_unstim"
)

sc.pl.umap(
    query[query.obs['Cell_Types'] == "LD_K562_2hr"].copy(),
    color=['celltypes_unified','celltypes_unified_predicted', 'Cell_Types'],
    frameon=False,
    ncols=1,
    save="_compare_fig7_LD_K562_2hr"
)

sc.pl.umap(
    query[query.obs['Cell_Types'] == "LD_K562_4hr"].copy(),
    color=['celltypes_unified','celltypes_unified_predicted', 'Cell_Types'],
    frameon=False,
    ncols=1,
    save="_compare_fig7_LD_K562_4hr"
)

sc.pl.umap(
    query[query.obs['Cell_Types'] == "LD_unstim"].copy(),
    color=['celltypes_unified','celltypes_unified_predicted', 'Cell_Types'],
    frameon=False,
    ncols=1,
    save="_compare_fig7_LD_unstim"
)

sc.pl.umap(
    query[query.obs['Cell_Types'] == "ML_K562_2hr"].copy(),
    color=['celltypes_unified','celltypes_unified_predicted', 'Cell_Types'],
    frameon=False,
    ncols=1,
    save="_compare_fig7_ML_K562_2hr"
)

sc.pl.umap(
    query[query.obs['Cell_Types'] == "ML_K562_4hr"].copy(),
    color=['celltypes_unified','celltypes_unified_predicted', 'Cell_Types'],
    frameon=False,
    ncols=1,
    save="_compare_fig7_ML_K562_4hr"
)

'''

# https://www.analyticsvidhya.com/blog/2021/12/evaluation-of-classification-model/
# generate confusion matrix for all runs and compare ======================================================================================
# df = query.obs.groupby(["celltype.l2", "leiden"]).size().unstack(fill_value=0)
df =  query.obs.groupby(["celltype.l2", "seurat_clusters"]).size().unstack(fill_value=0)
norm_df = df / df.sum(axis=0)

plt.figure(figsize=(8, 8))
# Plot heatmap
heatmap = plt.pcolor(norm_df, cmap='viridis')
# Add numerical values to each cell
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        plt.text(j + 0.5, i + 0.5, f'{df.iloc[i, j]}', ha='center', va='center', color='white')
# Customize ticks and labels
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90, ha='right')  # <-- Modified line
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.ylabel("Predicted")
plt.xlabel("Observed")
# Add colorbar
plt.colorbar(heatmap)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(object + "figures/query_confusion_matrix.jpg")
plt.show()

# Plot Heatmap with values ================================================================================================================
adata_full.obs['celltypes_unified_predicted'] = adata_full.obs['celltypes_unified_predicted'].astype("category")
df = adata_full.obs.groupby(["ref_celltypes_unified", "celltypes_unified_predicted"]).size().unstack(fill_value=0)
norm_df = df / df.sum(axis=0)

plt.figure(figsize=(8, 8))
# Plot heatmap
heatmap = plt.pcolor(norm_df, cmap='viridis')

# Add numerical values to each cell
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        plt.text(j + 0.5, i + 0.5, f'{df.iloc[i, j]}', ha='center', va='center', color='white')

# Customize ticks and labels
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.ylabel("Predicted")
plt.xlabel("Observed")
# Add colorbar
plt.colorbar(heatmap)
plt.savefig(object + "figures/full_confusion_matrix.jpg") 


# Subset just the unstimulated Cells and see predictions  ===========================================================================
# Only for full model
unstim_query = query[query.obs["Cell_Types"].isin(['ML_unstim','LD_unstim']),:]
sc.metrics.confusion_matrix("celltypes_unified", "celltypes_unified_predicted", unstim_query.obs)


# Bar plot of Acc for cell type and cell states ===================================================================================

# Assuming query and plots are defined
obs_df = pd.DataFrame({
    'ref_celltypes_unified': query.obs['ref_celltypes_unified'],
    'celltypes_unified_predicted': query.obs['celltypes_unified_predicted'],
    'Cell_Types': query.obs['Cell_Types']
})

obs_df['ref_celltypes_unified'] = obs_df['ref_celltypes_unified'].astype(str)
obs_df['celltypes_unified_predicted'] = obs_df['celltypes_unified_predicted'].astype(str)

# Calculate the mean for each category
mean_accuracy_by_category = obs_df.groupby(['Cell_Types', 'ref_celltypes_unified']) \
   .apply(lambda x: np.mean(x['ref_celltypes_unified'] == x['celltypes_unified_predicted']))


fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

mean_accuracy_by_category.unstack().plot(kind='bar', width=0.8, colormap='viridis', edgecolor='black', ax=ax)

ax.legend(title='Ref Cell Type Unified', bbox_to_anchor=(1.05, 1), loc='upper left')

fig.savefig(object + "figures/acc-by-celltype.jpg", bbox_inches='tight')
plt.show()

# loop and get Accuracies



# Loop and create confusion matrices 

