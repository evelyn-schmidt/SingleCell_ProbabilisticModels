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
import scvi
import pandas as pd
import time
import warnings
import matplotlib.font_manager
import seaborn as sns
import logging


# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/temp/myapp.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)


    
N_EPOCHS=250
scvi.settings.seed = 0



def classify_from_latent(
    ref: sc.AnnData, 
    query: sc.AnnData, 
    ref_obsm_key: str = "X_totalvi_scarches", 
    labels_obs_key: str = "celltype.l2",
    classifier: str = "random_forest",
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    y_train = ref.obs[labels_obs_key].astype("category").cat.codes.to_numpy()
    X_train = ref.obsm[ref_obsm_key]
    if classifier == "random_forest":
        clf = RandomForestClassifier(
            random_state=1, 
            class_weight = "balanced_subsample",
            verbose=1,
            n_jobs=-1,
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32,), 
            random_state=1, 
            max_iter=300, 
            verbose=True, 
            early_stopping=True, 
            learning_rate_init=1e-3
        )
    clf.fit(X_train, y_train)
    predictions = clf.predict(query.obsm[ref_obsm_key])
    categories = ref.obs[labels_obs_key].astype("category").cat.categories
    cat_preds = [categories[i] for i in predictions]
    
    return cat_preds

def run_totalvi_scarches(ref, query):
    """Run online totalVI."""

    # initialize and train model
    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        n_layers_decoder=2,
        n_layers_encoder=2,
    )

    start = time.time()
    vae = scvi.model.TOTALVI(
        ref, 
        **arches_params
    )
    vae.train(max_epochs=N_EPOCHS, batch_size=256, lr=4e-3)
    end = time.time()
    print("\n Total reference train time: {}".format(end-start))
    logging.info("\n Total reference train time: {}".format(end-start))

    plt.plot(vae.history["elbo_validation"][10:], label="validation")
    plt.title("Negative ELBO over training epochs")
    plt.legend()

    ref.obsm["X_totalvi_scarches"] = vae.get_latent_representation()

    dir_path = "yosef_ref_model/"
    vae.save(dir_path, overwrite=True)

    start = time.time()
    vae_q = scvi.model.TOTALVI.load_query_data(
        query, 
        vae,
    )

    vae_q.train(
        150, 
        lr=4e-3, 
        batch_size=256, 
        plan_kwargs=dict(
            weight_decay=0.0,
            scale_adversarial_loss=0.0
        ),
        # n_steps_kl_warmup=1,
    )
    end = time.time()
    print("\n Total query train time: {}".format(end-start))
    logging.info("\n Total query train time: {}".format(end-start))

    dir_path = "yosef_q_model/"
    vae_q.save(dir_path, overwrite=True)

    query.obsm["X_totalvi_scarches"] = vae_q.get_latent_representation(query)

    # predict cell types of query
    query.obs["predicted_l2_scarches"] = classify_from_latent(ref, query, ref_obsm_key="X_totalvi_scarches")
    query.obs["celltype.l2"] = query.obs["predicted_l2_scarches"]

    adata_full_new = anndata.concat([ref, query])
    adata_full_new.uns["_scvi"] = query.uns["_scvi"].copy()
    adata_full_new.obsm["X_totalvi_scarches"] = vae_q.get_latent_representation(adata_full_new)

    print("Computing full umap")
    sc.pp.neighbors(adata_full_new, use_rep="X_totalvi_scarches", metric="cosine")
    sc.tl.umap(adata_full_new, min_dist=0.3)

    return vae, vae_q, adata_full_new

def main():

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
    
    TF_CPP_MIN_LOG_LEVEL=0
    sca.models.TOTALVI.setup_anndata(
        adata_ref,
        batch_key="batch",
        protein_expression_obsm_key="protein_expression",
    )

    sca.models.TOTALVI.setup_anndata(
        adata_query,
        batch_key="batch",
        protein_expression_obsm_key="protein_expression",
    )

    adata_query = adata_query[:, adata_ref.var_names].copy()
    adata_query.obs["celltype.l3"] = "Unknown"
    adata_query.obs["celltype.l2"] = "Unknown"

    # reorganize query proteins, missing proteins become all 0
    for p in adata_ref.obsm["protein_expression"].columns:
        if p not in adata_query.obsm["protein_expression"].columns:
            adata_query.obsm["protein_expression"][p] = 0.0
    adata_query.obsm["protein_expression"] = adata_query.obsm["protein_expression"].loc[:, adata_ref.obsm["protein_expression"].columns]



    vae_arches, vae_q_arches, full_data_arches = run_totalvi_scarches(adata_ref, adata_query)






if __name__ == "__main__":
    main()