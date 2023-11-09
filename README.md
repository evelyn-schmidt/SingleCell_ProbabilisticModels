# Single Cell Probabilistic Models

Attempting to use scANVI and TOTALVI to label cell types of single cell data

## Tutorials followed
### scANVI
- https://docs.scarches.org/en/latest/scanvi_surgery_pipeline.html
### TOTALVI
- https://yoseflab.github.io/scvi-tools-reproducibility/scarches_totalvi_seurat_data/#totalvi-scarches
- https://docs.scvi-tools.org/en/1.0.2/tutorials/notebooks/totalVI_reference_mapping.html
- https://docs.scarches.org/en/latest/totalvi_surgery_pipeline.html

## Data Preperation Notes (Model_Adata_Preperation.ipynb)
We began with a seurat object so the first step is to convert the suerat object to an adata object. 
This is done by exporting csv from the meta, ADT, and RNA assays within the orginal sueart object. 
After those csvs are combined into the the adata object, there is some processing done to make sure 
the adata object is correctly formatted and matches the second adata object which will be used
for an "offline" query. 

## Models Run
To test how best to utlized these twos we have come up with the tests:
- Training on fig4 data and using fig7 data as query (refered to as offline  accoridng to yosef)
- Training on the 3 cominations of the batches present in teh fig4 data (451, 730, 3228)

