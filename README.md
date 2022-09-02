# Modeling and analyzing single-cell multimodal data with deep parametric inference
The proliferation of single-cell multimodal sequencing technologies has enabled us to understand cellular heterogeneity with multiple views, providing novel and actionable biological insights into the disease-driving mechanisms. Here, we propose a comprehensive end-to-end single-cell multimodal data analysis framework named Deep Parametric Inference (DPI). The python packages, datasets and user-friendly manuals of DPI are freely available at https://github.com/studentiz/dpi.

## The dpi framework works with scanpy and supports the following single-cell multimodal analyses
* Multimodal data integration
* Multimodal data noise reduction
* Cell clustering and visualization
* Reference and query cell types
* Cell state vector field visualization
## Pip install
```python
pip install dpi-sc
```
## Datasets
The dataset participating in "Single-cell multimodal modeling with deep parametric inference" can be downloaded at [DPI data warehouse](http://101.34.64.251:88/)
## Tutorial
We use Peripheral Blood Mononuclear Cell (PBMC) dataset to demonstrate the process of DPI analysis of single cell multimodal data. The following code is recommended to run on a computer with more than 64G memory.
### Import dependencies
```python
import scanpy as sc
import dpi
```
### Retina image output (optional)
```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```
### Load dataset
```python
# The dataset can be downloaded from [Datasets] above.
sc_data = sc.read_h5ad("PBMC_COVID19_Healthy_Annotated.h5ad")
```
### Set marker collection
```python
rna_markers = ["CCR7", "CD19", "CD3E", "CD4"]
protein_markers = ["AB_CCR7", "AB_CD19", "AB_CD3", "AB_CD4"]
```
### Preprocessing
```python
dpi.preprocessing(sc_data)
dpi.normalize(sc_data, protein_expression_obsm_key="protein_expression")
sc_data.var_names_make_unique()
sc.pp.highly_variable_genes(
    sc_data,
    n_top_genes=3000,
    flavor="seurat_v3",
    subset=False
)
dpi.add_genes(sc_data, rna_markers)
sc_data = sc_data[:,sc_data.var["highly_variable"]]
dpi.scale(sc_data)
```
### Prepare and run DPI model
Configure DPI model parameters
```python
dpi.build_mix_model(sc_data, net_dim_rna_list=[512, 128], net_dim_pro_list=[128], net_dim_rna_mean=128, net_dim_pro_mean=128, net_dim_mix=128, lr=0.0001)
```
Run DPI model
```python
dpi.fit(sc_data, batch_size=256)
```
Visualize the loss
```python
dpi.loss_plot(sc_data)
```
### Save DPI model (optional)
```python
dpi.saveobj2file(sc_data, "COVID19PBMC_healthy.dpi")
#sc_data = dpi.loadobj("COVID19PBMC_healthy.dpi")
```
### Visualize the latent space
Extract latent spaces
```python
dpi.get_spaces(sc_data)
```
Visualize the spaces
```python
dpi.space_plot(sc_data, "mm_parameter_space", color="green", kde=True, bins=30)
dpi.space_plot(sc_data, "rna_latent_space", color="orange", kde=True, bins=30)
dpi.space_plot(sc_data, "pro_latent_space", color="blue", kde=True, bins=30)
```
### Preparation for downstream analysis
Extract features
```python
dpi.get_features(sc_data)
```
Get denoised datas
```python
dpi.get_denoised_rna(sc_data)
dpi.get_denoised_pro(sc_data)
```
### Cell clustering and visualization
Cell clustering
```python
sc.pp.neighbors(sc_data, use_rep="mix_features")
dpi.umap_run(sc_data, min_dist=0.4)
sc.tl.leiden(sc_data)
```
Cell cluster visualization
```python
sc.pl.umap(sc_data, color="leiden")
```
### Observe multimodal data markers
RNA markers
```python
dpi.umap_plot(sc_data, featuretype="rna", color=rna_markers, ncols=2)
dpi.umap_plot(sc_data, featuretype="rna", color=rna_markers, ncols=2, layer="rna_denoised")
```
Protein markers
```python
dpi.umap_plot(sc_data, featuretype="protein", color=protein_markers, ncols=2)
dpi.umap_plot(sc_data, featuretype="protein", color=protein_markers, ncols=2, layer="pro_denoised")
```
### Reference and query
Reference objects need to be pre-set with cell labels.
```python
sc.pl.umap(sc_data, color="initial_clustering", frameon=False, title="PBMC COVID19 Healthy labels")
```
Demonstrate reference and query capabilities with unannotated asymptomatic COVID-19 PBMCs.
```python
# The dataset can be downloaded from [Datasets] above.
filepath = "COVID19_Asymptomatic.h5ad"
sc_data_COVID19_Asymptomatic = sc.read_h5ad(filepath)
```
Unannotated data also needs to be normalized.
```python
dpi.normalize(sc_data_COVID19_Asymptomatic, protein_expression_obsm_key="protein_expression")
```
Referenced and queried objects require alignment features.
```python
sc_data_COVID19_Asymptomatic = sc_data_COVID19_Asymptomatic[:,sc_data.var.index]
```
Run the automated annotation function.
```python
dpi.annotate(sc_data, ref_labelname="initial_clustering", sc_data_COVID19_Asymptomatic)
```
Visualize the annotated object.
```python
sc.pl.umap(sc_data_COVID19_Asymptomatic, color="labels", frameon=False, title="PBMC COVID19 Asymptomatic Annotated")
```
### Cell state vector field
Simulate and visualize the cellular state when the CCR7 protein is amplified 2-fold.
```python
dpi.cell_state_vector_field(sc_data, feature="AB_CCR7", amplitude=2, obs="initial_clustering", featuretype="protein")
```
