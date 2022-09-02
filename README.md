# Single-cell multimodal modeling with deep parametric inference
The proliferation of single-cell multimodal sequencing technologies has enabled us to understand cellular heterogeneity with multiple views, providing novel and actionable biological insights into the disease-driving mechanisms. Here, we propose the deep parametric inference (DPI) model, an end-to-end framework for single-cell multimodal data analysis. At the heart of DPI is the multimodal parameter space, where the parameters from each modal are inferred by neural networks. 
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
* We use pbmc1k data set to demonstrate the process of DPI analysis of single cell multimodal data.
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
sc_data = sc.read_h5ad("PBMC1k_data.h5ad")
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
sc_data = sc_data[:,sc_data.var["highly_variable"]]
dpi.scale(sc_data)
```
### Prepare and run DPI model
* Configure DPI model parameters
```python
dpi.build_mix_model(sc_data, net_dim_rna_list=[512, 128], net_dim_pro_list=[128], net_dim_rna_mean=128, net_dim_pro_mean=128, net_dim_mix=128, lr=0.0001)
```
* Run DPI model
```python
dpi.fit(sc_data)
```
* Visualize the loss
```python
dpi.loss_plot(sc_data)
```
### Visualize the latent space
* Extract latent spaces
```python
dpi.get_spaces(sc_data)
```
* Visualize the spaces
```python
dpi.space_plot(sc_data, "mm_parameter_space", color="green", kde=True, bins=30)
dpi.space_plot(sc_data, "rna_latent_space", color="orange", kde=True, bins=30)
dpi.space_plot(sc_data, "pro_latent_space", color="blue", kde=True, bins=30)
```
### Preparation for downstream analysis
* Extract features
```python
dpi.get_features(sc_data)
```
* Get denoised datas
```python
dpi.get_denoised_rna(sc_data)
dpi.get_denoised_pro(sc_data)
```
### Cell clustering and visualization
* Cell clustering
```python
sc.pp.neighbors(sc_data, use_rep="mix_features")
dpi.umap_run(sc_data, min_dist=0.4)
sc.tl.leiden(sc_data)
```
* Cell cluster visualization
```python
sc.pl.umap(sc_data, color="leiden")
```
