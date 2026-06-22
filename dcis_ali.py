#!/usr/bin/env python
# coding: utf-8

# # Loading

# In[1]:


import scanpy as sc
import pandas as pd
import anndata
import numpy as np
#import scvi-tools
from scipy.sparse import csr_matrix
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster #, sch
from scipy.spatial.distance import squareform, pdist
sc.settings.verbosity = 0             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')
sc.settings.seed = 1
import warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import LinearSegmentedColormap
from gseapy import enrichr, barplot, dotplot
import scrublet


# In[2]:


import infercnvpy as cnv
import pybiomart
import scipy.sparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from gseapy import enrichr, barplot, dotplot
from matplotlib.colors import LinearSegmentedColormap


# In[3]:


import importlib.metadata
print(importlib.metadata.version("scvi-tools"))


# In[4]:


import gseapy as gp


# In[6]:


# save csv file of all packages & versions used
import pkg_resources

packages = pd.DataFrame(
    [(dist.project_name, dist.version)
     for dist in pkg_resources.working_set],
    columns=["package", "version"]
)

packages = packages.sort_values("package")

packages.to_csv("environment_packages.csv", index=False)


# In[66]:


# set colour palette
celltype_colors = {
    "Basal": "#17becf",# teal
    "Luminal Progenitor": "#e377c2",# pink
    "Luminal Mature": "8c564b",# brown
    "Endothelial": "#9467bd",# purple
    "Fibroblast": "#2ca02c",# green
    "General Myeloid": "#fee08b",# yellow
    "T-Cell": "#bcbd22",# olive
    "B-Cell": "#d62728",# red
    "Macrophage": "#1f77b4",# blue
    "Monocyte": "#ffbb78" ,# light blue
    "Non-epithelial": "#1c1c84",# navy
    "Epithelial": "#91bfdb",# lighter blue
    "DCIS": "#fdbf6f",# gold
    "normal": "#7f7f7f"#grey
}


# In[5]:


sc.settings.set_figure_params(
    dpi=300,      
    dpi_save=600, 
    fontsize=12,
    frameon=True,
    vector_friendly=True
)


# In[67]:


def set_celltype_colors(adata, obs_key, color_dict, default="#cccccc"):
    if not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")

    cats = adata.obs[obs_key].cat.categories
    adata.uns[f"{obs_key}_colors"] = [
        color_dict.get(c, default) for c in cats
    ]


# In[74]:


cell_type_order = ['Basal', 'Luminal Progenitor', 'Luminal Mature', 'Endothelial', 'Fibroblast', 'General Myeloid', 'T-Cell', 'B-Cell', 'Macrophage', 'Monocyte']

alldata_N2025.obs['cell type'] = pd.Categorical(
    alldata_N2025.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)
alldata_Q2025.obs['cell type'] = pd.Categorical(
    alldata_Q2025.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)
alldata_T2022.obs['cell type'] = pd.Categorical(
    alldata_T2022.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)
alldata_W2022.obs['cell type'] = pd.Categorical(
    alldata_W2022.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)
alldata_G2017.obs['cell type'] = pd.Categorical(
    alldata_G2017.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)

alldata.obs['cell type'] = pd.Categorical(
    alldata.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)


# In[75]:


set_celltype_colors(alldata_N2025, "cell type", celltype_colors)
set_celltype_colors(alldata_N2025, "Epithelial_vs_NonEpithelial", celltype_colors)
set_celltype_colors(alldata_N2025, "cnv_status", celltype_colors)

set_celltype_colors(alldata_Q2025, "cell type", celltype_colors)
set_celltype_colors(alldata_Q2025, "Epithelial_vs_NonEpithelial", celltype_colors)
set_celltype_colors(alldata_Q2025, "cnv_status", celltype_colors)

set_celltype_colors(alldata_T2022, "cell type", celltype_colors)
set_celltype_colors(alldata_T2022, "Epithelial_vs_NonEpithelial", celltype_colors)
set_celltype_colors(alldata_T2022, "cnv_status", celltype_colors)

set_celltype_colors(alldata_W2022, "cell type", celltype_colors)
set_celltype_colors(alldata_W2022, "Epithelial_vs_NonEpithelial", celltype_colors)
set_celltype_colors(alldata_W2022, "cnv_status", celltype_colors)

set_celltype_colors(alldata_G2017, "cell type", celltype_colors)
set_celltype_colors(alldata_G2017, "Epithelial_vs_NonEpithelial", celltype_colors)
set_celltype_colors(alldata_G2017, "cnv_status", celltype_colors)

set_celltype_colors(alldata, "cell type", celltype_colors)


# ### Q2025 (fresh)

# In[2]:


#HTA6_2495_1
ind1_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2495_1-")
#HTA6_2498_2
ind2_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2498_2-")
#HTA6_2499_1
ind3_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2499_1-")
#HTA6_2500_1
ind4_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2500_1-")
#HTA6_2502_1
ind5_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2502_1-")
#HTA6_2502_2
ind6_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2502_2-")
#HTA6_2502_3
ind7_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2502_3-")
#HTA6_2502_4
ind8_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2502_4-")
#HTA6_2503_1
ind9_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2503_1-")
#HTA6_2503_2
ind10_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2503_2-")
#HTA6_2504_1
ind11_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2504_1-")
#HTA6_2504_2
ind12_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2504_2-")
#HTA6_2505_1
ind13_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2505_1-")
#HTA6_2505_2
ind14_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2505_2-")
#HTA6_2506_1
ind15_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2506_1-")
#HTA6_2508_2
ind16_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2508_2-")
#HTA6_2509_1
ind17_Q2025_data = sc.read_10x_mtx("./data/AAA_DCIS/Q2025/", prefix="HTA6_2509_1-")


# ### T2022 (fresh)

# In[3]:


#NCCBC2
ind1_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852268_NCCBC2_filtered_feature_bc_matrix_NCCBC2_")
#NCCBC3
ind2_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852269_NCCBC3_filtered_feature_bc_matrix_")
#NCCBC5
ind3_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852270_NCCBC5_filtered_feature_bc_matrix_")
#NCCBC6
ind4_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852271_NCCBC6_filtered_feature_bc_matrix_")
#NCCBC11
ind5_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852273_NCCBC11_filtered_feature_bc_matrix_")
#NCCBC13
ind6_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852274_NCCBC13_filtered_feature_bc_matrix_")
#NCCBC14
ind7_T2022_data = sc.read_10x_mtx("./data/AAA_DCIS/T2022/", prefix="GSM5852275_NCCBC14_filtered_feature_bc_matrix_")


# ### W2022 (fresh)

# In[4]:


#DCIS1_scRNA
ind1_W2022_data = sc.read_csv("./data/AAA_DCIS/W2022/GSM5493628_DCIS1.sc.count.matrix.csv.gz").T
#DCIS2_scRNA
ind2_W2022_data = sc.read_csv("./data/AAA_DCIS/W2022/GSM5493630_DCIS2.sc.count.matrix.csv.gz").T


# ### G2017 (fresh)

# In[5]:


#DCIS1
ind1_G2017_data = pd.read_csv("./data/AAA_DCIS/G2017/GSM4476485_combined_UMIcount_CellTypes_DCIS1.txt.gz", sep="\t", index_col=0, skiprows=[1, 2])
ind1_G2017_data = sc.AnnData(ind1_G2017_data.T)


# ### N2025 (FFPE)

# In[587]:


#FFPE DCIS 4
ind1_N2025_data = sc.read_10x_mtx("./data/AAA_DCIS/N2025/", prefix="GSM8768145_")
#FFPE DCIS 12
ind2_N2025_data = sc.read_10x_mtx("./data/AAA_DCIS/N2025/", prefix="GSM8768146_")
#FFPE DCIS 14
ind3_N2025_data = sc.read_10x_mtx("./data/AAA_DCIS/N2025/", prefix="GSM8768147_")
#FFPE DCIS 17
ind4_N2025_data = sc.read_10x_mtx("./data/AAA_DCIS/N2025/", prefix="GSM8768148_")
#FFPE DCIS 25
ind5_N2025_data = sc.read_10x_mtx("./data/AAA_DCIS/N2025/", prefix="GSM8768149_")
#FFPE DCIS 15
ind6_N2025_data = sc.read_10x_mtx("./data/AAA_DCIS/N2025/", prefix="GSM8768150_")


# # Doublet Removal (on single samples, not integrated)

# In[6]:


ind1_Q2025_data
sc.pp.scrublet(ind1_Q2025_data)
ind1_Q2025_data


# In[7]:


sc.pp.scrublet(ind2_Q2025_data)
sc.pp.scrublet(ind3_Q2025_data)
sc.pp.scrublet(ind4_Q2025_data)
sc.pp.scrublet(ind5_Q2025_data)
sc.pp.scrublet(ind6_Q2025_data)
sc.pp.scrublet(ind7_Q2025_data)
sc.pp.scrublet(ind8_Q2025_data)
sc.pp.scrublet(ind9_Q2025_data)
sc.pp.scrublet(ind10_Q2025_data)
sc.pp.scrublet(ind11_Q2025_data)
sc.pp.scrublet(ind12_Q2025_data)
sc.pp.scrublet(ind13_Q2025_data)
sc.pp.scrublet(ind14_Q2025_data)
sc.pp.scrublet(ind15_Q2025_data)
sc.pp.scrublet(ind16_Q2025_data)
sc.pp.scrublet(ind17_Q2025_data)


# In[8]:


sc.pp.scrublet(ind1_T2022_data)
sc.pp.scrublet(ind2_T2022_data)
sc.pp.scrublet(ind3_T2022_data)
sc.pp.scrublet(ind4_T2022_data)
sc.pp.scrublet(ind5_T2022_data)
sc.pp.scrublet(ind6_T2022_data)
sc.pp.scrublet(ind7_T2022_data)


# In[9]:


sc.pp.scrublet(ind1_W2022_data)
sc.pp.scrublet(ind2_W2022_data)


# In[10]:


sc.pp.scrublet(ind1_G2017_data)


# # PreProcessing

# In[588]:


ribo_url = "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt"
ribo_genes = pd.read_table(ribo_url, skiprows=2, header = None)


# In[589]:


def pp(adata, samplename, batch, expdesign=None, thr_mito=20, thr_ribo=2, verbose=True):
    if not 'Sample' in adata.obs:
        adata.obs['Sample'] = samplename
    else:
        print("sample is already there, which might be ok")
    adata.obs['Batch'] = batch
    adata.obs['Experiment'] = expdesign
    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=3) #get rid of genes that are found in fewer than 3 cells
    if verbose:
        print(adata.shape)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
    if verbose:
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_ribo')
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
    adata = adata[adata.obs.n_genes_by_counts < upper_lim]
    adata = adata[adata.obs.pct_counts_mt < thr_mito]
    if verbose:
        print(adata.shape)
    adata = adata[adata.obs.pct_counts_ribo > thr_ribo]
    if verbose:
        print(adata.shape)
    return adata


# In[13]:


# Q2025
designs = ["10x Genomics"]*17
samples = [f"ind{i}_Q2025" for i in range(1,18)]
dataobjects_Q2025 = ["ind1_Q2025_data", "ind2_Q2025_data", "ind3_Q2025_data",
                    "ind4_Q2025_data", "ind5_Q2025_data", "ind6_Q2025_data",
                    "ind7_Q2025_data", "ind8_Q2025_data", "ind9_Q2025_data",
                    "ind10_Q2025_data", "ind11_Q2025_data", "ind12_Q2025_data",
                     "ind13_Q2025_data", "ind14_Q2025_data", "ind15_Q2025_data",
                     "ind16_Q2025_data", "ind17_Q2025_data"]

for i,d in enumerate(dataobjects_Q2025):
    exec(f"{dataobjects_Q2025[i]} = pp({d}, samples[i], 'Q2025', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=True)")


# In[14]:


# T2022
designs = ["10x Genomics"]*7
samples = [f"ind{i}_T2022" for i in range(1,8)]
dataobjects_T2022 = ["ind1_T2022_data", "ind2_T2022_data", "ind3_T2022_data",
                    "ind4_T2022_data", "ind5_T2022_data", "ind6_T2022_data",
                    "ind7_T2022_data"]

for i,d in enumerate(dataobjects_T2022):
    exec(f"{dataobjects_T2022[i]} = pp({d}, samples[i], 'T2022', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=True)")


# In[15]:


# G2017
designs = ["10x Genomics"]*1
samples = [f"ind{i}_G2017" for i in range(1,2)]
dataobjects_G2017 = ["ind1_G2017_data"]

for i,d in enumerate(dataobjects_G2017):
    exec(f"{dataobjects_G2017[i]} = pp({d}, samples[i], 'G2017', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=True)")


# In[16]:


# W2022
designs = ["10x Genomics"]*2
samples = [f"ind{i}_W2022" for i in range(1,3)]
dataobjects_W2022 = ["ind1_W2022_data", "ind2_W2022_data"]

for i,d in enumerate(dataobjects_W2022):
    exec(f"{dataobjects_W2022[i]} = pp({d}, samples[i], 'W2022', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=True)")


# In[564]:


# prints counts at each step to diagnose issue
def npp(adata, samplename, batch, expdesign=None, thr_mito=20, thr_ribo=2, verbose=True):
    if not 'Sample' in adata.obs:
        adata.obs['Sample'] = samplename
    else:
        print("sample is already there, which might be ok")
    adata.obs['Batch'] = batch
    adata.obs['Experiment'] = expdesign
    print(f"Initial shape: {adata.shape}")
    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    print(f"After min_genes filter: {adata.shape}")
    sc.pp.filter_genes(adata, min_cells=3) #get rid of genes that are found in fewer than 3 cells
    print(f"After min_cells filter: {adata.shape}")
    if verbose:
        print(adata.shape)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
    print(f"Before mito/ribo/upper_lim filtering: {adata.shape}")
    if verbose:
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_ribo')
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
    adata = adata[adata.obs.n_genes_by_counts < upper_lim]
    print(f"After 98th percentile n_genes filter: {adata.shape}")
    adata = adata[adata.obs.pct_counts_mt < thr_mito]
    print(f"After MT% filter: {adata.shape}")
    if verbose:
        print(adata.shape)
    adata = adata[adata.obs.pct_counts_ribo > thr_ribo]
    print(f"After ribo% filter: {adata.shape}")
    if verbose:
        print(adata.shape)
    return adata


# In[590]:


# N2025
designs = ["10x Genomics"]*6
samples = [f"ind{i}_N2025" for i in range(1,7)]
dataobjects_N2025 = ["ind1_N2025_data", "ind2_N2025_data", "ind3_N2025_data",
                    "ind4_N2025_data", "ind5_N2025_data", "ind6_N2025_data"]

for i,d in enumerate(dataobjects_N2025):
    exec(f"{dataobjects_N2025[i]} = pp({d}, samples[i], 'N2025', expdesign=designs[i], thr_mito=20, thr_ribo=0, verbose=True)")


# # Concatenating

# In[17]:


alldatasets = dataobjects_W2022 + dataobjects_Q2025 + dataobjects_G2017 + dataobjects_T2022
alldata = sc.concat([globals()[d] for d in alldatasets])


# In[592]:


alldatasets_FFPE = dataobjects_N2025 # + dataobjects_F2019
alldata_FFPE = sc.concat([globals()[d] for d in alldatasets_FFPE])


# In[18]:


# saved for later, samples just concat after pp ran (& doublet removed per sample)
alldata.write_h5ad('260121_dcis_combined_after_pp.h5ad')


# In[722]:


#alldata_Q2025 = alldata[alldata.obs['Batch'] == 'Q2025']
alldata_Q2025.write_h5ad('260203_dcis_Q2025_only.h5ad')


# In[723]:


#alldata_W2022 = alldata[alldata.obs['Batch'] == 'W2022']
alldata_W2022.write_h5ad('260203_dcis_W2022_only.h5ad')


# In[724]:


#alldata_G2017 = alldata[alldata.obs['Batch'] == 'G2017']
alldata_G2017.write_h5ad('260203_dcis_G2017_only.h5ad')


# In[725]:


#alldata_T2022 = alldata[alldata.obs['Batch'] == 'T2022']
alldata_T2022.write_h5ad('260203_dcis_T2022_only.h5ad')


# In[726]:


#alldata_N2025 = alldata_FFPE[alldata_FFPE.obs['Batch'] == 'N2025']
alldata_N2025.write_h5ad('260205_dcis_N2025_only.h5ad')


# In[97]:


file_list = ['260205_dcis_N2025_only.h5ad', '260203_dcis_T2022_only.h5ad', '260203_dcis_G2017_only.h5ad', '260203_dcis_W2022_only.h5ad', '260203_dcis_Q2025_only.h5ad']
adatas = [ad.read_h5ad(f) for f in file_list]
merged_adata = ad.concat(
    adatas, 
    join='outer', 
    label='batch', 
    keys=['batch1', 'batch2', 'batch3', 'batch4', 'batch5'],
    fill_value=0
)


# In[99]:


sc.pp.filter_genes(merged_adata, min_cells = 50) # 7 samples so only keep genes if in min 100 cells
merged_adata.X = csr_matrix(merged_adata.X) # convert dense to sparse matrix, less memory
print(merged_adata.shape)


# In[100]:


merged_adata.obs.groupby('Sample').count() # cells you have for each sample


# In[101]:


merged_adata.layers['counts'] = merged_adata.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(merged_adata, target_sum = 1e4) # normalise counts
sc.pp.log1p(merged_adata) # convert to log
merged_adata.raw = merged_adata


# In[102]:


sc.pp.highly_variable_genes(merged_adata, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
merged_adata_hv = merged_adata[:, merged_adata.var['highly_variable']].copy() # subset hv
sc.pp.scale(merged_adata_hv)
sc.tl.pca(merged_adata_hv, svd_solver='arpack')
sc.pp.neighbors(merged_adata_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(merged_adata_hv)
# Copy UMAP coords back to full AnnData
merged_adata.obsm['X_umap'] = merged_adata_hv.obsm['X_umap']


# In[103]:


sc.tl.leiden(merged_adata_hv, resolution = 1.0)
merged_adata.obs['leiden'] = merged_adata_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(merged_adata_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(merged_adata_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(merged_adata, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[110]:


study_colors = {'Q2025': 'steelblue','T2022': 'orange','W2022': 'yellow','G2021': 'lightgreen', 'N2025': 'purple'}


# In[111]:


set_celltype_colors(merged_adata, "Batch", study_colors)


# In[112]:


sc.pl.umap(merged_adata, color=['Batch'], ncols = 1)


# # N2025 FFPE

# In[65]:


alldata_N2025 = sc.read_h5ad('260205_dcis_N2025_only.h5ad')


# In[594]:


print(alldata_N2025.shape)
sc.pp.filter_genes(alldata_N2025, min_cells = 50) # 7 samples so only keep genes if in min 100 cells
alldata_N2025.X = csr_matrix(alldata_N2025.X) # convert dense to sparse matrix, less memory
print(alldata_N2025.shape)


# In[595]:


alldata_N2025.obs.groupby('Sample').count() # cells you have for each sample


# In[596]:


alldata_N2025.layers['counts'] = alldata_N2025.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(alldata_N2025, target_sum = 1e4) # normalise counts
sc.pp.log1p(alldata_N2025) # convert to log
alldata_N2025.raw = alldata_N2025
alldata_N2025.obs.head() # inspect


# In[597]:


sc.pp.highly_variable_genes(alldata_N2025, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldata_N2025_hv = alldata_N2025[:, alldata_N2025.var['highly_variable']].copy() # subset hv
sc.pp.scale(alldata_N2025_hv)
sc.tl.pca(alldata_N2025_hv, svd_solver='arpack')
sc.pp.neighbors(alldata_N2025_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(alldata_N2025_hv)
# Copy UMAP coords back to full AnnData
alldata_N2025.obsm['X_umap'] = alldata_N2025_hv.obsm['X_umap']


# In[614]:


sc.tl.leiden(alldata_N2025_hv, resolution = 1.0)
alldata_N2025.obs['leiden'] = alldata_N2025_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(alldata_N2025_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(alldata_N2025_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(alldata_N2025, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[615]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['CD74'], #No HLA-DPA1 and HLA-DRA
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(alldata_N2025, var_names=marker_genes, groupby='leiden')


# In[617]:


sc.pl.rank_genes_groups(alldata_N2025_hv, n_genes=20, sharey=False)


# In[623]:


sc.pl.umap(alldata_N2025, color = ['KRT14', 'ELF5', 'FOXA1', 'EPCAM', 'leiden'], frameon = False, legend_loc = "on data", ncols =3)
# Basal (KRT14), LP (ELF5), LM (FOXA1), Epithelial (EPCAM)


# In[624]:


cell_type = {'0': 'General Myeloid', '1': 'Luminal Mature', '2': 'Luminal Mature', '3': 'T-Cell', '4': 'Luminal Mature',
 '5': 'General Myeloid', '6': 'Fibroblast', '7': 'Luminal Mature', '8': 'B-Cell', '9': 'T-Cell',
 '10': 'Basal', '11': 'B-Cell', '12': 'Luminal Mature', '13': 'Luminal Progenitor', '14': 'Endothelial',
 '15': 'General Myeloid', '16': 'Luminal Mature', '17': 'General Myeloid', '18': 'Basal', '19': 'General Myeloid',
 '20': 'General Myeloid', '21': 'General Myeloid', '22': 'Luminal Mature', '23': 'General Myeloid', '24': 'General Myeloid',
 '25': 'General Myeloid', '26': 'Luminal Mature', '27': 'Luminal Progenitor', '28': 'T-Cell', '29': 'General Myeloid',
 '30': 'Luminal Mature', '31': 'Fibroblast'}


# In[663]:


set_celltype_colors(alldata_N2025, "cell type", celltype_colors)
set_celltype_colors(alldata_N2025, "Epithelial_vs_NonEpithelial", epith_vs_non_colors)
set_celltype_colors(alldata_N2025, "cnv_status", DCIS_vs_non_colors)


# In[660]:


alldata_N2025.obs['cell type'] = alldata_N2025.obs.leiden.map(cell_type)
sc.pl.umap(alldata_N2025, color = ['cell type'], frameon = False)


# In[626]:


alldata_N2025.obs['Epithelial_vs_NonEpithelial'] = alldata_N2025.obs['cell type'].apply(
    lambda x: "Epithelial" if x in epithelial_types else "Non-epithelial")
sc.pl.umap(alldata_N2025, color=['Epithelial_vs_NonEpithelial'])


# In[627]:


alldata_N2025.obs['cell type'].value_counts()


# In[1290]:


N2025_Ep = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(N2025_Ep, color = ['cnv_status', 'cell type'], frameon = False)


# In[1291]:


N2025_Ep_DD = N2025_Ep[N2025_Ep.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_Ep_DD, color = ['cnv_status', 'cell type'], frameon = False)


# In[187]:


alldata_N2025.write_h5ad('260205_dcis_N2025_only.h5ad')


# In[188]:


alldata_N2025.obs['cell_id_orig'] = alldata_N2025.obs.index


# In[358]:


# Make unique index
alldata_N2025.obs.index = (
    alldata_N2025.obs.index
    + "-" 
    + alldata_N2025.obs.groupby(alldata_N2025.obs.index).cumcount().astype(str))
subpops_N2025 = [N2025_DN_B, N2025_DD_B, N2025_DN_LM, N2025_DD_LM, N2025_DN_LP, N2025_DD_LP]
for ad in subpops_N2025:
    ad.obs['cell_id_orig'] = ad.obs.index  # save original index
    ad.obs.index = ad.obs.index + "-" + ad.obs.groupby(ad.obs.index).cumcount().astype(str)


# In[362]:


alldata_N2025.obs['annotation'] = pd.NA

subpops = [N2025_DN_B, N2025_DD_B, N2025_DN_LM, N2025_DD_LM, N2025_DN_LP, N2025_DD_LP]
# Transfer annotations
for ad in subpops:
    alldata_N2025.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']

alldata_N2025.obs['annotation'] = alldata_N2025.obs['annotation'].fillna('unassigned')


# In[94]:


sc.pl.umap(alldata_N2025, color = ['cell type'], frameon = True, ncols=1)


# In[95]:


sc.pl.umap(alldata_N2025, color = ['cnv_status'], frameon = True, ncols=1)


# ### DN Basal N2025 FFPE

# In[133]:


N2025_DN_B = N2025_B[N2025_B.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(N2025_DN_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[134]:


sc.pl.pca_variance_ratio(N2025_DN_B, n_pcs=50)


# In[135]:


sc.pp.highly_variable_genes(N2025_DN_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DN_B_hv = N2025_DN_B[:, N2025_DN_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DN_B_hv)
sc.tl.pca(N2025_DN_B_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DN_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DN_B_hv)
N2025_DN_B.obsm['X_umap'] = N2025_DN_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[136]:


sc.tl.leiden(N2025_DN_B_hv, resolution = 0.2)
N2025_DN_B.obs['leiden'] = N2025_DN_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DN_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DN_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DN_B, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon = None)


# In[137]:


sc.pl.umap(N2025_DN_B, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'SPARC'], ncols=4)


# In[138]:


basal_leiden_DN_N2025 = {"0":"0", "1":"1", "2":"2"}
N2025_DN_B.obs['subcluster'] = N2025_DN_B.obs.leiden.map(basal_leiden_DN_N2025)
N2025_DN_B.obs['subcluster'].value_counts()


# In[141]:


save_top_marker_genes(N2025_DN_B, 'N2025', 'BDN')


# In[142]:


N2025_DN_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDN.h5ad")


# In[346]:


N2025_DN_B = sc.read_h5ad("./data/AAA_DCIS/260216_N2025_BDN.h5ad")
N2025_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "XX",
    "1": "XX",
    "2": "XX"
}
N2025_DN_B.obs['annotation'] = N2025_DN_B.obs['subcluster'].map(annotation_map)
N2025_DN_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDN.h5ad")


# ### DD N2025 FFPE Basal

# In[143]:


N2025_B = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Basal'])]
sc.pl.umap(N2025_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[144]:


N2025_DD_B = N2025_B[N2025_B.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_DD_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[145]:


sc.pl.pca_variance_ratio(N2025_DD_B, n_pcs=50)


# In[146]:


sc.pp.highly_variable_genes(N2025_DD_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DD_B_hv = N2025_DD_B[:, N2025_DD_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DD_B_hv)
sc.tl.pca(N2025_DD_B_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DD_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DD_B_hv)
N2025_DD_B.obsm['X_umap'] = N2025_DD_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[147]:


sc.tl.leiden(N2025_DD_B_hv, resolution = 0.5)
N2025_DD_B.obs['leiden'] = N2025_DD_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DD_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DD_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DD_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[148]:


sc.pl.umap(N2025_DD_B, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'SPARC'], ncols=4)


# In[149]:


basal_leiden_DD_N2025 = {"0":"0", "1":"1"}
N2025_DD_B.obs['subcluster'] = N2025_DD_B.obs.leiden.map(basal_leiden_DD_N2025)
N2025_DD_B.obs['subcluster'].value_counts()


# In[150]:


save_top_marker_genes(N2025_DD_B, 'N2025', 'BDD')


# In[151]:


N2025_DD_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDD.h5ad")


# ### DN N2025 FFPE LM

# In[153]:


N2025_DN_LM = N2025_LM[N2025_LM.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(N2025_DN_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[154]:


sc.pl.pca_variance_ratio(N2025_DN_LM, n_pcs=50)


# In[155]:


sc.pp.normalize_total(N2025_DN_LM, target_sum=1e4)
sc.pp.log1p(N2025_DN_LM)

sc.pp.highly_variable_genes(N2025_DN_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DN_LM_hv = N2025_DN_LM[:, N2025_DN_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DN_LM_hv)
sc.tl.pca(N2025_DN_LM_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DN_LM_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DN_LM_hv)
N2025_DN_LM.obsm['X_umap'] = N2025_DN_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[160]:


sc.tl.leiden(N2025_DN_LM_hv, resolution = 0.2)
N2025_DN_LM.obs['leiden'] = N2025_DN_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DN_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DN_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DN_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon=None)


# In[161]:


LM_leiden_DN_N2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
N2025_DN_LM.obs['subcluster'] = N2025_DN_LM.obs.leiden.map(LM_leiden_DN_N2025)
N2025_DN_LM.obs['subcluster'].value_counts()


# In[162]:


save_top_marker_genes(N2025_DN_LM, 'N2025', 'LMDN')


# In[163]:


N2025_DN_LM.write_h5ad("./data/AAA_DCIS/260217_N2025_LMDN.h5ad")


# ### DD N2025 FFPE LM

# In[164]:


N2025_LM = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(N2025_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[165]:


N2025_DD_LM = N2025_LM[N2025_LM.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_DD_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[166]:


sc.pl.pca_variance_ratio(N2025_DD_LM, n_pcs=50)


# In[171]:


sc.pp.normalize_total(N2025_DD_LM, target_sum=1e4)
sc.pp.log1p(N2025_DD_LM)

sc.pp.highly_variable_genes(N2025_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DD_LM_hv = N2025_DD_LM[:, N2025_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DD_LM_hv)
sc.tl.pca(N2025_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DD_LM_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(N2025_DD_LM_hv)
N2025_DD_LM.obsm['X_umap'] = N2025_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[172]:


sc.tl.leiden(N2025_DD_LM_hv, resolution = 0.5)
N2025_DD_LM.obs['leiden'] = N2025_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon = None)


# In[173]:


LM_leiden_DD_N2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7", "8":"8", "9":"9", "10":"10"}
N2025_DD_LM.obs['subcluster'] = N2025_DD_LM.obs.leiden.map(LM_leiden_DD_N2025)
N2025_DD_LM.obs['subcluster'].value_counts()


# In[174]:


save_top_marker_genes(N2025_DD_LM, 'N2025', 'LMDD')


# In[175]:


N2025_DD_LM.write_h5ad("./data/AAA_DCIS/260216_N2025_LMDD.h5ad")


# ### DD N2025 FFPE LP

# In[176]:


N2025_LP = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(N2025_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[177]:


N2025_DD_LP = N2025_LP[N2025_LP.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_DD_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[178]:


sc.pl.pca_variance_ratio(N2025_DD_LP, n_pcs=50)


# In[181]:


sc.pp.normalize_total(N2025_DD_LM, target_sum=1e4)
sc.pp.log1p(N2025_DD_LM)
sc.pp.highly_variable_genes(N2025_DD_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DD_LP_hv = N2025_DD_LP[:, N2025_DD_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DD_LP_hv)
sc.tl.pca(N2025_DD_LP_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DD_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DD_LP_hv)
N2025_DD_LP.obsm['X_umap'] = N2025_DD_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[184]:


sc.tl.leiden(N2025_DD_LP_hv, resolution = 0.3)
N2025_DD_LP.obs['leiden'] = N2025_DD_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DD_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DD_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DD_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[185]:


LP_leiden_DD_N2025 = {"0":"0", "1":"1"}
N2025_DD_LP.obs['subcluster'] = N2025_DD_LP.obs.leiden.map(LP_leiden_DD_N2025)
N2025_DD_LP.obs['subcluster'].value_counts()


# In[186]:


save_top_marker_genes(N2025_DD_LP, 'N2025', 'LPDD')


# In[1332]:


N2025_DD_LP.write_h5ad("./data/AAA_DCIS/260216_N2025_LPDD.h5ad")


# ### DN N2025 LP

# In[187]:


N2025_DN_LP = N2025_LP[N2025_LP.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(N2025_DN_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[188]:


sc.pl.pca_variance_ratio(N2025_DN_LP, n_pcs=50)


# In[189]:


sc.pp.highly_variable_genes(N2025_DN_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DN_LP_hv = N2025_DN_LP[:, N2025_DN_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DN_LP_hv)
sc.tl.pca(N2025_DN_LP_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DN_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DN_LP_hv)
N2025_DN_LP.obsm['X_umap'] = N2025_DN_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[191]:


sc.tl.leiden(N2025_DN_LP_hv, resolution = 0.3)
N2025_DN_LP.obs['leiden'] = N2025_DN_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DN_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DN_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DN_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon=None)


# In[192]:


LP_leiden_DN_N2025 = {"0":"0", "1":"1", "2":"2"}
N2025_DN_LP.obs['subcluster'] = N2025_DN_LP.obs.leiden.map(LP_leiden_DN_N2025)
N2025_DN_LP.obs['subcluster'].value_counts()


# In[193]:


save_top_marker_genes(N2025_DN_LP, 'N2025', 'LPDN')


# In[194]:


N2025_DN_LP.write_h5ad("./data/AAA_DCIS/260217_N2025_LPDN.h5ad")


# # Q2025

# In[4]:


alldata_Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')


# In[25]:


print(alldata_Q2025.shape)
sc.pp.filter_genes(alldata_Q2025, min_cells = 100) # 17 samples so only keep genes if in min 100 cells
alldata_Q2025.X = csr_matrix(alldata_Q2025.X) # convert dense to sparse matrix, less memory
print(alldata_Q2025.shape)


# In[26]:


alldata_Q2025.obs.groupby('Sample').count() # cells you have for each sample


# In[28]:


alldata_Q2025.layers['counts'] = alldata_Q2025.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(alldata_Q2025, target_sum = 1e4) # normalise counts
sc.pp.log1p(alldata_Q2025) # convert to log
alldata_Q2025.raw = alldata_Q2025
alldata_Q2025.obs.head() # inspect


# In[29]:


sc.pp.highly_variable_genes(alldata_Q2025, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldata_Q2025_hv = alldata_Q2025[:, alldata_Q2025.var['highly_variable']].copy() # subset hv


# In[30]:


sc.pp.scale(alldata_Q2025_hv)
sc.tl.pca(alldata_Q2025_hv, svd_solver='arpack')
sc.pp.neighbors(alldata_Q2025_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(alldata_Q2025_hv)

# Copy UMAP coords back to full AnnData
alldata_Q2025.obsm['X_umap'] = alldata_Q2025_hv.obsm['X_umap']


# In[38]:


sc.tl.leiden(alldata_Q2025_hv, resolution = 1.0)
alldata_Q2025.obs['leiden'] = alldata_Q2025_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(alldata_Q2025_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(alldata_Q2025_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(alldata_Q2025, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[8]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(alldata_Q2025, var_names=marker_genes, groupby='leiden', )


# In[9]:


sc.pl.dotplot(alldata_Q2025, var_names=marker_genes, groupby='leiden', dendrogram=True)


# In[80]:


cell_type = {
 '0': 'Luminal Mature', '1': 'T-Cell', '2': 'Basal', '3': 'T-Cell', '4': 'Fibroblast',
 '5': 'Endothelial', '6': 'Luminal Mature', '7': 'General Myeloid', '8': 'General Myeloid', '9': 'Luminal Progenitor',
 '10': 'Luminal Mature', '11': 'Endothelial', '12': 'Basal', '13': 'Fibroblast', '14': 'Luminal Mature',
 '15': 'Luminal Mature', '16': 'Macrophage', '17': 'Fibroblast', '18': 'Luminal Mature', '19': 'T-Cell',
 '20': 'Fibroblast', '21': 'General Myeloid', '22': 'Luminal Progenitor', '23': 'T-Cell', '24': 'Luminal Mature',
 '25': 'Basal', '26': 'Endothelial', '27': 'Luminal Mature', '28': 'Luminal Mature', '29': 'Basal',
 '30': 'Basal', '31': 'Luminal Mature', '32': 'Endothelial', '33': 'Basal', '34': 'Endothelial',
 '35': 'Luminal Mature', '36': 'Basal', '37': 'B-Cell', '38': 'B-Cell', '39':'T-Cell'
}


# In[55]:


sc.pl.rank_genes_groups(alldata_Q2025_hv, n_genes=20, sharey=False)


# In[19]:


sc.pl.umap(alldata_Q2025, color = ['KRT14', 'ELF5', 'FOXA1', 'EPCAM', 'leiden'], frameon = False, legend_loc = "on data", ncols =3)
# Basal (KRT14), LP (ELF5), LM (FOXA1), Epithelial (EPCAM)


# In[76]:


#alldata_Q2025.obs['cell type'] = alldata_Q2025.obs.leiden.map(cell_type)
sc.pl.umap(alldata_Q2025, color = ['cell type'], frameon = True, title = 'Q2025')


# In[25]:


sc.pl.umap(alldata_Q2025, color = ['Sample'], frameon = True, title = 'Q2025')


# In[26]:


sc.pl.umap(alldata_Q2025, color = ['Epithelial_vs_NonEpithelial'], frameon = True, title = 'Q2025')


# In[27]:


sc.pl.umap(alldata_Q2025, color = ['cnv_status'], frameon = True, title = 'Q2025')


# In[343]:


alldata_Q2025.obs['Epithelial_vs_NonEpithelial'] = alldata_Q2025.obs['cell type'].apply(
    lambda x: "Epithelial" if x in epithelial_types else "Non-epithelial"
)

# Inspect
alldata_Q2025.obs[['cell type', 'Epithelial_vs_NonEpithelial']].head()

# Optional: plot UMAP colored by this new annotation
sc.pl.umap(alldata_Q2025, color=['Epithelial_vs_NonEpithelial', 'cnv_status'], ncols=1)


# In[106]:


alldata_Q2025.obs['cell type'].value_counts()


# In[157]:


Q2025_Ep = alldata_Q2025[alldata_Q2025.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(Q2025_Ep, color = ['cnv_status', 'cell type'], frameon = False)


# In[158]:


Q2025_Ep_DD = Q2025_Ep[Q2025_Ep.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(Q2025_Ep_DD, color = ['cnv_status', 'cell type'], frameon = False)


# In[250]:


alldata_Q2025.write_h5ad('260203_dcis_Q2025_only.h5ad')


# In[13]:


alldata_Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')


# In[97]:


# Create column if it doesn't exist
alldata_Q2025.obs['annotation'] = pd.NA

subpops = [Q2025_DD_B, Q2025_DN_B, Q2025_DN_LM, Q2025_DD_LM, Q2025_DN_LP, Q2025_DD_LP]
# Transfer annotations
for ad in subpops:
    alldata_Q2025.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']


# In[98]:


alldata_Q2025.obs['annotation'] = alldata_Q2025.obs['annotation'].fillna('unassigned')


# In[99]:


sc.pl.umap(alldata_Q2025, color = ['annotation', 'Epithelial_vs_NonEpithelial', 'cnv_status', 'cell type'], frameon = False, ncols=1)


# In[100]:


alldata_Q2025.write_h5ad('260203_dcis_Q2025_only.h5ad')


# ### Q2025 DN Basal

# In[160]:


Q2025_DN_B = Q2025_B[Q2025_B.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(Q2025_DN_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[161]:


sc.pp.highly_variable_genes(Q2025_DN_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
Q2025_DN_B_hv = Q2025_DN_B[:, Q2025_DN_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(Q2025_DN_B_hv)
sc.tl.pca(Q2025_DN_B_hv, svd_solver='arpack')
sc.pp.neighbors(Q2025_DN_B_hv, n_neighbors=15, n_pcs=20)
sc.tl.umap(Q2025_DN_B_hv)
Q2025_DN_B.obsm['X_umap'] = Q2025_DN_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[163]:


sc.tl.leiden(Q2025_DN_B_hv, resolution = 0.4)
Q2025_DN_B.obs['leiden'] = Q2025_DN_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(Q2025_DN_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(Q2025_DN_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(Q2025_DN_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[164]:


basal_leiden_DN_Q2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5"}
Q2025_DN_B.obs['subcluster'] = Q2025_DN_B.obs.leiden.map(basal_leiden_DN_Q2025)
Q2025_DN_B.obs['subcluster'].value_counts()


# In[167]:


save_top_marker_genes(Q2025_DN_B, 'Q2025', 'BDN')


# In[168]:


Q2025_DN_B.write_h5ad("./data/AAA_DCIS/260217_Q2025_BDN.h5ad")


# In[5]:


Q2025_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_Q2025_BDN.h5ad")


# In[91]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDN_na",
    "1": "BDN_Mesenchymal-like",
    "2": "BDN_Inflammatory_ECM-remodelling",
    "3": "BDN_Inflammatory_immune-reactive_apoptosis-primed_adhering",
    "4": "BDN_na",
    "5": "BDN_ER-stressed_metabolically_adapted"
}
Q2025_DN_B.obs['annotation'] = Q2025_DN_B.obs['subcluster'].map(annotation_map)
Q2025_DN_B.write_h5ad("./data/AAA_DCIS/260217_Q2025_BDN.h5ad")


# In[261]:


sc.pl.umap(Q2025_DN_B, color=['annotation', 'Batch', 'Sample'], ncols = 1)


# ### Q2025 DD Basal

# In[140]:


Q2025_B = alldata_Q2025[alldata_Q2025.obs['cell type'].isin(['Basal'])]
sc.pl.umap(Q2025_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[142]:


Q2025_B17 = Q2025_B[Q2025_B.obs['Sample'].isin(['ind17_Q2025'])]
sc.pl.umap(Q2025_B17, color = ['annotation'], frameon = False)


# In[143]:


Q2025_B17D = Q2025_B17[Q2025_B17.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(Q2025_B17D, color = ['annotation'], frameon = False)


# In[144]:


Q2025_B1 = Q2025_B[Q2025_B.obs['Sample'].isin(['ind1_Q2025'])]
sc.pl.umap(Q2025_B1, color = ['annotation'], frameon = False)


# In[145]:


Q2025_B1D = Q2025_B1[Q2025_B1.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(Q2025_B1D, color = ['annotation'], frameon = False)


# In[170]:


Q2025_DD_B = Q2025_B[Q2025_B.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(Q2025_DD_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[171]:


sc.pl.pca_variance_ratio(Q2025_DD_B, n_pcs=50)


# In[173]:


sc.pp.highly_variable_genes(Q2025_DD_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
Q2025_DD_B_hv = Q2025_DD_B[:, Q2025_DD_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(Q2025_DD_B_hv)
sc.tl.pca(Q2025_DD_B_hv, svd_solver='arpack')
sc.pp.neighbors(Q2025_DD_B_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(Q2025_DD_B_hv)
Q2025_DD_B.obsm['X_umap'] = Q2025_DD_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[175]:


sc.tl.leiden(Q2025_DD_B_hv, resolution = 0.3)
Q2025_DD_B.obs['leiden'] = Q2025_DD_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(Q2025_DD_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(Q2025_DD_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(Q2025_DD_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[176]:


sc.pl.umap(Q2025_DD_B, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'SPARC'], ncols=4)


# In[177]:


basal_leiden_DD_Q2025 = {"0":"0", "1":"1", "2":"2"}
Q2025_DD_B.obs['subcluster'] = Q2025_DD_B.obs.leiden.map(basal_leiden_DD_Q2025)
Q2025_DD_B.obs['subcluster'].value_counts()


# In[140]:


def save_top_marker_genes(adata, study_name, cell_type, gene_counts=[20, 30, 100, 1000]):
    """
    Saves the top differentially expressed marker genes for each subcluster to CSV files.
    
    Parameters:
    - adata: AnnData object
    - study_name: str, the name of the study (e.g., 'N2021')
    - cell_type: str, the cell type (e.g., 'Basal')
    - gene_counts: list, number of top genes to save (default: [20, 30, 100, 1000])
    """
    # Create the output directory if it doesn't exist
    output_dir = f"{cell_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run differential expression analysis
    sc.tl.rank_genes_groups(adata, 'leiden')
    
    # Iterate through each leiden cluster
    for group in adata.obs['leiden'].unique():
        for n_genes in gene_counts:
            # Get top differentially expressed genes
            top_genes_df = sc.get.rank_genes_groups_df(adata, group=group)[:n_genes]
            
            # Define filename
            filename = f"{output_dir}/up{n_genes}_{study_name}_{cell_type}_{group}.csv"
            
            # Save to CSV
            top_genes_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")


# In[178]:


save_top_marker_genes(Q2025_DD_B, 'Q2025', 'BDD')


# In[179]:


Q2025_DD_B.write_h5ad("./data/AAA_DCIS/260205_Q2025_BDD.h5ad")


# In[92]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
Q2025_DD_B = sc.read_h5ad("./data/AAA_DCIS/260205_Q2025_BDD.h5ad")
#Q2025_DD_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDD_Inflammatory_immune-responsive_EMT-high",
    "1": "BDD_na",
    "2": "BDD_EMT/invasive-like_ECM-remodelling_contractile"
}
Q2025_DD_B.obs['annotation'] = Q2025_DD_B.obs['subcluster'].map(annotation_map)
Q2025_DD_B.write_h5ad("./data/AAA_DCIS/260205_Q2025_BDD.h5ad")


# In[180]:


markers = sc.get.rank_genes_groups_df(Q2025_DD_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.rank_genes_groups(Q2025_DD_B_hv, n_genes=20, sharey=False, ncols =2)


# ### Q2025 DN LM

# In[182]:


Q2025_DN_LM = Q2025_LM[Q2025_LM.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(Q2025_DN_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[183]:


sc.pp.highly_variable_genes(Q2025_DN_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
Q2025_DN_LM_hv = Q2025_DN_LM[:, Q2025_DN_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(Q2025_DN_LM_hv)
sc.tl.pca(Q2025_DN_LM_hv, svd_solver='arpack')
sc.pp.neighbors(Q2025_DN_LM_hv, n_neighbors=15, n_pcs=20)
sc.tl.umap(Q2025_DN_LM_hv)
Q2025_DN_LM.obsm['X_umap'] = Q2025_DN_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[185]:


sc.tl.leiden(Q2025_DN_LM_hv, resolution = 0.4)
Q2025_DN_LM.obs['leiden'] = Q2025_DN_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(Q2025_DN_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(Q2025_DN_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(Q2025_DN_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[186]:


LM_leiden_DN_Q2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6"}
Q2025_DN_LM.obs['subcluster'] = Q2025_DN_LM.obs.leiden.map(LM_leiden_DN_Q2025)
Q2025_DN_LM.obs['subcluster'].value_counts()


# In[187]:


save_top_marker_genes(Q2025_DN_LM, 'Q2025', 'LMDN')


# In[188]:


Q2025_DN_LM.write_h5ad("./data/AAA_DCIS/260217_Q2025_LMDN.h5ad")


# In[93]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
Q2025_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_Q2025_LMDN.h5ad")
#Q2025_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDN_Inflammatory_EMT-active_oxidative-stressed",
    "1": "LMDN_na",
    "2": "LMDN_na",
    "3": "LMDN_Immune-surveilling_invasion-like",
    "4": "LMDN_IFN-active_antigen-presenting",
    "5": "LMDN_Immune-modulating_stem-like",
    "6": "LMDN_na"
}
Q2025_DN_LM.obs['annotation'] = Q2025_DN_LM.obs['subcluster'].map(annotation_map)
Q2025_DN_LM.write_h5ad("./data/AAA_DCIS/260217_Q2025_LMDN.h5ad")


# ### Q2025 DD Luminal Mature

# In[189]:


Q2025_LM = alldata_Q2025[alldata_Q2025.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(Q2025_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[190]:


Q2025_DD_LM = Q2025_LM[Q2025_LM.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(Q2025_DD_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[191]:


sc.pl.pca_variance_ratio(Q2025_DD_LM, n_pcs=50)


# In[192]:


sc.pp.highly_variable_genes(Q2025_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
Q2025_DD_LM_hv = Q2025_DD_LM[:, Q2025_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(Q2025_DD_LM_hv)
sc.tl.pca(Q2025_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(Q2025_DD_LM_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(Q2025_DD_LM_hv)
Q2025_DD_LM.obsm['X_umap'] = Q2025_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[193]:


sc.tl.leiden(Q2025_DD_LM_hv, resolution = 0.6)
Q2025_DD_LM.obs['leiden'] = Q2025_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(Q2025_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(Q2025_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(Q2025_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[194]:


LM_leiden_DD_Q2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7", "8":"8", "9":"9", "10":"10"}
Q2025_DD_LM.obs['subcluster'] = Q2025_DD_LM.obs.leiden.map(LM_leiden_DD_Q2025)
Q2025_DD_LM.obs['subcluster'].value_counts()


# In[195]:


save_top_marker_genes(Q2025_DD_LM, 'Q2025', 'LMDD')


# In[196]:


Q2025_DD_LM.write_h5ad("./data/AAA_DCIS/260205_Q2025_LMDD.h5ad")


# In[94]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
Q2025_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_Q2025_LMDD.h5ad")
#Q2025_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDD_na",
    "1": "LMDD_na",
    "2": "LMDD_na",
    "3": "LMDD_Plastic_immune-modulating",
    "4": "LMDD_na",
    "5": "LMDD_na",
    "6": "LMDD_na",
    "7": "LMDD_Hormone-responsive_secretory_immune-modulating",
    "8": "LMDD_na",
    "9": "LMDD_Stress-adapted_ROS-high",
    "10": "LMDD_na",
}
Q2025_DD_LM.obs['annotation'] = Q2025_DD_LM.obs['subcluster'].map(annotation_map)
Q2025_DD_LM.write_h5ad("./data/AAA_DCIS/260205_Q2025_LMDD.h5ad")


# In[197]:


markers = sc.get.rank_genes_groups_df(Q2025_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.rank_genes_groups(Q2025_DD_LM_hv, n_genes=20, sharey=False, ncols =3)


# ### Q2025 DN LP

# In[200]:


Q2025_DN_LP = Q2025_LP[Q2025_LP.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(Q2025_DN_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[202]:


sc.pl.pca_variance_ratio(Q2025_DN_LP, n_pcs=50)


# In[203]:


sc.pp.highly_variable_genes(Q2025_DN_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
Q2025_DN_LP_hv = Q2025_DN_LP[:, Q2025_DN_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(Q2025_DN_LP_hv)
sc.tl.pca(Q2025_DN_LP_hv, svd_solver='arpack')
sc.pp.neighbors(Q2025_DN_LP_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(Q2025_DN_LP_hv)
Q2025_DN_LP.obsm['X_umap'] = Q2025_DN_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[205]:


sc.tl.leiden(Q2025_DN_LP_hv, resolution = 0.3)
Q2025_DN_LP.obs['leiden'] = Q2025_DN_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(Q2025_DN_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(Q2025_DN_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(Q2025_DN_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[206]:


LP_leiden_DN_Q2025 = {"0":"0", "1":"1", "2":"2"}
Q2025_DN_LP.obs['subcluster'] = Q2025_DN_LP.obs.leiden.map(LP_leiden_DN_Q2025)
Q2025_DN_LP.obs['subcluster'].value_counts()


# In[207]:


save_top_marker_genes(Q2025_DN_LP, 'Q2025', 'LPDN')


# In[208]:


Q2025_DN_LP.write_h5ad("./data/AAA_DCIS/260217_Q2025_LPDN.h5ad")


# In[95]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
Q2025_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_Q2025_LPDN.h5ad")
#Q2025_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDN_na",
    "1": "LPDN_Redox-stressed_tumour_protective",
    "2": "LPDN_Type_1_interferon_signaling_growth-suppressed"
}
Q2025_DN_LP.obs['annotation'] = Q2025_DN_LP.obs['subcluster'].map(annotation_map)
Q2025_DN_LP.write_h5ad("./data/AAA_DCIS/260217_Q2025_LPDN.h5ad")


# ### Q2025 DD Luminal Progenitor

# In[209]:


Q2025_LP = alldata_Q2025[alldata_Q2025.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(Q2025_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[210]:


Q2025_DD_LP = Q2025_LP[Q2025_LP.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(Q2025_DD_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[211]:


sc.pl.pca_variance_ratio(Q2025_DD_LP, n_pcs=50)


# In[212]:


sc.pp.highly_variable_genes(Q2025_DD_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
Q2025_DD_LP_hv = Q2025_DD_LP[:, Q2025_DD_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(Q2025_DD_LP_hv)
sc.tl.pca(Q2025_DD_LP_hv, svd_solver='arpack')
sc.pp.neighbors(Q2025_DD_LP_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(Q2025_DD_LP_hv)
Q2025_DD_LP.obsm['X_umap'] = Q2025_DD_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[214]:


sc.tl.leiden(Q2025_DD_LP_hv, resolution = 0.3)
Q2025_DD_LP.obs['leiden'] = Q2025_DD_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(Q2025_DD_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(Q2025_DD_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(Q2025_DD_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[215]:


LP_leiden_DD_Q2025 = {"0":"0", "1":"1", "2":"2", "3":"3"}
Q2025_DD_LP.obs['subcluster'] = Q2025_DD_LP.obs.leiden.map(LP_leiden_DD_Q2025)
Q2025_DD_LP.obs['subcluster'].value_counts()


# In[216]:


save_top_marker_genes(Q2025_DD_LP, 'Q2025', 'LPDD')


# In[217]:


Q2025_DD_LP.write_h5ad("./data/AAA_DCIS/260205_Q2025_LPDD.h5ad")


# In[96]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
Q2025_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_Q2025_LPDD.h5ad")
#Q2025_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDD_Antigen-presenting_protein-synthesising",
    "1": "LPDD_na",
    "2": "LPDD_Proliferative_basal-like",
    "3": "LPDD_Proliferative_basal-like"
}
Q2025_DD_LP.obs['annotation'] = Q2025_DD_LP.obs['subcluster'].map(annotation_map)
Q2025_DD_LP.write_h5ad("./data/AAA_DCIS/260205_Q2025_LPDD.h5ad")


# # T2022

# In[3]:


alldata_T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')


# In[87]:


print(alldata_T2022.shape)
sc.pp.filter_genes(alldata_T2022, min_cells = 50) # 7 samples so only keep genes if in min 100 cells
alldata_T2022.X = csr_matrix(alldata_T2022.X) # convert dense to sparse matrix, less memory
print(alldata_T2022.shape)


# In[88]:


alldata_T2022.obs.groupby('Sample').count() # cells you have for each sample


# In[89]:


alldata_T2022.layers['counts'] = alldata_T2022.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(alldata_T2022, target_sum = 1e4) # normalise counts
sc.pp.log1p(alldata_T2022) # convert to log
alldata_T2022.raw = alldata_T2022
sc.pp.highly_variable_genes(alldata_T2022, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldata_T2022_hv = alldata_T2022[:, alldata_T2022.var['highly_variable']].copy() # subset hv
sc.pp.scale(alldata_T2022_hv)
sc.tl.pca(alldata_T2022_hv, svd_solver='arpack')
sc.pp.neighbors(alldata_T2022_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(alldata_T2022_hv)

# Copy UMAP coords back to full AnnData
alldata_T2022.obsm['X_umap'] = alldata_T2022_hv.obsm['X_umap']


# In[94]:


sc.tl.leiden(alldata_T2022_hv, resolution = 1.0)
alldata_T2022.obs['leiden'] = alldata_T2022_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(alldata_T2022_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(alldata_T2022_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(alldata_T2022, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[95]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(alldata_T2022, var_names=marker_genes, groupby='leiden')


# In[98]:


cell_type = {
 '0': 'Luminal Mature', '1': 'Luminal Mature', '2': 'Luminal Mature', '3': 'Luminal Progenitor', '4': 'General Myeloid',
 '5': 'Luminal Mature', '6': 'T-Cell', '7': 'Luminal Mature', '8': 'Luminal Mature', '9': 'Luminal Mature',
 '10': 'B-Cell', '11': 'Luminal Mature', '12': 'T-Cell', '13': 'General Myeloid', '14': 'Luminal Mature',
 '15': 'Luminal Mature', '16': 'Luminal Mature', '17': 'Luminal Mature', '18': 'Luminal Mature', '19': 'Luminal Mature',
 '20': 'Basal', '21': 'B-Cell', '22': 'B-Cell', '23': 'General Myeloid', '24': 'Luminal Progenitor',
 '25': 'Basal', '26': 'Monocyte', '27': 'Luminal Progenitor', '28': 'Luminal Progenitor', '29': 'Luminal Mature'
}


# In[96]:


sc.pl.rank_genes_groups(alldata_T2022_hv, n_genes=20, sharey=False)


# In[97]:


sc.pl.umap(alldata_T2022, color = ['KRT14', 'ELF5', 'FOXA1', 'EPCAM', 'leiden'], frameon = False, legend_loc = "on data", ncols =3)
# Basal (KRT14), LP (ELF5), LM (FOXA1), Epithelial (EPCAM)


# In[77]:


alldata_T2022.obs['cell type'] = alldata_T2022.obs.leiden.map(cell_type)
sc.pl.umap(alldata_T2022, color = ['cell type'], frameon = False)


# In[346]:


alldata_T2022.obs['Epithelial_vs_NonEpithelial'] = alldata_T2022.obs['cell type'].apply(
    lambda x: "Epithelial" if x in epithelial_types else "Non-epithelial"
)
alldata_T2022.obs[['cell type', 'Epithelial_vs_NonEpithelial']].head()
sc.pl.umap(alldata_T2022, color=['Epithelial_vs_NonEpithelial', 'cnv_status'], ncols=1)


# In[105]:


alldata_T2022.obs['cell type'].value_counts()


# In[219]:


T2022_Ep = alldata_T2022[alldata_T2022.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(T2022_Ep, color = ['cnv_status', 'cell type'], frameon = False)


# In[220]:


T2022_Ep_DD = T2022_Ep[T2022_Ep.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(T2022_Ep_DD, color = ['cnv_status', 'cell type'], frameon = False)


# In[150]:


alldata_T2022.write_h5ad('260203_dcis_T2022_only.h5ad')


# In[39]:


alldata_T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')


# In[336]:


alldata_T2022.obs['cell_id_orig'] = alldata_T2022.obs.index

# Make unique index
alldata_T2022.obs.index = (
    alldata_T2022.obs.index
    + "-" 
    + alldata_T2022.obs.groupby(alldata_T2022.obs.index).cumcount().astype(str)
)


# In[337]:


subpops_T2022 = [T2022_DN_B, T2022_DD_B, T2022_DN_LM, T2022_DD_LM, T2022_DN_LP, T2022_DD_LP]

for ad in subpops_T2022:
    ad.obs['cell_id_orig'] = ad.obs.index  # save original index
    ad.obs.index = ad.obs.index + "-" + ad.obs.groupby(ad.obs.index).cumcount().astype(str)


# In[147]:


alldata_T2022.obs['annotation'] = pd.NA

subpops = [T2022_DN_B, T2022_DD_B, T2022_DN_LM, T2022_DD_LM, T2022_DN_LP, T2022_DD_LP]
# Transfer annotations
for ad in subpops:
    alldata_T2022.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']


# In[148]:


alldata_T2022.obs['annotation'] = alldata_T2022.obs['annotation'].fillna('unassigned')


# In[78]:


sc.pl.umap(alldata_T2022, color = ['annotation', 'Epithelial_vs_NonEpithelial', 'cnv_status', 'cell type'], frameon = False, ncols=1)


# ### T2022 DN Basal

# In[222]:


T2022_DN_B = T2022_B[T2022_B.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(T2022_DN_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[223]:


sc.pl.pca_variance_ratio(T2022_DN_B, n_pcs=50)


# In[224]:


sc.pp.highly_variable_genes(T2022_DN_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
T2022_DN_B_hv = T2022_DN_B[:, T2022_DN_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(T2022_DN_B_hv)
sc.tl.pca(T2022_DN_B_hv, svd_solver='arpack')
sc.pp.neighbors(T2022_DN_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(T2022_DN_B_hv)
T2022_DN_B.obsm['X_umap'] = T2022_DN_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[225]:


sc.tl.leiden(T2022_DN_B_hv, resolution = 0.4)
T2022_DN_B.obs['leiden'] = T2022_DN_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(T2022_DN_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(T2022_DN_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(T2022_DN_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[226]:


basal_leiden_DN_T2022 = {"0":"0", "1":"1", "2":"2"}
T2022_DN_B.obs['subcluster'] = T2022_DN_B.obs.leiden.map(basal_leiden_DN_T2022)
T2022_DN_B.obs['subcluster'].value_counts()


# In[227]:


save_top_marker_genes(T2022_DN_B, 'T2022', 'BDN')


# In[109]:


T2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_BDN.h5ad")
T2022_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDN_na",
    "1": "BDN_Inflammatory_ECM-remodelling",
    "2": "BDN_ER-stressed_metabolically_adapted"
}
T2022_DN_B.obs['annotation'] = T2022_DN_B.obs['subcluster'].map(annotation_map)
T2022_DN_B.write_h5ad("./data/AAA_DCIS/260217_T2022_BDN.h5ad")


# In[228]:


T2022_DN_B.write_h5ad("./data/AAA_DCIS/260217_T2022_BDN.h5ad")


# In[329]:


T2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_BDN.h5ad")


# ### T2022 DD Basal

# In[229]:


T2022_B = alldata_T2022[alldata_T2022.obs['cell type'].isin(['Basal'])]
sc.pl.umap(T2022_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[230]:


T2022_DD_B = T2022_B[T2022_B.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(T2022_DD_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[231]:


sc.pp.highly_variable_genes(T2022_DD_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
T2022_DD_B_hv = T2022_DD_B[:, T2022_DD_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(T2022_DD_B_hv)
sc.tl.pca(T2022_DD_B_hv, svd_solver='arpack')
sc.pp.neighbors(T2022_DD_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(T2022_DD_B_hv)
T2022_DD_B.obsm['X_umap'] = T2022_DD_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[232]:


sc.tl.leiden(T2022_DD_B_hv, resolution = 0.4)
T2022_DD_B.obs['leiden'] = T2022_DD_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(T2022_DD_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(T2022_DD_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(T2022_DD_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[233]:


basal_leiden_DD_T2022 = {"0":"0", "1":"1", "2":"2"}
T2022_DD_B.obs['subcluster'] = T2022_DD_B.obs.leiden.map(basal_leiden_DD_T2022)
T2022_DD_B.obs['subcluster'].value_counts()


# In[234]:


save_top_marker_genes(T2022_DD_B, 'T2022', 'BDD')


# In[235]:


T2022_DD_B.write_h5ad("./data/AAA_DCIS/260205_T2022_BDD.h5ad")


# In[101]:


T2022_DD_B = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_BDD.h5ad")
T2022_DD_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDD_na",
    "1": "BDD_Inflammatory_immune-responsive_EMT-high",
    "2": "BDD_EMT/invasive-like_ECM-remodelling_contractile"
}
T2022_DD_B.obs['annotation'] = T2022_DD_B.obs['subcluster'].map(annotation_map)
T2022_DD_B.write_h5ad("./data/AAA_DCIS/260205_T2022_BDD.h5ad")


# In[330]:


T2022_DD_B = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_BDD.h5ad")


# ### T2022 DN LM

# In[237]:


T2022_DN_LM = T2022_LM[T2022_LM.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(T2022_DN_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[238]:


sc.pp.highly_variable_genes(T2022_DN_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
T2022_DN_LM_hv = T2022_DN_LM[:, T2022_DN_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(T2022_DN_LM_hv)
sc.tl.pca(T2022_DN_LM_hv, svd_solver='arpack')
sc.pp.neighbors(T2022_DN_LM_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(T2022_DN_LM_hv)
T2022_DN_LM.obsm['X_umap'] = T2022_DN_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[240]:


sc.tl.leiden(T2022_DN_LM_hv, resolution = 0.4)
T2022_DN_LM.obs['leiden'] = T2022_DN_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(T2022_DN_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(T2022_DN_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(T2022_DN_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[241]:


LM_leiden_DN_T2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7"}
T2022_DN_LM.obs['subcluster'] = T2022_DN_LM.obs.leiden.map(LM_leiden_DN_T2022)
T2022_DN_LM.obs['subcluster'].value_counts()


# In[242]:


save_top_marker_genes(T2022_DN_LM, 'T2022', 'LMDN')


# In[243]:


T2022_DN_LM.write_h5ad("./data/AAA_DCIS/260217_T2022_LMDN.h5ad")


# In[146]:


T2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_LMDN.h5ad")
T2022_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDN_OXPHOS_metabolic",
    "1": "LMDN_na",
    "2": "LMDN_na",
    "3": "LMDN_Inflammatory_EMT-active_oxidative-stressed",
    "4": "LMDN_Hormone-responsive_epigenetically_active",
    "5": "LMDN_ER/UPR-stressed_secretory",
    "6": "LMDN_na",
    "7": "LMDN_Immune-surveilling_invasion-like"
}
T2022_DN_LM.obs['annotation'] = T2022_DN_LM.obs['subcluster'].map(annotation_map)
T2022_DN_LM.write_h5ad("./data/AAA_DCIS/260217_T2022_LMDN.h5ad")


# In[331]:


T2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_LMDN.h5ad")


# ### T2022 DD Luminal Mature

# In[244]:


T2022_LM = alldata_T2022[alldata_T2022.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(T2022_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[245]:


T2022_DD_LM = T2022_LM[T2022_LM.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(T2022_DD_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[246]:


sc.pl.pca_variance_ratio(T2022_DD_LM, n_pcs=50)


# In[247]:


sc.pp.highly_variable_genes(T2022_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
T2022_DD_LM_hv = T2022_DD_LM[:, T2022_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(T2022_DD_LM_hv)
sc.tl.pca(T2022_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(T2022_DD_LM_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(T2022_DD_LM_hv)
T2022_DD_LM.obsm['X_umap'] = T2022_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[249]:


sc.tl.leiden(T2022_DD_LM_hv, resolution = 0.3)
T2022_DD_LM.obs['leiden'] = T2022_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(T2022_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(T2022_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(T2022_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[250]:


LM_leiden_DD_T2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7"}
T2022_DD_LM.obs['subcluster'] = T2022_DD_LM.obs.leiden.map(LM_leiden_DD_T2022)
T2022_DD_LM.obs['subcluster'].value_counts()


# In[251]:


save_top_marker_genes(T2022_DD_LM, 'T2022', 'LMDD')


# In[252]:


T2022_DD_LM.write_h5ad("./data/AAA_DCIS/260205_T2022_LMDD.h5ad")


# In[103]:


T2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LMDD.h5ad")
T2022_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDD_ECM-remodelling_EMT-primed/plastic",
    "1": "LMDD_ECM-remodelling_EMT-primed/plastic",
    "2": "LMDD_Hormone-responsive_secretory_immune-modulating",
    "3": "LMDD_Proteostasis-active",
    "4": "LMDD_Stress-adapted_ROS-high",
    "5": "LMDD_Hormone-responsive_translationally_active",
    "6": "LMDD_Plastic_immune-modulating",
    "7": "LMDD_na"
}
T2022_DD_LM.obs['annotation'] = T2022_DD_LM.obs['subcluster'].map(annotation_map)
T2022_DD_LM.write_h5ad("./data/AAA_DCIS/260205_T2022_LMDD.h5ad")


# In[332]:


T2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LMDD.h5ad")


# ### T2022 DN LP

# In[255]:


T2022_DN_LP = T2022_LP[T2022_LP.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(T2022_DN_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[256]:


sc.pp.highly_variable_genes(T2022_DN_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
T2022_DN_LP_hv = T2022_DN_LP[:, T2022_DN_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(T2022_DN_LP_hv)
sc.tl.pca(T2022_DN_LP_hv, svd_solver='arpack')
sc.pp.neighbors(T2022_DN_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(T2022_DN_LP_hv)
T2022_DN_LP.obsm['X_umap'] = T2022_DN_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[257]:


sc.tl.leiden(T2022_DN_LP_hv, resolution = 0.3)
T2022_DN_LP.obs['leiden'] = T2022_DN_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(T2022_DN_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(T2022_DN_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(T2022_DN_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[258]:


LP_leiden_DN_T2022 = {"0":"0", "1":"1", "2":"2", "3":"3"}
T2022_DN_LP.obs['subcluster'] = T2022_DN_LP.obs.leiden.map(LP_leiden_DN_T2022)
T2022_DN_LP.obs['subcluster'].value_counts()


# In[259]:


save_top_marker_genes(T2022_DN_LP, 'T2022', 'LPDN')


# In[260]:


T2022_DN_LP.write_h5ad("./data/AAA_DCIS/260217_T2022_LPDN.h5ad")


# In[104]:


T2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_LPDN.h5ad")
T2022_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDN_na",
    "1": "LPDN_na",
    "2": "LPDN_na",
    "3": "LPDN_Type_1_interferon_signaling_growth-suppressed"
}
T2022_DN_LP.obs['annotation'] = T2022_DN_LP.obs['subcluster'].map(annotation_map)
T2022_DN_LP.write_h5ad("./data/AAA_DCIS/260217_T2022_LPDN.h5ad")


# In[333]:


T2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_LPDN.h5ad")


# ### T2022 DD Luminal Progenitor

# In[261]:


T2022_LP = alldata_T2022[alldata_T2022.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(T2022_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[262]:


T2022_DD_LP = T2022_LP[T2022_LP.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(T2022_DD_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[263]:


sc.pl.pca_variance_ratio(T2022_DD_LP, n_pcs=50)


# In[264]:


sc.pp.highly_variable_genes(T2022_DD_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
T2022_DD_LP_hv = T2022_DD_LP[:, T2022_DD_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(T2022_DD_LP_hv)
sc.tl.pca(T2022_DD_LP_hv, svd_solver='arpack')
sc.pp.neighbors(T2022_DD_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(T2022_DD_LP_hv)
T2022_DD_LP.obsm['X_umap'] = T2022_DD_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[266]:


sc.tl.leiden(T2022_DD_LP_hv, resolution = 0.3)
T2022_DD_LP.obs['leiden'] = T2022_DD_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(T2022_DD_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(T2022_DD_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(T2022_DD_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[267]:


LP_leiden_DD_T2022 = {"0":"0", "1":"1", "2":"2"}
T2022_DD_LP.obs['subcluster'] = T2022_DD_LP.obs.leiden.map(LP_leiden_DD_T2022)
T2022_DD_LP.obs['subcluster'].value_counts()


# In[268]:


save_top_marker_genes(T2022_DD_LP, 'T2022', 'LPDD')


# In[269]:


T2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")


# In[105]:


T2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")
T2022_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDD_na",
    "1": "LPDD_Proliferative_basal-like",
    "2": "LPDD_Antigen-presenting_protein-synthesising"
}
T2022_DD_LP.obs['annotation'] = T2022_DD_LP.obs['subcluster'].map(annotation_map)
T2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")


# In[334]:


T2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")


# # W2022

# In[71]:


alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[70]:


print(alldata_W2022.shape)
sc.pp.filter_genes(alldata_W2022, min_cells = 10) # 2 samples so only keep genes if in min 100 cells
alldata_W2022.X = csr_matrix(alldata_W2022.X) # convert dense to sparse matrix, less memory
print(alldata_W2022.shape)


# In[108]:


alldata_W2022.obs.groupby('Sample').count() # cells you have for each sample


# In[109]:


alldata_W2022.layers['counts'] = alldata_W2022.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(alldata_W2022, target_sum = 1e4) # normalise counts
sc.pp.log1p(alldata_W2022) # convert to log
alldata_W2022.raw = alldata_W2022
sc.pp.highly_variable_genes(alldata_W2022, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldata_W2022_hv = alldata_W2022[:, alldata_W2022.var['highly_variable']].copy() # subset hv
sc.pp.scale(alldata_W2022_hv)
sc.tl.pca(alldata_W2022_hv, svd_solver='arpack')
sc.pp.neighbors(alldata_W2022_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(alldata_W2022_hv)

# Copy UMAP coords back to full AnnData
alldata_W2022.obsm['X_umap'] = alldata_W2022_hv.obsm['X_umap']


# In[110]:


sc.tl.leiden(alldata_W2022_hv, resolution = 1.0)
alldata_W2022.obs['leiden'] = alldata_W2022_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(alldata_W2022_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(alldata_W2022_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(alldata_W2022, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[111]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(alldata_W2022, var_names=marker_genes, groupby='leiden')


# In[117]:


cell_type = {
 '0': 'Luminal Mature', '1': 'Luminal Mature', '2': 'T-Cell', '3': 'Luminal Mature', '4': 'Luminal Mature',
 '5': 'Luminal Progenitor', '6': 'T-Cell', '7': 'Luminal Mature', '8': 'B-Cell', '9': 'Macrophage',
 '10': 'T-Cell', '11': 'Luminal Mature', '12': 'T-Cell', '13': 'Endothelial', '14': 'Fibroblast',
 '15': 'General Myeloid', '16': 'T-Cell', '17': 'Luminal Mature', '18': 'Basal', '19': 'T-Cell',
 '20': 'Luminal Progenitor', '21': 'Basal', '22': 'Endothelial', '23': 'General Myeloid'
}


# In[113]:


sc.pl.umap(alldata_W2022, color = ['KRT14', 'ELF5', 'FOXA1', 'EPCAM', 'leiden'], frameon = False, legend_loc = "on data", ncols =3)
# Basal (KRT14), LP (ELF5), LM (FOXA1), Epithelial (EPCAM)


# In[112]:


sc.pl.rank_genes_groups(alldata_W2022_hv, n_genes=20, sharey=False)


# In[118]:


alldata_W2022.obs['cell type'] = alldata_W2022.obs.leiden.map(cell_type)
sc.pl.umap(alldata_W2022, color = ['cell type'], frameon = False)


# In[344]:


alldata_W2022.obs['Epithelial_vs_NonEpithelial'] = alldata_W2022.obs['cell type'].apply(
    lambda x: "Epithelial" if x in epithelial_types else "Non-epithelial"
)
alldata_W2022.obs[['cell type', 'Epithelial_vs_NonEpithelial']].head()
sc.pl.umap(alldata_W2022, color=['Epithelial_vs_NonEpithelial', 'cnv_status'], ncols=1)


# In[120]:


alldata_W2022.obs['cell type'].value_counts()


# In[271]:


W2022_Ep = alldata_W2022[alldata_W2022.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(W2022_Ep, color = ['cnv_status', 'cell type'], frameon = False)


# In[272]:


W2022_Ep_DD = W2022_Ep[W2022_Ep.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(W2022_Ep_DD, color = ['cnv_status', 'cell type'], frameon = False)


# In[120]:


alldata_W2022.write_h5ad('260203_dcis_W2022_only.h5ad')


# In[67]:


alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[381]:


W2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_BDN.h5ad")
W2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LMDN.h5ad")
W2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")
W2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LPDN.h5ad")
W2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")


# In[119]:


alldata_W2022.obs['annotation'] = pd.NA
subpops = [W2022_DN_B, W2022_DN_LM, W2022_DD_LM, W2022_DN_LP, W2022_DD_LP]
# Transfer annotations
for ad in subpops:
    alldata_W2022.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']
alldata_W2022.obs['annotation'] = alldata_W2022.obs['annotation'].fillna('unassigned')
sc.pl.umap(alldata_W2022, color = ['annotation', 'Epithelial_vs_NonEpithelial', 'cnv_status', 'cell type'], frameon = False, ncols=1)


# ### W2022 DN Basal

# In[274]:


W2022_DN_B = W2022_B[W2022_B.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(W2022_DN_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[275]:


sc.pl.pca_variance_ratio(W2022_DN_B, n_pcs=50)


# In[276]:


sc.pp.highly_variable_genes(W2022_DN_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DN_B_hv = W2022_DN_B[:, W2022_DN_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DN_B_hv)
sc.tl.pca(W2022_DN_B_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DN_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(W2022_DN_B_hv)
W2022_DN_B.obsm['X_umap'] = W2022_DN_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[278]:


sc.tl.leiden(W2022_DN_B_hv, resolution = 0.3)
W2022_DN_B.obs['leiden'] = W2022_DN_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DN_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DN_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DN_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[279]:


B_leiden_DN_W2022 = {"0":"0", "1":"1"}
W2022_DN_B.obs['subcluster'] = W2022_DN_B.obs.leiden.map(B_leiden_DN_W2022)
W2022_DN_B.obs['subcluster'].value_counts()


# In[280]:


save_top_marker_genes(W2022_DN_B, 'W2022', 'BDN')


# In[281]:


W2022_DN_B.write_h5ad("./data/AAA_DCIS/260217_W2022_BDN.h5ad")


# In[114]:


W2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_BDN.h5ad")
W2022_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDN_na",
    "1": "BDN_Inflammatory_ECM-remodelling"
}
W2022_DN_B.obs['annotation'] = W2022_DN_B.obs['subcluster'].map(annotation_map)
W2022_DN_B.write_h5ad("./data/AAA_DCIS/260217_W2022_BDN.h5ad")


# ### W2022 DD Basal

# In[282]:


W2022_B = alldata_W2022[alldata_W2022.obs['cell type'].isin(['Basal'])]
sc.pl.umap(W2022_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[330]:


# No DCIS Basal! DD Basal = 0


# ### W2022 DN LM

# In[284]:


W2022_DN_LM = W2022_LM[W2022_LM.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(W2022_DN_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[285]:


sc.pp.highly_variable_genes(W2022_DN_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DN_LM_hv = W2022_DN_LM[:, W2022_DN_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DN_LM_hv)
sc.tl.pca(W2022_DN_LM_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DN_LM_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(W2022_DN_LM_hv)
W2022_DN_LM.obsm['X_umap'] = W2022_DN_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[286]:


sc.tl.leiden(W2022_DN_LM_hv, resolution = 0.5)
W2022_DN_LM.obs['leiden'] = W2022_DN_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DN_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DN_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DN_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[287]:


LM_leiden_DN_W2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
W2022_DN_LM.obs['subcluster'] = W2022_DN_LM.obs.leiden.map(LM_leiden_DN_W2022)
W2022_DN_LM.obs['subcluster'].value_counts()


# In[288]:


save_top_marker_genes(W2022_DN_LM, 'W2022', 'LMDN')


# In[289]:


W2022_DN_LM.write_h5ad("./data/AAA_DCIS/260217_W2022_LMDN.h5ad")


# In[115]:


W2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LMDN.h5ad")
W2022_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDN_OXPHOS_metabolic",
    "1": "LMDN_ER/UPR-stressed_secretory",
    "2": "LMDN_Immune-modulating_stem-like",
    "3": "LMDN_na",
    "4":"LMDN_na"
}
W2022_DN_LM.obs['annotation'] = W2022_DN_LM.obs['subcluster'].map(annotation_map)
W2022_DN_LM.write_h5ad("./data/AAA_DCIS/260217_W2022_LMDN.h5ad")


# ### W2022 DD LM

# In[290]:


W2022_LM = alldata_W2022[alldata_W2022.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(W2022_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[291]:


W2022_DD_LM = W2022_LM[W2022_LM.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(W2022_DD_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[292]:


sc.pl.pca_variance_ratio(W2022_DD_LM, n_pcs=50)


# In[50]:


sc.pp.highly_variable_genes(W2022_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DD_LM_hv = W2022_DD_LM[:, W2022_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DD_LM_hv)
sc.tl.pca(W2022_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DD_LM_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(W2022_DD_LM_hv)
W2022_DD_LM.obsm['X_umap'] = W2022_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[51]:


sc.tl.leiden(W2022_DD_LM_hv, resolution = 0.4)
W2022_DD_LM.obs['leiden'] = W2022_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[52]:


LM_leiden_DD_W2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
W2022_DD_LM.obs['subcluster'] = W2022_DD_LM.obs.leiden.map(LM_leiden_DD_W2022)
W2022_DD_LM.obs['subcluster'].value_counts()


# In[295]:


save_top_marker_genes(W2022_DD_LM, 'W2022', 'LMDD')


# In[53]:


W2022_DD_LM.write_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")


# In[116]:


W2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")
W2022_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDD_Hormone-responsive_translationally_active",
    "1": "LMDD_Proteostasis-active",
    "2": "LMDD_ECM-remodelling_EMT-primed/plastic",
    "3": "LMDD_na",
    "4": "LMDD_na"
}
W2022_DD_LM.obs['annotation'] = W2022_DD_LM.obs['subcluster'].map(annotation_map)
W2022_DD_LM.write_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")


# ### W2022 DN LP

# In[299]:


W2022_DN_LP = W2022_LP[W2022_LP.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(W2022_DN_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[300]:


sc.pp.highly_variable_genes(W2022_DN_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DN_LP_hv = W2022_DN_LP[:, W2022_DN_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DN_LP_hv)
sc.tl.pca(W2022_DN_LP_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DN_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(W2022_DN_LP_hv)
W2022_DN_LP.obsm['X_umap'] = W2022_DN_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[301]:


sc.tl.leiden(W2022_DN_LP_hv, resolution = 0.3)
W2022_DN_LP.obs['leiden'] = W2022_DN_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DN_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DN_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DN_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[302]:


LP_leiden_DN_W2022 = {"0":"0", "1":"1", "2":"2"}
W2022_DN_LP.obs['subcluster'] = W2022_DN_LP.obs.leiden.map(LP_leiden_DN_W2022)
W2022_DN_LP.obs['subcluster'].value_counts()


# In[303]:


save_top_marker_genes(W2022_DN_LP, 'W2022', 'LPDN')


# In[304]:


W2022_DN_LP.write_h5ad("./data/AAA_DCIS/260217_W2022_LPDN.h5ad")


# In[117]:


W2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LPDN.h5ad")
W2022_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDN_na",
    "1": "LPDN_Redox-stressed_tumour_protective",
    "2": "LPDN_na"
}
W2022_DN_LP.obs['annotation'] = W2022_DN_LP.obs['subcluster'].map(annotation_map)
W2022_DN_LP.write_h5ad("./data/AAA_DCIS/260217_W2022_LPDN.h5ad")


# ### W2022 DD LP

# In[305]:


W2022_LP = alldata_W2022[alldata_W2022.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(W2022_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[306]:


W2022_DD_LP = W2022_LP[W2022_LP.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(W2022_DD_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[307]:


sc.pl.pca_variance_ratio(W2022_DD_LP, n_pcs=50)


# In[57]:


sc.pp.highly_variable_genes(W2022_DD_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DD_LP_hv = W2022_DD_LP[:, W2022_DD_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DD_LP_hv)
sc.tl.pca(W2022_DD_LP_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DD_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(W2022_DD_LP_hv)
W2022_DD_LP.obsm['X_umap'] = W2022_DD_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[58]:


sc.tl.leiden(W2022_DD_LP_hv, resolution = 0.5)
W2022_DD_LP.obs['leiden'] = W2022_DD_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DD_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DD_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DD_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[63]:


LP_leiden_DD_W2022 = {"0":"0"}
W2022_DD_LP.obs['subcluster'] = W2022_DD_LP.obs.leiden.map(LP_leiden_DD_W2022)
W2022_DD_LP.obs['subcluster'].value_counts()


# In[60]:


save_top_marker_genes(W2022_DD_LP, 'W2022', 'LPDD')


# In[64]:


W2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")


# In[118]:


W2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")
W2022_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDD_na"
}
W2022_DD_LP.obs['annotation'] = W2022_DD_LP.obs['subcluster'].map(annotation_map)
W2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")


# # G2017 (G2021)

# In[73]:


alldata_G2017 = sc.read_h5ad('260203_dcis_G2017_only.h5ad')


# In[5]:


# Replace 'G2017' with 'G2021' in the 'Batch' column
alldata_G2017.obs['Batch'] = alldata_G2017.obs['Batch'].replace('G2017', 'G2021')


# In[7]:


alldata_G2017.obs['Sample'] = alldata_G2017.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')


# In[121]:


print(alldata_G2017.shape)
sc.pp.filter_genes(alldata_G2017, min_cells = 5) # 2 samples so only keep genes if in min 100 cells
alldata_G2017.X = csr_matrix(alldata_G2017.X) # convert dense to sparse matrix, less memory
print(alldata_G2017.shape)


# In[122]:


alldata_G2017.obs.groupby('Sample').count() # cells you have for each sample


# In[123]:


alldata_G2017.layers['counts'] = alldata_G2017.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(alldata_G2017, target_sum = 1e4) # normalise counts
sc.pp.log1p(alldata_G2017) # convert to log
alldata_G2017.raw = alldata_G2017
sc.pp.highly_variable_genes(alldata_G2017, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldata_G2017_hv = alldata_G2017[:, alldata_G2017.var['highly_variable']].copy() # subset hv
sc.pp.scale(alldata_G2017_hv)
sc.tl.pca(alldata_G2017_hv, svd_solver='arpack')
sc.pp.neighbors(alldata_G2017_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(alldata_G2017_hv)

# Copy UMAP coords back to full AnnData
alldata_G2017.obsm['X_umap'] = alldata_G2017_hv.obsm['X_umap']


# In[124]:


sc.tl.leiden(alldata_G2017_hv, resolution = 1.0)
alldata_G2017.obs['leiden'] = alldata_G2017_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(alldata_G2017_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(alldata_G2017_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(alldata_G2017, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[126]:


sc.pl.umap(alldata_G2017, color = ['KRT14', 'ELF5', 'FOXA1', 'EPCAM', 'leiden'], frameon = False, legend_loc = "on data", ncols =3)
# Basal (KRT14), LP (ELF5), LM (FOXA1), Epithelial (EPCAM)


# In[8]:


sc.pl.umap(alldata_G2017, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[125]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(alldata_G2017, var_names=marker_genes, groupby='leiden')


# In[127]:


sc.pl.rank_genes_groups(alldata_G2017_hv, n_genes=20, sharey=False)


# In[128]:


cell_type = {'0': 'Luminal Mature', '1': 'Luminal Mature', '2': 'Macrophage', '3': 'Luminal Mature', '4': 'Luminal Mature',
 '5': 'Luminal Mature', '6': 'Luminal Mature', '7': 'Luminal Mature', '8': 'Luminal Mature', '9': 'Fibroblast',
 '10': 'Luminal Mature', '11': 'Endothelial', '12': 'Luminal Mature', '13': 'Luminal Progenitor', '14': 'Basal',
 '15': 'Luminal Mature', '16': 'Endothelial', '17': 'Basal'}


# In[129]:


alldata_G2017.obs['cell type'] = alldata_G2017.obs.leiden.map(cell_type)
sc.pl.umap(alldata_G2017, color = ['cell type'], frameon = False)


# In[345]:


alldata_G2017.obs['Epithelial_vs_NonEpithelial'] = alldata_G2017.obs['cell type'].apply(
    lambda x: "Epithelial" if x in epithelial_types else "Non-epithelial"
)
alldata_G2017.obs[['cell type', 'Epithelial_vs_NonEpithelial']].head()
sc.pl.umap(alldata_G2017, color=['Epithelial_vs_NonEpithelial', 'cnv_status'], ncols=1)


# In[131]:


alldata_G2017.obs['cell type'].value_counts()


# In[1251]:


G2017_Ep = alldata_G2017[alldata_G2017.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(G2017_Ep, color = ['cnv_status', 'cell type'], frameon = False)


# In[1252]:


G2017_Ep_DD = G2017_Ep[G2017_Ep.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(G2017_Ep_DD, color = ['cnv_status', 'cell type'], frameon = False)


# In[133]:


alldata_G2017.write_h5ad('260203_dcis_G2017_only.h5ad')


# In[392]:


alldata_G2017.obs['cell_id_orig'] = alldata_G2017.obs.index


# In[74]:


alldata_G2017 = sc.read_h5ad('260203_dcis_G2017_only.h5ad')


# In[131]:


alldata_G2017.obs['annotation'] = pd.NA

subpops = [G2017_DN_B, G2017_DN_LM, G2017_DD_LM, G2017_DN_LP]
# Transfer annotations
for ad in subpops:
    alldata_G2017.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']

alldata_G2017.obs['annotation'] = alldata_G2017.obs['annotation'].fillna('unassigned')


# In[132]:


sc.pl.umap(alldata_G2017, color = ['annotation', 'Epithelial_vs_NonEpithelial', 'cnv_status', 'cell type'], frameon = False, ncols=1)


# ### G2017 DN Basal
# 

# In[315]:


G2017_DN_B = G2017_B[G2017_B.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(G2017_DN_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[316]:


sc.pp.highly_variable_genes(G2017_DN_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
G2017_DN_B_hv = G2017_DN_B[:, G2017_DN_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(G2017_DN_B_hv)
sc.tl.pca(G2017_DN_B_hv, svd_solver='arpack')
sc.pp.neighbors(G2017_DN_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(G2017_DN_B_hv)
G2017_DN_B.obsm['X_umap'] = G2017_DN_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[317]:


sc.tl.leiden(G2017_DN_B_hv, resolution = 0.4)
G2017_DN_B.obs['leiden'] = G2017_DN_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(G2017_DN_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(G2017_DN_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(G2017_DN_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[318]:


B_leiden_DN_G2017 = {"0":"0", "1":"1"}
G2017_DN_B.obs['subcluster'] = G2017_DN_B.obs.leiden.map(B_leiden_DN_G2017)
G2017_DN_B.obs['subcluster'].value_counts()


# In[319]:


save_top_marker_genes(G2017_DN_B, 'G2017', 'BDN')


# In[320]:


G2017_DN_B.write_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")


# In[121]:


G2017_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")
G2017_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDN_Mesenchymal-like",
    "1": "BDN_Inflammatory_immune-reactive_apoptosis-primed_adhering"
}
G2017_DN_B.obs['annotation'] = G2017_DN_B.obs['subcluster'].map(annotation_map)
G2017_DN_B.write_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")


# In[10]:


G2017_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")
G2017_DN_B.obs['Batch'] = G2017_DN_B.obs['Batch'].replace('G2017', 'G2021')
G2017_DN_B.obs['Sample'] = G2017_DN_B.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DN_B.write_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")


# ### G2017 DD Basal

# In[123]:


G2017_B = alldata_G2017[alldata_G2017.obs['cell type'].isin(['Basal'])]
sc.pl.umap(G2017_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[124]:


G2017_DD_B = G2017_B[G2017_B.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(G2017_DD_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[127]:


G2017_DD_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "BDN_na"
}
G2017_DD_B.obs['annotation'] = pd.NA
G2017_DD_B.obs['annotation'] = G2017_DD_B.obs['annotation'].fillna('BDN_na')
G2017_DD_B.write_h5ad("./data/AAA_DCIS/260217_G2017_BDD.h5ad")


# In[333]:


G2017_DD_B = sc.read_h5ad("./data/AAA_DCIS/260205_G2017_BDD.h5ad")
G2017_DD_B.obs['Batch'] = G2017_DD_B.obs['Batch'].replace('G2017', 'G2021')
G2017_DD_B.obs['Sample'] = G2017_DD_B.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DD_B.write_h5ad("./data/AAA_DCIS/260205_G2017_BDD.h5ad")


# ### G2017 DN LM

# In[325]:


G2017_DN_LM = G2017_LM[G2017_LM.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(G2017_DN_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[326]:


sc.pp.highly_variable_genes(G2017_DN_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
G2017_DN_LM_hv = G2017_DN_LM[:, G2017_DN_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(G2017_DN_LM_hv)
sc.tl.pca(G2017_DN_LM_hv, svd_solver='arpack')
sc.pp.neighbors(G2017_DN_LM_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(G2017_DN_LM_hv)
G2017_DN_LM.obsm['X_umap'] = G2017_DN_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[327]:


sc.tl.leiden(G2017_DN_LM_hv, resolution = 0.4)
G2017_DN_LM.obs['leiden'] = G2017_DN_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(G2017_DN_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(G2017_DN_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(G2017_DN_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[328]:


LM_leiden_DN_G2017 = {"0":"0", "1":"1", "2":"2"}
G2017_DN_LM.obs['subcluster'] = G2017_DN_LM.obs.leiden.map(LM_leiden_DN_G2017)
G2017_DN_LM.obs['subcluster'].value_counts()


# In[329]:


save_top_marker_genes(G2017_DN_LM, 'G2017', 'LMDN')


# In[330]:


G2017_DN_LM.write_h5ad("./data/AAA_DCIS/260217_G2017_LMDN.h5ad")


# In[128]:


G2017_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_LMDN.h5ad")
G2017_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDN_na",
    "1": "LMDN_Hormone-responsive_epigenetically_active",
    "2": "LMDN_IFN-active_antigen-presenting"
}
G2017_DN_LM.obs['annotation'] = G2017_DN_LM.obs['subcluster'].map(annotation_map)
G2017_DN_LM.write_h5ad("./data/AAA_DCIS/260217_G2017_LMDN.h5ad")


# In[332]:


G2017_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_LMDN.h5ad")
G2017_DN_LM.obs['Batch'] = G2017_DN_LM.obs['Batch'].replace('G2017', 'G2021')
G2017_DN_LM.obs['Sample'] = G2017_DN_LM.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DN_LM.write_h5ad("./data/AAA_DCIS/260217_G2017_LMDN.h5ad")


# ### G2017 DD LM

# In[334]:


G2017_LM = alldata_G2017[alldata_G2017.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(G2017_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[335]:


G2017_DD_LM = G2017_LM[G2017_LM.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(G2017_DD_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[336]:


sc.pl.pca_variance_ratio(G2017_DD_LM, n_pcs=50)


# In[337]:


sc.pp.highly_variable_genes(G2017_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
G2017_DD_LM_hv = G2017_DD_LM[:, G2017_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(G2017_DD_LM_hv)
sc.tl.pca(G2017_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(G2017_DD_LM_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(G2017_DD_LM_hv)
G2017_DD_LM.obsm['X_umap'] = G2017_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[339]:


sc.tl.leiden(G2017_DD_LM_hv, resolution = 0.3)
G2017_DD_LM.obs['leiden'] = G2017_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(G2017_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(G2017_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(G2017_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[340]:


LM_leiden_DD_G2017 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5"}
G2017_DD_LM.obs['subcluster'] = G2017_DD_LM.obs.leiden.map(LM_leiden_DD_G2017)
G2017_DD_LM.obs['subcluster'].value_counts()


# In[341]:


save_top_marker_genes(G2017_DD_LM, 'G2017', 'LMDD')


# In[342]:


G2017_DD_LM.write_h5ad("./data/AAA_DCIS/260205_G2017_LMDD.h5ad")


# In[129]:


G2017_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_G2017_LMDD.h5ad")
G2017_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LMDD_na",
    "1": "LMDD_na",
    "2": "LMDD_na",
    "3": "LMDD_na",
    "4": "LMDD_na",
    "5": "LMDD_Hormone-responsive_translationally_active"
}
G2017_DD_LM.obs['annotation'] = G2017_DD_LM.obs['subcluster'].map(annotation_map)
G2017_DD_LM.write_h5ad("./data/AAA_DCIS/260205_G2017_LMDD.h5ad")


# In[343]:


G2017_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_G2017_LMDD.h5ad")
G2017_DD_LM.obs['Batch'] = G2017_DD_LM.obs['Batch'].replace('G2017', 'G2021')
G2017_DD_LM.obs['Sample'] = G2017_DD_LM.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DD_LM.write_h5ad("./data/AAA_DCIS/260205_G2017_LMDD.h5ad")


# ### G2017 DN LP

# In[345]:


G2017_DN_LP = G2017_LP[G2017_LP.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(G2017_DN_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[346]:


sc.pp.highly_variable_genes(G2017_DN_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
G2017_DN_LP_hv = G2017_DN_LP[:, G2017_DN_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(G2017_DN_LP_hv)
sc.tl.pca(G2017_DN_LP_hv, svd_solver='arpack')
sc.pp.neighbors(G2017_DN_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(G2017_DN_LP_hv)
G2017_DN_LP.obsm['X_umap'] = G2017_DN_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[347]:


sc.tl.leiden(G2017_DN_LP_hv, resolution = 0.5)
G2017_DN_LP.obs['leiden'] = G2017_DN_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(G2017_DN_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(G2017_DN_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(G2017_DN_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[348]:


LP_leiden_DN_G2017 = {"0":"0", "1":"1"}
G2017_DN_LP.obs['subcluster'] = G2017_DN_LP.obs.leiden.map(LP_leiden_DN_G2017)
G2017_DN_LP.obs['subcluster'].value_counts()


# In[349]:


save_top_marker_genes(G2017_DN_LP, 'G2017', 'LPDN')


# In[350]:


G2017_DN_LP.write_h5ad("./data/AAA_DCIS/260217_G2017_LPDN.h5ad")


# In[130]:


G2017_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_LPDN.h5ad")
G2017_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "LPDN_na",
    "1": "LPDN_na"
}
G2017_DN_LP.obs['annotation'] = G2017_DN_LP.obs['subcluster'].map(annotation_map)
G2017_DN_LP.write_h5ad("./data/AAA_DCIS/260217_G2017_LPDN.h5ad")


# In[351]:


G2017_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_LPDN.h5ad")
G2017_DN_LP.obs['Batch'] = G2017_DN_LP.obs['Batch'].replace('G2017', 'G2021')
G2017_DN_LP.obs['Sample'] = G2017_DN_LP.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DN_LP.write_h5ad("./data/AAA_DCIS/260217_G2017_LPDN.h5ad")


# ### G2017 DD LP

# In[352]:


G2017_LP = alldata_G2017[alldata_G2017.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(G2017_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[353]:


#no LP DD G2021/17 cells


# In[1287]:


G2017_DD_LP = G2017_LP[G2017_LP.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(G2017_DD_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[478]:


G2017_DD_LP.write_h5ad("./data/AAA_DCIS/260205_G2017_LPDD.h5ad")


# In[15]:


G2017_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_G2017_LPDD.h5ad")
G2017_DD_LP.obs['Batch'] = G2017_DD_LP.obs['Batch'].replace('G2017', 'G2021')
G2017_DD_LP.obs['Sample'] = G2017_DD_LP.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DD_LP.write_h5ad("./data/AAA_DCIS/260205_G2017_LPDD.h5ad")


# # Clustering

# In[24]:


# reloaded
alldata = sc.read_h5ad('260121_dcis_combined_after_pp.h5ad')


# In[42]:


print(alldata.shape)
sc.pp.filter_genes(alldata, min_cells = 200) #lot of samples, so higher filtering of genes: only keep genes if in min 200 cells
alldata.X = csr_matrix(alldata.X) # convert dense to sparse matrix, less memory
print(alldata.shape)


# In[44]:


alldata.obs.groupby('Sample').count() # cells you have for each sample


# In[45]:


alldata.layers['counts'] = alldata.X.copy() # save data before normalise/log transform, need later for scvi
sc.pp.normalize_total(alldata, target_sum = 1e4) # normalise counts
sc.pp.log1p(alldata) # convert to log
alldata.raw = alldata
alldata.obs.head() # inspect


# In[46]:


sc.pp.highly_variable_genes(alldata, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldata_hv = alldata[:, alldata.var['highly_variable']].copy() # subset hv


# In[8]:


alldata_hv.write_h5ad('260121_dcis_alldata_hv.h5ad') # save for later


# In[56]:


sc.pp.scale(alldata_hv)
sc.tl.pca(alldata_hv, svd_solver='arpack')
sc.pp.neighbors(alldata_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(alldata_hv)

# Copy UMAP coords back to full AnnData
alldata.obsm['X_umap'] = alldata_hv.obsm['X_umap']


# In[62]:


sc.tl.leiden(alldata_hv, resolution = 0.6)
alldata.obs['leiden'] = alldata_hv.obs['leiden']  # copy clusters to full AnnData


# In[63]:


sc.tl.rank_genes_groups(alldata_hv, groupby='leiden', method='t-test')
markers = sc.get.rank_genes_groups_df(alldata_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]


# In[64]:


sc.pl.umap(alldata, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[65]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(alldata, var_names=marker_genes, groupby='leiden')


# In[132]:


cell_type = {
    "0":"Luminal Mature", "1":"T-Cell", "2":"Fibroblast", "3":"Luminal Progenitor", "4":"Luminal Mature", 
    "5":"T-Cell", "6":"General Myeloid", "7":"Endothelial", "8":"Endothelial", 
    "9":"Luminal Mature", "10":"Luminal Mature", "11":"General Myeloid", "12":"B-Cell",
    "13":"Luminal Mature", "14":"Luminal Mature", "15":"Luminal Progenitor", "16":"Basal", "17":"Luminal Mature",
    "18":"Basal", "19":"Luminal Mature", "20":"Endothelial", "21":"Monocyte", "22":"Luminal Mature",
    "23":"Luminal Mature", "24":"General Myeloid", "25":"Luminal Mature", "26":"Luminal Mature", "27":"Luminal Mature",
    "28":"Endothelial", "29":"Luminal Mature", "30":"Luminal Mature", "31":"Luminal Mature", "32":"Luminal Mature",
    "33":"Luminal Mature", "34":"General Myeloid"
}


# In[94]:


sc.pl.rank_genes_groups(alldata_hv, n_genes=20, sharey=False)


# In[133]:


alldata.obs['cell type'] = alldata.obs.leiden.map(cell_type)


# In[134]:


sc.pl.umap(alldata, color = ['cell type'], frameon = False)


# In[432]:


sc.pl.umap(alldata, color = ['cell type'], frameon = False)


# In[19]:


sc.pl.umap(alldata, color = ['Batch'], frameon = False)


# In[20]:


sc.pl.umap(alldata, color = ['Sample'], frameon = False)


# In[18]:


alldata.write_h5ad("260130_dcis_alldata_annotated.h5ad")


# In[16]:


alldata = sc.read_h5ad("260130_dcis_alldata_annotated.h5ad")


# In[17]:


alldata.obs['Batch'] = alldata.obs['Batch'].replace('G2017', 'G2021')
alldata.obs['Sample'] = alldata.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')


# In[95]:


(alldata.obs
    .query("`cell type` == 'Basal'")
    .groupby('Batch')
    .size()
    .rename('n_cells')
)


# In[96]:


(alldata.obs
    .query("`cell type` == 'Luminal Mature'")
    .groupby('Batch')
    .size()
    .rename('n_cells')
)


# In[97]:


(alldata.obs
    .query("`cell type` == 'Luminal Progenitor'")
    .groupby('Batch')
    .size()
    .rename('n_cells')
)


# # Annotate Cell Types (pre-SCVI)

# In[72]:


sc.pl.umap(alldata, color=['leiden'], legend_loc = 'on data')


# In[69]:


sc.pl.umap(alldata, color = ['EPCAM', 'MUC1'], frameon = False, vmax = 5)


# In[70]:


sc.pl.umap(alldata, color = ['PTPRC', 'CD3E', 'CD4', 'CD8A'], frameon = False, vmax = 5) 
# PTPRC = CD45 (blood cells), CD3E = T-cells, CD4 = CD4+ T-cells, CD8A = CD8+ T-cells


# In[71]:


#B-Cell
sc.pl.umap(alldata, color = ['IGKC', 'CD79B'], frameon = False, vmax = 5) 


# In[147]:


# check if a gene was ever present
"AGR2" in alldata.var_names


# In[148]:


"AGR2" in alldata.raw.var_names


# In[74]:


#T-Cell
sc.pl.umap(alldata, color = ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'], frameon = False, vmax = 5) 


# In[75]:


#Macrophage
sc.pl.umap(alldata, color = ['APOE', 'CCL3', 'CCL4', 'IL1B'], frameon = False, vmax = 5) 


# In[76]:


#Monocyte
sc.pl.umap(alldata, color = ['VCAN', 'CD14'], frameon = False, vmax = 5) 


# In[77]:


#General Myeloid Markers
sc.pl.umap(alldata, color = ['HLA-DRA', 'HLA-DPA1', 'CD74'], frameon = False, vmax = 5) 


# In[78]:


#Fibroblast
sc.pl.umap(alldata, color = ['DCN', 'APOD', 'COL1A1'], frameon = False, vmax = 5) 


# In[79]:


#Endothelial
sc.pl.umap(alldata, color = ['PECAM1', 'CLDN5'], frameon = False, vmax = 5) 


# In[80]:


#Adipocytes
sc.pl.umap(alldata, color = ['APOE'], frameon = False, vmax = 5) 


# In[81]:


#Basal
sc.pl.umap(alldata, color = ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'], frameon = False, vmax = 5) 


# In[82]:


#Luminal Progenitor
sc.pl.umap(alldata, color = ['ELF5', 'KRT15', 'LTF', 'SLPI'], frameon = False, vmax = 5) 


# In[74]:


#Luminal Mature
sc.pl.umap(alldata, color = ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'], frameon = False, vmax = 5) 


# In[84]:


#Epithelial
sc.pl.umap(alldata, color = ['EPCAM'], frameon = False, vmax = 5) 


# # Run SCVI to Integrate

# In[3]:


scvi.model.SCVI.setup_anndata(alldata, layer = "counts",
                             categorical_covariate_keys=["Sample", "Batch"],
                             continuous_covariate_keys=['pct_counts_mt', 'total_counts', 'pct_counts_ribo'])


# In[4]:


model = scvi.model.SCVI(alldata)


# In[5]:


model.train()


# In[6]:


alldata.obsm['X_scVI'] = model.get_latent_representation()


# In[7]:


alldata.layers['scvi_normalized'] = model.get_normalized_expression(library_size = 1e4)


# In[8]:


sc.pp.neighbors(alldata, use_rep = 'X_scVI')


# In[54]:


sc.tl.umap(alldata)
sc.tl.leiden(alldata, resolution = 0.8)


# In[55]:


sc.pl.umap(alldata, color = ['leiden', 'Sample', 'Batch'], frameon = False, ncols = 1)


# In[56]:


sc.pl.umap(alldata, color = ['leiden'], frameon = False, ncols = 1)


# In[57]:


sc.pl.umap(alldata, color = ['Batch'], frameon = False, ncols = 1)


# In[58]:


sc.pl.umap(alldata, color = ['Sample'], frameon = False, ncols = 1)


# In[26]:


alldata.write_h5ad('260121_dcis_scvi_integrated.h5ad')


# In[61]:


import numpy as np
import scanpy as sc
import anndata as ad

print(np.__version__)
print(sc.__version__)
print(ad.__version__)


# In[60]:


# reload
alldata = sc.read_h5ad('260121_dcis_scvi_integrated.h5ad')


# In[ ]:


alldata.obs['Batch'] = alldata.obs['Batch'].replace('G2017', 'G2021')
alldata.obs['Sample'] = alldata.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')


# In[ ]:


sc.pl.umap(alldata, color = ['Batch'], frameon = True, ncols = 1)


# In[ ]:


sc.pl.umap(alldata, color = ['Sample'], frameon = True, ncols = 1)


# # Annotate Cell Types (post-SCVI)

# In[49]:


sc.tl.leiden(alldata, resolution = 0.8)
sc.tl.rank_genes_groups(alldata, 'leiden')
markers = sc.get.rank_genes_groups_df(alldata, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > .5)]
markers


# In[50]:


markers_scvi = model.differential_expression(groupby = 'leiden')
markers_scvi


# In[51]:


markers_scvi = markers_scvi[(markers_scvi['is_de_fdr_0.05']) & (markers_scvi.lfc_mean > .5)]
markers_scvi


# In[52]:


sc.pl.umap(alldata, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[72]:


#Basal
sc.pl.umap(alldata, color = ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'], frameon = False, vmax = 5) 


# In[73]:


#Luminal Progenitor
sc.pl.umap(alldata, color = ['ELF5', 'KRT15', 'LTF', 'SLPI'], frameon = False, vmax = 5) 


# In[75]:


#Luminal Mature
sc.pl.umap(alldata, color = ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'], frameon = False, vmax = 5) 


# In[71]:


#inspect top genes some cluster of interest
cluster = "38"  
top_cluster32 = (
    markers_scvi
    .query("group1 == @cluster")
    .sort_values("lfc_mean", ascending=False)
    .head(8)
)
top_cluster32


# In[60]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK'],
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP'],
    "Luminal_Progenitor": ['ELF5', 'KRT15', 'LTF', 'SLPI'],
    "Adipocyte": ['APOE'],
    "Endothelial": ['PECAM1', 'CLDN5'],
    "Fibroblast": ['DCN', 'APOD', 'COL1A1'],
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B'],
    "T-Cell": ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2'],
    "B-Cell": ['IGKC', 'CD79B']
}

sc.pl.dotplot(
    alldata,
    var_names=marker_genes,
    groupby='leiden'
)


# In[85]:


cell_type = {"0":"T-Cell", "1":"Fibroblast", "2":"Fibroblast", "3":"Luminal Progenitor", "4":"Luminal Mature", "5":"Luminal Mature",
             "6":"Luminal Mature", "7":"Luminal Mature", "8":"Endothelial", "9":"General Myeloid", "10":"Endothelial", "11":"T-Cell",
             "12":"General Myeloid", "13":"Fibroblast", "14":"Luminal Mature", "15":"Luminal Progenitor", "16":"Luminal Progenitor", "17":"Basal",
             "18":"Luminal Mature", "19":"Luminal Mature", "20":"Luminal Progenitor", "21":"Luminal Mature", "22":"Luminal Mature", "23":"Luminal Mature",
             "24":"General Myeloid", "25":"Luminal Mature", "26":"Luminal Mature", "27":"Luminal Mature", "28":"B-Cell", "29":"Luminal Mature",
             "30":"Endothelial", "31":"Luminal Mature", "32":"Luminal Mature", "33":"B-Cell", "34":"Luminal Mature", "35":"Luminal Progenitor", "36":"Basal", "37":"General Myeloid", "38":"Fibroblast"}


# In[86]:


alldata.obs['cell type'] = alldata.obs.leiden.map(cell_type)


# In[87]:


sc.pl.umap(alldata, color = ['cell type'], frameon = False)


# In[88]:


alldata.write_h5ad('260121_dcis_scvi_integrated.h5ad')


# In[91]:


(alldata.obs
    .query("`cell type` == 'Basal'")
    .groupby('Batch')
    .size()
    .rename('n_cells')
)


# In[92]:


(alldata.obs
    .query("`cell type` == 'Luminal Mature'")
    .groupby('Batch')
    .size()
    .rename('n_cells')
)


# In[93]:


(alldata.obs
    .query("`cell type` == 'Luminal Progenitor'")
    .groupby('Batch')
    .size()
    .rename('n_cells')
)


# # InferCNVpy

# In[6]:


import infercnvpy as cnv
import pybiomart
import scipy.sparse
sc.logging.print_header()


# ## Q2025

# In[6]:


#alldata = alldata_Q2025
alldata = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')


# In[145]:


#rename old cnv_status (conducted on whole study)
#to cnv_status_old & drop cnv_status (so can rename
#with new samples subsets names
alldata.obs['cnv_status_old'] = alldata.obs['cnv_status']


# ### Subset Q2025 1 (ind5, ind6, ind7, ind8 = ALL SAME PATIENT)

# In[622]:


Q2025_subset1 = ["ind5_Q2025", "ind6_Q2025", "ind7_Q2025", "ind8_Q2025"]
Q2025_subset1 = alldata[alldata.obs['Sample'].isin(Q2025_subset1)].copy()


# In[925]:


nuclear_genes = ~Q2025_subset1.var['chromosome'].isin(['MT', 'chrMT', 'M'])
Q2025_subset1 = Q2025_subset1[:, nuclear_genes]
cnv.tl.infercnv(
    Q2025_subset1,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
    
)


# In[926]:


cnv.pl.chromosome_heatmap(Q2025_subset1, groupby="cell type", figsize=(30,10))


# In[927]:


cnv.tl.pca(Q2025_subset1)
cnv.pp.neighbors(Q2025_subset1)
cnv.tl.leiden(Q2025_subset1)


# In[928]:


sc.tl.dendrogram(Q2025_subset1, groupby='cnv_leiden')


# In[929]:


cnv.tl.umap(Q2025_subset1)
cnv.tl.cnv_score(Q2025_subset1)


# In[930]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))
ax4.axis("off")
cnv.pl.umap(
    Q2025_subset1,
    color="cnv_leiden",
    legend_loc="on data",
    legend_fontoutline=2,
    ax=ax1,
    show=False,
)
cnv.pl.umap(Q2025_subset1, color="cnv_score", ax=ax2, show=False)
cnv.pl.umap(Q2025_subset1, color="cell type", ax=ax3)


# In[931]:


cnv.pl.umap(Q2025_subset1, color="cnv_score", show=False)
cnv.pl.umap(Q2025_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(Q2025_subset1, color="cell type", show=False)


# In[932]:


cnv.pl.chromosome_heatmap(Q2025_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,40))


# In[933]:


sc.pl.umap(Q2025_subset1, color="cnv_score")
sc.pl.umap(Q2025_subset1, color="cell type")


# In[934]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[935]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = Q2025_subset1.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_subset_1_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[936]:


mean_cnv = Q2025_subset1.obs['cnv_score'].mean()
median_cnv = Q2025_subset1.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[943]:


Q2025_subset1.obs["cnv_status"] = "normal"
Q2025_subset1.obs.loc[Q2025_subset1.obs["cnv_leiden"].isin(["1"]), "cnv_status"] = (
    "DCIS"
)


# In[944]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(Q2025_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(Q2025_subset1, color="cnv_status", ax=ax2)


# In[117]:


cnv.pl.chromosome_heatmap(Q2025_subset1[Q2025_subset1.obs["cnv_status"] == "DCIS", :], figsize=(40,10)) 


# In[115]:


cnv.pl.chromosome_heatmap(Q2025_subset1[Q2025_subset1.obs["cnv_status"] == "normal", :], figsize=(30,40)) 


# In[947]:


Q2025_subset1.write_h5ad('260309_dcis_Q2025_subset_1.h5ad')


# In[113]:


Q2025_subset1 = sc.read_h5ad('260309_dcis_Q2025_subset_1.h5ad')


# In[948]:


#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[Q2025_subset1.obs_names, 'cnv_status'] = Q2025_subset1.obs['cnv_status']


# ### Subset Q2025 2 (ind13, ind14 = ALL SAME PATIENT)

# In[773]:


Q2025_subset2 = ["ind13_Q2025", "ind14_Q2025"]
Q2025_subset2 = alldata[alldata.obs['Sample'].isin(Q2025_subset2)].copy()


# In[918]:


#nuclear_genes = ~Q2025_subset2.var['chromosome'].isin(['MT', 'chrMT', 'M'])
#Q2025_subset2 = Q2025_subset2[:, nuclear_genes]
cnv.tl.infercnv(
    Q2025_subset2,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
    
)


# In[919]:


cnv.pl.chromosome_heatmap(Q2025_subset2, groupby="cell type", figsize=(30,50))


# In[920]:


cnv.tl.pca(Q2025_subset2)
cnv.pp.neighbors(Q2025_subset2)
cnv.tl.leiden(Q2025_subset2)
sc.tl.dendrogram(Q2025_subset2, groupby='cnv_leiden')


# In[921]:


sc.tl.dendrogram(Q2025_subset2, groupby='cnv_leiden')


# In[110]:


cnv.pl.chromosome_heatmap(Q2025_subset2, groupby="cnv_leiden", dendrogram=True, figsize=(30,40))


# In[923]:


cnv.tl.umap(Q2025_subset2)
cnv.tl.cnv_score(Q2025_subset2)


# In[924]:


cnv.pl.umap(Q2025_subset2, color="cnv_score", show=False)
cnv.pl.umap(Q2025_subset2, color="cnv_leiden", show=False)
cnv.pl.umap(Q2025_subset2, color="cell type", show=False)
sc.pl.umap(Q2025_subset2, color="cnv_score")
sc.pl.umap(Q2025_subset2, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[941]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = Q2025_subset2.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_subset_2_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[942]:


mean_cnv = Q2025_subset2.obs['cnv_score'].mean()
median_cnv = Q2025_subset2.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[950]:


Q2025_subset2.obs["cnv_status"] = "normal"
Q2025_subset2.obs.loc[Q2025_subset2.obs["cnv_leiden"].isin(["3"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(Q2025_subset2[Q2025_subset2.obs["cnv_status"] == "DCIS", :], figsize = (40, 10)])


# In[119]:


cnv.pl.chromosome_heatmap(Q2025_subset2[Q2025_subset2.obs["cnv_status"] == "DCIS", :], figsize = (40, 10))


# In[121]:


cnv.pl.chromosome_heatmap(Q2025_subset2[Q2025_subset2.obs["cnv_status"] == "normal", :],  figsize = (30, 40))


# In[952]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(Q2025_subset2, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(Q2025_subset2, color="cnv_status", ax=ax2)


# In[953]:


cnv.pl.umap(Q2025_subset2, color="cell type")


# In[954]:


Q2025_subset2.write_h5ad('260309_dcis_Q2025_subset_2.h5ad')


# In[109]:


Q2025_subset2 = sc.read_h5ad('260309_dcis_Q2025_subset_2.h5ad')


# In[955]:


#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[Q2025_subset2.obs_names, 'cnv_status'] = Q2025_subset2.obs['cnv_status']


# ### Subset Q2025 3 (ind9, ind10 = ALL SAME PATIENT)

# In[215]:


Q2025_subset3 = ["ind9_Q2025", "ind10_Q2025"]
#Q2025_subset3 = ["ind10_Q2025"]
Q2025_subset3 = alldata[alldata.obs['Sample'].isin(Q2025_subset3)].copy()


# In[221]:


nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    Q2025_subset3,
    reference_key="cell type",
    reference_cat=[
        "General Myeloid",
        "T-Cell",
        "Macrophage",
        "B-Cell"],
    window_size=100,
    dynamic_threshold=2
)


# In[222]:


cnv.pl.chromosome_heatmap(Q2025_subset3, groupby="cell type", figsize=(30,30))


# In[223]:


cnv.tl.pca(Q2025_subset3)
cnv.pp.neighbors(Q2025_subset3)
cnv.tl.leiden(Q2025_subset3)
cnv.tl.umap(Q2025_subset3)
cnv.tl.cnv_score(Q2025_subset3)
sc.tl.dendrogram(Q2025_subset3, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(Q2025_subset3, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[224]:


cnv.pl.umap(Q2025_subset3, color="cnv_score", show=False)
cnv.pl.umap(Q2025_subset3, color="cnv_leiden", show=False)
cnv.pl.umap(Q2025_subset3, color="cell type", show=False)
sc.pl.umap(Q2025_subset3, color="cnv_score")
sc.pl.umap(Q2025_subset3, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[225]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = Q2025_subset3.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_subset_3_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[128]:


mean_cnv = Q2025_subset3.obs['cnv_score'].mean()
median_cnv = Q2025_subset3.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[236]:


Q2025_subset3.obs["cnv_status"] = "normal"
Q2025_subset3.obs.loc[Q2025_subset3.obs["cnv_leiden"].isin(["2", "3", "10", "11", "15"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(Q2025_subset3[Q2025_subset3.obs["cnv_status"] == "DCIS", :], figsize=(40,10))


# In[237]:


cnv.pl.chromosome_heatmap(Q2025_subset3[Q2025_subset3.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[228]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(Q2025_subset3, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(Q2025_subset3, color="cnv_status", ax=ax2)


# In[988]:


sc.pl.umap(Q2025_subset3, color="cell type")


# In[989]:


Q2025_subset3.write_h5ad('260309_dcis_Q2025_subset_3.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[Q2025_subset3.obs_names, 'cnv_status'] = Q2025_subset3.obs['cnv_status']


# ### Subset Q2025 4 (ind11, ind12 = ALL SAME PATIENT)

# In[83]:


Q2025_subset4 = ["ind11_Q2025", "ind12_Q2025"]
Q2025_subset4 = alldata[alldata.obs['Sample'].isin(Q2025_subset4)].copy()


# In[908]:


nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    Q2025_subset4,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
    
)


# In[909]:


cnv.pl.chromosome_heatmap(Q2025_subset4, groupby="cell type", figsize=(30,50))


# In[910]:


cnv.tl.pca(Q2025_subset4)
cnv.pp.neighbors(Q2025_subset4)
cnv.tl.leiden(Q2025_subset4)
cnv.tl.umap(Q2025_subset4)
cnv.tl.cnv_score(Q2025_subset4)
sc.tl.dendrogram(Q2025_subset4, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(Q2025_subset4, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[911]:


cnv.pl.umap(Q2025_subset4, color="cnv_score", show=False)
cnv.pl.umap(Q2025_subset4, color="cnv_leiden", show=False)
cnv.pl.umap(Q2025_subset4, color="cell type", show=False)
sc.pl.umap(Q2025_subset4, color="cnv_score")
sc.pl.umap(Q2025_subset4, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[912]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = Q2025_subset4.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_subset_4_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[939]:


mean_cnv = Q2025_subset4.obs['cnv_score'].mean()
median_cnv = Q2025_subset4.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[962]:


Q2025_subset4.obs["cnv_status"] = "normal"
Q2025_subset4.obs.loc[Q2025_subset4.obs["cnv_leiden"].isin(["0"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(Q2025_subset4[Q2025_subset4.obs["cnv_status"] == "DCIS", :], figsize=(40,10))


# In[963]:


cnv.pl.chromosome_heatmap(Q2025_subset4[Q2025_subset4.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[964]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(Q2025_subset4, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(Q2025_subset4, color="cnv_status", ax=ax2)


# In[965]:


Q2025_subset4.write_h5ad('260309_dcis_Q2025_subset_4.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[Q2025_subset4.obs_names, 'cnv_status'] = Q2025_subset4.obs['cnv_status']


# In[994]:


sc.pl.umap(alldata, color=["cnv_status_old", "cnv_status", "cell type"])


# ### Individual remaining samples Q2025

# In[24]:


# Trial Subset 5: ind1, ind3, ind16
Q2025_subset5 = ["ind16_Q2025"]
Q2025_subset5 = alldata[alldata.obs['Sample'].isin(Q2025_subset5)].copy()
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    Q2025_subset5,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
)
cnv.pl.chromosome_heatmap(Q2025_subset5, groupby="cell type", figsize=(30,50))


# In[25]:


cnv.tl.pca(Q2025_subset5)
cnv.pp.neighbors(Q2025_subset5)
cnv.tl.leiden(Q2025_subset5)
cnv.tl.umap(Q2025_subset5)
cnv.tl.cnv_score(Q2025_subset5)
sc.tl.dendrogram(Q2025_subset5, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(Q2025_subset5, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[26]:


cnv.pl.umap(Q2025_subset5, color="cnv_score", show=False)
cnv.pl.umap(Q2025_subset5, color="cnv_leiden", show=False)
cnv.pl.umap(Q2025_subset5, color="cell type", show=False)
sc.pl.umap(Q2025_subset5, color="cnv_score")
sc.pl.umap(Q2025_subset5, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[27]:


cnv_leiden = Q2025_subset5.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_subset_8_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[28]:


mean_cnv = Q2025_subset5.obs['cnv_score'].mean()
median_cnv = Q2025_subset5.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[29]:


Q2025_subset5.obs["cnv_status"] = "normal"
Q2025_subset5.obs.loc[Q2025_subset5.obs["cnv_leiden"].isin(["0", "1", "4"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(Q2025_subset5[Q2025_subset5.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(Q2025_subset5[Q2025_subset5.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[30]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(Q2025_subset5, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(Q2025_subset5, color="cnv_status", ax=ax2)


# In[31]:


Q2025_subset5.write_h5ad('260309_dcis_Q2025_subset_8.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[Q2025_subset5.obs_names, 'cnv_status'] = Q2025_subset5.obs['cnv_status']


# In[56]:


# Trial Subset 6: ind2, ind4, ind15, ind17
Q2025_subset6 = ["ind17_Q2025"]
Q2025_subset6 = alldata[alldata.obs['Sample'].isin(Q2025_subset6)].copy()
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    Q2025_subset6,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
)
cnv.pl.chromosome_heatmap(Q2025_subset6, groupby="cell type", figsize=(30,50))


# In[57]:


cnv.tl.pca(Q2025_subset6)
cnv.pp.neighbors(Q2025_subset6)
cnv.tl.leiden(Q2025_subset6)
cnv.tl.umap(Q2025_subset6)
cnv.tl.cnv_score(Q2025_subset6)
sc.tl.dendrogram(Q2025_subset6, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(Q2025_subset6, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[58]:


cnv.pl.umap(Q2025_subset6, color="cnv_score", show=False)
cnv.pl.umap(Q2025_subset6, color="cnv_leiden", show=False)
cnv.pl.umap(Q2025_subset6, color="cell type", show=False)
sc.pl.umap(Q2025_subset6, color="cnv_score")
sc.pl.umap(Q2025_subset6, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[59]:


cnv_leiden = Q2025_subset6.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_subset_11_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[60]:


mean_cnv = Q2025_subset6.obs['cnv_score'].mean()
median_cnv = Q2025_subset6.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[61]:


Q2025_subset6.obs["cnv_status"] = "normal"
Q2025_subset6.obs.loc[Q2025_subset6.obs["cnv_leiden"].isin(["5"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(Q2025_subset6[Q2025_subset6.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(Q2025_subset6[Q2025_subset6.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[62]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(Q2025_subset6, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(Q2025_subset6, color="cnv_status", ax=ax2)


# In[63]:


Q2025_subset6.write_h5ad('260309_dcis_Q2025_subset_11.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[Q2025_subset6.obs_names, 'cnv_status'] = Q2025_subset6.obs['cnv_status']


# ## T2022

# In[234]:


#alldata = alldata_T2022
alldata = sc.read_h5ad('260203_dcis_T2022_only.h5ad')


# In[139]:


#rename old cnv_status (conducted on whole study)
#to cnv_status_old & drop cnv_status (so can rename
#with new samples subsets names
alldata.obs['cnv_status_old'] = alldata.obs['cnv_status']


# In[547]:


sc.pl.umap(alldata, color=["cnv_status_old", "cnv_status", "cell type"])


# In[128]:


alldata.write_h5ad('260203_dcis_T2022_only.h5ad')


# ### Subsets T2022 Samples

# In[253]:


T2022_subset1 = ["ind3_T2022"]
T2022_subset1 = alldata[alldata.obs['Sample'].isin(T2022_subset1)].copy()
nuclear_genes = ~T2022_subset1.var['chromosome'].isin(['MT', 'chrMT', 'M'])
T2022_subset1 = T2022_subset1[:, nuclear_genes]
cnv.tl.infercnv(
    T2022_subset1,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell"],
    window_size=100,
    dynamic_threshold=2,
    calculate_gene_values=True
)
cnv.pl.chromosome_heatmap(T2022_subset1, groupby="cell type", figsize=(30,50))


# In[254]:


cnv.tl.pca(T2022_subset1)
cnv.pp.neighbors(T2022_subset1)
cnv.tl.leiden(T2022_subset1)
cnv.tl.umap(T2022_subset1)
cnv.tl.cnv_score(T2022_subset1)
sc.tl.dendrogram(T2022_subset1, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(T2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[255]:


cnv.pl.umap(T2022_subset1, color="cnv_score", show=False)
cnv.pl.umap(T2022_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(T2022_subset1, color="cell type", show=False)
sc.pl.umap(T2022_subset1, color="cnv_score")
sc.pl.umap(T2022_subset1, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[257]:


cnv_leiden = T2022_subset1.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_T2022_ind_3_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[258]:


mean_cnv = T2022_subset1.obs['cnv_score'].mean()
median_cnv = T2022_subset1.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[259]:


T2022_subset1.obs["cnv_status"] = "normal"
T2022_subset1.obs.loc[T2022_subset1.obs["cnv_leiden"].isin(["12", "0"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(T2022_subset1[T2022_subset1.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(T2022_subset1[T2022_subset1.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[9]:


T2022_subset1 = sc.read_h5ad('260309_dcis_T2022_subset_3.h5ad')


# In[10]:


cnv.pl.chromosome_heatmap(T2022_subset1, groupby="cell_type_annotation", dendrogram=True, figsize=(20,5))


# In[268]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(T2022_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(T2022_subset1, color="cnv_status", ax=ax2)


# In[151]:


# Choose cell type/subpopulation of interest
celltype = 'LMDN_OXPHOS_metabolic'

# Subset cells
subset = W2022_subset1[
    W2022_subset1.obs['cell_type_annotation'] == celltype
].copy()

# Mean CNV per gene
mean_cnv = np.array(
    subset.layers['gene_values_cnv'].mean(axis=0)
).flatten()

# Build dataframe
cnv_df = pd.DataFrame({
    'gene': subset.var_names,
    'mean_cnv': mean_cnv
})

# Clean values
cnv_df['mean_cnv'] = pd.to_numeric(
    cnv_df['mean_cnv'],
    errors='coerce'
)

cnv_df = cnv_df.loc[
    np.isfinite(cnv_df['mean_cnv'])
].copy()

# Rank by absolute CNV magnitude
cnv_df['abs_cnv'] = cnv_df['mean_cnv'].abs()

cnv_df = cnv_df.sort_values(
    'abs_cnv',
    ascending=False
)

# Save
cnv_df.to_csv(
    f"{celltype}_W2022_mean_gene_cnv_ranked.csv",
    index=False
)

cnv_df.head(20)


# In[152]:


celltype = 'LMDD_Proteostasis-active'

# Subset cells
subset = W2022_subset1[
    W2022_subset1.obs['cell_type_annotation'] == celltype
].copy()

# Mean CNV per gene
mean_cnv = np.array(
    subset.layers['gene_values_cnv'].mean(axis=0)
).flatten()

# Build dataframe
cnv_df = pd.DataFrame({
    'gene': subset.var_names,
    'mean_cnv': mean_cnv
})

# Clean values
cnv_df['mean_cnv'] = pd.to_numeric(
    cnv_df['mean_cnv'],
    errors='coerce'
)

cnv_df = cnv_df.loc[
    np.isfinite(cnv_df['mean_cnv'])
].copy()

# Rank by absolute CNV magnitude
cnv_df['abs_cnv'] = cnv_df['mean_cnv'].abs()

cnv_df = cnv_df.sort_values(
    'abs_cnv',
    ascending=False
)

# Save
cnv_df.to_csv(
    f"{celltype}_W2022_mean_gene_cnv_ranked.csv",
    index=False
)

cnv_df.head(20)


# In[148]:


cnv_df_1 = pd.read_csv('LMDD_Proteostasis-active_mean_gene_cnv_ranked.csv')
cnv_df_2 = pd.read_csv('LMDN_OXPHOS_metabolic_mean_gene_cnv_ranked.csv')
comparison = (
    cnv_df_1[['gene', 'mean_cnv']]
    .rename(columns={'mean_cnv': 'mean_cnv_type1'})
    .merge(
        cnv_df_2[['gene', 'mean_cnv']]
        .rename(columns={'mean_cnv': 'mean_cnv_type2'}),
        on='gene',
        how='inner'
    )
)

# Difference between cell types
comparison['cnv_diff'] = (
    comparison['mean_cnv_type1']
    - comparison['mean_cnv_type2']
)

# Absolute difference for ranking
comparison['abs_cnv_diff'] = comparison['cnv_diff'].abs()

# Rank by largest difference
comparison = comparison.sort_values(
    'abs_cnv_diff',
    ascending=False
)

comparison.head(20)


# In[149]:


comparison.to_csv(
    'LMDN_vs_LMDD_T2022_in3_CNV_comparison.csv',
    index=False
)


# In[269]:


T2022_subset1.write_h5ad('260309_dcis_T2022_subset_3.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[T2022_subset1.obs_names, 'cnv_status'] = T2022_subset1.obs['cnv_status']


# In[261]:


# add annotation back for ind3 similar DN DD
#T2022_subset1 = sc.read_h5ad('260309_dcis_T2022_subset_3.h5ad')
T2022_subset1.obs['annotation'] = alldata.obs.loc[
    T2022_subset1.obs.index, 'annotation'
]


# In[262]:


cnv.pl.umap(T2022_subset1, color="annotation", show=False)


# In[263]:


T2022_subset1.obs['annotation'] = T2022_subset1.obs['annotation'].replace('unassigned', 'na')


# In[264]:


T2022_subset1.obs['cell_type_annotation'] = T2022_subset1.obs['cell type'].astype(str)

lm_annotations = [
    'LMDD_Proteostasis-active',
    'LMDN_OXPHOS_metabolic'
]

# Mask for Luminal Mature cells
lm_mask = T2022_subset1.obs['cell type'] == 'Luminal Mature'
T2022_subset1.obs.loc[
    lm_mask &
    (T2022_subset1.obs['annotation'] == 'LMDD_Proteostasis-active'),
    'cell_type_annotation'
] = 'LMDD_Proteostasis-active'

T2022_subset1.obs.loc[
    lm_mask &
    (T2022_subset1.obs['annotation'] == 'LMDN_OXPHOS_metabolic'),
    'cell_type_annotation'
] = 'LMDN_OXPHOS_metabolic'

T2022_subset1.obs.loc[
    lm_mask &
    ~T2022_subset1.obs['annotation'].isin(lm_annotations),
    'cell_type_annotation'
] = 'Other Luminal Mature'

# Check counts
T2022_subset1.obs['cell_type_annotation'].value_counts()


# In[265]:


cnv.pl.umap(T2022_subset1, color="cell_type_annotation", show=False)


# In[266]:


sc.pl.umap(T2022_subset1, color="cell_type_annotation", show=False)


# In[243]:


T2022_subset1.write_h5ad('260309_dcis_T2022_subset_3.h5ad')


# ## W2022

# In[153]:


#alldata = alldata_W2022
alldata = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[128]:


sc.pl.umap(alldata, color = 'annotation')


# In[154]:


W2022_subset1 = ["ind1_W2022"]
W2022_subset1 = alldata[alldata.obs['Sample'].isin(W2022_subset1)].copy()
nuclear_genes = ~W2022_subset1.var['chromosome'].isin(['MT', 'chrMT', 'M'])
W2022_subset1 = W2022_subset1[:, nuclear_genes]
cnv.tl.infercnv(
    W2022_subset1,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell", 
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2,
        calculate_gene_values=True
)
cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cell type", figsize=(30,30))


# In[130]:


cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cell type", figsize=(25,17))


# In[43]:


cnv.tl.pca(W2022_subset1)
cnv.pp.neighbors(W2022_subset1)
cnv.tl.leiden(W2022_subset1)
cnv.tl.umap(W2022_subset1)
cnv.tl.cnv_score(W2022_subset1)
#sc.tl.dendrogram(W2022_subset1, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[41]:


sc.tl.dendrogram(W2022_subset1, groupby='cnv_leiden')


# In[52]:


cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(20,18))


# In[53]:


cnv.pl.umap(W2022_subset1, color="cnv_score", show=False)
cnv.pl.umap(W2022_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(W2022_subset1, color="cell type", show=False)
sc.pl.umap(W2022_subset1, color="cnv_score")
sc.pl.umap(W2022_subset1, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[56]:


cnv.pl.umap(W2022_subset1, color=["cnv_score", 'cell type', 'cnv_leiden', 'cnv_status'], show=False, ncols=2)


# In[147]:


cnv_leiden = W2022_subset1.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_W2022_ind_2_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[148]:


mean_cnv = W2022_subset1.obs['cnv_score'].mean()
median_cnv = W2022_subset1.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[149]:


W2022_subset1.obs["cnv_status"] = "normal"
W2022_subset1.obs.loc[W2022_subset1.obs["cnv_leiden"].isin(["4"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(W2022_subset1[W2022_subset1.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(W2022_subset1[W2022_subset1.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[59]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(W2022_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(W2022_subset1, color="cnv_status", ax=ax2)


# In[57]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(W2022_subset1, color="cnv_status", ax=ax1, show=False)
cnv.pl.umap(W2022_subset1, color="cell type", ax=ax2, show=False)


# In[58]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(W2022_subset1, color="cnv_score", ax=ax1, show=False)
cnv.pl.umap(W2022_subset1, color="cnv_leiden", ax=ax2, show=False)


# In[151]:


W2022_subset1.write_h5ad('260309_dcis_W2022_subset_2.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[W2022_subset1.obs_names, 'cnv_status'] = W2022_subset1.obs['cnv_status']


# In[127]:


sc.pl.umap(W2022_subset1, color = 'annotation')


# In[171]:


W2022_subset1.obs['cell_type_annotation'] = W2022_subset1.obs['cell type'].astype(str)

lm_annotations = [
    'LMDD_Proteostasis-active',
    'LMDN_OXPHOS_metabolic'
]

# Mask for Luminal Mature cells
lm_mask = W2022_subset1.obs['cell type'] == 'Luminal Mature'
W2022_subset1.obs.loc[
    lm_mask &
    (W2022_subset1.obs['annotation'] == 'LMDD_Proteostasis-active'),
    'cell_type_annotation'
] = 'LMDD_Proteostasis-active'

W2022_subset1.obs.loc[
    lm_mask &
    (W2022_subset1.obs['annotation'] == 'LMDN_OXPHOS_metabolic'),
    'cell_type_annotation'
] = 'LMDN_OXPHOS_metabolic'

W2022_subset1.obs.loc[
    lm_mask &
    ~W2022_subset1.obs['annotation'].isin(lm_annotations),
    'cell_type_annotation'
] = 'Other Luminal Mature'


# In[172]:


cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cell_type_annotation", dendrogram = True, figsize=(20,5))


# In[173]:


celltypes = ["LMDN_OXPHOS_metabolic", "LMDD_Proteostasis-active"]

adata_sub = W2022_subset1[
    W2022_subset1.obs["cell_type_annotation"].isin(celltypes)
].copy()

cnv.pl.chromosome_heatmap(
    adata_sub,
    groupby="cell_type_annotation",
    dendrogram=False,
    figsize=(20, 3)
)


# In[174]:


# Choose cell type/subpopulation of interest
celltype = 'LMDN_OXPHOS_metabolic'

# Subset cells
subset = W2022_subset1[
    W2022_subset1.obs['cell_type_annotation'] == celltype
].copy()

# Mean CNV per gene
mean_cnv = np.array(
    subset.layers['gene_values_cnv'].mean(axis=0)
).flatten()

# Build dataframe
cnv_df = pd.DataFrame({
    'gene': subset.var_names,
    'mean_cnv': mean_cnv
})

# Clean values
cnv_df['mean_cnv'] = pd.to_numeric(
    cnv_df['mean_cnv'],
    errors='coerce'
)

cnv_df = cnv_df.loc[
    np.isfinite(cnv_df['mean_cnv'])
].copy()

# Rank by absolute CNV magnitude
cnv_df['abs_cnv'] = cnv_df['mean_cnv'].abs()

cnv_df = cnv_df.sort_values(
    'abs_cnv',
    ascending=False
)

# Save
cnv_df.to_csv(
    f"{celltype}_W2022_mean_gene_cnv_ranked.csv",
    index=False
)

cnv_df.head(20)


# In[175]:


# Choose cell type/subpopulation of interest
celltype = 'LMDD_Proteostasis-active'

# Subset cells
subset = W2022_subset1[
    W2022_subset1.obs['cell_type_annotation'] == celltype
].copy()

# Mean CNV per gene
mean_cnv = np.array(
    subset.layers['gene_values_cnv'].mean(axis=0)
).flatten()

# Build dataframe
cnv_df = pd.DataFrame({
    'gene': subset.var_names,
    'mean_cnv': mean_cnv
})

# Clean values
cnv_df['mean_cnv'] = pd.to_numeric(
    cnv_df['mean_cnv'],
    errors='coerce'
)

cnv_df = cnv_df.loc[
    np.isfinite(cnv_df['mean_cnv'])
].copy()

# Rank by absolute CNV magnitude
cnv_df['abs_cnv'] = cnv_df['mean_cnv'].abs()

cnv_df = cnv_df.sort_values(
    'abs_cnv',
    ascending=False
)

# Save
cnv_df.to_csv(
    f"{celltype}_W2022_mean_gene_cnv_ranked.csv",
    index=False
)

cnv_df.head(20)


# In[176]:


cnv_df_1 = pd.read_csv('LMDD_Proteostasis-active_W2022_mean_gene_cnv_ranked.csv')
cnv_df_2 = pd.read_csv('LMDN_OXPHOS_metabolic_W2022_mean_gene_cnv_ranked.csv')
comparison = (
    cnv_df_1[['gene', 'mean_cnv']]
    .rename(columns={'mean_cnv': 'mean_cnv_LMDD'})
    .merge(
        cnv_df_2[['gene', 'mean_cnv']]
        .rename(columns={'mean_cnv': 'mean_cnv_LMDN'}),
        on='gene',
        how='inner'
    )
)

# Difference between cell types
comparison['cnv_diff'] = (
    comparison['mean_cnv_LMDD']
    - comparison['mean_cnv_LMDN']
)

# Absolute difference for ranking
comparison['abs_cnv_diff'] = comparison['cnv_diff'].abs()

# Rank by largest difference
comparison = comparison.sort_values(
    'abs_cnv_diff',
    ascending=False
)


# In[177]:


comparison.head(20)


# In[178]:


comparison.to_csv(
    'LMDN_vs_LMDD_W2022_ind1_CNV_comparison.csv',
    index=False
)


# In[180]:


#save csvs to one excel workbook
folder = "./data/AAA_DCIS/LMDNvsLMDDcsvs"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("LMDD_vs_LMDN_T2022_W2022.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[46]:


W2022_subset1 = sc.read_h5ad('260309_dcis_W2022_subset_2.h5ad')


# In[168]:


cnv.io.genomic_position_from_biomart(
    alldata,
    species="hsapiens",
    biomart_gene_id="hgnc_symbol"
)


# In[169]:


alldata.var[['chromosome', 'start', 'end']].head()


# In[436]:


sc.pl.umap(alldata, color="cell type")


# In[154]:


sc.tl.dendrogram(W2022_subset1, groupby="cnv_leiden")


# In[155]:


cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(18,12))


# In[152]:


alldata.write_h5ad('260203_dcis_W2022_only.h5ad')


# ## G2017

# In[1058]:


#alldata = alldata_G2017
alldata = sc.read_h5ad('260203_dcis_G2017_only.h5ad')


# In[1059]:


#rename old cnv_status (conducted on whole study)
#to cnv_status_old & drop cnv_status (so can rename
#with new samples subsets names
alldata.obs['cnv_status_old'] = alldata.obs['cnv_status']


# In[1051]:


cnv.io.genomic_position_from_biomart(
    alldata,
    species="hsapiens",
    biomart_gene_id="hgnc_symbol"
)


# In[1052]:


alldata.var[['chromosome', 'start', 'end']].head()


# In[1056]:


sc.pl.umap(alldata, color="cnv_status")


# In[1060]:


#biomart adds MT genes back in, only want nuclear genes for analysis
#nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
#alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
)


# In[1061]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type")


# In[1062]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden", figsize=(24,24))


# In[1063]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[1065]:


sc.tl.dendrogram(alldata, groupby="cnv_leiden")


# In[1066]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(24,24))


# In[1067]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[1068]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[1069]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[1070]:


alldata.obs["cnv_status"] = "normal"
alldata.obs.loc[alldata.obs["cnv_leiden"].isin(["0", "1", "4"]), "cnv_status"] = (
    "DCIS"
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_status", ax=ax2)


# In[1071]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "DCIS", :])


# In[1072]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "normal", :])


# In[1075]:


alldata.write_h5ad('260203_dcis_G2017_only.h5ad')


# In[1073]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_G2017_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[1074]:


mean_cnv = alldata.obs['cnv_score'].mean()
median_cnv = alldata.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# ## N2025

# In[57]:


#alldata = alldata_N2025
alldata = sc.read_h5ad('260205_dcis_N2025_only.h5ad')


# In[129]:


alldata.write_h5ad('260205_dcis_N2025_only.h5ad')


# ### N2025 Subsets

# In[121]:


N2025_subset1 = ["ind6_N2025"]
N2025_subset1 = alldata[alldata.obs['Sample'].isin(N2025_subset1)].copy()
nuclear_genes = ~N2025_subset1.var['chromosome'].isin(['MT', 'chrMT', 'M'])
N2025_subset1 = N2025_subset1[:, nuclear_genes]
cnv.tl.infercnv(
    N2025_subset1,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell"],
    window_size=100,
    dynamic_threshold=2
)
cnv.pl.chromosome_heatmap(N2025_subset1, groupby="cell type", figsize=(30,20))


# In[122]:


cnv.tl.pca(N2025_subset1)
cnv.pp.neighbors(N2025_subset1)
cnv.tl.leiden(N2025_subset1)
cnv.tl.umap(N2025_subset1)
cnv.tl.cnv_score(N2025_subset1)
sc.tl.dendrogram(N2025_subset1, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(N2025_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[123]:


cnv.pl.umap(N2025_subset1, color="cnv_score", show=False)
cnv.pl.umap(N2025_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(N2025_subset1, color="cell type", show=False)
sc.pl.umap(N2025_subset1, color="cnv_score")
sc.pl.umap(N2025_subset1, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[124]:


cnv_leiden = N2025_subset1.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_N2025_subset_ind6.csv', index=False)
cnv_leiden


# In[125]:


mean_cnv = N2025_subset1.obs['cnv_score'].mean()
median_cnv = N2025_subset1.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[126]:


N2025_subset1.obs["cnv_status"] = "normal"
N2025_subset1.obs.loc[N2025_subset1.obs["cnv_leiden"].isin(["0", "10", "3"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(N2025_subset1[N2025_subset1.obs["cnv_status"] == "DCIS", :], figsize=(40,5))
cnv.pl.chromosome_heatmap(N2025_subset1[N2025_subset1.obs["cnv_status"] == "normal", :], figsize=(30,30))


# In[127]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(N2025_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(N2025_subset1, color="cnv_status", ax=ax2)


# In[128]:


N2025_subset1.write_h5ad('260309_dcis_N2025_subset_6.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[N2025_subset1.obs_names, 'cnv_status'] = N2025_subset1.obs['cnv_status']


# # scMalignantFinder

# In[ ]:


#run in own env (scMalig) due to dependencies


# In[163]:


alldata = sc.read_h5ad("./260130_dcis_alldata_annotated.h5ad")


# In[164]:


sc.pl.umap(
    alldata,
    color=["scMalignantFinder_prediction", "cell type"],
    frameon=False,
    ncols = 1
)


# In[165]:


sc.pl.umap(
    alldata,
    color=["malignancy_probability", "cell type"],
    frameon=False,
    ncols = 1
)


# # SequencingCancerFinder

# In[ ]:


#run in terminal with sbash but transpose first


# In[21]:


#need to transpose so cells x genes, not genes x cells
adata = sc.read_h5ad("260130_dcis_alldata_annotated.h5ad")
print(adata.shape)  # should be (#cells, #genes)
print(adata.obs_names[:5])  # cell IDs
print(adata.var_names[:5])  # gene names


# In[22]:


adata = adata.T


# In[ ]:


#adata.var_names_make_unique


# In[25]:


# save new file transposed
adata.write_h5ad("./data/260202_dcis_transposed_alldata.h5ad")


# In[26]:


adata = sc.read_h5ad("./260130_dcis_alldata_annotated.h5ad")
#use not transposed version for analysis of results


# In[27]:


pred_df = pd.read_csv("~/work/SequencingCancerFinder/data/out_260202.csv")

# Make sure the cell barcodes are the index
pred_df = pred_df.set_index('sample')

# Check alignment with adata
common_cells = adata.obs_names.intersection(pred_df.index)
print(f"Cells in both adata and predictions: {len(common_cells)} / {adata.n_obs}")


# In[28]:


adata.obs['scf_prediction'] = pred_df.reindex(adata.obs_names)['predict']

print(adata.obs['scf_prediction'].notna().sum(), "cells have predictions")


# In[33]:


sc.pl.umap(adata, color=['scf_prediction', 'cell type'], ncols = 1, frameon = False)


# In[34]:


adata.write_h5ad("260130_dcis_alldata_annotated.h5ad")


# In[254]:


# transpose individual studies

#Q2025
alldata_Q2025 = sc.read_h5ad("260203_dcis_Q2025_only.h5ad")
alldata_Q2025 = alldata_Q2025.T
alldata_Q2025.write_h5ad("./data/260203_dcis_Q2025_only_transposed.h5ad")

#T2022
alldata_T2022 = sc.read_h5ad("260203_dcis_T2022_only.h5ad")
alldata_T2022 = alldata_T2022.T
alldata_T2022.write_h5ad("./data/260203_dcis_T2022_only_transposed.h5ad")

#W2022
alldata_W2022 = sc.read_h5ad("260203_dcis_W2022_only.h5ad")
alldata_W2022 = alldata_W2022.T
alldata_W2022.write_h5ad("./data/260203_dcis_W2022_only_transposed.h5ad")

#G2017
alldata_G2017 = sc.read_h5ad("260203_dcis_G2017_only.h5ad")
alldata_G2017 = alldata_G2017.T
alldata_G2017.write_h5ad("./data/260203_dcis_G2017_only_transposed.h5ad")


# In[647]:


alldata_N2025 = sc.read_h5ad("260205_dcis_N2025_only.h5ad")
alldata_N2025T = alldata_N2025.T
alldata_N2025T.write_h5ad("./data/260205_dcis_N2025_only_transposed.h5ad")


# ## Q2025

# In[188]:


alldata_Q2025 = sc.read_h5ad("260203_dcis_Q2025_only.h5ad")


# In[257]:


pred_df = pd.read_csv("~/work/SequencingCancerFinder/data/out_260204_Q2025.csv")

# Make sure the cell barcodes are the index
pred_df = pred_df.set_index('sample')

# Check alignment with adata
common_cells = alldata_Q2025.obs_names.intersection(pred_df.index)
print(f"Cells in both adata and predictions: {len(common_cells)} / {alldata_Q2025.n_obs}")


# In[258]:


alldata_Q2025.obs['scf_prediction'] = pred_df.reindex(alldata_Q2025.obs_names)['predict']

print(alldata_Q2025.obs['scf_prediction'].notna().sum(), "cells have predictions")


# In[198]:


sc.pl.umap(alldata_Q2025, color=['scf_prediction', 'malignancy_probability', 'cnv_status', 'cnv_score', 'cell type'], ncols = 2, frameon = True)


# In[203]:


sc.pl.umap(alldata_N2025, color=['scf_prediction', 'malignancy_probability', 'cnv_status', 'cnv_score', 'cell type'], ncols = 3, frameon = True)


# In[200]:


sc.pl.umap(alldata_T2022, color=['scf_prediction', 'malignancy_probability', 'cnv_status', 'cnv_score', 'cell type'], ncols = 2, frameon = True)


# In[201]:


sc.pl.umap(alldata_W2022, color=['scf_prediction', 'malignancy_probability', 'cnv_status', 'cnv_score', 'cell type'], ncols = 2, frameon = True)


# In[202]:


sc.pl.umap(alldata_G2017, color=['scf_prediction', 'malignancy_probability', 'cnv_status', 'cnv_score', 'cell type'], ncols = 2, frameon = True)


# In[260]:


alldata_Q2025.write_h5ad("260203_dcis_Q2025_only.h5ad")


# ## T2022

# In[261]:


alldata_T2022 = sc.read_h5ad("260203_dcis_T2022_only.h5ad")


# In[262]:


pred_df = pd.read_csv("~/work/SequencingCancerFinder/data/out_260204_T2022.csv")

# Make sure the cell barcodes are the index
pred_df = pred_df.set_index('sample')

# Check alignment with adata
common_cells = alldata_T2022.obs_names.intersection(pred_df.index)
print(f"Cells in both adata and predictions: {len(common_cells)} / {alldata_T2022.n_obs}")


# In[263]:


alldata_T2022.obs['scf_prediction'] = pred_df.reindex(alldata_T2022.obs_names)['predict']
print(alldata_T2022.obs['scf_prediction'].notna().sum(), "cells have predictions")
sc.pl.umap(alldata_T2022, color=['scf_prediction', 'cell type'], ncols = 1, frameon = False)


# In[264]:


alldata_T2022.write_h5ad("260203_dcis_T2022_only.h5ad")


# ## W2022

# In[265]:


alldata_W2022 = sc.read_h5ad("260203_dcis_W2022_only.h5ad")


# In[266]:


pred_df = pd.read_csv("~/work/SequencingCancerFinder/data/out_260204_W2022.csv")

# Make sure the cell barcodes are the index
pred_df = pred_df.set_index('sample')

# Check alignment with adata
common_cells = alldata_W2022.obs_names.intersection(pred_df.index)
print(f"Cells in both adata and predictions: {len(common_cells)} / {alldata_W2022.n_obs}")


# In[267]:


alldata_W2022.obs['scf_prediction'] = pred_df.reindex(alldata_W2022.obs_names)['predict']
print(alldata_W2022.obs['scf_prediction'].notna().sum(), "cells have predictions")
sc.pl.umap(alldata_W2022, color=['scf_prediction', 'cell type'], ncols = 1, frameon = False)


# In[268]:


alldata_W2022.write_h5ad("260203_dcis_W2022_only.h5ad")


# ## G2017

# In[270]:


alldata_G2017 = sc.read_h5ad("260203_dcis_G2017_only.h5ad")


# In[271]:


pred_df = pd.read_csv("~/work/SequencingCancerFinder/data/out_260204_G2017.csv")

# Make sure the cell barcodes are the index
pred_df = pred_df.set_index('sample')

# Check alignment with adata
common_cells = alldata_G2017.obs_names.intersection(pred_df.index)
print(f"Cells in both adata and predictions: {len(common_cells)} / {alldata_G2017.n_obs}")


# In[272]:


alldata_G2017.obs['scf_prediction'] = pred_df.reindex(alldata_G2017.obs_names)['predict']
print(alldata_G2017.obs['scf_prediction'].notna().sum(), "cells have predictions")
sc.pl.umap(alldata_G2017, color=['scf_prediction', 'cell type'], ncols = 1, frameon = False)


# In[273]:


alldata_G2017.write_h5ad("260203_dcis_G2017_only.h5ad")


# ## N2025

# In[648]:


alldata_N2025 = sc.read_h5ad("260205_dcis_N2025_only.h5ad")


# In[649]:


pred_df = pd.read_csv("~/work/SequencingCancerFinder/data/out_260206_N2025.csv")

# Make sure the cell barcodes are the index
pred_df = pred_df.set_index('sample')

# Check alignment with adata
common_cells = alldata_N2025.obs_names.intersection(pred_df.index)
print(f"Cells in both adata and predictions: {len(common_cells)} / {alldata_N2025.n_obs}")


# In[650]:


alldata_N2025.obs['scf_prediction'] = pred_df.reindex(alldata_N2025.obs_names)['predict']
print(alldata_N2025.obs['scf_prediction'].notna().sum(), "cells have predictions")
sc.pl.umap(alldata_N2025, color=['scf_prediction', 'cell type'], ncols = 1, frameon = False)


# In[651]:


alldata_N2025.write_h5ad("260205_dcis_N2025_only.h5ad")


# # Compare Epithelial Subclusters

# In[181]:


#save csvs to one excel workbook
folder = "./BDD/up100/"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("BDD_up100_markers.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[182]:


folder = "./BDN/up100/"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("BDN_up100_markers.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[183]:


folder = "./LMDD/up100/"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("LMDD_up100_markers.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[184]:


folder = "./LMDN/up100/"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("LMDN_up100_markers.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[185]:


folder = "./LPDD/up100/"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("LPDD_up100_markers.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[186]:


folder = "./LPDN/up100/"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

with pd.ExcelWriter("LPDN_up100_markers.xlsx", engine="openpyxl") as writer:
    for file in csv_files:
        sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[187]:


output_file = "Supplementary_File_1_Merged_Up100_Markers.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    
    # Process BDD files
    bdd_files = glob.glob("./BDD/up100/*.csv")
    for file in bdd_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        sheet_name = f"BDD_{base_name}"[:31]  # Excel sheet names max 31 chars
        
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Process BDN files
    bdn_files = glob.glob("./BDN/up100/*.csv")
    for file in bdn_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        sheet_name = f"BDN_{base_name}"[:31]  # Excel sheet names max 31 chars
        
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    lmdn_files = glob.glob("./LMDN/up100/*.csv")
    for file in lmdn_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        sheet_name = f"LMDN_{base_name}"[:31]  # Excel sheet names max 31 chars
        
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    lmdd_files = glob.glob("./LMDD/up100/*.csv")
    for file in lmdd_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        sheet_name = f"LMDD_{base_name}"[:31]  # Excel sheet names max 31 chars
        
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    lpdn_files = glob.glob("./LPDN/up100/*.csv")
    for file in lpdn_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        sheet_name = f"LPDN_{base_name}"[:31]  # Excel sheet names max 31 chars
        
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    lpdd_files = glob.glob("./LPDD/up100/*.csv")
    for file in lpdd_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        sheet_name = f"LPDD_{base_name}"[:31]  # Excel sheet names max 31 chars
        
        df = pd.read_csv(file)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# ## Rank Dendrograms Fresh DD

# In[79]:


def build_rank_dendrogram(folder_path, top_n=None, title=None, method='average', min_common_genes=5):
    """
    Build and plot a dendrogram from ranked gene signatures in CSV files.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing CSVs with columns ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj'].
    top_n : int or None
        Use only the top N genes per file (after filtering for upregulated). None uses all.
    title : str or None
        Title for the dendrogram plot.
    method : str
        Linkage method for hierarchical clustering.
    min_common_genes : int
        Minimum number of genes required to compute correlation between two clusters.
    """
    
    # Load CSV
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    # Build rank signatures
    def build_rank_signature(df):
        df = df.copy()
        df = df[df['logfoldchanges'] > 0]  # keep only upregulated genes
        df = df.sort_values('logfoldchanges', ascending=False)
        if top_n is not None:
            df = df.head(top_n)
        df['rank'] = np.arange(1, len(df) + 1)
        return df.set_index('names')['rank']

    rank_signatures = {}
    for file_path in csv_files:
        cluster_name = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        rank_signatures[cluster_name] = build_rank_signature(df)

    # Compute Spearman correlation matrix 
    cluster_names = list(rank_signatures.keys())
    S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

    for i, ki in enumerate(cluster_names):
        for j, kj in enumerate(cluster_names):
            if j < i:
                continue
            common_genes = rank_signatures[ki].index.intersection(rank_signatures[kj].index)
            if len(common_genes) < min_common_genes:
                corr_val = np.nan
            else:
                corr_val, _ = spearmanr(rank_signatures[ki].loc[common_genes],
                                        rank_signatures[kj].loc[common_genes])
            S.loc[ki, kj] = corr_val
            S.loc[kj, ki] = corr_val

    # Build dendrogram 
    dist_matrix = 1 - S.astype(float)
    dist_matrix = dist_matrix.fillna(1)
    np.fill_diagonal(dist_matrix.values, 0)

    dist_vector = squareform(dist_matrix.values)
    linkage_matrix = linkage(dist_vector, method=method)

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=cluster_names, leaf_rotation=90)
    plt.title(title or f"Cluster similarity dendrogram (top {top_n} genes)")
    plt.ylabel("1 - Spearman correlation")
    plt.tight_layout()
    plt.show()


# In[196]:


build_rank_dendrogram('./LMDD/up100', top_n=100, title="Fresh DCIS (DD) LM subcluster similarity (top 100)")


# In[208]:


build_rank_dendrogram('./LPDD/up100', top_n=100, title="Fresh DCIS (DD) LP subcluster similarity (top 100)")


# In[211]:


build_rank_dendrogram('./LPDN/up100', top_n=100, title="Fresh DCIS (DN) LP subcluster similarity (top 100)")


# In[212]:


build_rank_dendrogram('./LMDN/up100', top_n=100, title="Fresh DCIS (DN) LM subcluster similarity (top 100)")


# In[210]:


build_rank_dendrogram('./BDN/up100', top_n=100, title="Fresh DCIS (DN) Basal subcluster similarity (top 100)")


# In[83]:


build_rank_dendrogram('./BDD/up100', top_n=100, title="Fresh DCIS (DD) Basal subcluster similarity (top 100)")


# ## DD Basal (Fresh)

# In[396]:


input_folder = './BDD/up100/'
input_folder_up1000 = './BDD/up1000/'
output_folder = './BDD/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_T2022_BDD_1.csv", "up100_Q2025_BDD_0.csv"],
    "Group2": ["up100_T2022_BDD_2.csv", "up100_Q2025_BDD_2.csv"]}

# Colour Palette
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# Assisting Functions
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

def zscore_per_column(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

# Main Loop
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[9]:


def run_go_enrichment_intersect(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from gseapy import enrichr, barplot, dotplot

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_lists = []

        # Process each sample in the group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")

                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = set(df[df['logfoldchanges'] > 0]['names'])
                    if filtered_genes:
                        gene_lists.append(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")

        # Take the intersection across all samples in the group
        if gene_lists:
            intersect_genes = set.intersection(*gene_lists)
            print(f"Intersecting Genes in {group_name}: {intersect_genes}")
        else:
            intersect_genes = set()

        if intersect_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=list(intersect_genes),
                gene_sets=gene_sets,
                organism=organism
            )

            # Save enrichment results
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            # Helper function to style plots
            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

            # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")

        else:
            print(f"No intersecting genes for {group_name}. Skipping GO enrichment.")


# In[408]:


# Example of your sample groups
sample_groups = {
    "Group1": ["up100_T2022_BDD_1.csv", "up100_Q2025_BDD_0.csv"],
    "Group2": ["up100_T2022_BDD_2.csv", "up100_Q2025_BDD_2.csv"]}

# Example directories (make sure to adjust these to your file locations)
data_dir = './BDD/up100/'
output_dir = './BDD/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[311]:


#GSEApy
import gseapy as gp
import matplotlib.pyplot as plt


# In[316]:


yeast = gp.get_library_name(organism='Human')
yeast[:100]


# In[435]:


BDD_1_gene_list = ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB']
BDD_2_gene_list = ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN']


# In[437]:


enr = gp.enrichr(gene_list=BDD_2_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human',
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/BDD/BDD_2.csv', index=False)


# In[330]:


from gseapy import barplot, dotplot
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Basal DCIS Cluster 2",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )


# In[328]:


ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Basal DCIS Cluster 1",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# ## DD Basal (FFPE Additional)

# In[218]:


input_folder = './BDD/up100/'
input_folder_up1000 = './BDD/up1000/'
output_folder = './BDD/FFFPE_Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_N2025_BDD_0.csv", "up100_Q2025_BDD_1.csv"],
    "Group2": ["up100_T2022_BDD_2.csv", "up100_Q2025_BDD_2.csv", "up100_N2025_BDD_1.csv"]}

# Colour Palette
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# Assisting Functions
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

def zscore_per_column(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

# Main Loop
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[219]:


def run_go_enrichment_intersect(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from gseapy import enrichr, barplot, dotplot

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_lists = []

        # Process each sample in the group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")

                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = set(df[df['logfoldchanges'] > 0]['names'])
                    if filtered_genes:
                        gene_lists.append(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")

        # Take the intersection across all samples in the group
        if gene_lists:
            intersect_genes = set.intersection(*gene_lists)
            print(f"Intersecting Genes in {group_name}: {intersect_genes}")
        else:
            intersect_genes = set()

        if intersect_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=list(intersect_genes),
                gene_sets=gene_sets,
                organism=organism
            )

            # Save enrichment results
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            # Helper function to style plots
            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

            # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")

        else:
            print(f"No intersecting genes for {group_name}. Skipping GO enrichment.")


# In[220]:


sample_groups = {
    "Group1": ["up100_N2025_BDD_0.csv", "up100_Q2025_BDD_1.csv"],
    "Group2": ["up100_T2022_BDD_2.csv", "up100_Q2025_BDD_2.csv", "up100_N2025_BDD_1.csv"]}

# Example directories (make sure to adjust these to your file locations)
data_dir = './BDD/up100/'
output_dir = './BDD/FFPE_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[331]:


BDD_1_gene_list = ['ACTG1', 'MIF', 'KRT8', 'MT2A', 'KRT7']
BDD_2_gene_list = ['SPARC']

enr = gp.enrichr(gene_list=BDD_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Basal DCIS (w/FFPE) Cluster 1",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Basal DCIS (w/FFPE) Cluster 1",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


enr = gp.enrichr(gene_list=BDD_2_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Basal DCIS (w/FFPE) Cluster 2",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Basal DCIS (w/FFPE) Cluster 2",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# In[224]:


input_folder = './BDN/up100/'
input_folder_up1000 = './BDN/up1000/'
output_folder = './BDN/FFFPE_Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_N2025_BDN_1.csv", "up100_Q2025_BDN_4.csv"],
    "Group2": ["up100_G2017_BDN_1.csv", "up100_Q2025_BDN_3.csv", "up100_N2025_BDN_0.csv"]}

for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()
output_dir = './BDN/FFPE_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[226]:


groups = {
    "Group2": ["up100_G2017_BDN_1.csv", "up100_Q2025_BDN_3.csv", "up100_N2025_BDN_0.csv"]}
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[339]:


#BDN_1_gene_list = ['MT-CYB', 'MT-ATP6', 'MT-ND4', 'MT-ND2', 'MT-ND1', 'MT-ND3']
BDN_2_gene_list = ['NFIB', 'KRT7']

enr = gp.enrichr(gene_list=BDN_2_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Basal (w/ FFPE) Non-Malignant Cluster 2",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Basal (w/ FFPE) Non-Malignant Cluster 2",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# In[431]:


BDN_1_gene_list = ['MT-CYB', 'MT-ATP6', 'MT-ND4', 'MT-ND2', 'MT-ND1', 'MT-ND3']
BDN_2_gene_list = ['NFIB', 'KRT7']
enr = gp.enrichr(gene_list=BDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/FFPE_BDN/BDN_1.csv', index=False)

enr = gp.enrichr(gene_list=BDN_2_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/FFPE_BDN/BDN_2.csv', index=False)


# In[228]:


input_folder = './LMDD/up100/'
input_folder_up1000 = './LMDD/up1000/'
output_folder = './LDMM/FFFPE_Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_N2025_LMDD_6.csv", "up100_G2017_LMDD_0.csv"],
    "Group2": ["up100_T2022_LMDD_2.csv", "up100_N2025_LMDD_0.csv", "up100_Q2025_LMDD_7.csv"]}

for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()
output_dir = './LMDD/FFPE_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[340]:


LMDD_1_gene_list = ['MARCKSL1', 'CLDN4', 'JUN', 'RSF1', 'SOX4', 'TOB1']
#LMDD_2_gene_list = ['X'] No intersecting genes

enr = gp.enrichr(gene_list=LMDD_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Mature (w/ FFPE) DCIS Cluster 1",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Mature (w/ FFPE) DCIS Cluster 1",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# In[432]:


LMDD_1_gene_list = ['MARCKSL1', 'CLDN4', 'JUN', 'RSF1', 'SOX4', 'TOB1']
enr = gp.enrichr(gene_list=LMDD_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/FFPE_LMDD/LMDD_1.csv', index=False)


# In[228]:


input_folder = './LMDN/up100/'
input_folder_up1000 = './LMDN/up1000/'
output_folder = './LMDN/FFFPE_Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_N2025_LMDN_1.csv", "up100_W2022_LMDN_3.csv"],
    "Group2": ["up100_W2022_LMDN_1.csv", "up100_N2025_LMDN_3.csv", "up100_N2025_LMDN_4.csv", "up100_T_LMDN_7.csv"]}

for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()
output_dir = './LMDN/FFPE_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[341]:


LMDN_1_gene_list = ['LCN2', 'SLPI', 'S100A8', 'S100A9', 'IGKC', 'CXCR4', 'PDZK1IP1', 'CD74', 'APOE', 'GSTP1']
#LMDN_2_gene_list = ['X'] No intersecting genes

enr = gp.enrichr(gene_list=LMDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Mature (w/ FFPE) Non-Malignant Cluster 1",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Mature (w/ FFPE) Non-Malignant Cluster 1",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# In[433]:


LMDN_1_gene_list = ['LCN2', 'SLPI', 'S100A8', 'S100A9', 'IGKC', 'CXCR4', 'PDZK1IP1', 'CD74', 'APOE', 'GSTP1']
enr = gp.enrichr(gene_list=LMDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/FFPE_LMDN/LMDN_1.csv', index=False)


# In[342]:


LPDN_1_gene_list = ['ELF5', 'SRSF5', 'ZRANB2', 'KCTD3', 'NFIB', 'RSRP1', 'ARGLU1', 'KMT2C', 'RBM39', 'CCDC14', 'PNISR', 'LUC7L3', 'TC2N', 'SRRM2', 'RBM25', 'TRPS1', 'ATL2', 'NKTR', 'SLC25A37', 'PNN', 'ITPR2', 'AKAP9']
#LPDN_2_gene_list = ['X'] No intersecting genes

enr = gp.enrichr(gene_list=LPDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Progenitor (w/ FFPE) Non-Malignant Cluster 1",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Progenitor (w/ FFPE) Non-Malignant Cluster 1",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# In[434]:


LPDN_1_gene_list = ['ELF5', 'SRSF5', 'ZRANB2', 'KCTD3', 'NFIB', 'RSRP1', 'ARGLU1', 'KMT2C', 'RBM39', 'CCDC14', 'PNISR', 'LUC7L3', 'TC2N', 'SRRM2', 'RBM25', 'TRPS1', 'ATL2', 'NKTR', 'SLC25A37', 'PNN', 'ITPR2', 'AKAP9']
enr = gp.enrichr(gene_list=LPDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/FFPE_LPDN/LPDN_1.csv', index=False)


# ## DD LM (Fresh)

# In[409]:


input_folder = './LMDD/up100/'
input_folder_up1000 = './LMDD/up1000/'
output_folder = './LMDD/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_Q2025_LMDD_9.csv", "up100_T2022_LMDD_4.csv"],
    "Group2": ["up100_Q2025_LMDD_9.csv", "up100_T2022_LMDD_4.csv", "up100_Q2025_LMDD_5.csv"],
    "Group3": ["up100_T2022_LMDD_6.csv", "up100_Q2025_LMDD_3.csv"],
    "Group4": ["up100_G2017_LMDD_0.csv", "up100_Q2025_LMDD_1.csv"],
    "Group5": ["up100_T2022_LMDD_2.csv", "up100_Q2025_LMDD_7.csv"],
    "Group6": ["up100_Q2025_LMDD_6.csv", "up100_W2022_LMDD_1.csv", "up100_T2022_LMDD_3.csv"],
    "Group7": ["up100_W2022_LMDD_1.csv", "up100_T2022_LMDD_3.csv"],
    "Group8": ["up100_T2022_LMDD_3.csv", "up100_Q2025_LMDD_6.csv", "up100_W2022_LMDD_1.csv"],
    "Group9": ["up100_T2022_LMDD_3.csv", "up100_W2022_LMDD_1.csv"],
    "Group10": ["up100_T2022_LMDD_5.csv", "up100_W2022_LMDD_0.csv"],
    "Group11": ["up100_T2022_LMDD_5.csv", "up100_W2022_LMDD_0.csv", "up100_G2017_LMDD_5.csv"]
    }
# Colour Palette
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# Assisting Functions
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

def zscore_per_column(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

# Main Loop
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]
    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[414]:


def run_go_enrichment_intersect(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from gseapy import enrichr, barplot, dotplot

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_lists = []

        # Process each sample in the group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")

                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = set(df[df['logfoldchanges'] > 0]['names'])
                    if filtered_genes:
                        gene_lists.append(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")

        # Take the intersection across all samples in the group
        if gene_lists:
            intersect_genes = set.intersection(*gene_lists)
            print(f"Intersecting Genes in {group_name}: {intersect_genes}")
        else:
            intersect_genes = set()

        if intersect_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=list(intersect_genes),
                gene_sets=gene_sets,
                organism=organism
            )

            # Save enrichment results
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            # Helper function to style plots
            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

            # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(7, 5))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")

        else:
            print(f"No intersecting genes for {group_name}. Skipping GO enrichment.")
            
# Example of your sample groups
sample_groups = {
    "Group1": ["up100_Q2025_LMDD_9.csv", "up100_T2022_LMDD_4.csv"],
  #  "Group2": ["up100_Q2025_LMDD_9.csv", "up100_T2022_LMDD_4.csv", "up100_Q2025_LMDD_5.csv"],
    "Group3": ["up100_T2022_LMDD_6.csv", "up100_Q2025_LMDD_3.csv"],
 #   "Group4": ["up100_G2017_LMDD_0.csv", "up100_Q2025_LMDD_1.csv"],
    "Group5": ["up100_T2022_LMDD_2.csv", "up100_Q2025_LMDD_7.csv"],
    "Group7": ["up100_Q2025_LMDD_6.csv", "up100_W2022_LMDD_1.csv", "up100_T2022_LMDD_3.csv"],
    "Group6": ["up100_W2022_LMDD_1.csv", "up100_T2022_LMDD_3.csv"],
    "Group8": ["up100_T2022_LMDD_0.csv", "up100_T2022_LMDD_1.csv", "up100_W2022_LMDD_2.csv"],
    "Group9": ["up100_T2022_LMDD_1.csv", "up100_W2022_LMDD_2.csv"],
    "Group10": ["up100_T2022_LMDD_5.csv", "up100_W2022_LMDD_0.csv"],
    "Group11": ["up100_T2022_LMDD_5.csv", "up100_W2022_LMDD_0.csv", "up100_G2017_LMDD_5.csv"]
    }
# Example directories (make sure to adjust these to your file locations)
data_dir = './LMDD/up100/'
output_dir = './LMDD/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[415]:


# re-run Group8, as Group7 caused kill with no sig enriched GO Terms
sample_groups = {"Group11": ["up100_T2022_LMDD_5.csv", "up100_W2022_LMDD_0.csv", "up100_G2017_LMDD_5.csv"]}

# Example directories (make sure to adjust these to your file locations)
data_dir = './LMDD/up100/'
output_dir = './LMDD/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[343]:


LMDD_1_gene_list = ['GSTP1', 'NUDT8', 'PDLIM1', 'HEBP2', 'PPP1R14B', 'KRT7']
LMDD_2_gene_list = ['KRT7']
LMDD_3_gene_list = ['PPP1R1B', 'ADIRF', 'H3F3A', 'SRP9', 'TMSB10', 'ORMDL3', 'CD24', 'MRPL45', 'S100A14', 'MIEN1']
LMDD_4_gene_list = ['MGP', 'MT-CO1', 'MALAT1', 'MT-ND3', 'RPL3', 'MT-ND2', 'MT-CO2', 'NEAT1', 'MT-CO3', 'MT-ND4', 'MT-CYB', 'MT-ATP6', 'MT-ND1']
LMDD_5_gene_list = ['KTN1', 'XBP1', 'FLNB', 'BCAM', 'ELOVL5', 'STC2', 'AHNAK', 'AZGP1', 'DPP7', 'CIRBP', 'CHPT1', 'NME3', 'RUNX1', 'TRPS1']
LMDD_6_gene_list = ['CISD3', 'NDUFB9', 'RPL19', 'RPS26', 'PSMB3', 'UBL5', 'MDK']
LMDD_8_gene_list = ['CST3', 'MGP', 'RPL34']
LMDD_9_gene_list = ['MGP', 'RPL34', 'RPL32', 'CST3', 'RPS14', 'GRN']
LMDD_10_gene_list = ['WFDC2', 'MGP', 'FXYD3', 'XBP1', 'AGR3', 'AZGP1', 'RPL35A', 'CST3', 'KRT15', 'FTL']
LMDD_11_gene_list = ['RPL35A']


# In[355]:


enr = gp.enrichr(gene_list=LMDD_11_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Mature DCIS Cluster 11",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Mature DCIS Cluster 11",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# ## DD LP (Fresh)

# In[437]:


input_folder = './LPDD/up100/'
input_folder_up1000 = './LPDD/up1000/'
output_folder = './LPDD/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_T2022_LPDD_1.csv", "up100_Q2025_LPDD_3.csv", "up100_Q2025_LPDD_2.csv"],
    "Group2": ["up100_T2022_LPDD_1.csv", "up100_Q2025_LPDD_3.csv"],
    "Group3": ["up100_T2022_LPDD_2.csv", "up100_Q2025_LPDD_0.csv"]}
# Colour Palette
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# Assisting Functions
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

def zscore_per_column(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

# Main Loop
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[438]:


def run_go_enrichment_intersect(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from gseapy import enrichr, barplot, dotplot

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_lists = []

        # Process each sample in the group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")

                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = set(df[df['logfoldchanges'] > 0]['names'])
                    if filtered_genes:
                        gene_lists.append(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")

        # Take the intersection across all samples in the group
        if gene_lists:
            intersect_genes = set.intersection(*gene_lists)
            print(f"Intersecting Genes in {group_name}: {intersect_genes}")
        else:
            intersect_genes = set()

        if intersect_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=list(intersect_genes),
                gene_sets=gene_sets,
                organism=organism
            )

            # Save enrichment results
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            # Helper function to style plots
            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

            # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")

        else:
            print(f"No intersecting genes for {group_name}. Skipping GO enrichment.")
            
# Example of your sample groups
sample_groups = {
    "Group1": ["up100_T2022_LPDD_1.csv", "up100_Q2025_LPDD_3.csv", "up100_Q2025_LPDD_2.csv"],
    "Group2": ["up100_T2022_LPDD_1.csv", "up100_Q2025_LPDD_3.csv"],
    "Group3": ["up100_T2022_LPDD_2.csv", "up100_Q2025_LPDD_0.csv"]}

# Example directories (make sure to adjust these to your file locations)
data_dir = './LPDD/up100/'
output_dir = './LPDD/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[356]:


LPDD_1_gene_list = ['SON', 'RBM39', 'GTF2I']
LPDD_2_gene_list = ['MACF1', 'SON', 'GTF2I', 'CD46', 'RSRP1', 'NEAT1', 'ARID4B', 'RBM25', 'RBM39', 'PTPRK', 'ARID1B']
LPDD_3_gene_list = ['HLA-DRA', 'RPLP1', 'HLA-DPA1', 'FTH1', 'RPS3A', 'HLA-DQA1', 'RPL41', 'CD74', 'FTL', 'RPS3']


# In[359]:


enr = gp.enrichr(gene_list=LPDD_3_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human',
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Progenitor DCIS Cluster 3",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Progenitor DCIS Cluster 3",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# ## DN Basal (Fresh)

# In[7]:


input_folder = './BDN/up100/'
input_folder_up1000 = './BDN/up1000/'
output_folder = './BDN/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_Q2025_BDN_1.csv", "up100_G2017_BDN_0.csv"],
    "Group2": ["up100_T2022_BDN_2.csv", "up100_Q2025_BDN_5.csv"],
    "Group3": ["up100_W2022_BDN_1.csv", "up100_T2022_BDN_2.csv", "up100_Q2025_BDN_5.csv"],
    "Group4": ["up100_T2022_BDN_0.csv", "up100_W2022_BDN_1.csv", "up100_T2022_BDN_2.csv", "up100_Q2025_BDN_5.csv"],
    "Group5": ["up100_W2022_BDN_0.csv", "up100_Q2025_BDN_2.csv"],
    "Group6": ["up100_W2022_BDN_0.csv", "up100_Q2025_BDN_2.csv", "up100_T2022_BDN_1.csv"],
    "Group7": ["up100_Q2025_BDN_3.csv", "up100_G2017_BDN_1.csv"],
}
# Colour Palette
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])
# Assisting Functions
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

def zscore_per_column(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

from matplotlib.colors import LinearSegmentedColormap
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

# Main Loop
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]
    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[10]:


go_groups = {
    "Group1": ["up100_Q2025_BDN_1.csv", "up100_G2017_BDN_0.csv"],
    "Group2": ["up100_T2022_BDN_2.csv", "up100_Q2025_BDN_5.csv"],
    "Group3": ["up100_W2022_BDN_1.csv", "up100_T2022_BDN_2.csv", "up100_Q2025_BDN_5.csv"],
    "Group4": ["up100_T2022_BDN_0.csv", "up100_W2022_BDN_1.csv", "up100_T2022_BDN_2.csv", "up100_Q2025_BDN_5.csv"],
    "Group5": ["up100_W2022_BDN_0.csv", "up100_Q2025_BDN_2.csv"],
    "Group6": ["up100_W2022_BDN_0.csv", "up100_Q2025_BDN_2.csv", "up100_T2022_BDN_1.csv"],
    "Group7": ["up100_Q2025_BDN_3.csv", "up100_G2017_BDN_1.csv"],
}
output_dir = './BDN/GO_Enrichment_Results/'
# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[360]:


BDN_1_gene_list = ['MEF2C', 'COL4A2', 'ID3', 'PGF', 'EPAS1', 'CRIP1', 'SPARCL1', 'CD99', 'COL18A1', 'GJC1', 'ITGA1', 'NOTCH3', 'CRISPLD2', 'NR2F2', 'SPARC', 'COL6A2', 'UBA2', 'COL4A1', 'ZEB2', 'KANK2', 'ADIRF', 'MCAM', 'IGFBP7', 'PCOLCE', 'PTP4A3', 'RGS5', 'FILIP1L', 'APOLD1', 'KCNE4', 'LGALS1', 'MAP1B', 'PTK2']
BDN_2_gene_list = ['HSPA5', 'APLP2', 'APOE', 'CTSB', 'HSP90AB1']
BDN_3_gene_list = ['APOE']
#BDN_4_gene_list = []
BDN_5_gene_list = ['SDC4', 'GPRC5A', 'MUC1', 'KLF6', 'CLDN4', 'SAT1', 'HSPH1']
BDN_6_gene_list = ['SDC4', 'KLF6']
BDN_7_gene_list = ['LY6E', 'FXYD3', 'NFIB', 'CD9', 'IRX2', 'VAMP8', 'DDR1', 'JUP', 'SAA1', 'CRYAB', 'SFRP1', 'RCAN1', 'OCIAD2', 'KRT7', 'TNFSF10', 'TACSTD2', 'SPINT2', 'PERP']


# In[390]:


enr = gp.enrichr(gene_list=BDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Basal Non-Malignant Cluster 7",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Basal Non-Malignant Cluster 7",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/BDN_1.csv', index=False)

sig_results.head()


# In[430]:


enr = gp.enrichr(gene_list=LPDN_3_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
sig_results = enr.results[enr.results['Adjusted P-value'] <= 0.05]
sig_results.to_csv('./data/AAA_DCIS/final_ORA_results/LPDN_3.csv', index=False)

sig_results.head()


# ## DN LM (Fresh)

# In[461]:


input_folder = './LMDN/up100/'
input_folder_up1000 = './LMDN/up1000/'
output_folder = './LMDN/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_G2017_LMDN_1.csv", "up100_T2022_LMDN_4.csv"],
    "Group2": ["up100_W2022_LMDN_2.csv", "up100_Q2025_LMDN_5.csv"],
    "Group3": ["up100_W2022_LMDN_2.csv", "up100_Q2025_LMDN_5.csv", "up100_G2017_LMDN_1.csv", "up100_T2022_LMDN_4.csv"],
    "Group4": ["up100_Q2025_LMDN_3.csv", "up100_T2022_LMDN_7.csv"],
    "Group5": ["up100_Q2025_LMDN_0.csv", "up100_T2022_LMDN_3.csv"],
    "Group6": ["up100_Q2025_LMDN_4.csv", "up100_G2017_LMDN_2.csv"],
    "Group7": ["up100_Q2025_LMDN_4.csv", "up100_G2017_LMDN_2.csv", "up100_Q2025_LMDN_2.csv"],
    "Group8": ["up100_Q2025_LMDN_1.csv", "up100_T2022_LMDN_1.csv", "up100_T2022_LMDN_6.csv"],
    "Group9": ["up100_W2022_LMDN_1.csv", "up100_T2022_LMDN_5.csv"],
    "Group10": ["up100_W2022_LMDN_0.csv", "up100_T2022_LMDN_0.csv"]
}
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]
    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[22]:


input_folder = './LMDN/up100/'
input_folder_up1000 = './LMDN/up1000/'
output_folder = './LMDN/Heatmaps_up100/'
output_dir = './LMDN/GO_Enrichment_Results/'
groups = {
    "Group1": ["up100_G2017_LMDN_1.csv", "up100_T2022_LMDN_4.csv"],
    "Group2": ["up100_W2022_LMDN_2.csv", "up100_Q2025_LMDN_5.csv"],
    "Group3": ["up100_W2022_LMDN_2.csv", "up100_Q2025_LMDN_5.csv", "up100_G2017_LMDN_1.csv", "up100_T2022_LMDN_4.csv"],
    "Group4": ["up100_Q2025_LMDN_3.csv", "up100_T2022_LMDN_7.csv"],
    "Group5": ["up100_Q2025_LMDN_0.csv", "up100_T2022_LMDN_3.csv"],
    "Group6": ["up100_Q2025_LMDN_4.csv", "up100_G2017_LMDN_2.csv"],
    "Group7": ["up100_Q2025_LMDN_4.csv", "up100_G2017_LMDN_2.csv", "up100_Q2025_LMDN_2.csv"],
    "Group8": ["up100_Q2025_LMDN_1.csv", "up100_T2022_LMDN_1.csv", "up100_T2022_LMDN_6.csv"],
    "Group9": ["up100_W2022_LMDN_1.csv", "up100_T2022_LMDN_5.csv"],
    "Group10": ["up100_W2022_LMDN_0.csv", "up100_T2022_LMDN_0.csv"]
}
# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[369]:


LMDN_1_gene_list = ['ERBB3', 'GTF2I', 'KMT2A', 'PERP', 'TACSTD2']
LMDN_2_gene_list = ['CD46', 'APLP2', 'ALCAM', 'BZW1', 'MUC1', 'TBX3']
#LMDN_3_gene_list = []
LMDN_4_gene_list = ['MT-CO3', 'MT-CO1', 'FBN1', 'EMP1', 'SPARCL1', 'AKAP12', 'GSN', 'SPARC', 'MT-ND1', 'PECAM1', 'TIMP3', 'CALD1', 'HLA-B', 'MT-CYB', 'MT-ATP6', 'VIM', 'MT-ND4', 'MT-ND3', 'CAV1', 'IGFBP7', 'FSTL1', 'HLA-A', 'HLA-E', 'MT-ND2', 'TCF4', 'GNG11', 'MT-CO2']
LMDN_5_gene_list = ['B2M', 'LYZ', 'RPL28', 'VIM', 'CST3', 'RPL12', 'SAT1', 'MARCKS']
LMDN_6_gene_list = ['B2M', 'ACTG1', 'MZT2B', 'GPSM3', 'GAPDH']
LMDN_7_gene_list = ['GAPDH']
#LMDN_8_gene_list = []
LMDN_9_gene_list = ['CAPS', 'XBP1', 'MIF', 'CITED4', 'KRT15']
LMDN_10_gene_list = ['RPS26', 'COX6B1', 'LY6E', 'UBL5', 'RPL23A', 'NDUFS5', 'CHCHD2', 'COPE', 'PSMD8', 'NDUFA4', 'PSMB3', 'GAPDH']


# In[385]:


enr = gp.enrichr(gene_list=LMDN_10_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Mature Non-Malignant Cluster 10",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Mature Non-Malignant Cluster 10",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# ## DN LP (Fresh)

# In[463]:


input_folder = './LPDN/up100/'
input_folder_up1000 = './LPDN/up1000/'
output_folder = './LPDN/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_T2022_LPDN_3.csv", "up100_Q2025_LPDN_2.csv"],
    "Group2": ["up100_T2022_LPDN_3.csv", "up100_Q2025_LPDN_2.csv", "up100_G2017_LPDN_1.csv"],
    "Group3": ["up100_W2022_LPDN_1.csv", "up100_Q2025_LPDN_1.csv"]
    }
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(15, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]
    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(15, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[32]:


input_folder = './LPDN/up100/'
input_folder_up1000 = './LPDN/up1000/'
output_folder = './LPDN/Heatmaps_up100/'
groups = {
    "Group1": ["up100_T2022_LPDN_3.csv", "up100_Q2025_LPDN_2.csv"],
    "Group2": ["up100_T2022_LPDN_3.csv", "up100_Q2025_LPDN_2.csv", "up100_G2017_LPDN_1.csv"],
    "Group3": ["up100_W2022_LPDN_1.csv", "up100_Q2025_LPDN_1.csv"]
    }
output_dir = './LPDN/GO_Enrichment_Results/'
# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[386]:


LPDN_1_gene_list = ['TMSB10', 'ISG15', 'FBLN1', 'CDKN2A', 'IFI6', 'HSPB1', 'PGK1', 'SCD', 'BLVRB', 'S100A11']
LPDN_2_gene_list = ['PGK1', 'S100A11']
LPDN_3_gene_list = ['NDRG2', 'TOMM7', 'HINT1', 'EPB41L4A-AS1', 'PFDN5', 'PPA1', 'SNHG8', 'EIF3E', 'S100A1', 'PRDX2', 'BTF3', 'ZFAS1']


# In[389]:


enr = gp.enrichr(gene_list=LPDN_3_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2025'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(4, 10),
              title = "Luminal Progenitor Non-Malignant Cluster 3",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=10,
              figsize=(3,8),
              title = "Luminal Progenitor Non-Malignant Cluster 3",
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue', 'GO_Biological_Process_2025':'lightgreen'}
             )


# # Concatenate after Subpop Annotation

# In[151]:


adatas = [alldata_T2022, alldata_Q2025, alldata_W2022, alldata_G2017] 
combined = adatas[0].concatenate(
    *adatas[1:], 
    batch_key='dataset',       
    batch_categories=[f'D{i}' for i in range(len(adatas))], 
    index_unique=None           # preserve original cell IDs (already unique)
)
#cut out N2025 from above to just look at fresh tissues


# In[152]:


combined.obs['annotation'] = combined.obs['annotation'].astype('category')


# In[167]:


sc.pp.highly_variable_genes(combined, batch_key='dataset', n_top_genes=2000, flavor='seurat')
combined_hv = combined[:, combined.var['highly_variable']].copy()
sc.pp.scale(combined_hv)
sc.tl.pca(combined_hv, svd_solver='arpack')
sc.pp.neighbors(combined_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(combined_hv)
combined.obsm['X_umap'] = combined_hv.obsm['X_umap']
sc.pl.umap(combined, color='annotation', frameon=False)


# In[168]:


sc.tl.leiden(combined_hv, resolution = 1.0)
combined.obs['leiden'] = combined_hv.obs['leiden']  # copy clusters to full AnnData
sc.pl.umap(combined, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[173]:


sc.pl.umap(combined, color='annotation', frameon=False)


# In[174]:


sc.tl.rank_genes_groups(combined, groupby="annotation", method="wilcoxon")


# In[177]:


sc.pl.umap(combined, color='cell type', frameon=False)


# In[178]:


sc.pl.umap(combined, color='cnv_status', frameon=False)


# In[184]:


sc.pl.umap(combined, color='Epithelial_vs_NonEpithelial', frameon=False)


# In[179]:


sc.pl.umap(combined, color='Batch', frameon=False)


# In[180]:


combined.write_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")


# In[4]:


combined = sc.read_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")


# In[37]:


combined.obs['Batch'] = combined.obs['Batch'].replace('G2017', 'G2021')
combined.obs['Sample'] = combined.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
combined.write_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")

