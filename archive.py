#!/usr/bin/env python
# coding: utf-8

# # Loading

# In[1]:


import scanpy as sc
import pandas as pd
import anndata
import numpy as np
#import scvi
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


# In[477]:


import importlib.metadata

# replace 'scvi-tools' with the package name as installed
print(importlib.metadata.version("scvi-tools"))


# In[472]:


import gseapy as gp


# In[27]:


# set colour palette
celltype_colors = {
    "Basal": "#17becf",              # teal
    "Luminal Progenitor": "#e377c2", # pink
    "Luminal Mature": "8c564b",     # brown
    "Endothelial": "#9467bd",         # purple
    "Fibroblast": "#2ca02c",     # green
    "General Myeloid": "#fee08b", # yellow
    "T-Cell": "#bcbd22", # olive
    "B-Cell": "#d62728", # red
    "Macrophage": "#1f77b4", # blue
    "Monocyte": "#ffbb78" ,# light blue
    "Non-epithelial": "#1c1c84", # navy
    "Epithelial": "#91bfdb", # soft blue
    "DCIS": "#fdbf6f", # gold
    "normal": "#7f7f7f" # grey
}


# In[430]:


def set_celltype_colors(adata, obs_key, color_dict, default="#cccccc"):
    if not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")

    cats = adata.obs[obs_key].cat.categories
    adata.uns[f"{obs_key}_colors"] = [
        color_dict.get(c, default) for c in cats
    ]


# In[836]:


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


# In[431]:


cell_type_order = ['Basal', 'Luminal Progenitor', 'Luminal Mature', 'Endothelial', 'Fibroblast', 'General Myeloid', 'T-Cell', 'B-Cell', 'Macrophage', 'Monocyte']
alldata.obs['cell type'] = pd.Categorical(
    alldata.obs['cell type'],
    categories=cell_type_order,
    ordered=True
)
set_celltype_colors(alldata, "cell type", celltype_colors)


# In[837]:


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


# ### F2019 (FFPE)

# In[ ]:


#F2019


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


# In[19]:


#and saving separately
alldata = sc.read_h5ad('260121_dcis_combined_after_pp.h5ad')


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


# # N2025 FFPE

# In[1289]:


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


# In[363]:


sc.pl.umap(alldata_N2025, color = ['annotation', 'Epithelial_vs_NonEpithelial', 'cnv_status', 'cell type'], frameon = False, ncols=1)


# ### DN Basal N2025 FFPE

# In[1293]:


N2025_DN_B = N2025_B[N2025_B.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(N2025_DN_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[1294]:


sc.pl.pca_variance_ratio(N2025_DN_B, n_pcs=50)


# In[1295]:


sc.pp.highly_variable_genes(N2025_DN_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DN_B_hv = N2025_DN_B[:, N2025_DN_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DN_B_hv)
sc.tl.pca(N2025_DN_B_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DN_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DN_B_hv)
N2025_DN_B.obsm['X_umap'] = N2025_DN_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[1296]:


sc.tl.leiden(N2025_DN_B_hv, resolution = 0.2)
N2025_DN_B.obs['leiden'] = N2025_DN_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DN_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DN_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DN_B, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon = None)


# In[1297]:


sc.pl.umap(N2025_DN_B, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'SPARC'], ncols=4)


# In[1298]:


basal_leiden_DN_N2025 = {"0":"0", "1":"1", "2":"2", "3":"3"}
N2025_DN_B.obs['subcluster'] = N2025_DN_B.obs.leiden.map(basal_leiden_DN_N2025)
N2025_DN_B.obs['subcluster'].value_counts()


# In[1299]:


save_top_marker_genes(N2025_DN_B, 'N2025', 'BDN')


# In[1300]:


N2025_DN_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDN.h5ad")


# In[346]:


N2025_DN_B = sc.read_h5ad("./data/AAA_DCIS/260216_N2025_BDN.h5ad")
N2025_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "ITGA10/LENG8+_BDN",
    "1": "na",
    "2": "high_mitochondria/stress_responsive_BDN"
}
N2025_DN_B.obs['annotation'] = N2025_DN_B.obs['subcluster'].map(annotation_map)
N2025_DN_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDN.h5ad")


# ### DD N2025 FFPE Basal

# In[1292]:


N2025_B = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Basal'])]
sc.pl.umap(N2025_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[1301]:


N2025_DD_B = N2025_B[N2025_B.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_DD_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[1302]:


sc.pl.pca_variance_ratio(N2025_DD_B, n_pcs=50)


# In[1303]:


sc.pp.highly_variable_genes(N2025_DD_B, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DD_B_hv = N2025_DD_B[:, N2025_DD_B.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DD_B_hv)
sc.tl.pca(N2025_DD_B_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DD_B_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DD_B_hv)
N2025_DD_B.obsm['X_umap'] = N2025_DD_B_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[1304]:


sc.tl.leiden(N2025_DD_B_hv, resolution = 0.5)
N2025_DD_B.obs['leiden'] = N2025_DD_B_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DD_B_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DD_B_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DD_B, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[1305]:


sc.pl.umap(N2025_DD_B, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'SPARC'], ncols=4)


# In[1306]:


basal_leiden_DD_N2025 = {"0":"0", "1":"1", "2":"2"}
N2025_DD_B.obs['subcluster'] = N2025_DD_B.obs.leiden.map(basal_leiden_DD_N2025)
N2025_DD_B.obs['subcluster'].value_counts()


# In[1307]:


save_top_marker_genes(N2025_DD_B, 'N2025', 'BDD')


# In[1308]:


N2025_DD_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDD.h5ad")


# In[347]:


N2025_DD_B = sc.read_h5ad("./data/AAA_DCIS/260216_N2025_BDD.h5ad")
N2025_DD_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "na"
}
N2025_DD_B.obs['annotation'] = N2025_DD_B.obs['subcluster'].map(annotation_map)
N2025_DD_B.write_h5ad("./data/AAA_DCIS/260216_N2025_BDD.h5ad")


# ### DN N2025 FFPE LM

# In[1310]:


N2025_DN_LM = N2025_LM[N2025_LM.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(N2025_DN_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[1311]:


sc.pl.pca_variance_ratio(N2025_DN_LM, n_pcs=50)


# In[1312]:


sc.pp.normalize_total(N2025_DN_LM, target_sum=1e4)
sc.pp.log1p(N2025_DN_LM)

sc.pp.highly_variable_genes(N2025_DN_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DN_LM_hv = N2025_DN_LM[:, N2025_DN_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DN_LM_hv)
sc.tl.pca(N2025_DN_LM_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DN_LM_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(N2025_DN_LM_hv)
N2025_DN_LM.obsm['X_umap'] = N2025_DN_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[1313]:


sc.tl.leiden(N2025_DN_LM_hv, resolution = 0.5)
N2025_DN_LM.obs['leiden'] = N2025_DN_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DN_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DN_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DN_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon=None)


# In[1314]:


LM_leiden_DN_N2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7"}
N2025_DN_LM.obs['subcluster'] = N2025_DN_LM.obs.leiden.map(LM_leiden_DN_N2025)
N2025_DN_LM.obs['subcluster'].value_counts()


# In[1315]:


save_top_marker_genes(N2025_DN_LM, 'N2025', 'LMDN')


# In[1316]:


N2025_DN_LM.write_h5ad("./data/AAA_DCIS/260217_N2025_LMDN.h5ad")


# In[348]:


N2025_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_N2025_LMDN.h5ad")
N2025_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "growth_factor_responsive_LMDN",
    "2": "na",
    "3": "high_mitochondrial_gene_expressing_LMDN",
    "4": "MYC/mTOR_high/metabolic/proliferative_LMDN",
    "5": "na",
    "6": "na",
    "7": "na",
}
N2025_DN_LM.obs['annotation'] = N2025_DN_LM.obs['subcluster'].map(annotation_map)
N2025_DN_LM.write_h5ad("./data/AAA_DCIS/260217_N2025_LMDN.h5ad")


# ### DD N2025 FFPE LM

# In[1317]:


N2025_LM = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(N2025_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[1318]:


N2025_DD_LM = N2025_LM[N2025_LM.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_DD_LM, color = ['cnv_status', 'cell type'], frameon = False)


# In[1319]:


sc.pl.pca_variance_ratio(N2025_DD_LM, n_pcs=50)


# In[1320]:


sc.pp.highly_variable_genes(N2025_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DD_LM_hv = N2025_DD_LM[:, N2025_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DD_LM_hv)
sc.tl.pca(N2025_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DD_LM_hv, n_neighbors=15, n_pcs=10)
sc.tl.umap(N2025_DD_LM_hv)
N2025_DD_LM.obsm['X_umap'] = N2025_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[1321]:


sc.tl.leiden(N2025_DD_LM_hv, resolution = 0.5)
N2025_DD_LM.obs['leiden'] = N2025_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon = None)


# In[1322]:


LM_leiden_DD_N2025 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7", "8":"8", "9":"9", "10":"10", "11":"11", "12":"12", "13":"13"}
N2025_DD_LM.obs['subcluster'] = N2025_DD_LM.obs.leiden.map(LM_leiden_DD_N2025)
N2025_DD_LM.obs['subcluster'].value_counts()


# In[1323]:


save_top_marker_genes(N2025_DD_LM, 'N2025', 'LMDD')


# In[1324]:


N2025_DD_LM.write_h5ad("./data/AAA_DCIS/260216_N2025_LMDD.h5ad")


# In[349]:


N2025_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260216_N2025_LMDD.h5ad")
N2025_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "differentiating/plastic_LMDD",
    "1": "na",
    "2": "na",
    "3": "na",
    "4": "na",
    "5": "na",
    "6": "na",
    "7": "na",
    "8": "na",
    "9": "na",
    "10": "na",
}
N2025_DD_LM.obs['annotation'] = N2025_DD_LM.obs['subcluster'].map(annotation_map)
N2025_DD_LM.write_h5ad("./data/AAA_DCIS/260216_N2025_LMDD.h5ad")


# ### DD N2025 FFPE LP

# In[1325]:


N2025_LP = alldata_N2025[alldata_N2025.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(N2025_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[1326]:


N2025_DD_LP = N2025_LP[N2025_LP.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(N2025_DD_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[1327]:


sc.pl.pca_variance_ratio(N2025_DD_LP, n_pcs=50)


# In[1328]:


sc.pp.highly_variable_genes(N2025_DD_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DD_LP_hv = N2025_DD_LP[:, N2025_DD_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DD_LP_hv)
sc.tl.pca(N2025_DD_LP_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DD_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DD_LP_hv)
N2025_DD_LP.obsm['X_umap'] = N2025_DD_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[1329]:


sc.tl.leiden(N2025_DD_LP_hv, resolution = 0.6)
N2025_DD_LP.obs['leiden'] = N2025_DD_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DD_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DD_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DD_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[1330]:


LP_leiden_DD_N2025 = {"0":"0", "1":"1"}
N2025_DD_LP.obs['subcluster'] = N2025_DD_LP.obs.leiden.map(LP_leiden_DD_N2025)
N2025_DD_LP.obs['subcluster'].value_counts()


# In[1331]:


save_top_marker_genes(N2025_DD_LP, 'N2025', 'LPDD')


# In[1332]:


N2025_DD_LP.write_h5ad("./data/AAA_DCIS/260216_N2025_LPDD.h5ad")


# In[350]:


N2025_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260216_N2025_LPDD.h5ad")
N2025_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na"
}
N2025_DD_LP.obs['annotation'] = N2025_DD_LP.obs['subcluster'].map(annotation_map)
N2025_DD_LP.write_h5ad("./data/AAA_DCIS/260216_N2025_LPDD.h5ad")


# ### DN N2025 LP

# In[1333]:


N2025_DN_LP = N2025_LP[N2025_LP.obs['cnv_status'].isin(['normal'])]
sc.pl.umap(N2025_DN_LP, color = ['cnv_status', 'cell type'], frameon = False)


# In[1334]:


sc.pl.pca_variance_ratio(N2025_DN_LP, n_pcs=50)


# In[1335]:


sc.pp.highly_variable_genes(N2025_DN_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
N2025_DN_LP_hv = N2025_DN_LP[:, N2025_DN_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(N2025_DN_LP_hv)
sc.tl.pca(N2025_DN_LP_hv, svd_solver='arpack')
sc.pp.neighbors(N2025_DN_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(N2025_DN_LP_hv)
N2025_DN_LP.obsm['X_umap'] = N2025_DN_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[1336]:


sc.tl.leiden(N2025_DN_LP_hv, resolution = 0.4)
N2025_DN_LP.obs['leiden'] = N2025_DN_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(N2025_DN_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(N2025_DN_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(N2025_DN_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1, frameon=None)


# In[1337]:


LP_leiden_DN_N2025 = {"0":"0", "1":"1", "2":"2", "3":"3"}
N2025_DN_LP.obs['subcluster'] = N2025_DN_LP.obs.leiden.map(LP_leiden_DN_N2025)
N2025_DN_LP.obs['subcluster'].value_counts()


# In[1338]:


save_top_marker_genes(N2025_DN_LP, 'N2025', 'LPDN')


# In[1339]:


N2025_DN_LP.write_h5ad("./data/AAA_DCIS/260217_N2025_LPDN.h5ad")


# In[351]:


N2025_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_N2025_LPDN.h5ad")
N2025_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "proliferative/E2F_high_LPDN",
    "3": "na"
}
N2025_DN_LP.obs['annotation'] = N2025_DN_LP.obs['subcluster'].map(annotation_map)
N2025_DN_LP.write_h5ad("./data/AAA_DCIS/260217_N2025_LPDN.h5ad")


# # Q2025

# In[156]:


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


# In[46]:


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

sc.pl.dotplot(alldata_Q2025, var_names=marker_genes, groupby='leiden')


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


# In[85]:


sc.pl.umap(alldata_Q2025, color = ['KRT14', 'ELF5', 'FOXA1', 'EPCAM', 'leiden'], frameon = False, legend_loc = "on data", ncols =3)
# Basal (KRT14), LP (ELF5), LM (FOXA1), Epithelial (EPCAM)


# In[82]:


alldata_Q2025.obs['cell type'] = alldata_Q2025.obs.leiden.map(cell_type)
sc.pl.umap(alldata_Q2025, color = ['cell type'], frameon = False)


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


# In[249]:


alldata_Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')


# In[271]:


alldata_Q2025.obs.set_index('cell_id_orig', inplace=True)


# In[287]:


subpops = [Q2025_DD_B, Q2025_DN_LM, Q2025_DD_LM, Q2025_DN_LP, Q2025_DD_LP]

for ad in subpops:
    ad.obs.set_index('cell_id', inplace=True)


# In[305]:


# Create column if it doesn't exist
alldata_Q2025.obs['annotation'] = pd.NA

subpops = [Q2025_DD_B, Q2025_DN_B, Q2025_DN_LM, Q2025_DD_LM, Q2025_DN_LP, Q2025_DD_LP]
# Transfer annotations
for ad in subpops:
    alldata_Q2025.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']


# In[306]:


alldata_Q2025.obs['annotation'] = alldata_Q2025.obs['annotation'].fillna('unassigned')


# In[312]:


sc.pl.umap(alldata_Q2025, color = ['annotation', 'Epithelial_vs_NonEpithelial', 'cnv_status', 'cell type'], frameon = False, ncols=1)


# In[313]:


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


# In[205]:


Q2025_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_Q2025_BDN.h5ad")


# In[297]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "luminal_basal_like_antigen_presenting/immune_interacting_BDN",
    "3": "adhesion_enriched/stressed_BDN",
    "4": "ITGA10/LENG8+_BDN"
}
Q2025_DN_B.obs['annotation'] = Q2025_DN_B.obs['subcluster'].map(annotation_map)
Q2025_DN_B.write_h5ad("./data/AAA_DCIS/260217_Q2025_BDN.h5ad")


# In[261]:


sc.pl.umap(Q2025_DN_B, color=['annotation', 'Batch', 'Sample'], ncols = 1)


# ### Q2025 DD Basal

# In[169]:


Q2025_B = alldata_Q2025[alldata_Q2025.obs['cell type'].isin(['Basal'])]
sc.pl.umap(Q2025_B, color = ['cnv_status', 'cell type'], frameon = False)


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


# In[166]:


def save_top_marker_genes(adata, study_name, cell_type, gene_counts=[20, 30, 100, 1000]):
    """
    Saves the top differentially expressed marker genes for each subcluster to CSV files.
    
    Parameters:
    - adata: AnnData object
    - study_name: str, the name of the study (e.g., 'N2021')
    - cell_type: str, the cell type (e.g., 'basal')
    - gene_counts: list, number of top genes to save (default: [20, 30, 100])
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


# In[298]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DD_B = sc.read_h5ad("./data/AAA_DCIS/260205_Q2025_BDD.h5ad")
#Q2025_DD_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "interferon_responsive_BDD",
    "2": "na",
    "3": "na"
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


# In[299]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_Q2025_LMDN.h5ad")
#Q2025_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "na",
    "3": "lipid_metabolising_&_secreting_LMDN",
    "4": "highly_mitochondrial_gene_expressing_LMDN",
    "5": "oxidative_stress_responsive_LMDN",
    "6": "na"
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


# In[300]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_Q2025_LMDD.h5ad")
#Q2025_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "endothelial_like_&_angiogenic_LMDD",
    "1": "highly_biosynthesising/mTORC1_activated_LMDD",
    "2": "immune_modulating/KRAS_activated/inflammatory_LMDD",
    "3": "estrogen_responsive/tumour_associated_LMDD",
    "4": "na",
    "5": "MAPK_supressed_LMDD",
    "6": "highly_biosynthesising_LMDD",
    "7": "differentiating/plastic_LMDD",
    "8": "na",
    "9": "na",
    "10": "na",
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


# In[301]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_Q2025_LPDN.h5ad")
#Q2025_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "proliferative/E2F_high_LPDN"
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


# In[302]:


#add annotation based on GO/KEGG/MSigDB enrichment analysis & dendrograms
#Q2025_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_Q2025_LPDD.h5ad")
#Q2025_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "translation_primed_LPDD",
    "1": "immune_responsive_LPDD",
    "2": "motile/structured_LPDD",
    "3": "na",
    "4": "na"
}
Q2025_DD_LP.obs['annotation'] = Q2025_DD_LP.obs['subcluster'].map(annotation_map)
Q2025_DD_LP.write_h5ad("./data/AAA_DCIS/260205_Q2025_LPDD.h5ad")


# # T2022

# In[218]:


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


# In[99]:


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


# In[310]:


alldata_T2022.write_h5ad('260203_dcis_T2022_only.h5ad')


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


# In[339]:


T2022_DN_B.obs.head()


# In[340]:


alldata_T2022.obs.head()


# In[341]:


alldata_T2022.obs['annotation'] = pd.NA

subpops = [T2022_DN_B, T2022_DD_B, T2022_DN_LM, T2022_DD_LM, T2022_DN_LP, T2022_DD_LP]
# Transfer annotations
for ad in subpops:
    alldata_T2022.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']


# In[342]:


alldata_T2022.obs['annotation'] = alldata_T2022.obs['annotation'].fillna('unassigned')


# In[344]:


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


# In[316]:


T2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_BDN.h5ad")
T2022_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "contractile/myofibroblast_like_BDN",
    "1": "adhesion_enriched/stressed_BDN",
    "2": "high_mitochondria/stress_responsive_BDN"
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


# In[317]:


T2022_DD_B = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_BDD.h5ad")
T2022_DD_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "interferon_responsive_BDD",
    "1": "na"
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


# In[318]:


T2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_LMDN.h5ad")
T2022_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "phospholipid_metabolising_LMDN",
    "3": "na",
    "4": "na",
    "5": "na",
    "6": "lipid_metabolising_&_secreting_LMDN",
    "7": "highly_mitochondrial_gene_expressing_LMDN",
    "8": "MYC/mTOR_high/metabolic/proliferative_LMDN",
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


# In[319]:


T2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LMDD.h5ad")
T2022_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "immune_modulating/KRAS_activated/inflammatory_LMDD",
    "2": "MAPK_supressed_LMDD",
    "3": "highly_biosynthesising_LMDD",
    "4": "na",
    "5": "na",
    "6": "MAPK_supressed_LMDD",
    "7": "na",
    "8": "na",
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


# In[320]:


T2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_T2022_LPDN.h5ad")
T2022_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "na",
    "3": "na"
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


# In[321]:


T2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")
T2022_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "immune_responsive_LPDD",
    "2": "translation_primed_LPDD"
}
T2022_DD_LP.obs['annotation'] = T2022_DD_LP.obs['subcluster'].map(annotation_map)
T2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")


# In[334]:


T2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_T2022_LPDD.h5ad")


# # W2022

# In[270]:


alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[107]:


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


# In[387]:


alldata_W2022.write_h5ad('260203_dcis_W2022_only.h5ad')


# In[388]:


alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[381]:


W2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_BDN.h5ad")
W2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LMDN.h5ad")
W2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")
W2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LPDN.h5ad")
W2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")


# In[383]:


W2022_DD_LP.obs.head()


# In[385]:


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


# In[366]:


W2022_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_BDN.h5ad")
W2022_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "luminal_basal_like_antigen_presenting/immune_interacting_BDN",
    "1": "contractile/myofibroblast_like_BDN",
    "2": "mesenchymal_like_BDN"
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


# In[367]:


W2022_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LMDN.h5ad")
W2022_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na",
    "2": "phospholipid_metabolising_LMDN",
    "3": "highly_protein_synthesising/active_secretory_LMDN"
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


# In[293]:


sc.pp.highly_variable_genes(W2022_DD_LM, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DD_LM_hv = W2022_DD_LM[:, W2022_DD_LM.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DD_LM_hv)
sc.tl.pca(W2022_DD_LM_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DD_LM_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(W2022_DD_LM_hv)
W2022_DD_LM.obsm['X_umap'] = W2022_DD_LM_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[294]:


sc.tl.leiden(W2022_DD_LM_hv, resolution = 0.4)
W2022_DD_LM.obs['leiden'] = W2022_DD_LM_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DD_LM_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DD_LM_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DD_LM, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[1231]:


LM_leiden_DD_W2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
W2022_DD_LM.obs['subcluster'] = W2022_DD_LM.obs.leiden.map(LM_leiden_DD_W2022)
W2022_DD_LM.obs['subcluster'].value_counts()


# In[295]:


save_top_marker_genes(W2022_DD_LM, 'W2022', 'LMDD')


# In[296]:


W2022_DD_LM.write_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")


# In[368]:


W2022_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LMDD.h5ad")
W2022_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "endothelial_like_&_angiogenic_LMDN",
    "1": "stroma_regulating_LMDN",
    "2": "na",
    "3": "na",
    "4": "na"
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


# In[369]:


W2022_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_W2022_LPDN.h5ad")
W2022_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "na"
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


# In[308]:


sc.pp.highly_variable_genes(W2022_DD_LP, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
W2022_DD_LP_hv = W2022_DD_LP[:, W2022_DD_LP.var['highly_variable']].copy() # subset hv
sc.pp.scale(W2022_DD_LP_hv)
sc.tl.pca(W2022_DD_LP_hv, svd_solver='arpack')
sc.pp.neighbors(W2022_DD_LP_hv, n_neighbors=15, n_pcs=5)
sc.tl.umap(W2022_DD_LP_hv)
W2022_DD_LP.obsm['X_umap'] = W2022_DD_LP_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[310]:


sc.tl.leiden(W2022_DD_LP_hv, resolution = 0.5)
W2022_DD_LP.obs['leiden'] = W2022_DD_LP_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(W2022_DD_LP_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(W2022_DD_LP_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(W2022_DD_LP, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[1247]:


LP_leiden_DD_W2022 = {"0":"0", "1":"1"}
W2022_DD_LP.obs['subcluster'] = W2022_DD_LP.obs.leiden.map(LP_leiden_DD_W2022)
W2022_DD_LP.obs['subcluster'].value_counts()


# In[1248]:


save_top_marker_genes(W2022_DD_LP, 'W2022', 'LPDD')


# In[311]:


W2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")


# In[370]:


W2022_DD_LP = sc.read_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")
W2022_DD_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "motile/structured_LPDD",
    "1": "na"
}
W2022_DD_LP.obs['annotation'] = W2022_DD_LP.obs['subcluster'].map(annotation_map)
W2022_DD_LP.write_h5ad("./data/AAA_DCIS/260205_W2022_LPDD.h5ad")


# # G2017 (G2021)

# In[312]:


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


# In[9]:


alldata_G2017.write_h5ad('260203_dcis_G2017_only.h5ad')


# In[392]:


alldata_G2017.obs['cell_id_orig'] = alldata_G2017.obs.index


# In[395]:


alldata_G2017.obs['annotation'] = pd.NA

subpops = [G2017_DN_B, G2017_DN_LM, G2017_DD_LM, G2017_DN_LP]
# Transfer annotations
for ad in subpops:
    alldata_G2017.obs.loc[ad.obs.index, 'annotation'] = ad.obs['annotation']

alldata_G2017.obs['annotation'] = alldata_G2017.obs['annotation'].fillna('unassigned')


# In[396]:


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


# In[388]:


G2017_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")
G2017_DN_B.obs['annotation'] = np.nan 
annotation_map = {
    "0": "mesenchymal_like_BDN",
    "1": "adhesion_enriched/stressed_BDN"
}
G2017_DN_B.obs['annotation'] = G2017_DN_B.obs['subcluster'].map(annotation_map)
G2017_DN_B.write_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")


# In[10]:


G2017_DN_B = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")
G2017_DN_B.obs['Batch'] = G2017_DN_B.obs['Batch'].replace('G2017', 'G2021')
G2017_DN_B.obs['Sample'] = G2017_DN_B.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
G2017_DN_B.write_h5ad("./data/AAA_DCIS/260217_G2017_BDN.h5ad")


# ### G2017 DD Basal

# In[321]:


G2017_B = alldata_G2017[alldata_G2017.obs['cell type'].isin(['Basal'])]
sc.pl.umap(G2017_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[322]:


G2017_DD_B = G2017_B[G2017_B.obs['cnv_status'].isin(['DCIS'])]
sc.pl.umap(G2017_DD_B, color = ['cnv_status', 'cell type'], frameon = False)


# In[323]:


G2017_DD_B.write_h5ad("./data/AAA_DCIS/260205_G2017_BDD.h5ad")


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


# In[331]:


G2017_DN_LM = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_LMDN.h5ad")
G2017_DN_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "na",
    "1": "growth_factor_responsive_LMDN",
    "2": "highly_protein_synthesising/active_secretory_LMDN"
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


# In[390]:


G2017_DD_LM = sc.read_h5ad("./data/AAA_DCIS/260205_G2017_LMDD.h5ad")
G2017_DD_LM.obs['annotation'] = np.nan 
annotation_map = {
    "0": "estrogen_responsive/tumour_associated_LMDD",
    "1": "stroma_regulating_LMDD",
    "2": "highly_biosynthesising/mTORC1_activated_LMDD",
    "3": "na",
    "4": "na",
    "5": "na",
    "6": "stroma_regulating_LMDD"
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


# In[391]:


G2017_DN_LP = sc.read_h5ad("./data/AAA_DCIS/260217_G2017_LPDN.h5ad")
G2017_DN_LP.obs['annotation'] = np.nan 
annotation_map = {
    "0": "ribosome_high/protein_synthesis_active_LPDN",
    "1": "na"
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
#alldata = sc.read_h5ad('260121_dcis_combined_after_pp.h5ad')


# In[42]:


print(alldata.shape)
sc.pp.filter_genes(alldata, min_cells = 200) #lot of samples, so higher filtering of genes: only keep genes if in min 200 cells
alldata.X = csr_matrix(alldata.X) # convert dense to sparse matrix, less memory
print(alldata.shape)


# In[43]:


# save again here, less memory if need to


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


# alldata_hv.write_h5ad('260121_dcis_alldata_hv.h5ad') # save for later


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


# In[21]:


# reload
alldata = sc.read_h5ad('260121_dcis_scvi_integrated.h5ad')


# In[23]:


alldata.obs['Batch'] = alldata.obs['Batch'].replace('G2017', 'G2021')
alldata.obs['Sample'] = alldata.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')


# In[24]:


sc.pl.umap(alldata, color = ['Batch'], frameon = False, ncols = 1)


# In[25]:


sc.pl.umap(alldata, color = ['Sample'], frameon = False, ncols = 1)


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

# In[3]:


import infercnvpy as cnv
import pybiomart


# In[4]:


sc.logging.print_header()


# In[5]:


import scipy.sparse


# In[792]:


#export data so can be used in R inferCNV
Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')
Q2025.obs.to_csv("Q2025_meta.csv")
expr_matrix = Q2025.X.toarray() if scipy.sparse.issparse(Q2025.X) else Q2025.X
expr_df = pd.DataFrame(expr_matrix, index=Q2025.obs_names, columns=Q2025.var_names)
expr_df.to_csv("Q2025_counts.csv")


# In[793]:


T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')
T2022.obs.to_csv("T2022_meta.csv")
expr_matrix = T2022.X.toarray() if scipy.sparse.issparse(T2022.X) else T2022.X
expr_df = pd.DataFrame(expr_matrix, index=T2022.obs_names, columns=T2022.var_names)
expr_df.to_csv("T2022_counts.csv")


# In[794]:


W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')
W2022.obs.to_csv("W2022_meta.csv")
expr_matrix = W2022.X.toarray() if scipy.sparse.issparse(W2022.X) else W2022.X
expr_df = pd.DataFrame(expr_matrix, index=W2022.obs_names, columns=W2022.var_names)
expr_df.to_csv("W2022_counts.csv")


# In[796]:


G2021 = sc.read_h5ad('260203_dcis_G2017_only.h5ad')
G2021.obs.to_csv("G2021_meta.csv")
expr_matrix = G2021.X.toarray() if scipy.sparse.issparse(G2021.X) else G2021.X
expr_df = pd.DataFrame(expr_matrix, index=G2021.obs_names, columns=G2021.var_names)
expr_df.to_csv("G2021_counts.csv")


# ## Q2025

# In[6]:


#alldata = alldata_Q2025
alldata = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')


# In[145]:


#rename old cnv_status (conducted on whole study)
#to cnv_status_old & drop cnv_status (so can rename
#with new samples subsets names
alldata.obs['cnv_status_old'] = alldata.obs['cnv_status']


# In[135]:


cnv.io.genomic_position_from_biomart(
    alldata,
    species="hsapiens",
    biomart_gene_id="hgnc_symbol"
)


# In[197]:


alldata.var[['chromosome', 'start', 'end']].head()


# In[198]:


sc.pl.umap(alldata, color="cell type")


# In[199]:


sc.pl.umap(alldata, color= "Batch")


# In[139]:


#biomart adds MT genes back in, only want nuclear genes for analysis
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=250,
)


# In[140]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type")


# In[141]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden")


# In[142]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[143]:


sc.settings.set_figure_params(figsize=(5, 5))


# In[50]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[200]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[201]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[202]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[208]:


alldata.obs["cnv_status"] = "normal"
alldata.obs.loc[alldata.obs["cnv_leiden"].isin(["0", "1", "3", "4", "14", "22", "24", "33", "34"]), "cnv_status"] = (
    "DCIS"
)


# In[209]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_status", ax=ax2)


# In[210]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "DCIS", :])


# In[211]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "normal", :])


# In[64]:


alldata.write_h5ad('260203_dcis_Q2025_only.h5ad')


# In[149]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_Q2025_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


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


# Trial Subset 5: ind1, ind3, ind16 (young), "ind3_Q2025", "ind16_Q2025"
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


# Trial Subset 6: ind2, ind4, ind15, ind17 (remaining: older than Subset 5), "ind4_Q2025", "ind15_Q2025", "ind17_Q2025"
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

# In[65]:


#alldata = alldata_T2022
alldata = sc.read_h5ad('260203_dcis_T2022_only.h5ad')


# In[139]:


#rename old cnv_status (conducted on whole study)
#to cnv_status_old & drop cnv_status (so can rename
#with new samples subsets names
alldata.obs['cnv_status_old'] = alldata.obs['cnv_status']


# In[547]:


sc.pl.umap(alldata, color=["cnv_status_old", "cnv_status", "cell type"])


# In[153]:


cnv.io.genomic_position_from_biomart(
    alldata,
    species="hsapiens",
    biomart_gene_id="hgnc_symbol"
)


# In[154]:


alldata.var[['chromosome', 'start', 'end']].head()


# In[214]:


sc.pl.umap(alldata, color="cell type")


# In[215]:


sc.pl.umap(alldata, color="Batch")


# In[156]:


#biomart adds MT genes back in, only want nuclear genes for analysis
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Monocyte"],
    window_size=250,
)


# In[158]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type")


# In[159]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden")


# In[160]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[161]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(15,25))


# In[216]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[220]:


cnv.pl.umap(alldata, color="cnv_leiden", show=False, legend_loc = "on data")


# In[217]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[218]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[222]:


alldata.obs["cnv_status"] = "normal"
alldata.obs.loc[alldata.obs["cnv_leiden"].isin(["0", '2', '3', '5']), "cnv_status"] = (
    "DCIS"
)


# In[223]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_status", ax=ax2)


# In[224]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "DCIS", :])


# In[225]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "normal", :])


# In[128]:


alldata.write_h5ad('260203_dcis_T2022_only.h5ad')


# In[221]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_T2022_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# ### Subsets T2022 Samples Based on Similar Metadata

# In[120]:


T2022_subset1 = ["ind7_T2022"]
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
    dynamic_threshold=2
)
cnv.pl.chromosome_heatmap(T2022_subset1, groupby="cell type", figsize=(30,50))


# In[121]:


cnv.tl.pca(T2022_subset1)
cnv.pp.neighbors(T2022_subset1)
cnv.tl.leiden(T2022_subset1)
cnv.tl.umap(T2022_subset1)
cnv.tl.cnv_score(T2022_subset1)
sc.tl.dendrogram(T2022_subset1, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(T2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[122]:


cnv.pl.umap(T2022_subset1, color="cnv_score", show=False)
cnv.pl.umap(T2022_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(T2022_subset1, color="cell type", show=False)
sc.pl.umap(T2022_subset1, color="cnv_score")
sc.pl.umap(T2022_subset1, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[123]:


cnv_leiden = T2022_subset1.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_T2022_ind_7_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[124]:


mean_cnv = T2022_subset1.obs['cnv_score'].mean()
median_cnv = T2022_subset1.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[125]:


T2022_subset1.obs["cnv_status"] = "normal"
T2022_subset1.obs.loc[T2022_subset1.obs["cnv_leiden"].isin(["1", "5", "7", "8", "10", "11", "3"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(T2022_subset1[T2022_subset1.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(T2022_subset1[T2022_subset1.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[126]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(T2022_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(T2022_subset1, color="cnv_status", ax=ax2)


# In[127]:


T2022_subset1.write_h5ad('260309_dcis_T2022_subset_7.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[T2022_subset1.obs_names, 'cnv_status'] = T2022_subset1.obs['cnv_status']


# ## W2022

# In[129]:


#alldata = alldata_W2022
alldata = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[144]:


W2022_subset1 = ["ind2_W2022"]
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
    dynamic_threshold=2
)
cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cell type", figsize=(30,50))


# In[145]:


cnv.tl.pca(W2022_subset1)
cnv.pp.neighbors(W2022_subset1)
cnv.tl.leiden(W2022_subset1)
cnv.tl.umap(W2022_subset1)
cnv.tl.cnv_score(W2022_subset1)
sc.tl.dendrogram(W2022_subset1, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[ ]:


sc.tl.dendrogram(W2022_subset1, groupby='cnv_leiden')


# In[ ]:


cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[146]:


cnv.pl.umap(W2022_subset1, color="cnv_score", show=False)
cnv.pl.umap(W2022_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(W2022_subset1, color="cell type", show=False)
sc.pl.umap(W2022_subset1, color="cnv_score")
sc.pl.umap(W2022_subset1, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


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


# In[150]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(W2022_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(W2022_subset1, color="cnv_status", ax=ax2)


# In[151]:


W2022_subset1.write_h5ad('260309_dcis_W2022_subset_2.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[W2022_subset1.obs_names, 'cnv_status'] = W2022_subset1.obs['cnv_status']


# In[ ]:





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


# In[1021]:


#biomart adds MT genes back in, only want nuclear genes for analysis
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Macrophage"],
    window_size=100,
    dynamic_threshold=2
)


# In[1022]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type", figsize=(17,12))


# In[1023]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden")


# In[1024]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[153]:


cnv.tl.pca(W2022_subset1)
cnv.pp.neighbors(W2022_subset1)
cnv.tl.leiden(W2022_subset1)
cnv.tl.umap(W2022_subset1)
cnv.tl.cnv_score(W2022_subset1)


# In[154]:


sc.tl.dendrogram(W2022_subset1, groupby="cnv_leiden")


# In[155]:


cnv.pl.chromosome_heatmap(W2022_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(18,12))


# In[1026]:


sc.tl.dendrogram(alldata, groupby="cnv_leiden")


# In[1029]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(18,12))


# In[1028]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[1030]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[1031]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[1032]:


alldata.obs["cnv_status"] = "normal"
alldata.obs.loc[alldata.obs["cnv_leiden"].isin(["0", "1", "5"]), "cnv_status"] = (
    "DCIS"
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_status", ax=ax2)


# In[1033]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "DCIS", :])


# In[1034]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "normal", :])


# In[152]:


alldata.write_h5ad('260203_dcis_W2022_only.h5ad')


# In[1035]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_W2022_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[1036]:


mean_cnv = alldata.obs['cnv_score'].mean()
median_cnv = alldata.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


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

# In[1082]:


#alldata = alldata_N2025
alldata = sc.read_h5ad('260205_dcis_N2025_only.h5ad')


# In[ ]:


#rename old cnv_status (conducted on whole study)
#to cnv_status_old & drop cnv_status (so can rename
#with new samples subsets names
alldata.obs['cnv_status_old'] = alldata.obs['cnv_status']


# In[632]:


cnv.io.genomic_position_from_biomart(
    alldata,
    species="hsapiens",
    biomart_gene_id="hgnc_symbol"
)


# In[1079]:


#biomart adds MT genes back in, only want nuclear genes for analysis
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell"],
    window_size=100,
    dynamic_threshold=2
)


# In[ ]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type")


# In[ ]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden", figsize=(24,24))


# In[ ]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[ ]:


sc.tl.dendrogram(alldata, groupby="cnv_leiden")


# In[ ]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(24,24))


# In[ ]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[ ]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[642]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_N2025_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[643]:


alldata.obs["cnv_status"] = "normal"
alldata.obs.loc[alldata.obs["cnv_leiden"].isin(["0", "21", "26", "30"]), "cnv_status"] = (
    "DCIS"
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_status", ax=ax2)


# In[644]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "DCIS", :])


# In[645]:


cnv.pl.chromosome_heatmap(alldata[alldata.obs["cnv_status"] == "normal", :])


# In[183]:


alldata.write_h5ad('260205_dcis_N2025_only.h5ad')


# ### N2025 Subsets

# In[1083]:


N2025_subset1 = ["ind1_N2025", "ind3_N2025", "ind4_N2025", "ind6_N2025"]
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
cnv.pl.chromosome_heatmap(N2025_subset1, groupby="cell type", figsize=(30,50))


# In[1084]:


cnv.tl.pca(N2025_subset1)
cnv.pp.neighbors(N2025_subset1)
cnv.tl.leiden(N2025_subset1)
cnv.tl.umap(N2025_subset1)
cnv.tl.cnv_score(N2025_subset1)
sc.tl.dendrogram(N2025_subset1, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(N2025_subset1, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[1085]:


cnv.pl.umap(N2025_subset1, color="cnv_score", show=False)
cnv.pl.umap(N2025_subset1, color="cnv_leiden", show=False)
cnv.pl.umap(N2025_subset1, color="cell type", show=False)
sc.pl.umap(N2025_subset1, color="cnv_score")
sc.pl.umap(N2025_subset1, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[1086]:


cnv_leiden = N2025_subset1.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_N2025_subset_1_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[1087]:


mean_cnv = N2025_subset1.obs['cnv_score'].mean()
median_cnv = N2025_subset1.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[1093]:


N2025_subset1.obs["cnv_status"] = "normal"
N2025_subset1.obs.loc[N2025_subset1.obs["cnv_leiden"].isin(["15", "13", "17"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(N2025_subset1[N2025_subset1.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(N2025_subset1[N2025_subset1.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[1094]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(N2025_subset1, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(N2025_subset1, color="cnv_status", ax=ax2)


# In[1095]:


N2025_subset1.write_h5ad('260309_dcis_N2025_subset_1.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[N2025_subset1.obs_names, 'cnv_status'] = N2025_subset1.obs['cnv_status']


# In[1088]:


N2025_subset2 = ["ind2_N2025", "ind5_N2025"]
N2025_subset2 = alldata[alldata.obs['Sample'].isin(N2025_subset2)].copy()
nuclear_genes = ~N2025_subset2.var['chromosome'].isin(['MT', 'chrMT', 'M'])
N2025_subset2 = N2025_subset2[:, nuclear_genes]
cnv.tl.infercnv(
    N2025_subset2,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell"],
    window_size=100,
    dynamic_threshold=2
)
cnv.pl.chromosome_heatmap(N2025_subset2, groupby="cell type", figsize=(30,50))


# In[1089]:


cnv.tl.pca(N2025_subset2)
cnv.pp.neighbors(N2025_subset2)
cnv.tl.leiden(N2025_subset2)
cnv.tl.umap(N2025_subset2)
cnv.tl.cnv_score(N2025_subset2)
sc.tl.dendrogram(N2025_subset2, groupby='cnv_leiden')
cnv.pl.chromosome_heatmap(N2025_subset2, groupby="cnv_leiden", dendrogram=True, figsize=(30,50))


# In[1090]:


cnv.pl.umap(N2025_subset2, color="cnv_score", show=False)
cnv.pl.umap(N2025_subset2, color="cnv_leiden", show=False)
cnv.pl.umap(N2025_subset2, color="cell type", show=False)
sc.pl.umap(N2025_subset2, color="cnv_score")
sc.pl.umap(N2025_subset2, color="cell type")
sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[1091]:


cnv_leiden = N2025_subset2.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()
cnv_leiden.to_csv('./data/inferCNVpy_N2025_subset_2_output_cnv_leiden_cnv_score_cell_type.csv', index=False)
cnv_leiden


# In[1092]:


mean_cnv = N2025_subset2.obs['cnv_score'].mean()
median_cnv = N2025_subset2.obs['cnv_score'].median()

print("Mean CNV score:", mean_cnv)
print("Median CNV score:", median_cnv)


# In[1096]:


N2025_subset2.obs["cnv_status"] = "normal"
N2025_subset2.obs.loc[N2025_subset2.obs["cnv_leiden"].isin(["11", "10"]), "cnv_status"] = (
    "DCIS"
)
cnv.pl.chromosome_heatmap(N2025_subset2[N2025_subset2.obs["cnv_status"] == "DCIS", :], figsize=(40,10))
cnv.pl.chromosome_heatmap(N2025_subset2[N2025_subset2.obs["cnv_status"] == "normal", :], figsize=(30,50))


# In[1097]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(N2025_subset2, color="cnv_status", ax=ax1, show=False)
sc.pl.umap(N2025_subset2, color="cnv_status", ax=ax2)


# In[1098]:


N2025_subset2.write_h5ad('260309_dcis_N2025_subset_2.h5ad')
#add new classifications back to old alldata (all samples Q2025)
alldata.obs.loc[N2025_subset2.obs_names, 'cnv_status'] = N2025_subset2.obs['cnv_status']


# In[1099]:


sc.pl.umap(alldata, color=["cnv_status_old", "cnv_status", "cell type"])


# ## All Studies

# In[ ]:


alldata = sc.read_h5ad('260203_dcis_alldata_annotated.h5ad')


# In[ ]:


#cnv.io.genomic_position_from_biomart(
#    alldata,
#    species="hsapiens",
#    biomart_gene_id="hgnc_symbol"
#) #already done once


# In[ ]:


alldata.var[['chromosome', 'start', 'end']].head()


# In[ ]:


sc.pl.umap(alldata, color="cell type", "Batch")


# In[ ]:


#biomart adds MT genes back in, only want nuclear genes for analysis
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Monocyte"],
    window_size=250,
)


# In[ ]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type")


# In[ ]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden")


# In[ ]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[ ]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(15,25))


# In[ ]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[ ]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[ ]:


alldata.write_h5ad('260203_dcis_alldata_annotated.h5ad')


# In[ ]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


# In[ ]:





# ## All Datasets

# In[189]:


# do not do on scvi integrated! do on pre-integrated
alldata = sc.read_h5ad('260130_dcis_alldata_annotated.h5ad')


# In[190]:


cnv.io.genomic_position_from_biomart(
    alldata,
    species="hsapiens",
    biomart_gene_id="hgnc_symbol"
)


# In[191]:


alldata.var[['chromosome', 'start', 'end']].head()


# In[ ]:


sc.pl.umap(alldata, color="cell type")


# In[192]:


#biomart adds MT genes back in, only want nuclear genes for analysis
nuclear_genes = ~alldata.var['chromosome'].isin(['MT', 'chrMT', 'M'])
alldata = alldata[:, nuclear_genes]
cnv.tl.infercnv(
    alldata,
    reference_key="cell type",
    reference_cat=[
        "B-Cell",
        "General Myeloid",
        "T-Cell",
        "Monocyte"],
    window_size=250,
)


# In[193]:


cnv.pl.chromosome_heatmap(alldata, groupby="cell type")


# In[194]:


cnv.pl.chromosome_heatmap(alldata, groupby="leiden")


# In[195]:


cnv.tl.pca(alldata)
cnv.pp.neighbors(alldata)
cnv.tl.leiden(alldata)
cnv.tl.umap(alldata)
cnv.tl.cnv_score(alldata)


# In[203]:


sc.settings.set_figure_params(figsize=(5, 5))


# In[198]:


cnv.pl.chromosome_heatmap(alldata, groupby="cnv_leiden", dendrogram=True, figsize=(15,25))


# In[216]:


cnv.pl.umap(alldata, color="cnv_score", show=False)
cnv.pl.umap(alldata, color="cnv_leiden", show=False)
cnv.pl.umap(alldata, color="cell type", show=False)


# In[217]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.5})
cnv.pl.umap(alldata, color="cnv_score", ax=ax1, show=False)
sc.pl.umap(alldata, color="cnv_score", ax=ax2)


# In[204]:


sc.pl.umap(alldata, color="cnv_score")
sc.pl.umap(alldata, color="cell type")


# In[205]:


sc.pl.umap(alldata, color="leiden")


# In[206]:


sc.pl.umap(alldata, color="cnv_leiden")


# In[208]:


sc.pl.umap(alldata, color="leiden", legend_loc = "on data")


# In[209]:


sc.pl.umap(alldata, color="leiden")


# In[210]:


alldata.write_h5ad("260130_dcis_alldata_annotated.h5ad")


# In[211]:


# Group by leiden cluster and compute mean CNV score
cnv1 = alldata.obs.groupby('leiden')['cnv_score'].mean().to_frame()

# Add cell type annotation
cluster_celltype = alldata.obs[['leiden', 'cell type']].drop_duplicates().set_index('leiden')
cnv1 = cnv.merge(cluster_celltype, left_index=True, right_index=True)

# Save to CSV
cnv1.to_csv('./data/inferCNVpy_output_cnv_score_cell_type.csv')

# Optional: inspect
cnv1


# In[218]:


# Group by leiden cluster and compute mean CNV score
cnv_leiden = alldata.obs.groupby('cnv_leiden')['cnv_score'].mean().reset_index()

# Save to CSV
cnv_leiden.to_csv('./data/inferCNVpy_output_cnv_leiden_cnv_score_cell_type.csv', index=False)

# Optional: inspect
cnv_leiden


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

# In[255]:


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


# In[259]:


sc.pl.umap(alldata_Q2025, color=['scf_prediction', 'cell type'], ncols = 1, frameon = False)


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


# # Compare CNV Prediction Methods

# In[35]:


sc.pl.umap(
    adata,
    color=["malignancy_probability", "cnv_score", "scf_prediction", "cell type"],
    frameon=False,
    ncols = 1
)


# In[382]:


# G2017
alldata_G2017 = sc.read_h5ad('260203_dcis_G2017_only.h5ad')


# In[383]:


sc.pl.umap(
    alldata_G2017,
    color=["malignancy_probability", "scf_prediction", "cnv_score", "cnv_status"], 
    frameon=False,
    ncols = 2
)


# In[384]:


sc.pl.umap(
    alldata_G2017,
    color=["cell type", "Epithelial_vs_NonEpithelial"],
    frameon=False,
    ncols = 1
)


# In[385]:


# W2022
alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')


# In[386]:


sc.pl.umap(
    alldata_W2022,
    color=["malignancy_probability", "scf_prediction", "cnv_score", "cnv_status"],
    frameon=False,
    ncols = 2
)


# In[387]:


sc.pl.umap(
    alldata_W2022,
    color=["cell type", "Epithelial_vs_NonEpithelial"],
    frameon=False,
    ncols = 1
)


# In[388]:


# T2022
alldata_T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')


# In[389]:


sc.pl.umap(
    alldata_T2022,
    color=["malignancy_probability", "scf_prediction", "cnv_score", "cnv_status"],
    frameon=False,
    ncols = 2
)


# In[390]:


sc.pl.umap(
    alldata_T2022,
    color=["cell type", "Epithelial_vs_NonEpithelial"],
    frameon=False,
    ncols = 1
)


# In[391]:


#Q2025
alldata_Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')


# In[392]:


sc.pl.umap(
    alldata_Q2025,
    color=["malignancy_probability", "scf_prediction", "cnv_score", "cnv_status"],
    frameon=False,
    ncols = 2
)


# In[393]:


sc.pl.umap(
    alldata_Q2025,
    color=["cell type", "Epithelial_vs_NonEpithelial"],
    frameon=False,
    ncols = 1
)


# In[394]:


sc.pl.umap(
    alldata_Q2025,
    color=["ERBB2", "Sample"],
    frameon=False,
    ncols = 1
)


# In[21]:


#N2025
alldata_N2025 = sc.read_h5ad('260205_dcis_N2025_only.h5ad')


# In[22]:


sc.pl.umap(
    alldata_N2025,
    color=["malignancy_probability", "scf_prediction", "cnv_score", "cnv_status"],
    frameon=False,
    ncols = 2
)


# In[23]:


sc.pl.umap(
    alldata_N2025,
    color=["cell type", "Epithelial_vs_NonEpithelial"],
    frameon=False,
    ncols = 1
)


# In[24]:


natgen = pd.read_csv("./data/AAA_DCIS/NatGen_Supplementary_table_S4.csv")

subtype2genes = {}
for col in natgen.columns:
    genes = natgen[col].dropna().tolist()  # remove NaNs
    subtype2genes[col] = genes


for subtype, genes in subtype2genes.items():
    print(subtype, genes[:5])

# Compute per‑cell scores
# first normalize (e.g., log1p) if not already
if 'log1p' not in alldata_Q2025.uns:
    sc.pp.normalize_total(alldata_Q2025)
    sc.pp.log1p(alldata_Q2025)

# get expression matrix as DataFrame for ease
expr_df = pd.DataFrame(
    alldata_Q2025.X.toarray() if not isinstance(alldata_Q2025.X, np.ndarray) else alldata_Q2025.X,
    index=alldata_Q2025.obs_names,
    columns=alldata_Q2025.var_names
)

# function to compute mean expression of gene list per cell
def score_genes(genes, expr):
    present = [g for g in genes if g in expr.columns]
    if len(present)==0:
        return pd.Series(0, index=expr.index)
    return expr[present].mean(axis=1)

# score per subtype
scores = {}
for subtype, genes in subtype2genes.items():
    scores[subtype] = score_genes(genes, expr_df)

scores_df = pd.DataFrame(scores)

# Assign highest scoring subtype per cell
alldata_Q2025.obs['subtype_score'] = scores_df.values.argmax(axis=1)
alldata_Q2025.obs['subtype_call'] = scores_df.columns[scores_df.values.argmax(axis=1)]

print(alldata_Q2025.obs['subtype_call'].value_counts())


# In[376]:


sc.pl.umap(alldata_Q2025, color=['subtype_call'])


# In[377]:


if 'log1p' not in alldata_T2022.uns:
    sc.pp.normalize_total(alldata_T2022)
    sc.pp.log1p(alldata_T2022)

# get expression matrix as DataFrame for ease
expr_df = pd.DataFrame(
    alldata_T2022.X.toarray() if not isinstance(alldata_T2022.X, np.ndarray) else alldata_T2022.X,
    index=alldata_T2022.obs_names,
    columns=alldata_T2022.var_names
)
# score per subtype
scores = {}
for subtype, genes in subtype2genes.items():
    scores[subtype] = score_genes(genes, expr_df)

scores_df = pd.DataFrame(scores)

# Assign highest scoring subtype per cell
alldata_T2022.obs['subtype_score'] = scores_df.values.argmax(axis=1)
alldata_T2022.obs['subtype_call'] = scores_df.columns[scores_df.values.argmax(axis=1)]

print(alldata_T2022.obs['subtype_call'].value_counts())
sc.pl.umap(alldata_T2022, color=['subtype_call'])


# In[378]:


if 'log1p' not in alldata_W2022.uns:
    sc.pp.normalize_total(alldata_W2022)
    sc.pp.log1p(alldata_W2022)

# get expression matrix as DataFrame for ease
expr_df = pd.DataFrame(
    alldata_W2022.X.toarray() if not isinstance(alldata_W2022.X, np.ndarray) else alldata_W2022.X,
    index=alldata_W2022.obs_names,
    columns=alldata_W2022.var_names
)
# score per subtype
scores = {}
for subtype, genes in subtype2genes.items():
    scores[subtype] = score_genes(genes, expr_df)

scores_df = pd.DataFrame(scores)

# Assign highest scoring subtype per cell
alldata_W2022.obs['subtype_score'] = scores_df.values.argmax(axis=1)
alldata_W2022.obs['subtype_call'] = scores_df.columns[scores_df.values.argmax(axis=1)]

print(alldata_W2022.obs['subtype_call'].value_counts())
sc.pl.umap(alldata_W2022, color=['subtype_call'])


# In[379]:


if 'log1p' not in alldata_G2017.uns:
    sc.pp.normalize_total(alldata_G2017)
    sc.pp.log1p(alldata_G2017)

# get expression matrix as DataFrame for ease
expr_df = pd.DataFrame(
    alldata_G2017.X.toarray() if not isinstance(alldata_G2017.X, np.ndarray) else alldata_G2017.X,
    index=alldata_G2017.obs_names,
    columns=alldata_G2017.var_names
)
# score per subtype
scores = {}
for subtype, genes in subtype2genes.items():
    scores[subtype] = score_genes(genes, expr_df)

scores_df = pd.DataFrame(scores)

# Assign highest scoring subtype per cell
alldata_G2017.obs['subtype_score'] = scores_df.values.argmax(axis=1)
alldata_G2017.obs['subtype_call'] = scores_df.columns[scores_df.values.argmax(axis=1)]

print(alldata_G2017.obs['subtype_call'].value_counts())
sc.pl.umap(alldata_G2017, color=['subtype_call'])


# # Compare Each Study Epithelial Cell Markers

# In[97]:


# Fresh DD Epithelial Cells
alldatasets_DD_Ep = [Q2025_Ep_DD, T2022_Ep_DD, W2022_Ep_DD, G2017_Ep_DD]
alldatasets_DD_Ep = sc.concat(alldatasets_DD_Ep, join='outer')


# In[393]:


sc.pp.highly_variable_genes(alldatasets_DD_Ep, n_top_genes = 2000) # select top 2000 most variable/bio meaningful
alldatasets_DD_Ep_hv = alldatasets_DD_Ep[:, alldatasets_DD_Ep.var['highly_variable']].copy() # subset hv
sc.pp.scale(alldatasets_DD_Ep_hv)
sc.tl.pca(alldatasets_DD_Ep_hv, svd_solver='arpack')
sc.pp.neighbors(alldatasets_DD_Ep_hv, n_neighbors=15, n_pcs=20)
sc.tl.umap(alldatasets_DD_Ep_hv)
alldatasets_DD_Ep.obsm['X_umap'] = alldatasets_DD_Ep_hv.obsm['X_umap'] # Copy UMAP coords back to full AnnData


# In[394]:


sc.tl.leiden(alldatasets_DD_Ep_hv, resolution = 0.3)
alldatasets_DD_Ep.obs['leiden'] = alldatasets_DD_Ep_hv.obs['leiden']  # copy clusters to full AnnData
sc.tl.rank_genes_groups(alldatasets_DD_Ep_hv, groupby='leiden')
markers = sc.get.rank_genes_groups_df(alldatasets_DD_Ep_hv, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > 0.5)]
sc.pl.umap(alldatasets_DD_Ep, color=['leiden', 'Batch', 'Sample', 'cell type'], ncols = 1)


# In[395]:


#had trouble because G2017 Basal & LP is only 1 and 2 cells each, so comparing LM to 'rest' fails
def save_top_marker_genes_by_celltype(
    adata,
    study_name,
    cell_types,
    gene_counts=[20, 30, 100],
    min_cells=2,
):
    output_dir = "./data/AAA_DCIS/All_epithelial"
    os.makedirs(output_dir, exist_ok=True)

    for cell_type in cell_types:
        mask = adata.obs['cell type'] == cell_type
        n_target = mask.sum()
        n_rest = (~mask).sum()

        if n_target < min_cells or n_rest < min_cells:
            print(
                f"Skipping {study_name} | {cell_type}: "
                f"{n_target} target cells, {n_rest} rest cells"
            )
            continue

        adata_sub = adata[mask | (~mask)].copy()

        # convert to string to avoid categorical assignment issues
        adata_sub.obs['group'] = adata_sub.obs['cell type'].astype(str)
        adata_sub.obs['group'] = adata_sub.obs['group'].where(
            adata_sub.obs['group'] == cell_type, 'rest'
        )

        sc.tl.rank_genes_groups(
            adata_sub,
            groupby='group',
            reference='rest',
            method='t-test'
        )

        for n_genes in gene_counts:
            top_genes_df = sc.get.rank_genes_groups_df(
                adata_sub, group=cell_type
            ).head(n_genes)

            filename = f"{output_dir}/up{n_genes}_{study_name}_{cell_type}.csv"
            top_genes_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")


# In[397]:


save_top_marker_genes_by_celltype(Q2025_Ep_DD, 'Q2025', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(T2022_Ep_DD, 'T2022', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(W2022_Ep_DD, 'W2022', ['Basal', 'Luminal Progenitor', 'Luminal Mature']) # No Basal DD
save_top_marker_genes_by_celltype(G2017_Ep_DD, 'G2017', ['Basal', 'Luminal Progenitor', 'Luminal Mature']) # Only 1 or 2 cells DD Basal & LP


# # Compare Epithelial Subclusters

# ## Rank Dendrograms Fresh DD

# In[356]:


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


# In[354]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ---------- RBO implementation ----------
def rbo_score(list1, list2, p=0.9):
    """
    Compute Rank-Biased Overlap (RBO) score between two ranked lists.
    
    Parameters
    ----------
    list1, list2 : list
        Ranked gene lists (top = most important)
    p : float
        Weight parameter (0 < p < 1). Higher = more weight to top.
    
    Returns
    -------
    float
        RBO score (0–1)
    """
    s, t = list1, list2
    k = max(len(s), len(t))
    
    s_set, t_set = set(), set()
    overlap = 0
    summation = 0.0

    for d in range(1, k + 1):
        if d <= len(s):
            s_set.add(s[d - 1])
        if d <= len(t):
            t_set.add(t[d - 1])
        
        overlap = len(s_set.intersection(t_set))
        summation += overlap / d * (p ** (d - 1))

    return (1 - p) * summation


# ---------- Main function ----------
def build_rank_dendrogram_rbo(folder_path, top_n=None, title=None, method='average', p=0.9):
    """
    Build dendrogram using Rank-Biased Overlap (RBO) similarity.
    
    Parameters
    ----------
    folder_path : str
        Path to CSV files with marker genes.
    top_n : int or None
        Use top N genes per cluster.
    title : str
        Plot title.
    method : str
        Linkage method.
    p : float
        RBO parameter (higher = more weight to top genes).
    """

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    # ---------- Build ranked gene lists ----------
    def build_rank_list(df):
        df = df.copy()
        df = df[df['logfoldchanges'] > 0]
        df = df.sort_values('logfoldchanges', ascending=False)
        if top_n is not None:
            df = df.head(top_n)
        return df['names'].tolist()

    rank_lists = {}
    for file_path in csv_files:
        cluster_name = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        rank_lists[cluster_name] = build_rank_list(df)

    cluster_names = list(rank_lists.keys())
    S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

    # ---------- Compute RBO similarity ----------
    for i, ki in enumerate(cluster_names):
        for j, kj in enumerate(cluster_names):
            if j < i:
                continue
            
            rbo = rbo_score(rank_lists[ki], rank_lists[kj], p=p)
            S.loc[ki, kj] = rbo
            S.loc[kj, ki] = rbo

    # ---------- Convert to distance ----------
    dist_matrix = 1 - S
    np.fill_diagonal(dist_matrix.values, 0)

    dist_vector = squareform(dist_matrix.values)
    linkage_matrix = linkage(dist_vector, method=method)

    # ---------- Plot ----------
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=cluster_names, leaf_rotation=90)
    plt.title(title or f"Cluster similarity (RBO, p={p}, top {top_n})")
    plt.ylabel("1 - RBO")
    plt.tight_layout()
    plt.show()


# In[366]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ---------- RBO ----------
def rbo_score(list1, list2, p=0.9):
    s, t = list1, list2
    k = max(len(s), len(t))
    
    s_set, t_set = set(), set()
    summation = 0.0

    for d in range(1, k + 1):
        if d <= len(s):
            s_set.add(s[d - 1])
        if d <= len(t):
            t_set.add(t[d - 1])
        
        overlap = len(s_set.intersection(t_set))
        summation += overlap / d * (p ** (d - 1))

    return (1 - p) * summation


# ---------- Jaccard ----------
def jaccard_score(list1, list2):
    a, b = set(list1), set(list2)
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0


# ---------- Main ----------
def build_rank_dendrogram_dual(
    folder_path,
    top_n=None,
    title=None,
    method='average',
    p=0.9,
    plot_heatmaps=True
):

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    # ---------- Build ranked lists ----------
    def build_rank_list(df):
        df = df.copy()
        df = df[df['logfoldchanges'] > 0]
        df = df.sort_values('logfoldchanges', ascending=False)
        if top_n is not None:
            df = df.head(top_n)
        return df['names'].tolist()

    rank_lists = {}
    for file_path in csv_files:
        cluster_name = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        rank_lists[cluster_name] = build_rank_list(df)

    cluster_names = list(rank_lists.keys())

    # ---------- Matrices ----------
    S_rbo = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)
    S_jac = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

    for i, ki in enumerate(cluster_names):
        for j, kj in enumerate(cluster_names):
            if j < i:
                continue

            rbo = rbo_score(rank_lists[ki], rank_lists[kj], p=p)
            jac = jaccard_score(rank_lists[ki], rank_lists[kj])

            S_rbo.loc[ki, kj] = rbo
            S_rbo.loc[kj, ki] = rbo

            S_jac.loc[ki, kj] = jac
            S_jac.loc[kj, ki] = jac

    # ---------- Dendrograms ----------
    def plot_dendrogram(S, label):
        dist = 1 - S
        np.fill_diagonal(dist.values, 0)
        dist_vec = squareform(dist.values)
        Z = linkage(dist_vec, method=method)

        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=cluster_names, leaf_rotation=90)
        plt.title(f"{title or 'Cluster similarity'} ({label})")
        plt.ylabel(f"1 - {label}")
        plt.tight_layout()
        plt.show()

    plot_dendrogram(S_rbo, f"RBO p={p}")
    plot_dendrogram(S_jac, "Jaccard")

    # ---------- Heatmaps ----------
    if plot_heatmaps:
        plt.figure(figsize=(20, 15))
        sns.heatmap(S_rbo.astype(float), cmap="viridis", square=True)
        plt.title(f"RBO similarity (p={p})")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 15))
        sns.heatmap(S_jac.astype(float), cmap="magma", square=True)
        plt.title("Jaccard similarity")
        plt.tight_layout()
        plt.show()

    return S_rbo, S_jac


# In[367]:


S_rbo, S_jac = build_rank_dendrogram_dual(
    folder_path='./LMDD/up100',
    top_n=100,
    title="Subcluster similarity",
    p=0.9
)


# In[368]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ---------- Jaccard ----------
def jaccard_score(list1, list2):
    a, b = set(list1), set(list2)
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0


# ---------- Main ----------
def build_jaccard_dendrogram(
    folder_path,
    top_n=None,
    title=None,
    method='average'
):
    """
    Build dendrogram using Jaccard similarity of marker gene lists.
    """

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    # ---------- Build gene lists ----------
    def build_gene_list(df):
        df = df.copy()
        df = df[df['logfoldchanges'] > 0]
        df = df.sort_values('logfoldchanges', ascending=False)
        if top_n is not None:
            df = df.head(top_n)
        return df['names'].tolist()

    gene_lists = {}
    for file_path in csv_files:
        cluster_name = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        gene_lists[cluster_name] = build_gene_list(df)

    cluster_names = list(gene_lists.keys())

    # ---------- Compute Jaccard matrix ----------
    S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

    for i, ki in enumerate(cluster_names):
        for j, kj in enumerate(cluster_names):
            if j < i:
                continue

            jac = jaccard_score(gene_lists[ki], gene_lists[kj])
            S.loc[ki, kj] = jac
            S.loc[kj, ki] = jac

    # ---------- Convert to distance ----------
    dist_matrix = 1 - S
    np.fill_diagonal(dist_matrix.values, 0)

    dist_vector = squareform(dist_matrix.values)
    linkage_matrix = linkage(dist_vector, method=method)

    # ---------- Plot ----------
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=cluster_names, leaf_rotation=90)
    plt.title(title or f"Cluster similarity (Jaccard, top {top_n})")
    plt.ylabel("1 - Jaccard index")
    plt.tight_layout()
    plt.show()

    return S


# In[369]:


S_jaccard = build_jaccard_dendrogram(
    folder_path='./LMDD/up100',
    top_n=100,
    title="Jaccard similarity of subclusters"
)


# In[373]:


S_jaccard = build_jaccard_dendrogram(
    folder_path='./LMDD/up1000',
    top_n=100,
    title="Jaccard similarity of subclusters"
)


# In[378]:


build_rank_dendrogram_rbo(
    folder_path='./LMDD/up100',
    top_n=100,
    title="Cluster similarity (RBO)",
    p=0.9
)


# In[358]:


build_rank_dendrogram('./LMDD/up100', top_n=100, title="Fresh DCIS (DD) LM subcluster similarity (top 100)")


# In[26]:


build_rank_dendrogram('./data/AAA_DCIS/BDD/up1000', top_n=20, title="Fresh DCIS (DD) Basal subcluster similarity (top 20)")


# In[380]:


build_rank_dendrogram('./BDD/up100/', top_n=100, title="Fresh DCIS (DD) Basal subcluster similarity (top 100)")


# In[28]:


build_rank_dendrogram('./data/AAA_DCIS/BDD/up1000/', top_n=100, title="Fresh DCIS (DD) Basal subcluster similarity (top 100)")


# In[29]:


build_rank_dendrogram('./data/AAA_DCIS/BDD/up1000/', top_n=1000, title="Fresh DCIS (DD) Basal subcluster similarity (top 1000)")


# In[30]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD/up1000/', top_n=20, title="Fresh DCIS (DD) Luminal Mature subcluster similarity (top 20)")


# In[31]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD/up1000/', top_n=30, title="Fresh DCIS (DD) Luminal Mature subcluster similarity (top 30)")


# In[39]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD/up100/', top_n=100, title="Fresh DCIS (DD) Luminal Mature subcluster similarity (top 100)")


# In[33]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD/up1000/', top_n=1000, title="Fresh DCIS (DD) Luminal Mature subcluster similarity (top 1000)")


# In[34]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD/up1000/', top_n=20, title="Fresh DCIS (DD) Luminal Progenitor subcluster similarity (top 20)")


# In[35]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD/up1000/', top_n=30, title="Fresh DCIS (DD) Luminal Progenitor subcluster similarity (top 30)")


# In[381]:


build_rank_dendrogram('./LPDD/up100/', top_n=100, title="Fresh DCIS (DD) Luminal Progenitor subcluster similarity (top 100)")


# In[41]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD/up1000/', top_n=1000, title="Fresh DCIS (DD) Luminal Progenitor subcluster similarity (top 1000)")


# In[441]:


build_rank_dendrogram('./data/AAA_DCIS/All_epithelial/up20/', top_n=20, title="All DCIS Epithelial (DD) subcluster similarity (top 20)")


# In[442]:


build_rank_dendrogram('./data/AAA_DCIS/All_epithelial/up30/', top_n=30, title="All DCIS Epithelial (DD) subcluster similarity (top 30)")


# In[443]:


build_rank_dendrogram('./data/AAA_DCIS/All_epithelial/up100/', top_n=100, title="All DCIS Epithelial (DD) subcluster similarity (top 100)")


# ## Rank Dendrograms Fresh DN

# In[448]:


build_rank_dendrogram('./BDN/up100/', top_n=20, title="Fresh (DN) Basal subcluster similarity (top 20)")


# In[449]:


build_rank_dendrogram('./BDN/up100/', top_n=30, title="Fresh (DN) Basal subcluster similarity (top 30)")


# In[450]:


build_rank_dendrogram('./BDN/up100/', top_n=100, title="Fresh (DN) Basal subcluster similarity (top 100)")


# In[46]:


build_rank_dendrogram('./data/AAA_DCIS/BDN/up1000/', top_n=1000, title="Fresh (DN) Basal subcluster similarity (top 1000)")


# In[451]:


build_rank_dendrogram('./LMDN/up100/', top_n=20, title="Fresh (DN) Luminal Mature subcluster similarity (top 20)")


# In[452]:


build_rank_dendrogram('./LMDN/up100/', top_n=30, title="Fresh (DN) Luminal Mature subcluster similarity (top 30)")


# In[453]:


build_rank_dendrogram('./LMDN/up100/', top_n=100, title="Fresh (DN) Luminal Mature subcluster similarity (top 100)")


# In[50]:


build_rank_dendrogram('./data/AAA_DCIS/LMDN/up1000/', top_n=1000, title="Fresh (DN) Luminal Mature subcluster similarity (top 1000)")


# In[454]:


build_rank_dendrogram('./LPDN/up100/', top_n=20, title="Fresh (DN) Luminal Progenitor subcluster similarity (top 20)")


# In[455]:


build_rank_dendrogram('./LPDN/up100/', top_n=30, title="Fresh (DN) Luminal Progenitor subcluster similarity (top 30)")


# In[456]:


build_rank_dendrogram('./LPDN/up100/', top_n=100, title="Fresh (DN) Luminal Progenitor subcluster similarity (top 100)")


# In[54]:


build_rank_dendrogram('./data/AAA_DCIS/LPDN/up1000/', top_n=1000, title="Fresh (DN) Luminal Progenitor subcluster similarity (top 1000)")


# ## Rank Dendrograms Fresh & FFPE DD

# In[475]:


build_rank_dendrogram('./data/AAA_DCIS/BDD_FFPE/up20/', top_n=20, title="Fresh & FFPE DCIS (DD) Basal subcluster similarity (top 20)")


# In[473]:


build_rank_dendrogram('./data/AAA_DCIS/BDD_FFPE/up30/', top_n=30, title="Fresh & FFPE DCIS (DD) Basal subcluster similarity (top 30)")


# In[474]:


build_rank_dendrogram('./data/AAA_DCIS/BDD_FFPE/up100/', top_n=100, title="Fresh & FFPE DCIS (DD) Basal subcluster similarity (top 100)")


# In[472]:


build_rank_dendrogram('./data/AAA_DCIS/BDD_FFPE/up1000/', top_n=1000, title="Fresh & FFPE DCIS (DD) Basal subcluster similarity (top 1000)")


# In[477]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD_FFPE/up20/', top_n=20, title="Fresh & FFPE DCIS (DD) Luminal Mature subcluster similarity (top 20)")


# In[478]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD_FFPE/up30/', top_n=30, title="Fresh & FFPE DCIS (DD) Luminal Mature subcluster similarity (top 30)")


# In[479]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD_FFPE/up100/', top_n=100, title="Fresh & FFPE DCIS (DD) Luminal Mature subcluster similarity (top 100)")


# In[480]:


build_rank_dendrogram('./data/AAA_DCIS/LMDD_FFPE/up1000/', top_n=1000, title="Fresh & FFPE DCIS (DD) Luminal Mature subcluster similarity (top 1000)")


# In[481]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD_FFPE/up20/', top_n=20, title="Fresh & FFPE DCIS (DD) Luminal Progenitor subcluster similarity (top 20)")


# In[482]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD_FFPE/up20/', top_n=30, title="Fresh & FFPE DCIS (DD) Luminal Progenitor subcluster similarity (top 30)")


# In[483]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD_FFPE/up100/', top_n=100, title="Fresh & FFPE DCIS (DD) Luminal Progenitor subcluster similarity (top 100)")


# In[484]:


build_rank_dendrogram('./data/AAA_DCIS/LPDD_FFPE/up1000/', top_n=1000, title="Fresh & FFPE DCIS (DD) Luminal Progenitor subcluster similarity (top 1000)")


# ## Rank Dendrograms Fresh & FFPE DN

# In[485]:


build_rank_dendrogram('./data/AAA_DCIS/BDN_FFPE/up20/', top_n=20, title="Fresh & FFPE Normal (DN) Basal subcluster similarity (top 20)")


# In[486]:


build_rank_dendrogram('./data/AAA_DCIS/BDN_FFPE/up30/', top_n=30, title="Fresh & FFPE Normal (DN) Basal subcluster similarity (top 30)")


# In[487]:


build_rank_dendrogram('./data/AAA_DCIS/BDN_FFPE/up100/', top_n=100, title="Fresh & FFPE Normal (DN) Basal subcluster similarity (top 100)")


# In[488]:


build_rank_dendrogram('./data/AAA_DCIS/BDN_FFPE/up1000/', top_n=1000, title="Fresh & FFPE Normal (DN) Basal subcluster similarity (top 1000)")


# In[489]:


build_rank_dendrogram('./data/AAA_DCIS/LMDN_FFPE/up20/', top_n=20, title="Fresh & FFPE Normal (DN) Luminal Mature subcluster similarity (top 20)")


# In[490]:


build_rank_dendrogram('./data/AAA_DCIS/LMDN_FFPE/up30/', top_n=30, title="Fresh & FFPE Normal (DN) Luminal Mature subcluster similarity (top 30)")


# In[491]:


build_rank_dendrogram('./data/AAA_DCIS/LMDN_FFPE/up100/', top_n=100, title="Fresh & FFPE Normal (DN) Luminal Mature subcluster similarity (top 100)")


# In[492]:


build_rank_dendrogram('./data/AAA_DCIS/LMDN_FFPE/up20/', top_n=1000, title="Fresh & FFPE Normal (DN) Luminal Mature subcluster similarity (top 1000)")


# In[493]:


build_rank_dendrogram('./data/AAA_DCIS/LPDN_FFPE/up20/', top_n=20, title="Fresh & FFPE Normal (DN) Luminal Progenitor subcluster similarity (top 20)")


# In[494]:


build_rank_dendrogram('./data/AAA_DCIS/LPDN_FFPE/up30/', top_n=30, title="Fresh & FFPE Normal (DN) Luminal Progenitor subcluster similarity (top 30)")


# In[495]:


build_rank_dendrogram('./data/AAA_DCIS/LPDN_FFPE/up100/', top_n=100, title="Fresh & FFPE Normal (DN) Luminal Progenitor subcluster similarity (top 100)")


# In[ ]:


build_rank_dendrogram('./data/AAA_DCIS/LPDN_FFPE/up20/', top_n=20, title="Fresh & FFPE Normal (DN) Luminal Progenitor subcluster similarity (top 1000)")


# In[ ]:





# In[ ]:





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


# In[408]:


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
    "Group1": ["up100_T2022_BDD_1.csv", "up100_Q2025_BDD_0.csv"],
    "Group2": ["up100_T2022_BDD_2.csv", "up100_Q2025_BDD_2.csv"]}

# Example directories (make sure to adjust these to your file locations)
data_dir = './BDD/up100/'
output_dir = './BDD/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[398]:


#GSEApy
import gseapy as gp
import matplotlib.pyplot as plt


# In[ ]:





# In[399]:


BDD_1_gene_list = ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB']
BDD_2_gene_list = ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN']


# In[404]:


enr = gp.enrichr(gene_list=BDD_2_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )


# In[405]:


from gseapy import barplot, dotplot
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=20,
              top_term=10,
              figsize=(3, 5),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )


# In[406]:


ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=5,
              figsize=(3,5),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )


# In[403]:


ax = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=25, figsize=(4,5))


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


# In[432]:


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


# In[447]:


enr = gp.enrichr(gene_list=LMDD_9_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
#ax1.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_2_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
#ax2.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_2_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
#ax3.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_2_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


ax4 = dotplot(enr.res2d, title='MSigDB_Hallmark_2020',cmap='viridis_r', size=40, figsize=(4, 7))
#ax4.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_2_MSigDB_Hallmark_2020_dotplot.png", dpi=300, bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[ ]:





# In[789]:


enr = gp.enrichr(gene_list=LMDD_3_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_3_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_3_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_3_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# In[790]:


enr.results.head()


# In[800]:


enr = gp.enrichr(gene_list=LMDD_5_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_5_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_5_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_5_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# In[797]:


enr = gp.enrichr(gene_list=LMDD_7_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_7_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_7_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_7_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# In[796]:


enr = gp.enrichr(gene_list=LMDD_8_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_8_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_8_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_8_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# In[809]:


enr = gp.enrichr(gene_list=LMDD_9_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_9_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_9_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD/GSEApy/LMDD_9_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


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


# In[444]:


LPDD_1_gene_list = ['SON', 'RBM39', 'GTF2I']
LPDD_2_gene_list = ['MACF1', 'SON', 'GTF2I', 'CD46', 'RSRP1', 'NEAT1', 'ARID4B', 'RBM25', 'RBM39', 'PTPRK', 'ARID1B']
LPDD_3_gene_list = ['HLA-DRA', 'RPLP1', 'HLA-DPA1', 'FTH1', 'RPS3A', 'HLA-DQA1', 'RPL41', 'CD74', 'FTL', 'RPS3']


# In[445]:


enr = gp.enrichr(gene_list=LPDD_2_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
#ax1.get_figure().savefig("./data/AAA_DCIS/LPDD/GSEApy/LPDD_1_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
#ax2.get_figure().savefig("./data/AAA_DCIS/LPDD/GSEApy/LPDD_1_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
#ax3.get_figure().savefig("./data/AAA_DCIS/LPDD/GSEApy/LPDD_1_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# ## DD LM (FFPE) 

# In[862]:


input_folder = './LMDD_FFPE/up100/'
input_folder_up1000 = './data/AAA_DCIS/LMDD_FFPE/up1000/'
output_folder = './data/AAA_DCIS/LMDD_FFPE/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group11": ["up100_N2025_LMDD_9.csv", "up100_T2022_LMDD_0.csv", "up100_T2022_LMDD_5.csv"],
    "Group10": ["up100_N2025_LMDD_0.csv", "up100_Q2025_LMDD_7.csv"]
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


# In[861]:


sample_groups = {
    "Group11": ["up100_N2025_LMDD_9.csv", "up100_T2022_LMDD_0.csv", "up100_T2022_LMDD_5.csv"],
    "Group10": ["up100_N2025_LMDD_0.csv", "up100_Q2025_LMDD_7.csv"]
}
# Example directories (make sure to adjust these to your file locations)
data_dir = './data/AAA_DCIS/LMDD_FFPE/up100/'
output_dir = './data/AAA_DCIS/LMDD_FFPE/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[864]:


LMDD_11_gene_list = ['CD74']
LMDD_10_gene_list = ['REEP5', 'BCAM', 'MGP', 'AZGP1', 'SLC38A1']


# In[868]:


enr = gp.enrichr(gene_list=LMDD_10_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD_FFPE/GSEApy/LMDD_10_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD_FFPE/GSEApy/LMDD_10_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD_FFPE/GSEApy/LMDD_10_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# In[869]:


enr = gp.enrichr(gene_list=LMDD_11_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
ax1.get_figure().savefig("./data/AAA_DCIS/LMDD_FFPE/GSEApy/LMDD_11_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
ax2.get_figure().savefig("./data/AAA_DCIS/LMDD_FFPE/GSEApy/LMDD_11_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
ax3.get_figure().savefig("./data/AAA_DCIS/LMDD_FFPE/GSEApy/LMDD_11_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# ## DN Basal (Fresh)

# In[481]:


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


# In[482]:


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


# In[160]:


BDN_1_gene_list = ['ADIRF', 'MEF2C', 'ZEB2', 'ID3', 'SPARCL1', 'PTP4A3', 'COL4A1', 'CRISPLD2', 'NR2F2', 'RGS5', 'MAP1B', 'PTK2', 'KCNE4', 'NOTCH3', 'APOLD1', 'EPAS1', 'COL4A2', 'PGF', 'MCAM', 'KANK2', 'COL6A2', 'ITGA1', 'CRIP1', 'COL18A1', 'LGALS1', 'GJC1', 'PCOLCE', 'IGFBP7', 'UBA2', 'SPARC', 'FILIP1L', 'CD99']
BDN_2_gene_list = []
BDN_3_gene_list = []
BDN_4_gene_list = []
BDN_5_gene_list = []
BDN_6_gene_list = []
BDN_7_gene_list = []


# In[163]:


enr = gp.enrichr(gene_list=BDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
#ax1.get_figure().savefig("./data/AAA_DCIS/BDN/GSEApy/BDN_1_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
#ax2.get_figure().savefig("./data/AAA_DCIS/BDN/GSEApy/BDN_1_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
#ax3.get_figure().savefig("./data/AAA_DCIS/BDN/GSEApy/BDN_1_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


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


# In[462]:


output_dir = './LMDN/GO_Enrichment_Results/'
# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[174]:


LMDN_1_gene_list = ['KMT2A', 'GTF2I', 'PERP', 'ERBB3', 'TACSTD2']
LMDN_2_gene_list = []
LMDN_3_gene_list = []
LMDN_4_gene_list = []
LMDN_5_gene_list = []
LMDN_6_gene_list = []
LMDN_7_gene_list = []


# In[175]:


enr = gp.enrichr(gene_list=LMDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=40,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
#ax1.get_figure().savefig("./data/AAA_DCIS/LMDN/GSEApy/LMDN_1_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
#ax2.get_figure().savefig("./data/AAA_DCIS/LMDN/GSEApy/LMDN_1_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
#ax3.get_figure().savefig("./data/AAA_DCIS/LMDN/GSEApy/LMDN_1_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


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


# In[464]:


output_dir = './LPDN/GO_Enrichment_Results/'
# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(groups, input_folder, output_dir)


# In[185]:


LPDN_1_gene_list = ['IFI6', 'SCD', 'ISG15', 'PGK1', 'TMSB10', 'CDKN2A', 'HSPB1', 'BLVRB', 'FBLN1', 'S100A11']
LPDN_2_gene_list = []
LPDN_3_gene_list = []


# In[191]:


enr = gp.enrichr(gene_list=LPDN_1_gene_list, 
                 gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                 organism='human', 
                 outdir=None, # don't write to disk
                )
ax1 = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=25,
              top_term=10,
              figsize=(4, 7),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, 
              marker='o',
             )
#ax1.get_figure().savefig("./data/AAA_DCIS/LPDN/GSEApy/LPDN_1_KEGG_&_MSigDB_Human_dotplot.png", dpi=300, bbox_inches='tight')
ax2 = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=40,
              top_term=5,
              figsize=(4, 7),
              color = {'KEGG_2021_Human': 'salmon', 'MSigDB_Hallmark_2020':'darkblue'}
             )
#ax2.get_figure().savefig("./data/AAA_DCIS/LPDN/GSEApy/LPDN_1_KEGG_2021_Human_barplot.png", dpi=300, bbox_inches='tight')
ax3 = dotplot(enr.res2d, title='KEGG_2021_Human',cmap='viridis_r', size=40, figsize=(4, 7))
#ax3.get_figure().savefig("./data/AAA_DCIS/LPDN/GSEApy/LPDN_1_KEGG_2021_Human_dotplot.png", dpi=300, bbox_inches='tight')


# # Concatenate after Subpop Annotation

# In[407]:


adatas = [alldata_N2025, alldata_T2022, alldata_Q2025, alldata_W2022, alldata_G2017] 
combined = adatas[0].concatenate(
    *adatas[1:], 
    batch_key='dataset',        # creates 'dataset' column to track original source
    batch_categories=[f'D{i}' for i in range(len(adatas))],  # optional labels
    index_unique=None           # preserve original cell IDs (already unique)
)


# In[408]:


combined.obs['annotation'] = combined.obs['annotation'].astype('category')


# In[410]:


sc.pp.highly_variable_genes(combined, batch_key='dataset', n_top_genes=2000, flavor='seurat')
combined_hv = combined[:, combined.var['highly_variable']].copy()
sc.pp.scale(combined_hv)
sc.tl.pca(combined_hv, svd_solver='arpack')
sc.pp.neighbors(combined_hv, n_neighbors=15, n_pcs=50)
sc.tl.umap(combined_hv)
combined.obsm['X_umap'] = combined_hv.obsm['X_umap']
sc.pl.umap(combined, color='annotation', frameon=False)


# In[461]:


sc.tl.leiden(combined_hv, resolution = 1.0)
combined.obs['leiden'] = combined_hv.obs['leiden']  # copy clusters to full AnnData
sc.pl.umap(combined, color=['leiden', 'Batch', 'Sample'], ncols = 1)


# In[415]:


combined.obs['annotation'] = combined.obs['annotation'].replace('endothelial_like_&_angiogenic_LMDN', 'endothelial_like_&_angiogenic_LMDD')


# In[422]:


combined.obs['annotation'] = combined.obs['annotation'].replace('high_mitochondrial_gene_expressing_LMDN', 'highly_mitochondrial_gene_expressing_LMDN')


# In[418]:


combined.obs['annotation'] = combined.obs['annotation'].replace('stroma_regulating_LMDN', 'stroma_regulating_LMDD')


# In[420]:


combined.obs['annotation'] = combined.obs['annotation'].replace('oxidative_stress_responsive_LMDN', 'oxidative_stress_responsive_&_phospholipid_metabolising_LMDN')


# In[423]:


sc.pl.umap(combined, color='annotation', frameon=False)


# In[459]:


sc.tl.rank_genes_groups(combined, groupby="annotation", method="wilcoxon")


# In[460]:


sc.pl.rank_genes_groups_dotplot(combined, groupby="annotation", standard_scale="var", n_genes=5)


# In[468]:


combined.obs["annotation"].cat.categories


# In[467]:


marker_genes_subpops = {
    'ITGA10/LENG8+_BDN': ["X"],
    'MAPK_supressed_LMDD': ["X"],
    'MYC/mTOR_high/metabolic/proliferative_LMDN': ["X"],
    'adhesion_enriched/stressed_BDN': ["X"],
    'contractile/myofibroblast_like_BDN': ["X"],
    'differentiating/plastic_LMDD': ["X"],
    'endothelial_like_&_angiogenic_LMDD': ["X"],
    'estrogen_responsive/tumour_associated_LMDD': ["X"],
    'growth_factor_responsive_LMDN': ["X"],
    'high_mitochondria/stress_responsive_BDN': ["X"],
    'highly_mitochondrial_gene_expressing_LMDN': ["X"],
    'highly_biosynthesising/mTORC1_activated_LMDD': ["X"],
    'highly_biosynthesising_LMDD': ["X"],
    'highly_protein_synthesising/active_secretory_LMDN': ["X"],
    'immune_modulating/KRAS_activated/inflammatory_LMDD': ["X"],
    'immune_responsive_LPDD': ["X"],
    'interferon_responsive_BDD': ["X"],
    'lipid_metabolising_&_secreting_LMDN': ["X"],
    'luminal_basal_like_antigen_presenting/immune_interacting_BDN': ["X"],
    'mesenchymal_like_BDN': ["X"],
    'motile/structured_LPDD': ["X"],
    'oxidative_stress_responsive_&_phospholipid_metabolising_LMDN': ["X"],
    'phospholipid_metabolising_LMDN': ["X"],
    'proliferative/E2F_high_LPDN': ["X"],
    'ribosome_high/protein_synthesis_active_LPDN': ["X"],
    'stroma_regulating_LMDD': ["X"],
    'translation_primed_LPDD': ["X"],
    'X': ["X"],
    'X': ["X"]
}


# In[424]:


sc.pl.umap(combined, color='cell type', frameon=False)


# In[425]:


sc.pl.umap(combined, color='cnv_status', frameon=False)


# In[38]:


sc.pl.umap(combined, color='Batch', frameon=False)


# In[427]:


combined.write_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")


# In[36]:


combined = sc.read_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")


# In[37]:


combined.obs['Batch'] = combined.obs['Batch'].replace('G2017', 'G2021')
combined.obs['Sample'] = combined.obs['Sample'].replace('ind1_G2017', 'ind1_G2021')
combined.write_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")


# # scCODA

# In[1]:


# do in own env, need R etc dependancies


# In[13]:


cell_counts_N2025 = (
    alldata_N2025.obs
    .groupby(['Sample', 'cell type'])
    .size() 
    .unstack(fill_value=0)  
)
print(cell_counts_N2025.head())


# In[14]:


cell_counts_Q2025 = (
    alldata_Q2025.obs
    .groupby(['Sample', 'cell type'])
    .size() 
    .unstack(fill_value=0)  
)
cell_counts_T2022 = (
    alldata_T2022.obs
    .groupby(['Sample', 'cell type'])
    .size() 
    .unstack(fill_value=0)  
)
cell_counts_W2022 = (
    alldata_W2022.obs
    .groupby(['Sample', 'cell type'])
    .size() 
    .unstack(fill_value=0)  
)
cell_counts_G2017 = (
    alldata_G2017.obs
    .groupby(['Sample', 'cell type'])
    .size() 
    .unstack(fill_value=0)  
)


# In[19]:


adatas = [alldata_N2025, alldata_Q2025, alldata_T2022, alldata_W2022, alldata_G2017]

counts_list = []

for ad in adatas:
    counts = ad.obs.groupby(['Sample', 'cell type']).size().unstack(fill_value=0)
    counts_list.append(counts)
cell_counts = pd.concat(counts_list, axis=0)
print(cell_counts)


# In[34]:


cell_counts.to_csv("./data/AAA_DCIS/scCODA_cell_counts.csv")


# In[33]:


cell_counts = pd.read_csv("./data/AAA_DCIS/scCODA_cell_counts.csv")
cell_counts['Sample'] = cell_counts['Sample'].replace('ind1_G2017', 'ind1_G2021')


# In[39]:


# not using scCODA, just straight compositions
cell_counts = combined.obs.groupby(['Sample', 'cell type']).size().unstack(fill_value=0)
cell_counts.head()


# In[40]:


cell_props = cell_counts.div(cell_counts.sum(axis=1), axis=0)


# In[41]:


colors = plt.cm.tab20.colors  
cell_types = cell_props.columns
color_dict = {ct: colors[i % len(colors)] for i, ct in enumerate(cell_types)}


# In[42]:


cell_props['Batch'] = combined.obs.drop_duplicates('Sample').set_index('Sample')['Batch']

cell_props_sorted = cell_props.sort_values('Batch')
cell_props_sorted = cell_props_sorted.drop(columns='Batch') 


# In[43]:


celltype_colors = {
    "Basal": "#17becf",              # teal
    "Luminal Progenitor": "#e377c2", # pink
    "Luminal Mature": "#8c564b",     # brown
    "Endothelial": "#9467bd",        # purple
    "Fibroblast": "#2ca02c",         # green
    "General Myeloid": "#fee08b",    # yellow
    "T-Cell": "#bcbd22",             # olive
    "B-Cell": "#d62728",             # red
    "Macrophage": "#1f77b4",         # blue
    "Monocyte": "#ffbb78",           # light blue
}
color_dict = {ct: celltype_colors.get(ct, "#cccccc") for ct in cell_props.columns}

# plot stacked barplot
cell_props_sorted.plot(
    kind='bar',
    stacked=True,
    color=[color_dict[ct] for ct in cell_props.columns],
    figsize=(14,6)
)

plt.xlabel("Sample")
plt.ylabel("Fraction of cells")
plt.title("Cell Type Composition")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[44]:


epithelial_cells = ["Basal", "Luminal Progenitor", "Luminal Mature"]
epi_obs = combined.obs[combined.obs['cell type'].isin(epithelial_cells)].copy()
epi_obs['DCIS Cell Type'] = epi_obs['cell type'].astype(str) + ' ' + epi_obs['cnv_status'].str.capitalize()
cell_props = epi_obs.groupby(['Sample', 'DCIS Cell Type']).size().unstack(fill_value=0)
cell_props = cell_props.div(cell_props.sum(axis=1), axis=0)
celltype_colors = {
    "Basal Normal": "#17becf",              
    "Luminal Progenitor Normal": "#e377c2", 
    "Luminal Mature Normal": "#8c564b",     
    "Basal Dcis": "#0d727c",        
    "Luminal Progenitor Dcis": "#884774",         
    "Luminal Mature Dcis": "#54332d"   
}
color_dict = {ct: celltype_colors.get(ct, "#cccccc") for ct in cell_props.columns}
cell_props.plot(
    kind='bar', 
    stacked=True, 
    color=[color_dict[ct] for ct in cell_props.columns], 
    figsize=(15,6)
)
plt.xlabel("Sample")
plt.ylabel("Fraction of Epithelial Cells")
plt.title("Epithelial Cell Composition per sample (DCIS vs Normal)")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[113]:


lm_cells = alldata_Q2025[alldata_Q2025.obs['cell type'] == 'Luminal Mature']
lm_cells


# In[114]:


print(lm_cells.obs['cnv_status'].value_counts())
print(lm_cells.obs['Sample'].unique())


# In[117]:


bs = []  # store pseudo-bulk AnnData

for sample in lm_cells.obs['Sample'].unique():
    sample_subset = lm_cells[lm_cells.obs['Sample'] == sample].copy()
    
    for status in ['DCIS', 'normal']:
        status_subset = sample_subset[sample_subset.obs['cnv_status'] == status].copy()
        if status_subset.n_obs == 0:
            continue
        
        # Split into 2 pseudoreplicates
        indices = list(status_subset.obs_names)
        random.shuffle(indices)
        splits = np.array_split(np.array(indices), 2)
        
        for i, split in enumerate(splits):
            X_sum = status_subset[split].layers['counts'].sum(axis=0)
            
            # create a DataFrame first for obs
            obs_df = pd.DataFrame({
                'Sample': [sample],
                'cnv_status': [status],
                'replicate': [i]
            }, index=[f"{sample}_{status}_{i}"])
            
            rep_adata = sc.AnnData(
                X=np.array(X_sum).reshape(1, -1),
                var=status_subset.var.copy(),
                obs=obs_df
            )
            pbs.append(rep_adata)

# Concatenate all pseudobulks
pb = sc.concat(pbs, join='outer', label=None, index_unique=None)
print(pb.obs)


# In[118]:


counts = pd.DataFrame(pb.X.astype(int), columns=pb.var_names, index=pb.obs_names)
metadata = pb.obs[['Sample', 'cnv_status', 'replicate']].copy()

print(counts.shape)
print(metadata.shape)


# In[121]:


counts_int = counts.astype(int).copy()  # ensure int type
counts_int = pd.DataFrame(
    counts_int.values,  # make it dense numpy
    columns=counts_int.columns,
    index=counts_int.index
)

metadata_clean = metadata.copy()
for col in metadata_clean.columns:
    metadata_clean[col] = metadata_clean[col].astype(str)  # all strings are safe


# In[126]:


dds = DeseqDataSet(
    counts=counts_int,
    metadata=metadata_clean,
    design_factors=["cnv_status"]
)
sc.pp.filter_genes(dds, min_cells = 1)


# In[ ]:


dds = DeseqDataSet(
    counts=counts_int,
    metadata=metadata,
    design_factors=["cnv_status"],
    n_cpus=1
)
dds.deseq2()


# In[ ]:


sc.tl.pca(dds)
sc.pl.pca(dds, color = 'cnv_status', size = 200)


# In[ ]:


stat_res = DeseqStats(dds, n_cpus=8, contrast=('cnv_status', 'DCIS', 'normal'))
    
stat_res.summary()


# In[ ]:


de  = stat_res.results_df


# In[ ]:


de.sort_values('stat', ascending = False)


# In[ ]:




