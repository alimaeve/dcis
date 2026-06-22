#!/usr/bin/env python
# coding: utf-8

# In[3]:


import liana as li
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


from liana.method import singlecellsignalr, connectome, cellphonedb, natmi, logfc, cellchat, geometric_mean


# In[66]:


sc.settings.set_figure_params(
    dpi=300,      
    dpi_save=600, 
    fontsize=12,
    frameon=False,
    vector_friendly=True
)


# In[69]:


plt.rcParams['axes.grid'] = False


# # T2022: Test Run, all of T2022

# In[5]:


alldata_T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')


# In[9]:


alldata_T2022.obs['cellchat_celltypes'] = pd.NA


# In[11]:


# import all individual methods
from liana.method import singlecellsignalr, connectome, cellphonedb, natmi, logfc, cellchat, geometric_mean


# In[12]:


cellphonedb(alldata_T2022,
            groupby='cell type', 
            # NOTE by default the resource uses HUMAN gene symbols
            resource_name='consensus',
            expr_prop=0.1,
            verbose=True, key_added='cpdb_res')


# In[15]:


li.mt.rank_aggregate(alldata_T2022, 
                     groupby='cell type',
                     resource_name='consensus',
                     expr_prop=0.1,
                     verbose=True)


# In[16]:


li.pl.circle_plot(alldata_T2022,
                  groupby='cell type',
                  score_key='magnitude_rank',
                  inverse_score=True,
                  source_labels='Luminal Mature',
                  filter_fun=lambda x: x['specificity_rank'] <= 0.05,
                  pivot_mode='counts', # NOTE: this will simply count the interactions, 'mean' is also available
                  figure_size=(10, 10),
                  )


# In[21]:


li.pl.dotplot(adata = alldata_T2022, 
              colour='magnitude_rank',
              size='specificity_rank',
              inverse_size=True,
              inverse_colour=True,
              source_labels=['Luminal Mature', 'Luminal Progenitor', 'Basal', 'T-Cell', 'B-Cell', 'Monocyte', 'General Myeloid'],
              target_labels=['Luminal Mature', 'Luminal Progenitor', 'Basal', 'T-Cell', 'B-Cell', 'Monocyte', 'General Myeloid'],
              top_n=10, 
              orderby='magnitude_rank',
              orderby_ascending=True,
              figure_size= (20, 17)
             )


# # T2022: Focus on ind_3, LM, LMDD Proteostasis & LMDN OXPHOS

# In[6]:


T2022_ind_3 = alldata_T2022[alldata_T2022.obs['Sample'].isin(['ind3_T2022'])]


# In[8]:


T2022_ind_3 = T2022_ind_3.copy()
T2022_ind_3.obs['cellchat_labels'] = (
    T2022_ind_3.obs['cell type'].astype(str)
)
override_annotations = [
    'LMDN_OXPHOS_metabolic',
    'LMDD_Proteostasis-active'
]
mask = T2022_ind_3.obs['annotation'].isin(override_annotations)
T2022_ind_3.obs.loc[mask, 'cellchat_labels'] = (
    T2022_ind_3.obs.loc[mask, 'annotation'].astype(str)
)


# In[12]:


cellphonedb(T2022_ind_3,
            groupby='cellchat_labels', 
            resource_name='consensus',
            expr_prop=0.1,
            verbose=True, key_added='cpdb_res')


# In[14]:


T2022_ind_3.uns['cpdb_res'].to_csv(
    'T2022_ind_3_LM_cellphone_results.csv',
    index=False
)


# In[15]:


li.mt.rank_aggregate(T2022_ind_3, 
                     groupby='cellchat_labels',
                     resource_name='consensus',
                     expr_prop=0.1,
                     verbose=True)


# In[17]:


# save liana results table as CSV
T2022_ind_3.uns['liana_res'].to_csv(
    'T2022_ind_3_LM_liana_results.csv',
    index=False
)


# In[18]:


# save liana results as excel, order by cellphone_pval, only save <0.05
# filter significant interactions
liana_sig = T2022_ind_3.uns['liana_res'][
    T2022_ind_3.uns['liana_res']['cellphone_pvals'] < 0.05]
liana_sig = liana_sig.sort_values(by='cellphone_pvals')
liana_sig.to_excel(
    'T2022_ind_3_liana_significant.xlsx',
    index=False
)


# In[19]:


myplot2 = li.pl.dotplot(
    adata=T2022_ind_3,
    colour="magnitude_rank",
    size="specificity_rank",
    inverse_colour=True,  # we inverse sign since we want small p-values to have large sizes
    inverse_size=True,
    # only the cell types which we wish to plot
    source_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature', 'T-Cell', 'B-Cell', 'General Myeloid', 'Basal'],
    target_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature', 'T-Cell', 'B-Cell', 'General Myeloid', 'Basal'],
    # since the rank_aggregate can also be interpreted as a probability distribution
    # filtered according to how consistently highly-ranked is their specificity across the methods
    # filterby="specificity_rank",
    # filter_lambda=lambda x: x <= 0.05,
    filter_fun=lambda x: x["specificity_rank"] <= 0.05,
    orderby="magnitude_rank",
    orderby_ascending=True,  # prioritize those with lowest values
    top_n=20,  # and we want to keep only the top 20 interactions
    figure_size=(25, 9),
    size_range=(1, 6),
)
myplot2
myplot2.save('T2022_ind3_LM_plot_allcells.pdf')


# In[20]:


myplot3 = li.pl.dotplot(
    adata=T2022_ind_3,
    colour="magnitude_rank",
    size="specificity_rank",
    inverse_colour=True,  # we inverse sign since we want small p-values to have large sizes
    inverse_size=True,
    source_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature'],
    target_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature'],
    filter_fun=lambda x: x["specificity_rank"] <= 0.05,
    orderby="magnitude_rank",
    orderby_ascending=True,  # prioritize those with lowest values
    top_n=20,  # and we want to keep only the top 20 interactions
    figure_size=(13, 9),
    size_range=(1, 6),
)
myplot3
myplot3.save('T2022_ind3_LM_plot_just_LM.pdf')


# In[21]:


myplot3


# In[22]:


myplot2


# In[23]:


my_plot.save('T2022_ind3_LM_plot_top_mag_rank.pdf')


# In[24]:


liana_sig.to_csv(
    'T2022_ind_3_liana_significant.csv',
    index=False
)


# In[25]:


my_plot = li.pl.dotplot(adata = T2022_ind_3, 
                        colour='magnitude_rank',
                        inverse_colour=True,
                        size='specificity_rank',
                        inverse_size=True,
                        source_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature'],
                        target_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature'],
                        filter_fun=lambda x: x['specificity_rank'] <= 0.05,
                        figure_size=(13, 10),
                        orderby='magnitude_rank',
                        orderby_ascending=True,
                       )
my_plot
#0.4 is the smallest avaolable specificity rank? cant do 0.01 like in vignette


# In[26]:


li.pl.circle_plot(T2022_ind_3,
                  groupby='cellchat_labels',
                  score_key='magnitude_rank',
                  inverse_score=True,
                  source_labels=['LMDD_Proteostasis-active', 'LMDN_OXPHOS_metabolic', 'General Myeloid', 'Basal', 'B-Cell', 'T-Cell', 'Luminal Mature'],
                  filter_fun=lambda x: x['specificity_rank'] <= 0.05,
                  pivot_mode='mean', # NOTE: this will simply count the interactions, 'mean' is also available
                  figure_size=(10, 10),
                  )

#Note: Ind3_T2022 has a very small % basal cells, and no LP cells! Mainly LM


# In[27]:


li.pl.circle_plot(T2022_ind_3,
                  groupby='cellchat_labels',
                  score_key='magnitude_rank',
                  inverse_score=True,
                  source_labels=['LMDD_Proteostasis-active', 'LMDN_OXPHOS_metabolic', 'General Myeloid', 'Basal', 'B-Cell', 'T-Cell', 'Luminal Mature'],
                  filter_fun=lambda x: x['specificity_rank'] <= 0.05,
                  pivot_mode='counts', # NOTE: this will simply count the interactions, 'mean' is also available
                  figure_size=(10, 10),
                  )

#Note: Ind3_T2022 has a very small % basal cells, and no LP cells! Mainly LM


# In[28]:


li.pl.circle_plot(T2022_ind_3,
                  groupby='cellchat_labels',
                  score_key='magnitude_rank',
                  inverse_score=True,
                  source_labels='LMDN_OXPHOS_metabolic',
                  filter_fun=lambda x: x['specificity_rank'] <= 0.05,
                  pivot_mode='mean', # NOTE: 'counts' = this will simply count the interactions, 'mean' is also available
                  figure_size=(10, 10),
                  )


# In[95]:


def set_celltype_colors(adata, obs_key, color_dict, default="#cccccc"):
    if not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")

    cats = adata.obs[obs_key].cat.categories
    adata.uns[f"{obs_key}_colors"] = [
        color_dict.get(c, default) for c in cats
    ]

celltype_colors = {
    "Basal": "#1f77b4",# teal
    "Luminal Progenitor": "#ff7f0e",# pink
    "Luminal Mature": "#2ca02c",# brown
    "Endothelial": "#d62728",# purple
    "Fibroblast": "#9467bd",# green
    "General Myeloid": "#8c564b",# yellow
    "T-Cell": "#e377c2",# olive
    "B-Cell": "#bcbd22",# red
    "Macrophage": "#17becf",# blue
    "Monocyte": "#87CEEB" ,# light blue
    "LMDN_OXPHOS_metabolic": "#FF2290", #hotpink
    "LMDD_Proteostasis-active": "#39FF14" #neongreen
   
}
set_celltype_colors(T2022_ind_3, "cellchat_labels", celltype_colors)


# In[96]:


#Github: liana-py: Frequency Chord diagram and Heatmap in Python #85
import pycirclize
from pycirclize import Circos
#import pandas as pd
from pycirclize.parser import Matrix

# Extract liana_res
df1 = T2022_ind_3.uns['liana_res'][['source','target']].groupby(['source','target']).size().reset_index(name='counts')
df = df1.pivot(index='source',columns='target',values='counts')

matrix_data=df.values
row_names = list(df.index)
col_names = row_names
matrix_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)

# Define link_kws handler function to customize each link property
def link_kws_handler(from_label: str, to_label: str):
    ## Highlith Astro, CR, Microglia source (in example, change to my own)
    if from_label in ('LMDD_Proteostasis-active', 'LMDN_OXPHOS_metabolic', 'Luminal Mature'):
        # Set alpha, zorder values higher than other links for highlighting
        return dict(alpha=0.5, zorder=1.0)
    else:
     return dict(alpha=0.1, zorder=1.0)

# Initialize from matrix (Can also directly load tsv matrix file)
circos = Circos.initialize_from_matrix(
    matrix_df,
    space=3,
    cmap=celltype_colors,
    r_lim=(93, 100),
    ticks_interval=500,
    label_kws=dict(r=109, size=10, color="black"),
    link_kws=dict(direction=1, ec="black", lw=0.5),
    link_kws_handler=link_kws_handler,
)

#print(matrix_df)
fig = circos.plotfig()


# In[30]:


cellchat(T2022_ind_3,
            groupby='cellchat_labels', 
            resource_name='consensus',
            expr_prop=0.1,
            min_cells=5,
            verbose=True, key_added='cellchat_res')

T2022_ind_3.uns['cellchat_res'].head()


# In[31]:


T2022_ind_3.uns['cellchat_res'].to_csv(
    'T2022_ind_3_LM_cellchat_results.csv',
    index=False
)


# In[32]:


from pySankey.sankey import sankey
df1 = T2022_ind_3.uns['liana_res'][['source','target']].groupby(['source','target']).size().reset_index(name='counts')
sankey(
    left=df1["source"], right=df1["target"], 
    leftWeight= df1["counts"], rightWeight=df1["counts"], fontsize=10
)


# # Annotate LR pairs: Pathway Annotations

# In[39]:


import os
os.makedirs(os.path.expanduser("~/.cache/omnipathdb"), exist_ok=True)
os.makedirs(os.path.expanduser("~/.config"), exist_ok=True)
config_text = """
[https://omnipathdb.org]
cache_dir = ~/.cache/omnipathdb
license = academic
"""
with open(os.path.expanduser("~/.config/omnipathdb.ini"), "w") as f:
    f.write(config_text)


# In[40]:


import omnipath as op
import decoupler as dc


# In[41]:


progeny = dc.op.progeny(top=2500)
progeny.head()


# In[42]:


lr_pairs = li.resource.select_resource('consensus')


# In[43]:


lr_progeny = li.rs.generate_lr_geneset(lr_pairs, progeny, lr_sep="^")
lr_progeny.head()


# In[44]:


diseases = op.requests.Annotations.get(
    resources = ['DisGeNet']
    )


# In[45]:


diseases = diseases[['genesymbol', 'label', 'value']]
diseases = diseases.pivot_table(index='genesymbol',
                                columns='label', values='value',
                                aggfunc=lambda x: '; '.join(x)).reset_index()
diseases = diseases[['genesymbol', 'disease']]
diseases['disease'] = diseases['disease'].str.split('; ')
diseases = diseases.explode('disease')
lr_diseases = li.rs.generate_lr_geneset(lr_pairs, diseases, source='disease', target='genesymbol', weight=None, lr_sep="^")
lr_diseases.sort_values("interaction").head()


# In[46]:


T2022_ind_3.uns['liana_res'].head()


# In[47]:


liana_res = T2022_ind_3.uns['liana_res'].copy()
# Create interaction column
liana_res['interaction'] = (
    liana_res['ligand_complex'].astype(str)
    + "^" +
    liana_res['receptor_complex'].astype(str)
)


# In[48]:


mat = liana_res.pivot_table(
    index='source',
    columns='interaction',
    values='magnitude_rank',
    aggfunc='mean'
).fillna(0)


# In[49]:


lr_progeny = lr_progeny.rename(columns={
    'interaction': 'target'
})


# In[50]:


pathway_scores = dc.mt.mlm(
    data=mat,
    net=lr_progeny,
    verbose=True
)


# In[70]:


scores, pvals = pathway_scores
plt.figure(figsize=(8,6))
sns.heatmap(
    scores,
    cmap='coolwarm',
    center=0,
    linewidths=0,      
    linecolor=None
)
plt.title("T2022 Individual 3: Inferred Pathway Activity")
plt.tight_layout()
plt.show()


# # W2022 Individual 1 LM

# In[71]:


alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')
W2022_ind_1 = alldata_W2022[alldata_W2022.obs['Sample'].isin(['ind1_W2022'])]


# In[73]:


W2022_ind_1 = W2022_ind_1.copy()
W2022_ind_1.obs['cellchat_labels'] = (
    W2022_ind_1.obs['cell type'].astype(str)
)
override_annotations = [
    'LMDN_OXPHOS_metabolic',
    'LMDD_Proteostasis-active'
]
mask = W2022_ind_1.obs['annotation'].isin(override_annotations)
W2022_ind_1.obs.loc[mask, 'cellchat_labels'] = (
    W2022_ind_1.obs.loc[mask, 'annotation'].astype(str)
)


# In[74]:


cellphonedb(W2022_ind_1,
            groupby='cellchat_labels', 
            resource_name='consensus',
            expr_prop=0.1,
            verbose=True, key_added='cpdb_res')


# In[75]:


W2022_ind_1.uns['cpdb_res'].to_csv(
    'W2022_ind_1_LM_cellphone_results.csv',
    index=False
)


# In[76]:


li.mt.rank_aggregate(W2022_ind_1, 
                     groupby='cellchat_labels',
                     resource_name='consensus',
                     expr_prop=0.1,
                     verbose=True)


# In[77]:


liana_sig = W2022_ind_1.uns['liana_res'][
    W2022_ind_1.uns['liana_res']['cellphone_pvals'] < 0.05]
liana_sig = liana_sig.sort_values(by='cellphone_pvals')
liana_sig.to_excel(
    'W2022_ind_1_liana_significant.xlsx',
    index=False
)


# In[79]:


myplot2 = li.pl.dotplot(
    adata=W2022_ind_1,
    colour="magnitude_rank",
    size="specificity_rank",
    inverse_colour=True,  # we inverse sign since we want small p-values to have large sizes
    inverse_size=True,
    source_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature', 'T-Cell', 'General Myeloid', 'Basal'],
    target_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature', 'T-Cell', 'General Myeloid', 'Basal'],
    filter_fun=lambda x: x["specificity_rank"] <= 0.05,
    orderby="magnitude_rank",
    orderby_ascending=True,  # prioritize those with lowest values
    top_n=20,  # and we want to keep only the top 20 interactions
    figure_size=(25, 9),
    size_range=(1, 6),
)
myplot2
myplot2.save('W2022_ind1_LM_plot_allcells.pdf')


# In[80]:


myplot2


# In[81]:


myplot3 = li.pl.dotplot(
    adata=W2022_ind_1,
    colour="magnitude_rank",
    size="specificity_rank",
    inverse_colour=True,  # we inverse sign since we want small p-values to have large sizes
    inverse_size=True,
    source_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature'],
    target_labels=['LMDN_OXPHOS_metabolic', 'LMDD_Proteostasis-active', 'Luminal Mature'],
    filter_fun=lambda x: x["specificity_rank"] <= 0.05,
    orderby="magnitude_rank",
    orderby_ascending=True,  # prioritize those with lowest values
    top_n=20,  # and we want to keep only the top 20 interactions
    figure_size=(13, 9),
    size_range=(1, 6),
)
myplot3
myplot3.save('W2022_ind1_LM_plot_just_LM.pdf')


# In[97]:


df1 = W2022_ind_1.uns['liana_res'][['source','target']].groupby(['source','target']).size().reset_index(name='counts')
df = df1.pivot(index='source',columns='target',values='counts')

matrix_data=df.values
row_names = list(df.index)
col_names = row_names
matrix_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)

# Define link_kws handler function to customize each link property
def link_kws_handler(from_label: str, to_label: str):
    ## Highlith Astro, CR, Microglia source (in example, change to my own)
    if from_label in ('LMDD_Proteostasis-active', 'LMDN_OXPHOS_metabolic', 'Luminal Mature'):
        # Set alpha, zorder values higher than other links for highlighting
        return dict(alpha=0.5, zorder=1.0)
    else:
     return dict(alpha=0.1, zorder=1.0)

# Initialize from matrix (Can also directly load tsv matrix file)
circos = Circos.initialize_from_matrix(
    matrix_df,
    space=3,
    cmap=celltype_colors,
    r_lim=(93, 100),
    ticks_interval=2000,
    label_kws=dict(r=109, size=10, color="black"),
    link_kws=dict(direction=1, ec="black", lw=0.5),
    link_kws_handler=link_kws_handler,
)
fig = circos.plotfig()


# In[98]:


cellchat(W2022_ind_1,
            groupby='cellchat_labels', 
            resource_name='consensus',
            expr_prop=0.1,
            min_cells=5,
            verbose=True, key_added='cellchat_res')

W2022_ind_1.uns['cellchat_res'].head()
W2022_ind_1.uns['cellchat_res'].to_csv(
    'W2022_ind_1_LM_cellchat_results.csv',
    index=False
)


# In[99]:


liana_res = W2022_ind_1.uns['liana_res'].copy()
# Create interaction column
liana_res['interaction'] = (
    liana_res['ligand_complex'].astype(str)
    + "^" +
    liana_res['receptor_complex'].astype(str)
)


# In[100]:


mat = liana_res.pivot_table(
    index='source',
    columns='interaction',
    values='magnitude_rank',
    aggfunc='mean'
).fillna(0)


# In[101]:


lr_progeny = lr_progeny.rename(columns={
    'interaction': 'target'
})


# In[102]:


pathway_scores = dc.mt.mlm(
    data=mat,
    net=lr_progeny,
    verbose=True
)


# In[103]:


scores, pvals = pathway_scores
plt.figure(figsize=(8,6))
sns.heatmap(
    scores,
    cmap='coolwarm',
    center=0,
    linewidths=0,      
    linecolor=None
)
plt.title("W2022 Individual 1: Inferred Pathway Activity")
plt.tight_layout()
plt.show()

