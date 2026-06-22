#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import pandas as pd
import numpy as np


# In[2]:


pam50genes = pd.read_csv("NatGen_Supplementary_table_S4.csv")


# In[3]:


gene_sets = {
    col: pam50genes[col].dropna().tolist()
    for col in pam50genes.columns
}

gene_sets


# In[4]:


alldata_Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')
alldata_T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')
alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')
alldata_G2017 = sc.read_h5ad('260203_dcis_G2017_only.h5ad')
alldata_N2025 = sc.read_h5ad('260205_dcis_N2025_only.h5ad')


# In[5]:


def assign_scSubtype(adata, gene_sets, layer='counts'):
    
    X = adata.layers[layer] if layer else adata.X
    gene_names = adata.var_names
    
    scores = pd.DataFrame(index=adata.obs_names)

    for subtype, genes in gene_sets.items():
        
        valid_genes = [g for g in genes if g in gene_names]
        
        if len(valid_genes) == 0:
            print(f"Warning: no genes found for {subtype}")
            scores[subtype] = 0
            continue
        
        idx = [gene_names.get_loc(g) for g in valid_genes]
        
        # sparse-safe mean
        subtype_expr = X[:, idx]
        scores[subtype] = np.array(subtype_expr.mean(axis=1)).flatten()
    
    # assign a subtype
    adata.obs['scSubtype'] = scores.idxmax(axis=1)
    
    # store the scores
    for col in scores.columns:
        adata.obs[f"{col}_score"] = scores[col]
    
    return adata


# # Q2025

# In[9]:


DCIS_Q2025 = alldata_Q2025[alldata_Q2025.obs['cnv_status'] == 'DCIS'].copy()
DCIS_Q2025 = assign_scSubtype(DCIS_Q2025, gene_sets)


# In[10]:


alldata_Q2025.obs['scSubtype'] = 'Non-tumor'
alldata_Q2025.obs.loc[DCIS_Q2025.obs_names, 'scSubtype'] = DCIS_Q2025.obs['scSubtype']


# In[11]:


alldata_Q2025.obs['scSubtype'].value_counts()


# In[12]:


alldata_Q2025.obs.groupby(['Sample', 'scSubtype']).size().unstack(fill_value=0)


# In[13]:


alldata_Q2025.obs.groupby(['Sample', 'cnv_status']).size().unstack(fill_value=0)


# In[14]:


sc.pl.umap(alldata_Q2025, color='scSubtype')


# In[ ]:


alldata_Q2025.write_h5ad("260203_dcis_Q2025_only.h5ad")


# In[27]:


adata_list = [
    alldata_Q2025,
    alldata_T2022,
    alldata_W2022,
    alldata_G2017,
    alldata_N2025
]

obs = pd.concat([a.obs.copy() for a in adata_list], axis=0)


# In[28]:


compartment_col = "scSubtype"   
ordered_compartment = ["Basal_SC", "Her2E_SC", "LumA_SC", "LumB_SC", "Non-tumor"]


# In[29]:


pct = (
    pd.crosstab(obs[compartment_col], obs["Sample"])
    .T
)

# converts to proportions
pct = pct.div(pct.sum(axis=1), axis=0)

# ensures correct column order
pct = pct.reindex(columns=ordered_compartment, fill_value=0)


# In[30]:


alldata_Q2025.obs["scSubtype"]


# In[43]:


COLOR_PALETTE = {
    "scSubtype": {'Basal_SC': '#8da0cb','Her2E_SC': '#a6d854','LumA_SC': '#e78ac3','LumB_SC': '#66c2a5','Non-tumor': 'grey'}
}


# In[44]:


pct = pct.sort_index()


# In[45]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(4, 8), dpi=200)

pct.plot(
    kind="barh",
    stacked=True,
    color=[COLOR_PALETTE["scSubtype"][ct] for ct in ordered_compartment],
    width=0.8,
    ax=ax
)

ax.set_ylabel("")
ax.legend(loc=(1.01, 0), frameon=False)

ax.spines[["right", "top", "left"]].set_visible(False)


# In[46]:


custom_order = [
    "ind1_Q2025",
    "ind16_Q2025",
    "ind17_Q2025",
    "ind1_T2022",
    "ind2_T2022",
    "ind3_T2022",
    "ind4_T2022",
    "ind5_T2022",
    "ind6_T2022",
    "ind7_T2022",
    "ind1_W2022",
    "ind3_Q2025",
    "ind2_W2022",
    "ind1_G2021",
    "ind1_N2025",
    "ind2_N2025",
    "ind3_N2025",
    "ind4_N2025",
    "ind5_N2025",
    "ind6_N2025",
    "ind2_Q2025",
    "ind13_Q2025",
    "ind14_Q2025",
    "ind15_Q2025",
    "ind4_Q2025",
    "ind5_Q2025",
    "ind6_Q2025",
    "ind7_Q2025",
    "ind8_Q2025",
    "ind10_Q2025",
    "ind9_Q2025",
    "ind11_Q2025",
    "ind12_Q2025"
]


# In[47]:


pct = pct.reindex(custom_order)


# In[48]:


obs["Sample"] = pd.Categorical(
    obs["Sample"],
    categories=custom_order,
    ordered=True
)


# In[49]:


obs = obs.sort_values("Sample")


# In[50]:


samples = list(pct.index)


# In[51]:


fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)

pct.plot(
    kind="bar",
    stacked=True,
    color=[COLOR_PALETTE["scSubtype"][ct] for ct in ordered_compartment],
    width=0.8,
    ax=ax
)

ax.set_ylabel("PAM50 Molecular Composition (%)")
ax.set_xlabel("")

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5), 
    frameon=False
)
ax.spines[["top", "right"]].set_visible(False)

plt.xticks(rotation=90)


# In[52]:


bar_width = 0.85


# In[53]:


fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)

x = np.arange(len(samples)) 

bottom = np.zeros(len(samples))

for ct in ordered_compartment:
    ax.bar(
        x,
        pct[ct].values,
        bottom=bottom,
        width=0.85,   
        label=ct,
        color=COLOR_PALETTE["scSubtype"][ct]
    )
    bottom += pct[ct].values

ax.set_xticks(x)
ax.set_xticklabels(samples, rotation=90)

ax.set_ylabel("PAM50 Molecular Composition (%)")
ax.set_xlabel("")
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


# In[54]:


fig.savefig(
    "./data/AAA_DCIS/PAM50_Barplot.svg",
    bbox_inches="tight"
)


# # T2022

# In[15]:


DCIS_T2022 = alldata_T2022[alldata_T2022.obs['cnv_status'] == 'DCIS'].copy()
DCIS_T2022 = assign_scSubtype(DCIS_T2022, gene_sets)
alldata_T2022.obs['scSubtype'] = 'Non-tumor'
alldata_T2022.obs.loc[DCIS_T2022.obs_names, 'scSubtype'] = DCIS_T2022.obs['scSubtype']
alldata_T2022.obs['scSubtype'].value_counts()


# In[16]:


alldata_T2022.obs.groupby(['Sample', 'scSubtype']).size().unstack(fill_value=0)


# In[17]:


sc.pl.umap(alldata_T2022, color='scSubtype')


# In[ ]:


alldata_T2022.write_h5ad("260203_dcis_T2022_only.h5ad")


# # W2022

# In[18]:


DCIS_W2022 = alldata_W2022[alldata_W2022.obs['cnv_status'] == 'DCIS'].copy()
DCIS_W2022 = assign_scSubtype(DCIS_W2022, gene_sets)
alldata_W2022.obs['scSubtype'] = 'Non-tumor'
alldata_W2022.obs.loc[DCIS_W2022.obs_names, 'scSubtype'] = DCIS_W2022.obs['scSubtype']
alldata_W2022.obs['scSubtype'].value_counts()


# In[19]:


alldata_W2022.obs.groupby(['Sample', 'scSubtype']).size().unstack(fill_value=0)


# In[20]:


sc.pl.umap(alldata_W2022, color='scSubtype')


# In[35]:


alldata_W2022.write_h5ad("260203_dcis_W2022_only.h5ad")


# # G2017

# In[21]:


DCIS_G2017 = alldata_G2017[alldata_G2017.obs['cnv_status'] == 'DCIS'].copy()
DCIS_G2017 = assign_scSubtype(DCIS_G2017, gene_sets)
alldata_G2017.obs['scSubtype'] = 'Non-tumor'
alldata_G2017.obs.loc[DCIS_G2017.obs_names, 'scSubtype'] = DCIS_G2017.obs['scSubtype']
alldata_G2017.obs['scSubtype'].value_counts()


# In[22]:


alldata_G2017.obs.groupby(['Sample', 'scSubtype']).size().unstack(fill_value=0)


# In[23]:


sc.pl.umap(alldata_G2017, color='scSubtype')


# In[39]:


alldata_G2017.write_h5ad("260203_dcis_G2017_only.h5ad")


# # N2025

# In[6]:


alldata_N2025 = sc.read_h5ad('260205_dcis_N2025_only.h5ad')
DCIS_N2025 = alldata_N2025[alldata_N2025.obs['cnv_status'] == 'DCIS'].copy()
DCIS_N2025 = assign_scSubtype(DCIS_N2025, gene_sets)
alldata_N2025.obs['scSubtype'] = 'Non-tumor'
alldata_N2025.obs.loc[DCIS_N2025.obs_names, 'scSubtype'] = DCIS_N2025.obs['scSubtype']
alldata_N2025.obs['scSubtype'].value_counts()


# In[7]:


alldata_N2025.obs.groupby(['Sample', 'scSubtype']).size().unstack(fill_value=0)


# In[8]:


sc.pl.umap(alldata_N2025, color='scSubtype')


# In[9]:


alldata_N2025.write_h5ad("260205_dcis_N2025_only.h5ad")

