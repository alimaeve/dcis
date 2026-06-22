#!/usr/bin/env python
# coding: utf-8

# # Add Metadata to anndata obs

# In[2]:


import numpy as np
import pandas as pd
from comut import comut
import scanpy as sc
import seaborn as sns
import warnings
import sys
import matplotlib.pyplot as plt


# In[2]:


study_metadata = {
    "ind1_Q2025": {"Patient":"1", "Study": "Q2025", "Type": "Fresh", "Parity": "Nulliparous", "Age_(range)": '<50', "Age_(years)": 30, "ER_status": "Pos", "BMI_(kg/m2)": 24.26},
    "ind3_Q2025": {"Patient":"2", "Study": "Q2025", "Type": "Fresh", "Parity": "Parous", "Age_(range)": '<50', "Age_(years)":35, "ER_status": "Neg", "BMI_(kg/m2)": 22.64},
    "ind2_Q2025": {"Patient":"3", "Study": "Q2025", "Type": "Fresh", "Parity": "Parous", "Age_(range)": '≥50', "Age_(years)": 61, "ER_status": "Neg", "BMI_(kg/m2)": 29.37},
    "ind13_Q2025": {"Patient":"4", "Study": "Q2025", "Type": "Fresh", "Parity": "Parous", "Age_(range)": '≥50',"Age_(years)": 53, "ER_status": "Neg", "BMI_(kg/m2)": 24.00},
    "ind14_Q2025": {"Patient":"4", "Study": "Q2025", "Type": "Fresh", "Parity": "Parous", "Age_(range)": '≥50',"Age_(years)": 53, "ER_status": "Neg", "BMI_(kg/m2)": 24.00},
    "ind15_Q2025": {"Patient":"5", "Study": "Q2025", "Type": "Fresh", "Parity": "Parous", "Age_(range)": '<50',"Age_(years)": 44, "ER_status": "Pos", "BMI_(kg/m2)": 24.55},
    "ind4_Q2025": {"Patient":"6", "Study": "Q2025", "Type": "Fresh", "Parity": "Parous", "Age_(range)": '<50',"Age_(years)": 47, "ER_status": "Pos", "BMI_(kg/m2)": 23.7},
    "ind5_Q2025": {"Patient":"7", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50',"Age_(years)": 75, "ER_status": "Neg", "BMI_(kg/m2)": 24.00},
    "ind6_Q2025": {"Patient":"7", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50',"Age_(years)": 75, "ER_status": "Neg", "BMI_(kg/m2)": 24.00},
    "ind7_Q2025": {"Patient":"7", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50', "Age_(years)": 75, "ER_status": "Neg", "BMI_(kg/m2)": 24.00},
    "ind8_Q2025": {"Patient":"7", "Study": "Q2025", "Type": "Fresh","Age_(range)": '≥50',  "Age_(years)": 75, "ER_status": "Neg", "BMI_(kg/m2)": 24.00},
    "ind9_Q2025": {"Patient":"8", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50',"Parity": "Parous", "Age_(years)": 58, "ER_status": "Pos"},
    "ind10_Q2025": {"Patient":"8", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50',"Parity": "Parous", "Age_(years)": 58, "ER_status": "Pos"},
    "ind11_Q2025": {"Patient":"9", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50',"Age_(years)": 57, "ER_status": "Pos", "BMI_(kg/m2)": 31.16},
    "ind12_Q2025": {"Patient":"9", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '≥50',"Age_(years)": 57, "ER_status": "Pos", "BMI_(kg/m2)": 31.16},
    "ind16_Q2025": {"Patient":"10", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Nulliparous", "Age_(years)": 24, "ER_status": "Pos", "BMI_(kg/m2)": 28.52},
    "ind17_Q2025": {"Patient":"11", "Study": "Q2025", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Parous", "Age_(years)": 47, "ER_status": "Neg", "BMI_(kg/m2)": 23.66},
    "ind1_T2022": {"Patient":"12", "Study": "T2022", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Nulliparous", "Age_(years)": 31, "ER_status": "Pos", "PR_status": "Neg", "Menopausal_status": "Pre", "HER2_status": 'Neg', "Ki67_(%)": 32.2},
    "ind2_T2022": {"Patient":"13", "Study": "T2022", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Nulliparous", "Age_(years)": 45, "ER_status": "Pos", "PR_status": "Pos", "Menopausal_status": "Pre", "HER2_status": 'Neg', "Ki67_(%)": 10.6},
    "ind3_T2022": {"Patient":"14", "Study": "T2022", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Parous", "Age_(years)": 44, "ER_status": "Pos", "PR_status": "Pos", "Menopausal_status": "Pre", "HER2_status": 'Pos', "Ki67_(%)": 65},
    "ind4_T2022": {"Patient":"15", "Study": "T2022", "Type": "Fresh", "Age_(range)": '≥50',"Parity": "Parous", "Age_(years)": 60, "ER_status": "Pos", "PR_status": "Neg", "Menopausal_status": "Post", "HER2_status": 'Pos', "Ki67_(%)": 40},
    "ind5_T2022": {"Patient":"16", "Study": "T2022", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Nulliparous", "Age_(years)": 43, "ER_status": "Pos", "PR_status": "Pos", "Menopausal_status": "Pre", "HER2_status": 'Pos', "Ki67_(%)": 10.6},
    "ind6_T2022": {"Patient":"17", "Study": "T2022", "Type": "Fresh", "Age_(range)": '≥50',"Parity": "Parous", "Age_(years)": 54, "ER_status": "Pos", "PR_status": "Pos", "Menopausal_status": "Pre", "HER2_status": 'Neg', "Ki67_(%)": 0.8},
    "ind7_T2022": {"Patient":"18", "Study": "T2022", "Type": "Fresh", "Age_(range)": '<50',"Parity": "Parous", "Age_(years)": 39, "ER_status": "Pos", "PR_status": "Neg", "Menopausal_status": "Pre", "HER2_status": 'Pos', "Ki67_(%)": 25.8},
    "ind1_W2022": {"Patient":"19", "Study": "W2022", "Type": "Fresh", "ER_status": "Pos", "PR_status": "Pos"},
    "ind2_W2022": {"Patient":"20", "Study": "W2022", "Type": "Fresh", "ER_status": "Pos", "PR_status": "Neg"},
    "ind1_G2021": {"Patient":"21", "Study": "G2017", "Type": "Fresh", },
    "ind1_N2025": {"Patient":"22", "Study": "N2025", "Type": "FFPE", "Age_(range)": '<50',"Age_(years)": 46, "ER_status": "Pos", "PR_status": "Pos", "HER2_status": 'Neg', "Ki67_(%)": 29},
    "ind2_N2025": {"Patient":"23", "Study": "N2025", "Type": "FFPE", "Age_(range)": '≥50',"Age_(years)": 58, "ER_status": "Neg", "PR_status": "Neg"},
    "ind3_N2025": {"Patient":"24", "Study": "N2025", "Type": "FFPE", "Age_(range)": '≥50',"Age_(years)": 76, "ER_status": "Pos", "PR_status": "Pos", "HER2_status": 'Neg', "Ki67_(%)": 77},
    "ind4_N2025": {"Patient":"25", "Study": "N2025", "Type": "FFPE", "Age_(range)": '<50',"Age_(years)": 41, "ER_status": "Pos", "PR_status": "Pos", "HER2_status": 'Pos'},
    "ind5_N2025": {"Patient":"26", "Study": "N2025", "Type": "FFPE", "Age_(range)": '<50',"Age_(years)": 46, "ER_status": "Neg", "PR_status": "Neg", "HER2_status": 'Neg'},
    "ind6_N2025": {"Patient":"27", "Study": "N2025", "Type": "FFPE", "Age_(range)": '≥50',"Age_(years)": 55, "ER_status": "Pos", "PR_status": "Pos"}
}


# In[3]:


metadata = pd.DataFrame.from_dict(study_metadata, orient='index')


# In[4]:


metadata = metadata.drop(columns = 'Age_(years)')


# In[5]:


metadata = metadata.drop(columns = 'Ki67_(%)')


# In[6]:


metadata = metadata.drop(columns = 'BMI_(kg/m2)')


# In[8]:


visual_category_columns = ['Study','Type','ER_status','PR_status','HER2_status','Menopausal_status','Patient', 'Age_(range)', 'Parity']

Name_Map = {
    'Study':'Study',
    'Type':'Sample Type',
    'ER_status':'ER Status',
    'PR_status':'PR Status',
    'HER2_status':'HER2 Status',
    'Menopausal_status':'Menopausal Status',
    'Age_(range)':'Age',
    'Parity':"Parity"
}


# In[9]:


comut_plot = comut.CoMut()


# In[10]:


metadata.index.name = "sample"
metadata = metadata.reset_index()


# In[11]:


metadata["sample"] = metadata["sample"].astype(str)
metadata["Patient"] = metadata["Patient"].astype(str)


# In[12]:


samples = metadata.sort_values(["Study", "sample"])["sample"].tolist()


# In[13]:


indicator_df = metadata[["sample", "Patient"]].copy()

# numeric group required by CoMut
indicator_df["group"] = pd.factorize(indicator_df["Patient"])[0]

indicator_df = indicator_df[["sample", "group"]]

# enforce sample order
indicator_df = indicator_df.set_index("sample").loc[samples].reset_index()


# In[14]:


samples = (
    metadata
    .sort_values(["Patient", "Study", "sample"])["sample"]
    .tolist()
)


# In[15]:


comut_plot.samples =samples


# In[16]:


indicator_kwargs = {
    "color": "black",
    "marker": "o",
    "linewidth": 1,
    "markersize": 5,
}

comut_plot.add_sample_indicators(
    indicator_df,
    name="Same patient",
    plot_kwargs=indicator_kwargs,
)


# In[19]:


metadata = metadata.fillna("Unknown")


# In[4]:


#colour palette
COLOR_PALETTE = {
    "Study": {'Q2025': 'steelblue','T2022': 'orange','W2022': 'yellow','G2017': 'lightgreen','N2025': 'purple'},    
    'ER_status': {"Pos": 'darkblue', "Neg": "#fed9b7", "Unknown":"#F8F8F8"},
    'Parity':{
        'Parous':'pink',
        'Nulliparous':'plum',
        "Unknown":"#F8F8F8"
    },
    'PR_status': {'Neg': '#006d77','Pos': '#e29578', "Unknown":"#F8F8F8"},
   'HER2_status': {
        'Neg': "#352208",
        'Pos': "#E1BB80",
        "Unknown":"#F8F8F8"},
    'Menopausal_status': {
        "Pre": "#6c8152",
        "Post": "#685634",
        "Unknown":"#F8F8F8"},
    'Type': {
        "Fresh": "salmon",
        "FFPE": 'darkred'},
    'Age_(range)': {
        '<50': 'tan',
        '≥50':'goldenrod',
        "Unknown":"#F8F8F8"},
    'cell type': {
        "Basal": "#1f77b4",# teal
        "Luminal Progenitor": "#ff7f0e",# pink
        "Luminal Mature": "#2ca02c",# brown
        "Endothelial": "#d62728",# purple
        "Fibroblast": "#9467bd",# green
        "General Myeloid": "#8c564b",# yellow
        "T-Cell": "#e377c2",# olive
        "B-Cell": "#bcbd22",# red
        "Macrophage": "#17becf",# blue
        "Monocyte": "#87CEEB"},
    'annotation_plot': {
        "Non-malignant Basal": "#118d9a", #teal
        "Unassigned malignant Basal": "#19cee1", #BDD na
        "Inflammatory_immune-responsive_EMT-high": "#6de3ef", #BDD
        "EMT/invasive-like_ECM-remodelling_contractile": "#c5f4f9", #BDD
        "Non-malignant Luminal Progenitor": "#9d2077", #pink
        "Unassigned malignant Luminal Progenitor": "#dc56b3", #LPDD na
        "Proliferative_basal-like": "#eda8d8", #LPDD
        "Antigen-presenting_protein-synthesising": "#fae9f5", #LPDD,
        "Non-malignant Luminal Mature": "#261714", #brown
        "Unassigned malignant Luminal Mature": "#593730", #LMDD na
        "Stress-adapted_ROS-high": "#7f4e44", #LMDD
        "Plastic_immune-modulating": "#a66659", #LMDD
        "Hormone-responsive_secretory_immune-modulating": "#b37d72", #LMDD
        "Proteostasis-active": "#c8a098", #LMDD
        "ECM-remodelling_EMT-primed/plastic": "#ddc4bf", #LMDD
        "Hormone-responsive_translationally_active": "#f8f3f2" #LMDD      
    }
}


# In[21]:


for col_name in visual_category_columns:
    
    if col_name not in COLOR_PALETTE:
        print(f"Skipping {col_name} (no palette)")
        continue

    cat_df = metadata.melt(
        id_vars=["sample"],
        value_vars=[col_name],
        var_name="category",
        value_name="value"
    )

    cat_df["value"] = cat_df["value"].astype(str)

    name = Name_Map.get(col_name, col_name)
    cat_df["category"] = name

    comut_plot.add_categorical_data(
        cat_df,
        name=name,
        mapping=COLOR_PALETTE[col_name]
    )


# In[22]:


comut_plot.plot_comut(
    
    x_padding=0.04,
    y_padding=0.04,
    tri_padding=0.03,
    wspace=0.3,
    hspace=0.03,
    fig = plt.figure(figsize=(10, 2), tight_layout=False,dpi=150)
)
borders = ['N/A']
border_white = ['N']
pt_map = metadata.set_index('sample')['Patient'].to_dict()
pt_label_raw = [ pt_map[lb.get_text()] for lb in comut_plot.axes['Same patient'].get_xticklabels()]
seen=''
pt_label =[]
for l in pt_label_raw:
    if l==seen:
        pt_label.append('')
    else:
        pt_label.append(l)
    seen = l
comut_plot.axes['Same patient'].set_xticklabels(pt_label,rotation=0,ha='left',va='top')
# comut_plot.axes['Mutation type'].set_xticklabels([])
comut_plot.add_unified_legend(
    bbox_to_anchor=(1.1, -18), frameon=False, border_white=border_white,
    ncol=7)
comut_plot.figure.savefig(
    "./data/AAA_DCIS/comut_plot.svg",
    dpi=200,
    bbox_inches="tight"
)
#with index, but with sample names horizontal


# In[23]:


fig = plt.figure(figsize=(10, 3), dpi=150, tight_layout=False)

comut_plot.plot_comut(
    x_padding=0.07,
    y_padding=0.04,
    tri_padding=0.03,
    wspace=0.3,
    hspace=0.03,
    fig=fig
)
comut_plot.figure.savefig(
    "./data/AAA_DCIS/comut_plot_no_index.svg",
    dpi=200,
    bbox_inches="tight"
)


# In[9]:


# cell type composition stacked bar plots
alldata_Q2025 = sc.read_h5ad('260203_dcis_Q2025_only.h5ad')
alldata_T2022 = sc.read_h5ad('260203_dcis_T2022_only.h5ad')
alldata_W2022 = sc.read_h5ad('260203_dcis_W2022_only.h5ad')
alldata_G2017 = sc.read_h5ad('260203_dcis_G2017_only.h5ad')
alldata_N2025 = sc.read_h5ad('260205_dcis_N2025_only.h5ad')


# In[4]:


# calculate tumour purity of G2021, to compare to study
counts = alldata_G2017.obs["cnv_status"].value_counts()

print("DCIS cells:", counts.get("DCIS", 0))
print("Normal cells:", counts.get("normal", 0))


# In[25]:


# calculate PAM50 cell number
alldata_N2025_6 = alldata_N2025[alldata_N2025.obs['Sample'].isin(['ind6_N2025'])]
counts_N_6 = alldata_N2025_6.obs["scSubtype"].value_counts()
counts_N_6


# In[43]:


adata_list = [
    alldata_Q2025,
    alldata_T2022,
    alldata_W2022,
    alldata_G2017,
    alldata_N2025
]

obs = pd.concat([a.obs.copy() for a in adata_list], axis=0)


# In[45]:


dcis_counts = (
    obs[obs["cnv_status"] == "DCIS"]
    .groupby("cell type")
    .size()
)

print(dcis_counts)


# In[34]:


dcis_counts = (
    obs[obs["cnv_status"] == "normal"]
    .groupby("cell type")
    .size()
)

print(dcis_counts)


# In[7]:


compartment_col = "cell type"  
ordered_compartment = ["Basal", "Luminal Mature", "Luminal Progenitor"]


# In[8]:


pct = (
    pd.crosstab(obs[compartment_col], obs["Sample"])
    .T
)

# convert to proportions
pct = pct.div(pct.sum(axis=1), axis=0)

# ensure correct column orders
pct = pct.reindex(columns=ordered_compartment, fill_value=0)


# In[9]:


pct = pct.sort_index()


# In[30]:


fig, ax = plt.subplots(1, 1, figsize=(4, 8), dpi=200)

pct.plot(
    kind="barh",
    stacked=True,
    color=[COLOR_PALETTE["cell type"][ct] for ct in ordered_compartment],
    width=0.8,
    ax=ax
)

ax.set_ylabel("")
ax.legend(loc=(1.01, 0), frameon=False)

ax.spines[["right", "top", "left"]].set_visible(False)


# In[12]:


fig, ax = plt.subplots(1, 1, figsize=(4, 8), dpi=200)

celltype_colors = [
    COLOR_PALETTE["cell type"][ct]
    for ct in pct.columns
]

pct.plot(
    kind="barh",
    stacked=True,
    color=celltype_colors,
    width=0.8,
    ax=ax
)
ax.set_ylabel("")
ax.legend(loc=(1.01, 0), frameon=False)

ax.spines[["right", "top", "left"]].set_visible(False)


# In[31]:


epithelial_types = ["Basal", "Luminal Mature", "Luminal Progenitor"]


# In[13]:


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


# In[14]:


pct = pct.reindex(custom_order)


# In[15]:


obs["Sample"] = pd.Categorical(
    obs["Sample"],
    categories=custom_order,
    ordered=True
)


# In[16]:


obs = obs.sort_values("Sample")


# In[17]:


comut_plot.samples = custom_order


# In[19]:


#samples = comut_plot.samples 
samples = list(pct.index)  # fallback (but CoMut order is better)


# In[25]:


fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)

pct.plot(
    kind="bar",
    stacked=True,
    color=celltype_colors,
    width=0.8,
    ax=ax
)
ax.set_ylabel("")
ax.legend(loc=(1.01, 0), frameon=False)

ax.spines[["top", "right"]].set_visible(False)

plt.xticks(rotation=90)


# In[38]:


epith = obs[obs["cell type"].isin(epithelial_types)].copy()

pct = (
    pd.crosstab(epith["Sample"], epith["cell type"])
    .reindex(samples)              # forces CoMut order
    .fillna(0)
)

# convert to percent per sample
pct = pct.div(pct.sum(axis=1), axis=0) * 100

# force column order
pct = pct.reindex(columns=epithelial_types, fill_value=0)


# In[39]:


fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)

pct.plot(
    kind="bar",
    stacked=True,
    color=[COLOR_PALETTE["cell type"][ct] for ct in epithelial_types],
    width=0.8,
    ax=ax
)

ax.set_ylabel("Epithelial composition (%)")
ax.set_xlabel("")

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  # puts legend outside
    frameon=False
)
ax.spines[["top", "right"]].set_visible(False)

plt.xticks(rotation=90)


# In[40]:


fig.savefig(
    "./data/AAA_DCIS/epithelial_composition.svg",
    bbox_inches="tight"
)


# In[26]:


#try to match bar widths
bar_width = 0.85


# In[42]:


fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)

x = np.arange(len(samples))  

bottom = np.zeros(len(samples))

for ct in epithelial_types:
    ax.bar(
        x,
        pct[ct].values,
        bottom=bottom,
        width=0.85,   # matches CoMut feel
        label=ct,
        color=COLOR_PALETTE["cell type"][ct]
    )
    bottom += pct[ct].values

ax.set_xticks(x)
ax.set_xticklabels(samples, rotation=90)

ax.set_ylabel("Epithelial composition (%)")
ax.set_xlabel("")
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


# In[43]:


# Subcluster annotation as new obs 'annotation_plot'
obs['annotation_plot'] = np.nan
non_epi_mask = obs['Epithelial_vs_NonEpithelial'] == 'Non-Epithelial'

obs.loc[non_epi_mask, 'annotation_plot'] = 'Non-Epithelial'
normal_epi = (
    (obs['Epithelial_vs_NonEpithelial'] == 'Epithelial') &
    (obs['cnv_status'] == 'normal')
)

obs.loc[
    normal_epi & (obs['cell type'] == 'Basal'),
    'annotation_plot'
] = 'Non-malignant Basal'

obs.loc[
    normal_epi & (obs['cell type'] == 'Luminal Progenitor'),
    'annotation_plot'
] = 'Non-malignant Luminal Progenitor'

obs.loc[
    normal_epi & (obs['cell type'] == 'Luminal Mature'),
    'annotation_plot'
] = 'Non-malignant Luminal Mature'

malignant_epi = (
    (obs['Epithelial_vs_NonEpithelial'] == 'Epithelial') &
    (obs['cnv_status'] == 'DCIS')
)

valid_annotation = (
    malignant_epi &
    (obs['annotation'].notna()) &
    (obs['annotation'] != 'na')
)

obs.loc[valid_annotation, 'annotation_plot'] = obs.loc[
    valid_annotation, 'annotation'
]
unassigned = (
    malignant_epi &
    (
        obs['annotation'].isna() |
        (obs['annotation'] == 'na')
    )
)

obs.loc[
    unassigned & (obs['cell type'] == 'Basal'),
    'annotation_plot'
] = 'Unassigned malignant Basal'

obs.loc[
    unassigned & (obs['cell type'] == 'Luminal Progenitor'),
    'annotation_plot'
] = 'Unassigned malignant Luminal Progenitor'

obs.loc[
    unassigned & (obs['cell type'] == 'Luminal Mature'),
    'annotation_plot'
] = 'Unassigned malignant Luminal Mature'


# In[44]:


obs.loc[obs['annotation_plot'] == 'unassigned', 'annotation_plot'] = np.nan


# In[45]:


obs['annotation_plot'].value_counts(dropna=False)


# In[46]:


compartment_col = "annotation_plot"   

ordered_compartment = ["Non-malignant Basal", "Unassigned malignant Basal", 
                       "Inflammatory_immune-responsive_EMT-high", "EMT/invasive-like_ECM-remodelling_contractile", 
                       "Non-malignant Luminal Progenitor", "Unassigned malignant Luminal Progenitor",
                       "Proliferative_basal-like", "Antigen-presenting_protein-synthesising",
                       "Non-malignant Luminal Mature", "Unassigned malignant Luminal Mature",
                       "Stress-adapted_ROS-high", "Plastic_immune-modulating", "Hormone-responsive_secretory_immune-modulating", 
                       "Proteostasis-active", "ECM-remodelling_EMT-primed/plastic", "Hormone-responsive_translationally_active"]

pct = (
    pd.crosstab(obs[compartment_col], obs["Sample"])
    .T
)

# convert to proportions
pct = pct.div(pct.sum(axis=1), axis=0)

# force correct column order
pct = pct.reindex(columns=ordered_compartment, fill_value=0)


# In[47]:


pct = pct.sort_index()


# In[48]:


fig, ax = plt.subplots(1, 1, figsize=(4, 8), dpi=200)

pct.plot(
    kind="barh",
    stacked=True,
    color=[COLOR_PALETTE["annotation_plot"][ct] for ct in ordered_compartment],
    width=0.8,
    ax=ax
)

ax.set_ylabel("")
ax.legend(loc=(1.01, 0), frameon=False)

ax.spines[["right", "top", "left"]].set_visible(False)


# In[50]:


pct = pct.reindex(custom_order)
obs["Sample"] = pd.Categorical(
    obs["Sample"],
    categories=custom_order,
    ordered=True
)
obs = obs.sort_values("Sample")
comut_plot.samples = custom_order


# In[51]:


samples = list(pct.index)  # fallback

fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)

pct.plot(
    kind="bar",
    stacked=True,
    color=[COLOR_PALETTE["annotation_plot"][ct] for ct in ordered_compartment],
    width=0.8,
    ax=ax
)

ax.set_ylabel("Epithelial composition (%)")
ax.set_xlabel("")

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  # pushes legend outside
    frameon=False
)
ax.spines[["top", "right"]].set_visible(False)

plt.xticks(rotation=90)
# In[52]:


bar_width = 0.85
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
        color=COLOR_PALETTE["annotation_plot"][ct]
    )
    bottom += pct[ct].values

ax.set_xticks(x)
ax.set_xticklabels(samples, rotation=90)

ax.set_ylabel("Epithelial composition (%)")
ax.set_xlabel("")
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


# In[53]:


fig.savefig(
    "./data/AAA_DCIS/epithelial_subtype_noN2025_composition.svg",
    bbox_inches="tight"
)

