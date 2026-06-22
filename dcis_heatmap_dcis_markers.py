#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
sc.settings.set_figure_params(
    dpi=300,      
    dpi_save=600, 
    fontsize=12,
    frameon=True,
    vector_friendly=True
)


# In[2]:


BDD_inflam_imm_emt = ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB']
BDD_emt_inv_ecm_con = ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN']


# In[3]:


LMDD_str_ad_ros = ['GSTP1', 'NUDT8', 'PDLIM1', 'HEBP2', 'PPP1R14B', 'KRT7']
LMDD_plas_imm = ['PPP1R1B', 'ADIRF', 'H3F3A', 'SRP9', 'TMSB10', 'ORMDL3', 'CD24', 'MRPL45', 'S100A14', 'MIEN1']
LMDD_hor_sec_imm = ['KTN1', 'XBP1', 'FLNB', 'BCAM', 'ELOVL5', 'STC2', 'AHNAK', 'AZGP1', 'DPP7', 'CIRBP', 'CHPT1', 'NME3', 'RUNX1', 'TRPS1']
LMDD_proteost = ['CISD3', 'NDUFB9', 'RPL19', 'RPS26', 'PSMB3', 'UBL5', 'MDK']
LMDD_ecm_emt_pl = ['CST3', 'MGP', 'RPL34']
LMDD_hor_transl_act = ['RPL35A']


# In[4]:


LPDD_prolif_bas = ['SON', 'RBM39', 'GTF2I']
LPDD_ag_pres_prot_syn = ['HLA-DRA', 'RPLP1', 'HLA-DPA1', 'FTH1', 'RPS3A', 'HLA-DQA1', 'RPL41', 'CD74', 'FTL', 'RPS3']


# In[5]:


BDN_mesench = ['MEF2C', 'COL4A2', 'ID3', 'PGF', 'EPAS1', 'CRIP1', 'SPARCL1', 'CD99', 'COL18A1', 'GJC1', 'ITGA1', 'NOTCH3', 'CRISPLD2', 'NR2F2', 'SPARC', 'COL6A2', 'UBA2', 'COL4A1', 'ZEB2', 'KANK2', 'ADIRF', 'MCAM', 'IGFBP7', 'PCOLCE', 'PTP4A3', 'RGS5', 'FILIP1L', 'APOLD1', 'KCNE4', 'LGALS1', 'MAP1B', 'PTK2']
BDN_er_str_metab = ['HSPA5', 'APLP2', 'APOE', 'CTSB', 'HSP90AB1']
BDN_infl_ecm_rem = ['SDC4', 'KLF6']
BDN_infl_imm_apop_adh = ['LY6E', 'FXYD3', 'NFIB', 'CD9', 'IRX2', 'VAMP8', 'DDR1', 'JUP', 'SAA1', 'CRYAB', 'SFRP1', 'RCAN1', 'OCIAD2', 'KRT7', 'TNFSF10', 'TACSTD2', 'SPINT2', 'PERP']


# In[6]:


LMDN_horm_epig = ['ERBB3', 'GTF2I', 'KMT2A', 'PERP', 'TACSTD2']
LMDN_imm_mod_stem = ['CD46', 'APLP2', 'ALCAM', 'BZW1', 'MUC1', 'TBX3']
LMDN_imm_surv_inv = ['MT-CO3', 'MT-CO1', 'FBN1', 'EMP1', 'SPARCL1', 'AKAP12', 'GSN', 'SPARC', 'MT-ND1', 'PECAM1', 'TIMP3', 'CALD1', 'HLA-B', 'MT-CYB', 'MT-ATP6', 'VIM', 'MT-ND4', 'MT-ND3', 'CAV1', 'IGFBP7', 'FSTL1', 'HLA-A', 'HLA-E', 'MT-ND2', 'TCF4', 'GNG11', 'MT-CO2']
LMDN_infl_emt_ox = ['B2M', 'LYZ', 'RPL28', 'VIM', 'CST3', 'RPL12', 'SAT1', 'MARCKS']
LMDN_ifn_ag_pres = ['B2M', 'ACTG1', 'MZT2B', 'GPSM3', 'GAPDH']
LMDN_er_upr_str_sec = ['CAPS', 'XBP1', 'MIF', 'CITED4', 'KRT15']
LMDN_oxphos = ['RPS26', 'COX6B1', 'LY6E', 'UBL5', 'RPL23A', 'NDUFS5', 'CHCHD2', 'COPE', 'PSMD8', 'NDUFA4', 'PSMB3', 'GAPDH']


# In[7]:


LPDN_typ_1_grow_sup = ['PGK1', 'S100A11']
LPDN_redox_tum_prot = ['NDRG2', 'TOMM7', 'HINT1', 'EPB41L4A-AS1', 'PFDN5', 'PPA1', 'SNHG8', 'EIF3E', 'S100A1', 'PRDX2', 'BTF3', 'ZFAS1']


# In[8]:


marker_genes = {
    "BDD_inflam_imm_emt": ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB'],
    "BDD_emt_inv_ecm_con": ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN'],
    "LMDD_str_ad_ros": ['GSTP1', 'NUDT8', 'PDLIM1', 'HEBP2', 'PPP1R14B', 'KRT7'],
    "LMDD_plas_imm": ['PPP1R1B', 'ADIRF', 'H3F3A', 'SRP9', 'TMSB10', 'ORMDL3', 'CD24', 'MRPL45', 'S100A14', 'MIEN1'],
    "LMDD_hor_sec_imm": ['KTN1', 'XBP1', 'FLNB', 'BCAM', 'ELOVL5', 'STC2', 'AHNAK', 'AZGP1', 'DPP7', 'CIRBP', 'CHPT1', 'NME3', 'RUNX1', 'TRPS1'],
    "LMDD_proteost": ['CISD3', 'NDUFB9', 'RPL19', 'RPS26', 'PSMB3', 'UBL5', 'MDK'],
    "LMDD_ecm_emt_pl": ['CST3', 'MGP', 'RPL34'],
    "LMDD_hor_transl_act": ['RPL35A'],
    "LPDD_prolif_bas": ['SON', 'RBM39', 'GTF2I'],
    "LPDD_ag_pres_prot_syn": ['HLA-DRA', 'RPLP1', 'HLA-DPA1', 'FTH1', 'RPS3A', 'HLA-DQA1', 'RPL41', 'CD74', 'FTL', 'RPS3'],
    "BDN_mesench": ['MEF2C', 'COL4A2', 'ID3', 'PGF', 'EPAS1', 'CRIP1', 'SPARCL1', 'CD99', 'COL18A1', 'ITGA1', 'NOTCH3', 'CRISPLD2', 'NR2F2', 'SPARC', 'COL6A2', 'UBA2', 'COL4A1', 'ZEB2', 'KANK2', 'ADIRF', 'MCAM', 'IGFBP7', 'PCOLCE', 'PTP4A3', 'RGS5', 'FILIP1L', 'APOLD1', 'KCNE4', 'LGALS1', 'MAP1B', 'PTK2'], #'GJC1'
    "BDN_er_str_metab": ['HSPA5', 'APLP2', 'APOE', 'CTSB', 'HSP90AB1'],
    "BDN_infl_ecm_rem": ['SDC4', 'KLF6'],
    "BDN_infl_imm_apop_adh": ['LY6E', 'FXYD3', 'NFIB', 'CD9', 'IRX2', 'VAMP8', 'DDR1', 'JUP', 'SAA1', 'CRYAB', 'SFRP1', 'RCAN1', 'OCIAD2', 'KRT7', 'TNFSF10', 'TACSTD2', 'SPINT2', 'PERP'],
    "LMDN_horm_epig": ['ERBB3', 'GTF2I', 'KMT2A', 'PERP', 'TACSTD2'],
    "LMDN_imm_mod_stem": ['CD46', 'APLP2', 'ALCAM', 'BZW1', 'MUC1', 'TBX3'],
    "LMDN_imm_surv_inv": ['MT-CO3', 'MT-CO1', 'FBN1', 'EMP1', 'SPARCL1', 'AKAP12', 'GSN', 'SPARC', 'MT-ND1', 'PECAM1', 'TIMP3', 'CALD1', 'HLA-B', 'MT-CYB', 'MT-ATP6', 'VIM', 'MT-ND4', 'MT-ND3', 'CAV1', 'IGFBP7', 'FSTL1', 'HLA-A', 'HLA-E', 'MT-ND2', 'TCF4', 'GNG11', 'MT-CO2'],
    "LMDN_infl_emt_ox": ['B2M', 'LYZ', 'RPL28', 'VIM', 'CST3', 'RPL12', 'SAT1', 'MARCKS'],
    "LMDN_ifn_ag_pres": ['B2M', 'ACTG1', 'MZT2B', 'GPSM3', 'GAPDH'],
    "LMDN_er_upr_str_sec": ['CAPS', 'XBP1', 'MIF', 'CITED4', 'KRT15'],
    "LMDN_oxphos": ['RPS26', 'COX6B1', 'LY6E', 'UBL5', 'RPL23A', 'NDUFS5', 'CHCHD2', 'COPE', 'PSMD8', 'NDUFA4', 'PSMB3', 'GAPDH'],
    "LPDN_typ_1_grow_sup": ['PGK1', 'S100A11'],
    "LPDN_redox_tum_prot": ['NDRG2', 'TOMM7', 'HINT1', 'EPB41L4A-AS1', 'PFDN5', 'PPA1', 'SNHG8', 'EIF3E', 'S100A1', 'PRDX2', 'BTF3', 'ZFAS1']
}


# # Using combined data w/ annots added

# In[2]:


import numpy as np
import anndata


# In[3]:


combined = sc.read_h5ad("./data/AAA_DCIS/260220_dcis_combined_after_annot_subpops.h5ad")


# In[45]:


combined_basal = combined[combined.obs['cell type'] == 'Basal'].copy()
combined_basal = combined_basal[combined_basal.obs['annotation'] != 'unassigned'].copy()
combined_basal_DCIS = combined_basal[combined_basal.obs['cnv_status'] == 'DCIS'].copy()
combined_basal_norm = combined_basal[combined_basal.obs['cnv_status'] == 'normal'].copy()
marker_genes_basal = {
    "BDD_inflam_imm_emt": ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB'],
    "BDD_emt_inv_ecm_con": ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN'],
    "BDN_mesench": ['MEF2C', 'COL4A2', 'ID3', 'PGF', 'EPAS1', 'CRIP1', 'SPARCL1', 'CD99', 'COL18A1', 'ITGA1', 'NOTCH3', 'CRISPLD2', 'NR2F2', 'SPARC', 'COL6A2', 'UBA2', 'COL4A1', 'ZEB2', 'KANK2', 'ADIRF', 'MCAM', 'IGFBP7', 'PCOLCE', 'PTP4A3', 'RGS5', 'FILIP1L', 'APOLD1', 'KCNE4', 'LGALS1', 'MAP1B', 'PTK2'], #'GJC1'
    "BDN_er_str_metab": ['HSPA5', 'APLP2', 'APOE', 'CTSB', 'HSP90AB1'],
    "BDN_infl_ecm_rem": ['SDC4', 'KLF6'],
    "BDN_infl_imm_apop_adh": ['LY6E', 'FXYD3', 'NFIB', 'CD9', 'IRX2', 'VAMP8', 'DDR1', 'JUP', 'SAA1', 'CRYAB', 'SFRP1', 'RCAN1', 'OCIAD2', 'KRT7', 'TNFSF10', 'TACSTD2', 'SPINT2', 'PERP'],
}
marker_genes_basal_DCIS = {
    "BDD_inflam_imm_emt": ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB'],
    "BDD_emt_inv_ecm_con": ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN'],
}
marker_genes_basal_normal = {
    "BDN_mesench": ['MEF2C', 'COL4A2', 'ID3', 'PGF', 'EPAS1', 'CRIP1', 'SPARCL1', 'CD99', 'COL18A1', 'ITGA1', 'NOTCH3', 'CRISPLD2', 'NR2F2', 'SPARC', 'COL6A2', 'UBA2', 'COL4A1', 'ZEB2', 'KANK2', 'ADIRF', 'MCAM', 'IGFBP7', 'PCOLCE', 'PTP4A3', 'RGS5', 'FILIP1L', 'APOLD1', 'KCNE4', 'LGALS1', 'MAP1B', 'PTK2'], #'GJC1'
    "BDN_er_str_metab": ['HSPA5', 'APLP2', 'APOE', 'CTSB', 'HSP90AB1'],
    "BDN_infl_ecm_rem": ['SDC4', 'KLF6'],
    "BDN_infl_imm_apop_adh": ['LY6E', 'FXYD3', 'NFIB', 'CD9', 'IRX2', 'VAMP8', 'DDR1', 'JUP', 'SAA1', 'CRYAB', 'SFRP1', 'RCAN1', 'OCIAD2', 'KRT7', 'TNFSF10', 'TACSTD2', 'SPINT2', 'PERP'],
}


# In[46]:


sc.pl.dotplot(combined_basal, var_names=marker_genes_basal, groupby='annotation')


# In[47]:


#correctly order marker genes so heatmap has clearer diagonal shading 
marker_genes_basal = {
    "BDD_emt_inv_ecm_con": ['ADCY3', 'TCF4', 'CALD1', 'IGFBP4', 'ITGA1', 'SPARC', 'MFGE8', 'A2M', 'TAGLN', 'PDGFA', 'COL4A1', 'PLS3', 'GSN'],
    "BDD_inflam_imm_emt": ['CD46', 'SHROOM3', 'HOMER2', 'IQGAP1', 'SDC4', 'ITGB8', 'CCL28', 'WEE1', 'PLPP2', 'SLPI', 'CD55', 'ANPEP', 'PER2', 'SOX9', 'SON', 'EMP1', 'PNISR', 'SLC25A37', 'NFIB'],
    "BDN_er_str_metab": ['HSPA5', 'APLP2', 'APOE', 'CTSB', 'HSP90AB1'],
    "BDN_infl_ecm_rem": ['SDC4', 'KLF6'],
    "BDN_infl_imm_apop_adh": ['LY6E', 'FXYD3', 'NFIB', 'CD9', 'IRX2', 'VAMP8', 'DDR1', 'JUP', 'SAA1', 'CRYAB', 'SFRP1', 'RCAN1', 'OCIAD2', 'KRT7', 'TNFSF10', 'TACSTD2', 'SPINT2', 'PERP'],
    "BDN_mesench": ['MEF2C', 'COL4A2', 'ID3', 'PGF', 'EPAS1', 'CRIP1', 'SPARCL1', 'CD99', 'COL18A1', 'ITGA1', 'NOTCH3', 'CRISPLD2', 'NR2F2', 'SPARC', 'COL6A2', 'UBA2', 'COL4A1', 'ZEB2', 'KANK2', 'ADIRF', 'MCAM', 'IGFBP7', 'PCOLCE', 'PTP4A3', 'RGS5', 'FILIP1L', 'APOLD1', 'KCNE4', 'LGALS1', 'MAP1B', 'PTK2'], #'GJC1'
}
sc.tl.dendrogram(combined_basal, groupby = 'annotation')
sc.pl.matrixplot(
    combined_basal,
    marker_genes_basal,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="Scaled Gene\nExpression",
)


# In[11]:


sc.pl.dotplot(combined_basal_norm, var_names=marker_genes_basal_normal, groupby='annotation')


# In[17]:


sc.pl.matrixplot(
    combined_basal_norm,
    marker_genes_basal_normal,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="column scaled\nexpression",
)


# In[19]:


combined_lm = combined[combined.obs['cell type'] == 'Luminal Mature'].copy()
combined_lm_DCIS = combined_lm[combined_lm.obs['cnv_status'] == 'DCIS'].copy()
combined_lm_norm = combined_lm[combined_lm.obs['cnv_status'] == 'normal'].copy()
marker_genes_lm = {
    "LMDD_str_ad_ros": ['GSTP1', 'NUDT8', 'PDLIM1', 'HEBP2', 'PPP1R14B', 'KRT7'],
    "LMDD_plas_imm": ['PPP1R1B', 'ADIRF', 'H3F3A', 'SRP9', 'TMSB10', 'ORMDL3', 'CD24', 'MRPL45', 'S100A14', 'MIEN1'],
    "LMDD_hor_sec_imm": ['KTN1', 'XBP1', 'FLNB', 'BCAM', 'ELOVL5', 'STC2', 'AHNAK', 'AZGP1', 'DPP7', 'CIRBP', 'CHPT1', 'NME3', 'RUNX1', 'TRPS1'],
    "LMDD_proteost": ['CISD3', 'NDUFB9', 'RPL19', 'RPS26', 'PSMB3', 'UBL5', 'MDK'],
    "LMDD_ecm_emt_pl": ['CST3', 'MGP', 'RPL34'],
    "LMDD_hor_transl_act": ['RPL35A'],
    "LMDN_horm_epig": ['ERBB3', 'GTF2I', 'KMT2A', 'PERP', 'TACSTD2'],
    "LMDN_imm_mod_stem": ['CD46', 'APLP2', 'ALCAM', 'BZW1', 'MUC1', 'TBX3'],
    "LMDN_imm_surv_inv": ['MT-CO3', 'MT-CO1', 'FBN1', 'EMP1', 'SPARCL1', 'AKAP12', 'GSN', 'SPARC', 'MT-ND1', 'PECAM1', 'TIMP3', 'CALD1', 'HLA-B', 'MT-CYB', 'MT-ATP6', 'VIM', 'MT-ND4', 'MT-ND3', 'CAV1', 'IGFBP7', 'FSTL1', 'HLA-A', 'HLA-E', 'MT-ND2', 'TCF4', 'GNG11', 'MT-CO2'],
    "LMDN_infl_emt_ox": ['B2M', 'LYZ', 'RPL28', 'VIM', 'CST3', 'RPL12', 'SAT1', 'MARCKS'],
    "LMDN_ifn_ag_pres": ['B2M', 'ACTG1', 'MZT2B', 'GPSM3', 'GAPDH'],
    "LMDN_er_upr_str_sec": ['CAPS', 'XBP1', 'MIF', 'CITED4', 'KRT15'],
    "LMDN_oxphos": ['RPS26', 'COX6B1', 'LY6E', 'UBL5', 'RPL23A', 'NDUFS5', 'CHCHD2', 'COPE', 'PSMD8', 'NDUFA4', 'PSMB3', 'GAPDH'],
}
marker_genes_lm_dcis = {
    "LMDD_str_ad_ros": ['GSTP1', 'NUDT8', 'PDLIM1', 'HEBP2', 'PPP1R14B', 'KRT7'],
    "LMDD_plas_imm": ['PPP1R1B', 'ADIRF', 'H3F3A', 'SRP9', 'TMSB10', 'ORMDL3', 'CD24', 'MRPL45', 'S100A14', 'MIEN1'],
    "LMDD_hor_sec_imm": ['KTN1', 'XBP1', 'FLNB', 'BCAM', 'ELOVL5', 'STC2', 'AHNAK', 'AZGP1', 'DPP7', 'CIRBP', 'CHPT1', 'NME3', 'RUNX1', 'TRPS1'],
    "LMDD_proteost": ['CISD3', 'NDUFB9', 'RPL19', 'RPS26', 'PSMB3', 'UBL5', 'MDK'],
    "LMDD_ecm_emt_pl": ['CST3', 'MGP', 'RPL34'],
    "LMDD_hor_transl_act": ['RPL35A'],
}
marker_genes_lm_normal = {
    "LMDN_horm_epig": ['ERBB3', 'GTF2I', 'KMT2A', 'PERP', 'TACSTD2'],
    "LMDN_imm_mod_stem": ['CD46', 'APLP2', 'ALCAM', 'BZW1', 'MUC1', 'TBX3'],
    "LMDN_imm_surv_inv": ['MT-CO3', 'MT-CO1', 'FBN1', 'EMP1', 'SPARCL1', 'AKAP12', 'GSN', 'SPARC', 'MT-ND1', 'PECAM1', 'TIMP3', 'CALD1', 'HLA-B', 'MT-CYB', 'MT-ATP6', 'VIM', 'MT-ND4', 'MT-ND3', 'CAV1', 'IGFBP7', 'FSTL1', 'HLA-A', 'HLA-E', 'MT-ND2', 'TCF4', 'GNG11', 'MT-CO2'],
    "LMDN_infl_emt_ox": ['B2M', 'LYZ', 'RPL28', 'VIM', 'CST3', 'RPL12', 'SAT1', 'MARCKS'],
    "LMDN_ifn_ag_pres": ['B2M', 'ACTG1', 'MZT2B', 'GPSM3', 'GAPDH'],
    "LMDN_er_upr_str_sec": ['CAPS', 'XBP1', 'MIF', 'CITED4', 'KRT15'],
    "LMDN_oxphos": ['RPS26', 'COX6B1', 'LY6E', 'UBL5', 'RPL23A', 'NDUFS5', 'CHCHD2', 'COPE', 'PSMD8', 'NDUFA4', 'PSMB3', 'GAPDH'],
}


# In[20]:


sc.pl.dotplot(combined_lm, var_names=marker_genes_lm, groupby='annotation')


# In[43]:


marker_genes_lm = {
    "LMDD_str_ad_ros": ['GSTP1', 'NUDT8', 'PDLIM1', 'HEBP2', 'PPP1R14B', 'KRT7'],
    "LMDD_plas_imm": ['PPP1R1B', 'ADIRF', 'H3F3A', 'SRP9', 'TMSB10', 'ORMDL3', 'CD24', 'MRPL45', 'S100A14', 'MIEN1'],
    "LMDD_hor_sec_imm": ['KTN1', 'XBP1', 'FLNB', 'BCAM', 'ELOVL5', 'STC2', 'AHNAK', 'AZGP1', 'DPP7', 'CIRBP', 'CHPT1', 'NME3', 'RUNX1', 'TRPS1'],
    "LMDD_proteost": ['CISD3', 'NDUFB9', 'RPL19', 'RPS26', 'PSMB3', 'UBL5', 'MDK'],
    "LMDD_ecm_emt_pl": ['CST3', 'MGP', 'RPL34'],
    "LMDD_hor_transl_act": ['RPL35A'],
    "LMDN_horm_epig": ['ERBB3', 'GTF2I', 'KMT2A', 'PERP', 'TACSTD2'],
    "LMDN_imm_mod_stem": ['CD46', 'APLP2', 'ALCAM', 'BZW1', 'MUC1', 'TBX3'],
    "LMDN_imm_surv_inv": ['MT-CO3', 'MT-CO1', 'FBN1', 'EMP1', 'SPARCL1', 'AKAP12', 'GSN', 'SPARC', 'MT-ND1', 'PECAM1', 'TIMP3', 'CALD1', 'HLA-B', 'MT-CYB', 'MT-ATP6', 'VIM', 'MT-ND4', 'MT-ND3', 'CAV1', 'IGFBP7', 'FSTL1', 'HLA-A', 'HLA-E', 'MT-ND2', 'TCF4', 'GNG11', 'MT-CO2'],
    "LMDN_infl_emt_ox": ['B2M', 'LYZ', 'RPL28', 'VIM', 'CST3', 'RPL12', 'SAT1', 'MARCKS'],
    "LMDN_ifn_ag_pres": ['B2M', 'ACTG1', 'MZT2B', 'GPSM3', 'GAPDH'],
    "LMDN_er_upr_str_sec": ['CAPS', 'XBP1', 'MIF', 'CITED4', 'KRT15'],
    "LMDN_oxphos": ['RPS26', 'COX6B1', 'LY6E', 'UBL5', 'RPL23A', 'NDUFS5', 'CHCHD2', 'COPE', 'PSMD8', 'NDUFA4', 'PSMB3', 'GAPDH'],
}
sc.pl.matrixplot(
    combined_lm,
    marker_genes_lm,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="Scaled Gene Expression",
)


# In[24]:


sc.pl.dotplot(combined_lm_DCIS, var_names=marker_genes_lm_dcis, groupby='annotation')


# In[26]:


sc.pl.matrixplot(
    combined_lm_DCIS,
    marker_genes_lm_dcis,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="column scaled\nexpression",
)


# In[27]:


sc.pl.dotplot(combined_lm_norm, var_names=marker_genes_lm_normal, groupby='annotation')


# In[29]:


sc.pl.matrixplot(
    combined_lm_norm,
    marker_genes_lm_normal,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="column scaled\nexpression",
)


# In[31]:


combined_lp = combined[combined.obs['cell type'] == 'Luminal Progenitor'].copy()
combined_lp_DCIS = combined_lp[combined_lp.obs['cnv_status'] == 'DCIS'].copy()
combined_lp_norm = combined_lp[combined_lp.obs['cnv_status'] == 'normal'].copy()
marker_genes_lp = {
    "LPDD_prolif_bas": ['SON', 'RBM39', 'GTF2I'],
    "LPDD_ag_pres_prot_syn": ['HLA-DRA', 'RPLP1', 'HLA-DPA1', 'FTH1', 'RPS3A', 'HLA-DQA1', 'RPL41', 'CD74', 'FTL', 'RPS3'],
    "LPDN_typ_1_grow_sup": ['PGK1', 'S100A11'],
    "LPDN_redox_tum_prot": ['NDRG2', 'TOMM7', 'HINT1', 'EPB41L4A-AS1', 'PFDN5', 'PPA1', 'SNHG8', 'EIF3E', 'S100A1', 'PRDX2', 'BTF3', 'ZFAS1']
}
marker_genes_lp_dcis = {
    "LPDD_prolif_bas": ['SON', 'RBM39', 'GTF2I'],
    "LPDD_ag_pres_prot_syn": ['HLA-DRA', 'RPLP1', 'HLA-DPA1', 'FTH1', 'RPS3A', 'HLA-DQA1', 'RPL41', 'CD74', 'FTL', 'RPS3'],
}
marker_genes_lp_normal = {
    "LPDN_typ_1_grow_sup": ['PGK1', 'S100A11'],
    "LPDN_redox_tum_prot": ['NDRG2', 'TOMM7', 'HINT1', 'EPB41L4A-AS1', 'PFDN5', 'PPA1', 'SNHG8', 'EIF3E', 'S100A1', 'PRDX2', 'BTF3', 'ZFAS1']
}


# In[32]:


sc.pl.dotplot(combined_lp, var_names=marker_genes_lp, groupby='annotation')


# In[44]:


sc.pl.matrixplot(
    combined_lp,
    marker_genes_lp,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="Scaled Gene Expression",
)


# In[35]:


sc.pl.dotplot(combined_lp_DCIS, var_names=marker_genes_lp_dcis, groupby='annotation')


# In[37]:


sc.pl.matrixplot(
    combined_lp_DCIS,
    marker_genes_lp_dcis,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="column scaled\nexpression",
)


# In[38]:


sc.pl.dotplot(combined_lp_norm, var_names=marker_genes_lp_normal, groupby='annotation')


# In[39]:


sc.pl.matrixplot(
    combined_lp_norm,
    marker_genes_lp_normal,
    "annotation",
    dendrogram=False,
    cmap="RdPu",
    standard_scale="var",
    colorbar_title="column scaled\nexpression",
)

