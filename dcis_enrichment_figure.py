#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


base_dir = Path("./data/AAA_DCIS/final_ORA_results")

# Output Excel file
output_file = "merged_ORA_results.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for csv_file in base_dir.rglob("*.csv"):
        df = pd.read_csv(csv_file)
        folder_name = csv_file.parent.name
        file_name = csv_file.stem
        sheet_name = f"{file_name}"[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[110]:


excel_file = "./data/AAA_DCIS/final_ORA_results/BDD/BDD_selected_ORA_results.xlsx"
sheets = pd.read_excel(excel_file, sheet_name=None)
# Order pathways by strongest significance observed

all_data = []
for sheet_name, df in sheets.items():
    df = df.copy()

    # Overlap fraction
    df["Overlap_fraction"] = df["Overlap"].apply(
        lambda x: int(str(x).split("/")[0]) / int(str(x).split("/")[1])
    )

    # Significance
    df["neglog10_adjP"] = -np.log10(df["Adjusted P-value"])

    df["Sheet"] = sheet_name
    all_data.append(
        df[
            [
                "Sheet",
                "Gene_set",
                "Term",
                "Adjusted P-value",
                "Overlap_fraction",
                "neglog10_adjP"
            ]
        ]
    )
all_data = pd.concat(all_data, ignore_index=True)
priority_map = {
    "MSigDB_Hallmark_2020": 0,
    "KEGG_2021_Human": 1,
    "GO_Biological_Process_2025": 2,
}
all_data["GeneSet_priority"] = all_data["Gene_set"].apply(
    lambda x: next((v for k, v in priority_map.items() if k in str(x)), 3)
)
term_order = (
    all_data
    .groupby("Term", as_index=False)
    .agg({
        "GeneSet_priority": "min",      
        "neglog10_adjP": "max"          
    })
    .sort_values(
        ["GeneSet_priority", "neglog10_adjP"],
        ascending=[True, False]
    )["Term"]
    .tolist()
)
sheet_order = list(all_data["Sheet"].unique())

term_to_y = {
    term: i
    for i, term in enumerate(term_order)
}
sheet_to_x = {
    sheet: i
    for i, sheet in enumerate(sheet_order)
}
all_data["x"] = all_data["Sheet"].map(sheet_to_x)
all_data["y"] = all_data["Term"].map(term_to_y)

fig_height = max(
    8,
    len(term_order) * 0.35
)

fig, ax = plt.subplots(
    figsize=(3, fig_height))

# ============================================================
# Scatter plot
# ============================================================

sc = ax.scatter(
    all_data["x"],
    all_data["y"],
    s=300,                          # fixed bubble size
    c=all_data["neglog10_adjP"],    # colour = significance
    cmap="plasma",
    linewidths=0.3
)
ax.invert_yaxis()
# ============================================================
# Axes
# ============================================================

ax.set_xticks(range(len(sheet_order)))
ax.set_xticklabels(
    sheet_order,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(term_order)))
ax.set_yticklabels(term_order)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Over-Representation Analysis Comparison: DCIS Basal",
    fontsize=20,
    pad=15
)

cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.5,
    pad=0.02
)

cbar.set_label(
    r"$-\log_{10}(\mathrm{FDR})$",
    fontsize=12
)

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)

plt.tight_layout()
ax.set_xlim(-0.45, len(sheet_order) - 0.65)

plt.subplots_adjust(left=0.22, right=0.86)

plt.savefig(
    "./data/AAA_DCIS/final_ORA_results/plots/Selected_BDD_Fresh_ORA_bubbleplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[111]:


excel_file = "./data/AAA_DCIS/final_ORA_results/BDN/BDN_selected_ORA_results.xlsx"
sheets = pd.read_excel(excel_file, sheet_name=None)
# Order pathways by strongest significance observed

all_data = []
for sheet_name, df in sheets.items():
    df = df.copy()

    # Overlap fraction
    df["Overlap_fraction"] = df["Overlap"].apply(
        lambda x: int(str(x).split("/")[0]) / int(str(x).split("/")[1])
    )

    # Significance
    df["neglog10_adjP"] = -np.log10(df["Adjusted P-value"])

    df["Sheet"] = sheet_name
    all_data.append(
        df[
            [
                "Sheet",
                "Gene_set",
                "Term",
                "Adjusted P-value",
                "Overlap_fraction",
                "neglog10_adjP"
            ]
        ]
    )
all_data = pd.concat(all_data, ignore_index=True)
priority_map = {
    "MSigDB_Hallmark_2020": 0,
    "KEGG_2021_Human": 1,
    "GO_Biological_Process_2025": 2,
}
all_data["GeneSet_priority"] = all_data["Gene_set"].apply(
    lambda x: next((v for k, v in priority_map.items() if k in str(x)), 3)
)
term_order = (
    all_data
    .groupby("Term", as_index=False)
    .agg({
        "GeneSet_priority": "min",      
        "neglog10_adjP": "max"          
    })
    .sort_values(
        ["GeneSet_priority", "neglog10_adjP"],
        ascending=[True, False]
    )["Term"]
    .tolist()
)
sheet_order = list(all_data["Sheet"].unique())
term_to_y = {
    term: i
    for i, term in enumerate(term_order)
}
sheet_to_x = {
    sheet: i
    for i, sheet in enumerate(sheet_order)
}
all_data["x"] = all_data["Sheet"].map(sheet_to_x)
all_data["y"] = all_data["Term"].map(term_to_y)
fig_height = max(
    8,
    len(term_order) * 0.35
)

fig, ax = plt.subplots(
    figsize=(5, fig_height))

# ============================================================
# Scatter plot
# ============================================================

sc = ax.scatter(
    all_data["x"],
    all_data["y"],
    s=300,                          # fixed bubble size
    c=all_data["neglog10_adjP"],    # colour = significance
    cmap="plasma",
    linewidths=0.3
)
ax.invert_yaxis()

ax.set_xticks(range(len(sheet_order)))
ax.set_xticklabels(
    sheet_order,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(term_order)))
ax.set_yticklabels(term_order)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Over-Representation Analysis Comparison: Non-Malignant Basal",
    fontsize=20,
    pad=15
)


cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.5,
    pad=0.02
)

cbar.set_label(
    r"$-\log_{10}(\mathrm{FDR})$",
    fontsize=12
)


ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)

plt.tight_layout()
ax.set_xlim(-0.45, len(sheet_order) - 0.65)

plt.subplots_adjust(left=0.22, right=0.86)

plt.savefig(
    "./data/AAA_DCIS/final_ORA_results/plots/Selected_BDN_Fresh_ORA_bubbleplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[113]:


excel_file = "./data/AAA_DCIS/final_ORA_results/LMDD/LMDD_selected_ORA_results.xlsx"
sheets = pd.read_excel(excel_file, sheet_name=None)
# Order pathways by strongest significance observed

all_data = []
for sheet_name, df in sheets.items():
    df = df.copy()

    # Overlap fraction
    df["Overlap_fraction"] = df["Overlap"].apply(
        lambda x: int(str(x).split("/")[0]) / int(str(x).split("/")[1])
    )

    # Significance
    df["neglog10_adjP"] = -np.log10(df["Adjusted P-value"])

    df["Sheet"] = sheet_name
    all_data.append(
        df[
            [
                "Sheet",
                "Gene_set",
                "Term",
                "Adjusted P-value",
                "Overlap_fraction",
                "neglog10_adjP"
            ]
        ]
    )
all_data = pd.concat(all_data, ignore_index=True)
priority_map = {
    "MSigDB_Hallmark_2020": 0,
    "KEGG_2021_Human": 1,
    "GO_Biological_Process_2025": 2,
}
all_data["GeneSet_priority"] = all_data["Gene_set"].apply(
    lambda x: next((v for k, v in priority_map.items() if k in str(x)), 3)
)
term_order = (
    all_data
    .groupby("Term", as_index=False)
    .agg({
        "GeneSet_priority": "min",      
        "neglog10_adjP": "max"          
    })
    .sort_values(
        ["GeneSet_priority", "neglog10_adjP"],
        ascending=[True, False]
    )["Term"]
    .tolist()
)
sheet_order = list(all_data["Sheet"].unique())
term_to_y = {
    term: i
    for i, term in enumerate(term_order)
}
sheet_to_x = {
    sheet: i
    for i, sheet in enumerate(sheet_order)
}
all_data["x"] = all_data["Sheet"].map(sheet_to_x)
all_data["y"] = all_data["Term"].map(term_to_y)
fig_height = max(
    8,
    len(term_order) * 0.35
)

fig, ax = plt.subplots(
    figsize=(8, fig_height))

sc = ax.scatter(
    all_data["x"],
    all_data["y"],
    s=300,                          # fixed bubble size
    c=all_data["neglog10_adjP"],    # colour = significance
    cmap="plasma",
    linewidths=0.3
)
ax.invert_yaxis()

ax.set_xticks(range(len(sheet_order)))
ax.set_xticklabels(
    sheet_order,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(term_order)))
ax.set_yticklabels(term_order)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Over-Representation Analysis Comparison: DCIS Luminal Mature",
    fontsize=20,
    pad=15
)

cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.5,
    pad=0.02
)

cbar.set_label(
    r"$-\log_{10}(\mathrm{FDR})$",
    fontsize=12
)

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)

plt.tight_layout()
ax.set_xlim(-0.45, len(sheet_order) - 0.65)

plt.subplots_adjust(left=0.22, right=0.86)

plt.savefig(
    "./data/AAA_DCIS/final_ORA_results/plots/Selected_LMDD_Fresh_ORA_bubbleplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[10]:


excel_file = "./data/AAA_DCIS/final_ORA_results/LMDN/LMDN_selected_ORA_results.xlsx"
sheets = pd.read_excel(excel_file, sheet_name=None)
# Order pathways by strongest significance observed

all_data = []
for sheet_name, df in sheets.items():
    df = df.copy()

    # Overlap fraction
    df["Overlap_fraction"] = df["Overlap"].apply(
        lambda x: int(str(x).split("/")[0]) / int(str(x).split("/")[1])
    )

    # Significance
    df["neglog10_adjP"] = -np.log10(df["Adjusted P-value"])

    df["Sheet"] = sheet_name
    all_data.append(
        df[
            [
                "Sheet",
                "Gene_set",
                "Term",
                "Adjusted P-value",
                "Overlap_fraction",
                "neglog10_adjP"
            ]
        ]
    )
all_data = pd.concat(all_data, ignore_index=True)
priority_map = {
    "MSigDB_Hallmark_2020": 0,
    "KEGG_2021_Human": 1,
    "GO_Biological_Process_2025": 2,
}
all_data["GeneSet_priority"] = all_data["Gene_set"].apply(
    lambda x: next((v for k, v in priority_map.items() if k in str(x)), 3)
)
term_order = (
    all_data
    .groupby("Term", as_index=False)
    .agg({
        "GeneSet_priority": "min",      
        "neglog10_adjP": "max"          
    })
    .sort_values(
        ["GeneSet_priority", "neglog10_adjP"],
        ascending=[True, False]
    )["Term"]
    .tolist()
)
sheet_order = list(all_data["Sheet"].unique())
term_to_y = {
    term: i
    for i, term in enumerate(term_order)
}
sheet_to_x = {
    sheet: i
    for i, sheet in enumerate(sheet_order)
}
all_data["x"] = all_data["Sheet"].map(sheet_to_x)
all_data["y"] = all_data["Term"].map(term_to_y)

fig_height = max(
    8,
    len(term_order) * 0.35
)

fig, ax = plt.subplots(
    figsize=(8, fig_height))


sc = ax.scatter(
    all_data["x"],
    all_data["y"],
    s=300,                          # fixed bubble size
    c=all_data["neglog10_adjP"],    # colour = significance
    cmap="plasma",
    linewidths=0.3
)
ax.invert_yaxis()


ax.set_xticks(range(len(sheet_order)))
ax.set_xticklabels(
    sheet_order,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(term_order)))
ax.set_yticklabels(term_order)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Over-Representation Analysis Comparison: Non-Malignant Luminal Mature",
    fontsize=20,
    pad=15
)
cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.5,
    pad=0.02
)

cbar.set_label(
    r"$-\log_{10}(\mathrm{FDR})$",
    fontsize=12
)

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="y", labelsize=16)
ax.tick_params(axis="x", labelsize=15)

plt.tight_layout()
ax.set_xlim(-0.45, len(sheet_order) - 0.65)

plt.subplots_adjust(left=0.22, right=0.86)

plt.savefig(
    "./data/AAA_DCIS/final_ORA_results/plots/Selected_LMDN_Fresh_ORA_bubbleplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[117]:


excel_file = "./data/AAA_DCIS/final_ORA_results/LPDD/LPDD_selected_ORA_results.xlsx"
sheets = pd.read_excel(excel_file, sheet_name=None)
# Order pathways by strongest significance observed

all_data = []
for sheet_name, df in sheets.items():
    df = df.copy()

    # Overlap fraction
    df["Overlap_fraction"] = df["Overlap"].apply(
        lambda x: int(str(x).split("/")[0]) / int(str(x).split("/")[1])
    )

    # Significance
    df["neglog10_adjP"] = -np.log10(df["Adjusted P-value"])

    df["Sheet"] = sheet_name
    all_data.append(
        df[
            [
                "Sheet",
                "Gene_set",
                "Term",
                "Adjusted P-value",
                "Overlap_fraction",
                "neglog10_adjP"
            ]
        ]
    )
all_data = pd.concat(all_data, ignore_index=True)
priority_map = {
    "MSigDB_Hallmark_2020": 0,
    "KEGG_2021_Human": 1,
    "GO_Biological_Process_2025": 2,
}
all_data["GeneSet_priority"] = all_data["Gene_set"].apply(
    lambda x: next((v for k, v in priority_map.items() if k in str(x)), 3)
)
term_order = (
    all_data
    .groupby("Term", as_index=False)
    .agg({
        "GeneSet_priority": "min",      
        "neglog10_adjP": "max"          
    })
    .sort_values(
        ["GeneSet_priority", "neglog10_adjP"],
        ascending=[True, False]
    )["Term"]
    .tolist()
)
sheet_order = list(all_data["Sheet"].unique())
term_to_y = {
    term: i
    for i, term in enumerate(term_order)
}
sheet_to_x = {
    sheet: i
    for i, sheet in enumerate(sheet_order)
}
all_data["x"] = all_data["Sheet"].map(sheet_to_x)
all_data["y"] = all_data["Term"].map(term_to_y)
fig_height = max(
    8,
    len(term_order) * 0.35
)

fig, ax = plt.subplots(
    figsize=(4, fig_height))


sc = ax.scatter(
    all_data["x"],
    all_data["y"],
    s=300,                          # fixed bubble size
    c=all_data["neglog10_adjP"],    # colour = significance
    cmap="plasma",
    linewidths=0.3
)
ax.invert_yaxis()

ax.set_xticks(range(len(sheet_order)))
ax.set_xticklabels(
    sheet_order,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(term_order)))
ax.set_yticklabels(term_order)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Over-Representation Analysis Comparison: DCIS Luminal Progenitor",
    fontsize=20,
    pad=15
)

cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.5,
    pad=0.02
)

cbar.set_label(
    r"$-\log_{10}(\mathrm{FDR})$",
    fontsize=12
)


ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)

plt.tight_layout()
ax.set_xlim(-0.45, len(sheet_order) - 0.65)

plt.subplots_adjust(left=0.22, right=0.86)

plt.savefig(
    "./data/AAA_DCIS/final_ORA_results/plots/Selected_LPDD_Fresh_ORA_bubbleplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[118]:


excel_file = "./data/AAA_DCIS/final_ORA_results/LPDN/LPDN_selected_ORA_results.xlsx"
sheets = pd.read_excel(excel_file, sheet_name=None)
# Order pathways by strongest significance observed

all_data = []
for sheet_name, df in sheets.items():
    df = df.copy()

    # Overlap fraction
    df["Overlap_fraction"] = df["Overlap"].apply(
        lambda x: int(str(x).split("/")[0]) / int(str(x).split("/")[1])
    )

    # Significance
    df["neglog10_adjP"] = -np.log10(df["Adjusted P-value"])

    df["Sheet"] = sheet_name
    all_data.append(
        df[
            [
                "Sheet",
                "Gene_set",
                "Term",
                "Adjusted P-value",
                "Overlap_fraction",
                "neglog10_adjP"
            ]
        ]
    )
all_data = pd.concat(all_data, ignore_index=True)
priority_map = {
    "MSigDB_Hallmark_2020": 0,
    "KEGG_2021_Human": 1,
    "GO_Biological_Process_2025": 2,
}
all_data["GeneSet_priority"] = all_data["Gene_set"].apply(
    lambda x: next((v for k, v in priority_map.items() if k in str(x)), 3)
)
term_order = (
    all_data
    .groupby("Term", as_index=False)
    .agg({
        "GeneSet_priority": "min",      
        "neglog10_adjP": "max"          
    })
    .sort_values(
        ["GeneSet_priority", "neglog10_adjP"],
        ascending=[True, False]
    )["Term"]
    .tolist()
)
sheet_order = list(all_data["Sheet"].unique())
term_to_y = {
    term: i
    for i, term in enumerate(term_order)
}
sheet_to_x = {
    sheet: i
    for i, sheet in enumerate(sheet_order)
}
all_data["x"] = all_data["Sheet"].map(sheet_to_x)
all_data["y"] = all_data["Term"].map(term_to_y)
fig_height = max(
    8,
    len(term_order) * 0.35
)

fig, ax = plt.subplots(
    figsize=(4, fig_height))


sc = ax.scatter(
    all_data["x"],
    all_data["y"],
    s=300,                          # fixed bubble size
    c=all_data["neglog10_adjP"],    # colour = significance
    cmap="plasma",
    linewidths=0.3
)
ax.invert_yaxis()

ax.set_xticks(range(len(sheet_order)))
ax.set_xticklabels(
    sheet_order,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(term_order)))
ax.set_yticklabels(term_order)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Over-Representation Analysis Comparison: Non-Malignant Luminal Progenitor",
    fontsize=20,
    pad=15
)

cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.5,
    pad=0.02
)

cbar.set_label(
    r"$-\log_{10}(\mathrm{FDR})$",
    fontsize=12
)

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)

plt.tight_layout()
ax.set_xlim(-0.45, len(sheet_order) - 0.65)

plt.subplots_adjust(left=0.22, right=0.86)

plt.savefig(
    "./data/AAA_DCIS/final_ORA_results/plots/Selected_LPDN_Fresh_ORA_bubbleplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

