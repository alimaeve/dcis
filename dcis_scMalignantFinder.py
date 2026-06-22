#!/usr/bin/env python
# coding: utf-8

# # scMalignantFinder: Initialise

# In[1]:


import sys
print(sys.executable)
#check what environment you're in


# In[1]:


import scMalignantFinder
from scMalignantFinder import classifier
import scanpy as sc


# In[4]:


print(scMalignantFinder.__version__)


# In[4]:


# Initialize model, run on all datasets together
model = classifier.scMalignantFinder(
    test_input="./260130_dcis_alldata_annotated.h5ad",                                        
    pretrain_dir=None,      
    train_h5ad_path="./combine_training.h5ad",  # Path to training dataset (.h5ad)
    feature_path="./combined_tumor_up_down_degs.txt",        # Path to feature list.
    model_method="LogisticRegression",           
    norm_type=True,                              
    n_thread=1)                                  

model.load()

result_adata = model.predict()

print(result_adata.obs["scMalignantFinder_prediction"].head())


# In[6]:


# save
result_adata.write("./260130_dcis_alldata_annotated.h5ad")


# # Run on individual studies

# ## G2017

# In[3]:


model = classifier.scMalignantFinder(
    test_input="./260203_dcis_G2017_only.h5ad",                                        
    pretrain_dir=None,      
    train_h5ad_path="./combine_training.h5ad",  # Path to training dataset (.h5ad)
    feature_path="./combined_tumor_up_down_degs.txt",        # Path to feature list.
    model_method="LogisticRegression",           
    norm_type=True,                              
    n_thread=1)                                  

model.load()

result_adata = model.predict()

print(result_adata.obs["scMalignantFinder_prediction"].head())


# In[4]:


# save
result_adata.write("./260203_dcis_G2017_only.h5ad")


# ## W2022

# In[5]:


model = classifier.scMalignantFinder(
    test_input="./260203_dcis_W2022_only.h5ad",                                        
    pretrain_dir=None,      
    train_h5ad_path="./combine_training.h5ad",  # Path to training dataset (.h5ad)
    feature_path="./combined_tumor_up_down_degs.txt",        # Path to feature list.
    model_method="LogisticRegression",           
    norm_type=True,                              
    n_thread=1)                                  

model.load()

result_adata = model.predict()

print(result_adata.obs["scMalignantFinder_prediction"].head())


# In[6]:


# save
result_adata.write("./260203_dcis_W2022_only.h5ad")


# ## T2022

# In[7]:


model = classifier.scMalignantFinder(
    test_input="./260203_dcis_T2022_only.h5ad",                                        
    pretrain_dir=None,      
    train_h5ad_path="./combine_training.h5ad",  # Path to training dataset (.h5ad)
    feature_path="./combined_tumor_up_down_degs.txt",        # Path to feature list.
    model_method="LogisticRegression",           
    norm_type=True,                              
    n_thread=1)                                  

model.load()

result_adata = model.predict()

print(result_adata.obs["scMalignantFinder_prediction"].head())


# In[8]:


# save
result_adata.write("./260203_dcis_T2022_only.h5ad")


# ## Q2025

# In[9]:


model = classifier.scMalignantFinder(
    test_input="./260203_dcis_Q2025_only.h5ad",                                        
    pretrain_dir=None,      
    train_h5ad_path="./combine_training.h5ad",  # Path to training dataset (.h5ad)
    feature_path="./combined_tumor_up_down_degs.txt",        # Path to feature list.
    model_method="LogisticRegression",           
    norm_type=True,                              
    n_thread=1)                                  

model.load()

result_adata = model.predict()

print(result_adata.obs["scMalignantFinder_prediction"].head())


# In[10]:


# save
result_adata.write("./260203_dcis_Q2025_only.h5ad")


# In[12]:


import scanpy as sc
combine_training = sc.read_h5ad("./combine_training.h5ad")


# In[13]:


combine_training


# In[16]:


combine_training.obs['Raw_annotation'].value_counts()


# ## N2025

# In[ ]:


model = classifier.scMalignantFinder(
    test_input="./260205_dcis_N2025_only.h5ad",                                        
    pretrain_dir=None,      
    train_h5ad_path="./combine_training.h5ad",  # Path to training dataset (.h5ad)
    feature_path="./combined_tumor_up_down_degs.txt",        # Path to feature list.
    model_method="LogisticRegression",           
    norm_type=True,                              
    n_thread=1)                                  

model.load()

result_adata = model.predict()

print(result_adata.obs["scMalignantFinder_prediction"].head())


# In[ ]:


result_adata.write("./260205_dcis_N2025_only.h5ad")

