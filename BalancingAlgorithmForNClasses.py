#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
cobabalancing = pd.read_csv('kucinganjingmonyet.csv', header= None)


# In[2]:


cobabalancing=cobabalancing.to_numpy()
if cobabalancing[1]=='kucing':
    print('yes')


# In[3]:


unique, counts = np.unique(cobabalancing, return_counts=True)
dict(zip(unique, counts))


# In[4]:


kucingcounter = 21
anjingcounter = 0
monyetcounter = 0
indices_anjing_remove = []
indices_monyet_remove = []

for i in range(cobabalancing.shape[0]):
    if cobabalancing[i] =='anjing':
        anjingcounter +=1
        if anjingcounter > kucingcounter:
            indices_anjing_remove.append(i)
    if cobabalancing[i] =="monyet" :
        monyetcounter +=1
        if monyetcounter > kucingcounter:
            indices_monyet_remove.append(i)
            

indices_anjing_counter = np.array(indices_anjing_remove)
print(indices_anjing_counter.shape[0])

indices_monyet_counter = np.array(indices_monyet_remove)
print(indices_monyet_counter.shape[0])
print(indices_monyet_remove)
#unscalled_but_balanced_priors = np.delete(unscalleddata, indices_to_remove, axis=0)
#target_balanced_priors = np.delete(alltargets, indices_to_remove, axis=0)


# In[5]:


indicestoremove = np.append(indices_anjing_remove,indices_monyet_remove)


# In[6]:


indicestoremove


# In[7]:


balanceddata = np.delete(cobabalancing, indicestoremove, axis = 0)


# In[8]:


balanceddata.shape[0]


# In[ ]:




