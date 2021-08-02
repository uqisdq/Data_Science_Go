#!/usr/bin/env python
# coding: utf-8

# Basic NN

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


#Creating target

observations = 100
xs=np.random.uniform(low=-10,high=10,size=(observations,1))
zs=np.random.uniform(low=-10,high=10,size=(observations,1))

inputs=np.column_stack((xs,zs))

inputs[:,1].shape


# In[4]:


#targets = f(x,z)=2x - 3z + 5 + noise

noise = np.random.uniform(-1,1,(observations,1))
targets = 2*xs - 3*zs + 5 + noise
print(targets.shape)
plt.scatter(inputs[:,1],targets)
plt.show()


# In[5]:


targets = targets.reshape(observations,)
xs=xs.reshape(observations,)
zs=zs.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=500)
plt.show()
targets = targets.reshape(observations,1)
xs=xs.reshape(observations,1)
zs=zs.reshape(observations,1)


# In[6]:


#Initializing weights and biases for iterating
init_range =0.1

weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
biases = np.random.uniform(low=-init_range, high=init_range, size=1)

print (weights)
print (biases)
print(inputs.shape,weights.shape)


# In[7]:


#Learning rate

learning_rate=0.02


# In[8]:


#Iterating

iteration=100
scaleforloss=observations #Scale can be any number
for i in range (iteration):
    outputs=np.dot(inputs,weights)+biases
    delta=outputs-targets
    #Calculating the loss, you can choose between l1 or l2
    
   #l1loss=np.sum(delta)/scaleforloss 
    l2loss=np.sum(delta**2)/scaleforloss #Scaled l2loss
    
    #scale the delta
    delta_scaled=delta/scaleforloss
    #update weights and biases
    weights=weights - learning_rate*np.dot(inputs.T,delta_scaled)
    biases=biases - learning_rate*np.sum(delta_scaled)
print(weights,biases)
    
#targets = f(x,z)=2x - 3z + 5 + noise


# In[9]:


import seaborn as sns
sns.set()


# In[ ]:





# In[10]:


from sklearn.model_selection import train_test_split as tss


# In[12]:


from sklearn.cluster import KMeans


# In[16]:


kmeans = KMeans(2)

kmeans.fit(targets)


# In[17]:


targetss=kmeans.fit_predict(targets)


# In[19]:


plt.scatter(xs,targets,c=targetss,cmap='rainbow')


# In[ ]:




