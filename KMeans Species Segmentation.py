#!/usr/bin/env python
# coding: utf-8

# # Species Segmentation with Cluster Analysis

# The Iris flower dataset is one of the most popular ones for machine learning. You can read a lot about it online and have probably already heard of it: https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# We didn't want to use it in the lectures, but believe that it would be very interesting for you to try it out (and maybe read about it on your own).
# 
# There are 4 features: sepal length, sepal width, petal length, and petal width.
# 
# ***
# 
# You have already solved the first exercise, so you can start from there (you've done taken advantage of the Elbow Method).
# 
# Plot the data with 2, 3 and 5 clusters. What do you think that means?
# 
# Finally, import the CSV with the correct answers (iris_with_answers.csv) and check if the clustering worked as expected. Note that this is not how we usually go about clustering problems. If we have the answers prior to that, we would go for classification (e.g. a logistic regression).

# ## Import the relevant libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# ## Load the data

# Load data from the csv file: <i> 'iris_dataset.csv'</i>.

# In[3]:


# Load the data
data = pd.read_csv('iris_dataset.csv')
# Check the data
data


# ## Plot the data

# For this exercise, try to cluster the iris flowers by the shape of their sepal. 
# 
# <i> Use the 'sepal_length' and 'sepal_width' variables.</i> 

# In[4]:


# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(data['sepal_length'],data['sepal_width'])
# name your axes
plt.xlabel('Lenght of sepal')
plt.ylabel('Width of sepal')
plt.show()


# ## Clustering (unscaled data)

# In[5]:


# create a variable which will contain the data for the clustering
x = data.copy()
# create a k-means object with 2 clusters
kmeans = KMeans(2)
# fit the data
kmeans.fit(x)


# In[6]:


# create a copy of data, so we can see the clusters next to the original data
clusters = data.copy()
# predict the cluster for each observation
clusters['cluster_pred']=kmeans.fit_predict(x)


# In[7]:


# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c= clusters ['cluster_pred'], cmap = 'rainbow')


# ## Standardize the variables

# Import and use the <i> scale </i> method from sklearn to standardize the data. 

# In[8]:


# import some preprocessing module
from sklearn import preprocessing

# scale the data for better results
x_scaled = preprocessing.scale(data)


# ## Clustering (scaled data)

# In[12]:


# create a k-means object with 2 clusters
kmeans_scaled = KMeans(2)
# fit the data
kmeans_scaled.fit(x_scaled)


# In[13]:


# create a copy of data, so we can see the clusters next to the original data
clusters_scaled = data.copy()
# predict the cluster for each observation
clusters_scaled['cluster_pred']=kmeans_scaled.fit_predict(x_scaled)


# In[14]:


# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c= clusters_scaled ['cluster_pred'], cmap = 'rainbow')


# Looks like the two solutions are identical. That is because the original features have very similar scales to start with!

# ## Take Advantage of the Elbow Method

# ### WCSS

# In[15]:


wcss = []
# 'cl_num' is a that keeps track the highest number of clusters we want to use the WCSS method for. 
# We have it set at 10 right now, but it is completely arbitrary.
cl_num = 10
for i in range (1,cl_num):
    kmeans= KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
wcss


# ### The Elbow Method

# In[16]:


number_clusters = range(1,cl_num)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')


# ## Understanding the Elbow Curve
# 
# Construct and compare the scatter plots to determine which number of clusters is appropriate for further use in our analysis. Based on the Elbow Curve, 2, 3 or 5 seem the most likely.

# ## 2 clusters
# 
# Start by separating the standardized data into 2 clusters (you've already done that!)

# In[18]:


kmeans2=KMeans(2)
kmeans2.fit(x_scaled)
cluster2=data.copy()
cluster2['cluster2']=kmeans2.fit_predict(x_scaled)


# Construct a scatter plot of the original data using the standardized clusters

# In[19]:


plt.scatter(cluster2['sepal_length'],cluster2['sepal_width'],c=cluster2['cluster2'],cmap='rainbow')
plt.xlabel='Sepal Length'
plt.ylabel='Sepal Width'
plt.show()


# ## 3 clusters
# Redo the same for 3 and 5 clusters

# In[20]:


kmeans3=KMeans(3)
kmeans3.fit(x_scaled)
cluster2['cluster3']=kmeans3.fit_predict(x_scaled)


# In[21]:


plt.scatter(cluster2['sepal_length'],cluster2['sepal_width'],c=cluster2['cluster3'],cmap='rainbow')
plt.xlabel='Sepal Length'
plt.ylabel='Sepal Width'
plt.show()


# In[ ]:





# ## 5 clusters

# In[22]:


kmeans5=KMeans(5)
kmeans5.fit(x_scaled)
cluster2['cluster5']=kmeans5.fit_predict(x_scaled)


# In[24]:


plt.scatter(cluster2['sepal_length'],cluster2['sepal_width'],c=cluster2['cluster5'],cmap='rainbow')
plt.xlabel='Sepal Length'
plt.ylabel='Sepal Width'
plt.show()


# In[ ]:





# ## Compare your solutions to the original iris dataset
# 
# The original (full) iris data is located in <i>iris_with_answers.csv</i>. Load the csv, plot the data and compare it with your solution. 
# 
# Obviously there are only 3 types, because that's the original (truthful) iris dataset.
# 
# The 2-cluster solution seemed good, but in real life the iris dataset has 3 SPECIES (a 3-cluster solution). Therefore, clustering cannot be trusted at all times. Sometimes it seems like x clusters are a good solution, but in real life, there are more (or less).

# In[27]:


answer=pd.read_csv('iris_with_answers.csv')
answer.head()


# In[29]:


plt.scatter(answer['sepal_length'],answer['species'])
plt.xlabel='Sepal Length'
plt.ylabel='Sepal Width'
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




