#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[6]:


#declare number of observation
observation = 1000

#declare the random inputs

xs=np.random.uniform(low=-10,high=10,size=(observation,1))
zs=np.random.uniform(low=-10,high=10,size=(observation,1))

inputs_stacked = np.column_stack((xs,zs))

#declaring random noise
noise=np.random.uniform(low=-1,high=1,size=(observation,1))

#declaring the targets

generatedtargets=2*xs-3*zs+5+noise

#declaring save file
np.savez('TF_intro',inputs=inputs_stacked,targets=generatedtargets)


# In[50]:


#Load the nbz data
training_data=np.load('TF_intro.npz')

#Declare inputs and targets size

input_size = 2
output_size = 1

#Create the model, loss function and optimization algorithm
model = tf.keras.Sequential([
                                tf.keras.layers.Dense(
                                    output_size,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
                                    bias_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)
                                )
                            ])

customoptimizer=tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer='sgd', loss='mean_squared_error')

#fitting the model with inputs
model.fit(training_data['inputs'],training_data['targets'],epochs=100,verbose=2)
#model.fit(inputs,outputs,num of iteration a.k.a epochs, verbose{0: silence, 1: Complete, 2: one line per epoch})


# In[18]:


weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]

print(weights,biases)


# In[21]:


yhat=model.predict_on_batch(training_data['inputs']).round(1)
import seaborn as sns
sns.set()


# In[25]:


sns.distplot(training_data['targets']-yhat)


# In[44]:


plt.scatter(np.squeeze(yhat),np.squeeze(training_data['targets']))
plt.xlabel('test Y',size=10)
plt.ylabel('real Y',size=10)
plt.show()


# In[ ]:




