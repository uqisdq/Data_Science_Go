#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# In[5]:


mnistdataset, mnistinfo = tfds.load(name='mnist',with_info=True,as_supervised=True)


# In[42]:


mnisttrain, mnisttest = mnistdataset['train'],mnistdataset['test']


# In[21]:


#Declaring the number of validation and test
num_validation_samples = 0.1*mnistinfo.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples,tf.int64)

num_test_samples = mnistinfo.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)


# In[22]:


#Scaling Function

def scale(image,label) :
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

#make sure the function return image and label (we can use any other scaling functions, ex : sklearn, etc)
#dataset.map(*function*) --> Transform the dataset using certain function
#.map tupple can also used to rewrite the label of a dataset

#scale the datas
scaled_train_and_validation = mnisttrain.map(scale)
scaled_test_data = mnisttest.map(scale)


# In[24]:


#Shuffling the data
#Because in SGD, the data will be batched, it's better to shuffle the data, so the rows will not be uniform, which is better for propagating/regressing/etc

BUFFER_SIZE = 10000
#in some occasions, we cant shuffle the data at once, because of the processing power of the CPU, that is why we shuffled it little by little

shuffled_train_and_validation = scaled_train_and_validation.shuffle(BUFFER_SIZE)
#.shuffle tuple shuffle an input by a BUFFER_SIZE at a time

#take the validation data from shuffled_train_and_validation using .take() tuple
validationjadi=shuffled_train_and_validation.take(num_validation_samples)

#take the train data from shuffled_train_and_validation using .skip() tuple
trainjadi=shuffled_train_and_validation.skip(num_validation_samples)

#Do batching
BATCH_SIZE = 100

#Batch using .batch() tuple

trainjadi= trainjadi.batch(BATCH_SIZE)
validationjadi = validationjadi.batch(num_validation_samples) 
#-->Validation cuma dibikin satu batch, formalitas aja, karena si modelnya butuh validation di batch juga
testjadi=scaled_test_data.batch(num_test_samples)

#split the validation into targets and inputs, and transform it into iterator.
validationinputs,validationtargets = next(iter(validationjadi))
#.iter() transform data into iterator
#.next() Load the next data from iterator


# In[43]:


#Outlining/Build the NN model

#We have 28 x 28 pixel inputs per object, so the input size (width) will be 28 x 28 = 784
#We want 2 Hidden layer, with each hidden layer containing 50 I/O
#, so there will be 4 layers = inputs, hiddenlayer1, hiddenlayer2, outputs
# 4 layers mean Depth = 4
#We have 10 Output categories (0,1,...,8,9) --> Output size will be 10
    #for the optimal size of the model, further experimentation needed, but according to information above,
    #we could say that width = 784 and Depth = 4, is the suboptimal size

inputsize = 784
outputsize = 10
hiddenlayersize = 100

#tf.keras.sequential used to determine the layers we want to use
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)), #layer 1
                            tf.keras.layers.Dense(hiddenlayersize, activation = 'relu'), #hidden layer 1
                            tf.keras.layers.Dense(hiddenlayersize, activation = 'relu'), # hidden layer 2
                            tf.keras.layers.Dense(outputsize, activation = 'softmax') # layer 4
                            ])

    


# In[44]:


#Choosing Optimizer and Loss

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')


# TRAIN THE MODEL!

# In[45]:


#set epochs number
NUMEPOCHS = 6

#set input, output & other arguments
model.fit(trainjadi,epochs=NUMEPOCHS,validation_data=(validationinputs,validationtargets),verbose=2)


# In[46]:


test_loss, test_accuracy = model.evaluate(testjadi)


# In[47]:


print('test loss : {0:.2f}. Test accuracy : {1:.2f}%'.format(test_loss, test_accuracy*100.))


# In[58]:





# In[ ]:





# In[ ]:




