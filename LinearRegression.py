#!/usr/bin/env python
# coding: utf-8

# # Practical example

# ## Importing the relevant libraries

# In[3]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# ## Loading the raw data

# In[4]:


raw_data = pd.read_csv('1.04. Real-life example.csv')
raw_data.head()


# ## Preprocessing

# ### Exploring the descriptive statistics of the variables

# In[5]:


raw_data.describe(include='all')
raw_data.head


# ### Determining the variables of interest

# In[6]:


data = raw_data.drop(['Model'],axis=1)
data.describe(include='all')


# ### Dealing with missing values

# In[7]:


data.isnull().sum()


# In[8]:


data_no_mv = data.dropna(axis=0)


# In[9]:


data_no_mv.describe(include='all')


# ### Exploring the PDFs

# In[10]:


sns.distplot(data_no_mv['Price'])


# ### Dealing with outliers

# In[11]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[12]:


sns.distplot(data_1['Price'])


# In[13]:


sns.distplot(data_no_mv['Mileage'])


# In[14]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[15]:


sns.distplot(data_2['Mileage'])


# In[16]:


sns.distplot(data_no_mv['EngineV'])


# In[17]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[18]:


sns.distplot(data_3['EngineV'])


# In[19]:


sns.distplot(data_no_mv['Year'])


# In[20]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[21]:


sns.distplot(data_4['Year'])


# In[22]:


data_cleaned = data_4.reset_index(drop=True)


# In[23]:


data_cleaned.describe(include='all')


# ## Checking the OLS assumptions

# In[24]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


plt.show()


# In[25]:


sns.distplot(data_cleaned['Price'])


# ### Relaxing the assumptions

# In[26]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned.describe(include='all')


# In[27]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()


# In[28]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# ### Multicollinearity

# In[29]:


data_cleaned.columns.values


# In[30]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns


# In[31]:


vif


# In[32]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# ## Create dummy variables

# In[32]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[33]:


data_with_dummies.head()


# ### Rearrange a bit

# In[34]:


data_with_dummies.columns.values


# In[35]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[36]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# ## Linear regression model

# ### Declare the inputs and the targets

# In[37]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# ### Scale the data

# In[38]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[39]:


inputs_scaled = scaler.transform(inputs)


# ### Train Test Split

# In[40]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# ### Create the regression

# In[41]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[42]:


y_hat = reg.predict(x_train)


# In[43]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[44]:


sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size=18)


# In[45]:


reg.score(x_train,y_train)


# ### Finding the weights and bias

# In[46]:


reg.intercept_


# In[47]:


reg.coef_


# In[48]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[49]:


data_cleaned['Brand'].unique()


# ## Testing

# In[50]:


y_hat_test = reg.predict(x_test)


# In[52]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[54]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[56]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[58]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[59]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[60]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[61]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[62]:


df_pf.describe()


# In[64]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])


# In[ ]:




