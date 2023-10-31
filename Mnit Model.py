#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.figsize"]=(22,5)


# In[5]:


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


# In[9]:


data=load_breast_cancer()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df["target"]=data.target
df.head()


# In[10]:


#Coorelation Heatmap showing only those values that have a positive correlation


# In[19]:


corr=df.corr()
#filters out any negative correlation

corr=np.around(corr[corr>0.2],2)

#Gets rid of any other changes
mask=np.triu(np.ones_like(corr,dtype=bool))

f,ax=plt.subplots(figsize=(25,20))

cmap=sns.diverging_palette(220,20,as_cmap=True)

sns.heatmap(corr,mask=mask,cmap=cmap,center=0,annot=True,square=True,linewidth=5)

plt.show()


# In[25]:


input_cols = df.columns[:-1]
input_cols


# In[32]:


inputs_df=df[list(input_cols)].copy()
inputs_df


# In[37]:


scaler=MinMaxScaler()

scaler.fit(inputs_df[input_cols])
inputs_df[input_cols]=scaler.transform(inputs_df[input_cols])
inputs_df[input_cols].head()


# In[52]:


column_values=[]
for i in range(len(inputs_df.columns)):
    column_values.append(inputs_df.iloc[:,i].values)
    
covariance_matrix=np.cov(column_values)
eigen_values,eigen_vectors=np.linalg.eig(covariance_matrix)
   


# In[54]:


covariance_matrix[0]


# In[57]:


print(eigen_values/(np.sum(eigen_values))*100)


# In[ ]:




