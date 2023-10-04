#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import sklearn
from sklearn.cross_validation import train_test_split


# In[5]:


titanic_data=pd.read_csv("C:/Users/Lenovo/train.csv")


# In[6]:


titanic_data.head(10)


# In[7]:


print(len(titanic_data.index))


# In[8]:


sns.countplot(x="Survived",data=titanic_data)


# In[9]:


sns.countplot(x="Survived",hue="Sex",data=titanic_data)


# In[10]:


sns.countplot(x="Survived",hue="Pclass",data=titanic_data)


# In[11]:


titanic_data["Age"].plot.hist()


# In[12]:


titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))


# In[13]:


titanic_data.info()


# In[14]:


sns.countplot(x="SibSp",data=titanic_data)


# Data cleaning

# In[15]:


titanic_data.isnull()


# In[16]:


titanic_data.isnull().sum()


# In[17]:


sns.heatmap(titanic_data.isnull(), yticklabels=False)


# In[18]:


sns.boxplot(x="Pclass", y="Age", data=titanic_data)


# In[19]:


titanic_data.head()


# In[20]:


titanic_data.drop("Cabin",axis=1,inplace=True)


# In[21]:


titanic_data.head()


# In[22]:


titanic_data.dropna(inplace=True)


# In[23]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False)


# In[24]:


titanic_data.isnull().sum()


# In[25]:


titanic_data.head()


# In[26]:


pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[27]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head(5)


# In[28]:


embark=pd.get_dummies(titanic_data['Embarked'])
embark.head()


# In[29]:


pcl=pd.get_dummies(titanic_data['Pclass'])
pcl.head()


# In[30]:


titanic_data=pd.concat([titanic_data,sex,embark,pcl],axis=1)


# In[31]:


titanic_data.head()


# In[32]:


titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[33]:


titanic_data.head()


# Train

# In[47]:


x=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]


# In[49]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


logmodel=LogisticRegression()


# In[54]:


logmodel.fit(x_train,y_train)


# In[55]:


prediction=logmodel.predict(x_test)


# In[57]:


from sklearn.metrics import classification_report


# In[59]:


classification_report(y_test,prediction)


# In[60]:


from sklearn.metrics import confusion_matrix


# In[61]:


confusion_matrix(y_test,prediction)


# In[62]:


from sklearn.metrics import accuracy_score


# In[63]:


accuracy_score(y_test,prediction)


# In[ ]:




