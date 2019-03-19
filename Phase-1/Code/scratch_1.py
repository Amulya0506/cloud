#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
data = pd.read_csv("heart.csv")
data.head()
t = data[['target']]
t.head()


# In[26]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,t, test_size = 0.25, random_state = 0)


# In[25]:


#feature scaling:

from sklearn.preprocessing import StandardScaler
independent_scaler = StandardScaler()
x_train  = independent_scaler.fit_transform(x_train)
x_test = independent_scaler.fit_transform(x_test)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNN.fit(x_train, y_train)
prediction = KNN.predict(x_test)
print(prediction)


# In[19]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, prediction)
print(conf_matrix)


# In[22]:


# ACCURACY

print(KNN.score(x_train, y_train))
print(KNN.score(x_test,y_test))


# In[ ]:




