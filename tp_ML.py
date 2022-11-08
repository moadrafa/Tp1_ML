#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[7]:



plt.style.use("ggplot")


# In[8]:


df = pd.read_csv('C:/Users/Administrator/Desktop/ML/basetable.csv')
df = df.drop('nonsense404', axis=1)


# In[ ]:



x = df.loc[:, df.columns != 'REVIEW']
y = df['REVIEW']
x_train ,x_test , y_train ,y_test = train_test_split(x, y, train_size= 0.70, random_state=1)


# In[5]:


param_grid_lr = {
'max_iter': [20, 50, 100, 200, 500, 1000],
'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
'class_weight': ['balanced']
}


# In[4]:



logModel_grid = GridSearchCV(estimator=LogisticRegression(random_state=1234),scoring ='accuracy', param_grid=param_grid_lr, verbose=1, cv=5, n_jobs=-1)
logModel_grid.fit(x_train, y_train)
y_pred = logModel_grid.predict(x_test)
y_prob = logModel_grid.predict_proba(x_test)



# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# In[ ]:


print(metrics.classification_report(y_test, y_pred))


# In[ ]:


coef_table = pd.DataFrame(list(x_train.columns)).copy()
coef_table.insert(len(coef_table.columns),"Coefs",logModel_grid.best_estimator_.coef_.transpose())
coef_table["abs_Coefs"] = coef_table["Coefs"].abs()
coef_table = coef_table.sort_values(by=['abs_Coefs'], ascending=False).head(10)


# In[ ]:


x_train.to_csv('C:/Users/Administrator/Desktop/ML/v1/x_train.csv', index=False)
x_test.to_csv('C:/Users/Administrator/Desktop/ML/v1/x_test.csv', index=False)
y_train.to_csv('C:/Users/Administrator/Desktop/ML/v1/y_train.csv', index=False)
y_test.to_csv('C:/Users/Administrator/Desktop/ML/v1/y_test.csv', index=False)

joblib.dump(logModel_grid, 'C:/Users/Administrator/Desktop/ML/v1/logModel_grid.pkl')

