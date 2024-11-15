#!/usr/bin/env python
# coding: utf-8

# In[7]:


# get_ipython().system('pwd')


# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

from icecream import ic


# In[4]:


# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("karthickveerakumar/salary-data-simple-linear-regression")

# print("Path to dataset files:", path)


# ## setp 2: load dataset

# In[23]:

ic("Loading Data")

# Get dataset
df_sal = pd.read_csv('./data/1/Salary_Data.csv')
df_sal.head()


# ## step 3: data analysis

# In[24]:


# Describe data
df_sal.describe()


# In[25]:


# Data distribution
plt.title('Salary Distribution Plot')
sns.distplot(df_sal['Salary'])
plt.show()


# In[26]:


# Relationship between Salary and Experience
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color = 'lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()


# ## Step 4: Split the dataset into dependent/independent variables
# - Experience (X) is the independent variable
# - Salary (y) is dependent on experience

# In[27]:


# Splitting variables
X = df_sal.iloc[:, :1]  # independent
y = df_sal.iloc[:, 1:]  # dependent


# ## Step 4: Split data into Train/Test sets
# Further, split your data into training (80%) and test (20%) sets using train_test_split

# In[28]:


# Splitting dataset into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Step 5: Train the regression model
# Pass the X_train and y_train data into the regressor model by regressor.fit to train the model with our training data.

# In[30]:


# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Step 6: Predict the result
# Here comes the interesting part, when we are all set and ready to predict any value of y (Salary) dependent on X (Experience) with the trained model using regressor.predict

# In[31]:


# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train


# ## Step 7: Plot the training and test results
# Its time to test our predicted results by plotting graphs
# - Plot training set data vs predictions
# - First we plot the result of training sets (X_train, y_train) with X_train and predicted value of y_train (regressor.predict(X_train))

# In[32]:


# Prediction on training set
plt.scatter(X_train, y_train, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()


# ## Plot test set data vs predictions
# Secondly, we plot the result of test sets (X_test, y_test) with X_train and predicted value of y_train (regressor.predict(X_train))

# In[33]:


# Prediction on test set
plt.scatter(X_test, y_test, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()


# If you remember from the beginning of this article, we discussed the linear equation y = mx + c, we can also get the c (y-intercept) and m (slope/coefficient) from the regressor model.

# In[34]:


# Regressor coefficients and intercept
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')


# In[ ]:




