#!/usr/bin/env python
# coding: utf-8

# ## Homework 3
# ### Part 1: Imbalanced Dataset
# 
#  In this homework, you will be working with an imbalanced Dataset. The dataset is Credit Card Fraud Detection dataset which was hosted on Kaggle. The aim is to detect fraudlent transactions.

# ### Instructions
# 
# Please push the .ipynb, .py, and .pdf to Github Classroom prior to the deadline. Please include your UNI as well.

# ### Setup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Feel free to import any other packages you'd like to


# ### **Data Preprocessing and Exploration**
# Download the Kaggle Credit Card Fraud data set. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# In[3]:


raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head()


# #### **1.1 Examining the class Imbalance**
# **1.1.1 How many observations are in this dataset? How many are positive and negative?**
# (Note: Positive labels are labeled as 1)

# In[4]:


# Your Code here
raw_df.shape[0]


# In[5]:


raw_df['Class'].value_counts()


# **Ans: there are 284807 observations, while 492 are positive, and 284315 are negative**

# #### **1.2 Cleaning and normalizing the data**
# The raw data has a few issues. We are not sure what the time column actually means so drop the Time column. The Amount column also has a wide range of values covered so we take the log of the Amount column to reduce its range.

# In[6]:


cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps = 0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)


# In[7]:


cleaned_df


# In[8]:


cleaned_df["Class"].value_counts(normalize=True)


# **1.2.1 Split the dataset into development and test sets. Please set test size as 0.2 and random state as 42. Print the shape of your development and test features**

# In[9]:


# Your Code Here
X = cleaned_df.drop("Class", axis = 1)
y = cleaned_df["Class"]


# In[10]:


from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X, y==1, stratify = y, test_size = 0.2, random_state = 42 )
print(f"X_dev shape: {X_dev.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_dev shape: {y_dev.shape}")
print(f"y_test shape: { y_test.shape}")


# **1.2.2 Normalize the features using Standard Scaler from Sklearn.**

# In[11]:


# Your Code Here
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
X_test = scaler1.fit_transform(X_test)
X_dev = scaler1.fit_transform(X_dev)


# #### **1.3 Defining Model and Performance Metrics**

# **1.3.1 First, let us fit a default Decision tree classifier. ( use max_depth=10 and random_state=42). Print the AUC and Average Precision values of 5 Fold Cross Validation**

# In[12]:


# Your Code here
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
tree1 = DecisionTreeClassifier(max_depth = 10, random_state = 42)
score1 = cross_validate(tree1, X_dev, y_dev, cv = 5, scoring = ['roc_auc', 'average_precision'])


# In[13]:


print(score1["test_roc_auc"])
print(score1["test_average_precision"])


# In[14]:


print(f'mean AUC: {score1["test_roc_auc"].mean()}')
print(f'mean AP: {score1["test_average_precision"].mean()}')


# **1.3.2 Perform random oversampling on the development dataset.**
# 
# 
# *   What many positive and negative labels do you observe after random oversampling?
# *   What is the shape of your development dataset?
# 
# (Note: Set random state as 42 when performing oversampling)
# 

# In[15]:


get_ipython().system('pip install imblearn')


# In[16]:


# Your Code here
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state = 42)
X_dev_oversample1, y_dev_oversample1 = ros.fit_resample(X_dev, y_dev)
print(y_dev_oversample1.value_counts())
print(X_dev_oversample1.shape)


# **Ans: Both positive and negative labels have 227451 samples, and the shape is about (454902, 29)** 

# **1.3.3 Repeat 1.3.1 using the dataset you created in the above step(1.3.2 Random oversampling). (Make sure you use the same hyperparameters as 1.3.1. i.e., max_depth=10 and random_state=42. This will help us to compare the models)**

# In[17]:


tree2 = DecisionTreeClassifier(max_depth = 10, random_state = 42)


# In[18]:


score22 = cross_validate(tree2, X_dev_oversample1, y_dev_oversample1, cv = 5, scoring = ['roc_auc', 'average_precision'])


# In[19]:


print(score22["test_roc_auc"])
print(score22["test_average_precision"])


# In[20]:


print(f'mean AUC: {score22["test_roc_auc"].mean()}')
print(f'mean AP: {score22["test_average_precision"].mean()}')


# **1.3.4 Perform Random undersampling on the development dataset**. 
# - What many positive and negative labels do you observe after random undersampling?
# - What is the shape of your development dataset?
# (Note: Set random state as 42 when performing undersampling)

# In[21]:


# Your Code here
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(replacement = False, random_state = 42)
X_dev_subsample1, y_dev_subsample1 = rus.fit_resample(X_dev, y_dev)
print(y_dev_subsample1.value_counts())
print(X_dev_subsample1.shape)


# **Ans: Both positive and negative labels have 394 samples, and the shape is about (788, 29)** 

# **1.3.5 Repeat 1.3.1 using the dataset you created in the above step(1.3.4 Random undersampling). (Make sure you use the same hyperparameters as 1.3.1. i,e., max_depth=10 and random_state=42. This will help us to compare the models)**

# In[22]:


tree3 = DecisionTreeClassifier(max_depth = 10, random_state = 42)


# In[23]:


# tree3 = DecisionTreeClassifier(max_depth = 10, random_state = 42)

score33 = cross_validate(tree3, X_dev_subsample1, y_dev_subsample1, cv = 5, scoring = ['roc_auc', 'average_precision'])


# In[24]:


print(score33["test_roc_auc"])
print(score33["test_average_precision"])


# In[25]:


print(f'mean AUC: {score33["test_roc_auc"].mean()}')
print(f'mean AP: {score33["test_average_precision"].mean()}')


# **1.3.6 Perform Synthetic Minority Oversampling Technique(SMOTE) on the development dataset**
# - What many positive and negative labels do you observe after performing SMOTE?
# - What is the shape of your development dataset? (Note: Set random state as 42 when performing SMOTE)

# In[26]:


# Your code here
from imblearn.over_sampling import SMOTE
smote1 = SMOTE(random_state = 42)
X_dev_smote1, y_dev_smote1 = smote1.fit_resample(X_dev, y_dev)
print(y_dev_smote1.value_counts())
print(X_dev_smote1.shape)


# **Ans: Both positive and negative labels have 227451 samples, and the shape is about (454902, 29)** 

# **1.3.7 Repeat 1.3.1 using the dataset you created in the above step(1.3.6 SMOTE). (Make sure you use the same hyperparameters as 1.3.1. i.e., max_depth=10 and random_state=42. This will help us to compare the models)**

# In[27]:


tree4 = DecisionTreeClassifier(max_depth = 10, random_state = 42)


# In[28]:


score44 = cross_validate(tree4, X_dev_smote1, y_dev_smote1, cv = 5, scoring = ['roc_auc', 'average_precision'])


# In[29]:


print(score44["test_roc_auc"])
print(score44["test_average_precision"])


# In[30]:


print(f'mean AUC: {score44["test_roc_auc"].mean()}')
print(f'mean AP: {score44["test_average_precision"].mean()}')


# **1.3.8 Make predictions on the test set using the four models that you built and report their AUC values.**

# In[31]:


# Your Code here
from sklearn.metrics import roc_auc_score
tree1.fit(X_dev, y_dev)
score_normal = tree1.predict_proba(X_test)[:, 1]
print(f"The normal model has an AUC of {roc_auc_score(y_test, score_normal)}")

tree2.fit(X_dev_oversample1, y_dev_oversample1)
score_over = tree2.predict_proba(X_test)[:, 1]
print(f"The oversampling model has an AUC of {roc_auc_score(y_test, score_over)}")


tree3.fit(X_dev_subsample1, y_dev_subsample1)
score_under = tree3.predict_proba(X_test)[:, 1]
print(f"The undersampling model has an AUC of {roc_auc_score(y_test, score_under)}")


tree4.fit(X_dev_smote1, y_dev_smote1)
score_smote = tree4.predict_proba(X_test)[:, 1]
print(f"The smote model has an AUC of {roc_auc_score(y_test, score_smote)}")




# In[ ]:





# **1.3.9 Plot Confusion Matrices for all the four models on the test set. Comment your results**

# In[32]:


# Your Code here
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(tree1, X_test, y_test)


# In[34]:


plot_confusion_matrix(tree2, X_test, y_test)


# In[35]:


plot_confusion_matrix(tree3, X_test, y_test)


# In[36]:


plot_confusion_matrix(tree4, X_test, y_test)


# **Comment: the normal decision tree classifier has the highest accuracy, while the undersampling tree has the lowest accuracy. The undersampling tree has the best recall, while the normal decision tree has lowest recall. The normal decision tree has the highest precision, while the undersampling has the lowest precision.**

# **1.3.10 Plot ROC for all the four models on the test set in a single plot. Make sure you label axes and legend properly. Comment your results**

# In[37]:


# Your code
from sklearn.metrics import roc_curve
normal_fpr, normal_tpr, normal_thr = roc_curve(y_test, score_normal, pos_label = 1)
over_fpr, over_tpr, over_thr = roc_curve(y_test, score_over, pos_label = 1)
under_fpr, under_tpr, under_thr = roc_curve(y_test, score_under, pos_label = 1)
smote_fpr, smote_tpr, smote_thr = roc_curve(y_test, score_smote, pos_label = 1)

plt.figure(figsize = (10,5))
plt.plot(normal_fpr, normal_tpr, label = "Normal DT")
plt.plot(over_fpr, over_tpr, label = "Oversample")
plt.plot(under_fpr, under_tpr, label = "Undersample")
plt.plot(smote_fpr, smote_tpr, label = "Smote")
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# **ANS: The normal decision tree has the worst AUC, and undersampling has the best AUC, which is relatively higher than oversampling and smote. While smote and oversampling have similar AUC**

# **1.3.11 Train a balanced default Decision tree classifier, using max_depth = 10 and random_state = 42 (balance the class weights). Print the AUC and average precision on dev set**

# In[38]:


# Your code here
tree5 = DecisionTreeClassifier(max_depth = 10, random_state = 42, class_weight='balanced')
score_balanced_default = cross_validate(tree5, X_dev, y_dev, scoring = ['roc_auc', 'average_precision'])
print(score_balanced_default["test_roc_auc"])
print(score_balanced_default["test_average_precision"])


# In[39]:


print(f'Mean AUC: {score_balanced_default["test_roc_auc"].mean()}')
print(f'Mean AP: {score_balanced_default["test_average_precision"].mean()}')


# <!-- **1.3.12 Train a balanced Decision tree classifier. (You can use max_depth=10 and random_state=42)( balance the class weights). Print the AUC and average precision on test set** (Use Random state = 42) -->

# **1.3.12 Plot confusion matrix on test set using the above model and comment on your results**

# In[40]:


# Your code here
tree5.fit(X_dev, y_dev)
plot_confusion_matrix(tree5, X_test, y_test)


# **ANS:It has an accuracy similar to oversampling decision tree, which is only slightly less than the normal decision tree. Its recall is also similar to the oversampling decision tree, which is relatively less than other models. Its precision is also similar to the oversampling decision tree, which is the only slightly less than the normal DT**

# In[ ]:




