#!/usr/bin/env python
# coding: utf-8

# # **Applied Machine Learning Homework 5**
# **Due 12 Dec,2022 (Monday) 11:59PM EST**

# ### Natural Language Processing
# We will train a supervised model to predict if a movie has a positive or a negative review.

# ####  **Dataset loading & dev/test splits**

# **1.0) Load the movie reviews dataset from NLTK library**

# In[1]:


import nltk
nltk.download("movie_reviews")
import pandas as pd
from nltk.corpus import twitter_samples 
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop = stopwords.words('english')
import string
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


from nltk.corpus import movie_reviews


# In[3]:


len(movie_reviews.fileids())


# In[4]:


negative_fileids = movie_reviews.fileids('neg')
positive_fileids = movie_reviews.fileids('pos')

pos_document = [(' '.join(movie_reviews.words(file_id)),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id) if category == 'pos']
neg_document = [(' '.join(movie_reviews.words(file_id)),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id) if category == 'neg']

# List of postive and negative reviews
pos_list = [pos[0] for pos in pos_document]
neg_list = [neg[0] for neg in neg_document]


# In[5]:


len(pos_document)


# In[6]:


pos_document[0]


# In[7]:


pos_document[0][0]


# In[8]:


len(neg_document)


# In[9]:


pos_list


# In[10]:


neg_list


# **1.1) Make a data frame that has reviews and its label**

# In[38]:


# code here
movies = pd.DataFrame(pos_document+neg_document, columns = ["Review", "Label"])
movies


# **1.2 look at the class distribution of the movie reviews**

# In[39]:


# code here
movies["Label"].value_counts()


# **1.3) Create a development & test split (80/20 ratio):**

# In[40]:


# code here
x_dev, x_test, y_dev, y_test = train_test_split(movies["Review"], movies["Label"], 
                                                   test_size = 0.2, random_state = 42)


# #### **Data preprocessing**
# We will do some data preprocessing before we tokenize the data. We will remove `#` symbol, hyperlinks, stop words & punctuations from the data. You may use `re` package for this. 

# **1.4) Replace the `#` symbol with '' in every review**

# In[41]:


# code here
x_dev = x_dev.str.replace('#', '\"')
x_test = x_test.str.replace('#', '\"')


# In[42]:


x_dev[240]


# **1.5) Replace hyperlinks with '' in every review**

# In[43]:


# code here


# **1.6) Remove all stop words**

# In[44]:


# code here
stop


# In[46]:


for word in stop: 
    temp_string = ' ' + word + ' '
    x_dev = x_dev.str.replace(temp_string, ' ').replace(temp_string, ' ')
    x_test = x_test.str.replace(temp_string, ' ').replace(temp_string, ' ')


# **1.7) Remove all punctuations**

# In[49]:


# code here
punctuation_list = string.punctuation
punctuation_list


# In[50]:


for p in punctuation_list: 
    x_dev = x_dev.str.replace(p, ' ')
    x_test = x_test.str.replace(p, ' ')


# In[51]:


x_dev[240]


# **1.8) Apply stemming on the development & test datasets using Porter algorithm**

# In[58]:


#code here
def stemSentence(sentence):
    porter = PorterStemmer()
    token_words = word_tokenize(sentence)
    stem_sentence = [porter.stem(word) for word in token_words]
    return " ".join(stem_sentence)


# In[64]:


for index, sentence in x_dev.iteritems(): 
    x_dev[index] = stemSentence(sentence)

for index, sentence in x_test.iteritems():
    x_test[index] = stemSentence(sentence)


# In[65]:


x_dev


# In[66]:


x_test


# #### **Model training**

# **1.9) Create bag of words features for each review in the development dataset**

# In[87]:


#code here
vector = CountVectorizer(stop_words = 'english')
x_dev_transform = vector.fit_transform(x_dev)
x_dev_transform
feature_names = vector.get_feature_names()
feature_names
print(feature_names[:10])
print(feature_names[10000:10020])
print(feature_names[::3000])


# **1.10) Train a Logistic Regression model on the development dataset**

# In[88]:


#code here
lr = LogisticRegression().fit(x_dev_transform, y_dev)


# In[89]:


x_test_transform = vector.transform(x_test)


# **1.11) Create TF-IDF features for each review in the development dataset**

# In[82]:


#code here
vector2 = TfidfVectorizer()
x_dev_transform2 = vector2.fit_transform(x_dev)
x_test_transform2 = vector2.transform(x_test)
feature_names2 = vector2.get_feature_names()
feature_names2
print(feature_names2[:10])
print(feature_names2[10000:10020])
print(feature_names2[::3000])


# **1.12) Train the Logistic Regression model on the development dataset with TF-IDF features**

# In[84]:


#code here
lr2 = LogisticRegression().fit(x_dev_transform2, y_dev)


# **1.13) Compare the performance of the two models on the test dataset. Explain the difference in results obtained?**

# In[90]:


#code here
lr.score(x_test_transform, y_test)


# In[86]:


lr2.score(x_test_transform2, y_test)


# It is slightly better than the results obtained in the Bag of words, since it not only counts the world, but assign weight importance

# In[ ]:




