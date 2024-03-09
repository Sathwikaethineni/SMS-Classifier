#!/usr/bin/env python
# coding: utf-8

# In[18]:


#importing libraries
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[19]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[20]:


df.sample(10)


# In[21]:


df.shape


# In[22]:


df.info()


# In[23]:


df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)


# In[24]:


df.sample(5)


# In[25]:


df.columns = ['category','Message']


# In[26]:


df.head(10)


# In[27]:


df['category'].value_counts()


# In[28]:


df['category'].value_counts().plot(kind='bar')


# In[29]:


df['spam'] = df['category'].apply(lambda x: 'SPAM' if x == 'spam' else 'NOT SPAM')


# In[30]:


df.head(10)


# In[31]:


x = np.array(df["Message"]) 
y = np.array(df["spam"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)


# In[32]:


sample = input('Enter a message: ')
df = cv.transform([sample]).toarray()
print(clf.predict(df))


# In[33]:


sample = input('Enter a message: ')
df = cv.transform([sample]).toarray()
print(clf.predict(df))


# In[34]:


clf.score(X_test,y_test)


# In[ ]:




