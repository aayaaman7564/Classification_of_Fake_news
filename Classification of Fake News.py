#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install nlp_utils')
get_ipython().system('pip install scikit-learn')


# In[4]:


import nlp_utils
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


df = pd.read_csv('FNC_dataset.csv')


# In[11]:


df.shape
df


# In[12]:


pd.set_option('display.max_colwidth', -1)


# In[13]:


df['title']


# In[14]:


df['text']


# In[17]:


df['label'].value_counts()


# In[19]:


df.isnull().sum()


# In[20]:


df= df.dropna()


# In[21]:


df.isnull().sum()


# In[22]:


df.reset_index(inplace=True)


# In[23]:


df


# In[24]:


import re
import string


# In[29]:


alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub('\n', ' ', x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ',x)


# In[30]:


df['text'] = df['text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


# In[31]:


df['text']


# In[35]:


get_ipython().system('pip install nltk')


# In[41]:


import nltk
nltk.download('stopwords')


# In[ ]:


# for removing stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[42]:


Y = df['label']


# In[43]:


Y.head()


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(df['text'], Y, test_size=0.30, random_state=40)


# In[ ]:


tfidf_vect = TfidVectorizer(stop_words = 'english', max_df=0.7)
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)


# In[ ]:


print(tfidf_test)


# In[ ]:


print(tfidf.vect.get_feature_names()[-10:])


# In[ ]:


count_vect = CountVectorizer(stop_words = 'english')
count_train = count_vect.fit_transform(X_train.values)
count_test = count_vect.transform(X_test.values)


# In[ ]:


#get the feature names of count vectorizer
print(count_vect.get_feature_names()[0:10])


# In[ ]:


# data cleaning ends here and data modeling starts here
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[ ]:


clf = MultinomialNB()
clf.fit(tfidf_train, Y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(Y_test, pred)
print("accuracy:  %0.3f" % score)
cm = metrics.confusion_matrix(Y_test, pred)


# In[ ]:


print('worng prediction out of total')
print(Y_test != pred).sum(),'/',((Y_test == pred).sum()+(Y_test !=pred).sum())
print('Percentage accuracy: ', 100*accuracy_score(Y_test, pred))


# In[ ]:


sns.heatmap(cm, cmap="plasma", annot=True)


# In[ ]:


clf = MultinomialNB()
clf.fit(count_train, Y_train)
pred1 = clf.predict(count_test)
score = metrics.accuracy_score(Y_test, pred)
print("accuracy:  %0.3f" % score)
cm2 = metrics.confusion_matrix(Y_test, pred)
print(cm2)


# In[ ]:


print('worng prediction out of total')
print(Y_test != pred1).sum(),'/',((Y_test == pred1).sum()+(Y_test !=pred1).sum())
print('Percentage accuracy: ', 100*accuracy_score(Y_test, pred1))


# In[ ]:


sns.heatmap(cm2, cmap="plasma", annot=True)

