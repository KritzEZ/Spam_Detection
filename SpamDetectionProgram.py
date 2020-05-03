#!/usr/bin/env python
# coding: utf-8

# # Spam Dection using NLP

# By: Kritarth Shah
# 
# For this program, I have only used the first 4000 messages of the file since my system was unable to allocate an array with that large of a size. If you system can handle the space needed, feel free to remove the "[:4000]" line below.

# ## Step 1: Reading Data

# In[1]:


import pandas as pd

data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']][:4000]
data.columns = ['label', 'text']

data.head()


# ## Step 2: Cleaning Data
# 
# For the ease of understanding this project, the following methods are divided into their own functions and later called in one function

# ### Sub 1: Removing Punctuations

# In[2]:


import string

def punctuation_remove(text):
    no_punct = "".join([char for char in text if char not in string.punctuation])
    return(no_punct.lower())


# ### Sub 2: Tokenizing Words  

# In[3]:


import re

def tokenizer(text):
    tokens = re.split('\W+', text)
    return tokens


# ### Sub 3: Removing Stopwords

# In[4]:


import nltk

stopwords = nltk.corpus.stopwords.words('english')

def stopwords_remove(text):
    no_stopwords = [word for word in text if word not in stopwords]
    return no_stopwords


# ### Sub 4: Lemmatizing Words

# In[5]:


lemwords = nltk.WordNetLemmatizer()

def lemmatizer(text):
    lemmatized = [lemwords.lemmatize(word) for word in text]
    return lemmatized


# ### Cleaning Data (with previous four functions)

# In[6]:


def clean_data(text):
    return lemmatizer(stopwords_remove(tokenizer(punctuation_remove(text))))

data.head()


# ## Step 3: Adding Features

# In[7]:


#add data of length of every mail minus white spaces
data['body_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))

#function to get % of punctuations in the whole text minus white space
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

#add data of % of punctuations in text
data['punct%'] = data['text'].apply(lambda x: count_punct(x))


# ## Step 3: Vectorizing Data

# ### Sub 1: Spliting Data into test and train

# In[8]:


from sklearn.model_selection import train_test_split

#splitting the test and training data (using data of 5 (20%))
X_train, X_test, y_train, y_test = train_test_split(data[['text', 'body_len', 'punct%']], data['label'], test_size=0.2)


# ### Sub 2: Vectorizing test and training data

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_data)
tfidf_vect_fit = tfidf_vect.fit(X_train['text'])

tfidf_train = tfidf_vect_fit.transform(X_train['text'])
X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_train.toarray())], axis=1)

tfidf_test = tfidf_vect_fit.transform(X_test['text'])
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1)


# In[10]:


print("Training Data Vector")
X_train_vect.head()


# In[11]:


print("Testing Data Vector")
X_test_vect.head()


# In[12]:


print(tfidf_train.shape)
print(tfidf_test.shape)
print(tfidf_vect.get_feature_names())


# ## Step 4: Creating ML Classifier

# In[13]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix


# ###### Note
# The parameters use for the classifier below were determined to be the most efficient and effetive from a combination of variables that were tested. To find these tests, check out the classifier parameter file.

# In[14]:


rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

rf_model = rf.fit(X_train_vect, y_train) #fitting training data with label
y_pred = rf_model.predict(X_test_vect)


# In[15]:


precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

matrix = confusion_matrix(y_test, y_pred)

pHam_Ham = matrix[0][0]
pHam_Spam = matrix[1][0]
pSpam_Ham = matrix[0][1]
pSpam_Spam = matrix[1][1]

print("------------Results-----------")
print("Predicted to be Ham and is Ham ->", pHam_Ham, "  ---- GOOD")
print("Predicted to be Ham and is Spam ->", pHam_Spam, "  ---- False Negative")
print("Predicted to be Spam and is Ham ->", pSpam_Ham, "  ---- False Positive")
print("Predicted to be Spam and is Spam ->", pSpam_Spam, "  ---- GOOD")
print("------------Stats-------------")
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))


# Precision -> The number of mails that were predicted to be spam that are actually spam
# 
# Recall -> The amount of mails that were perdicted to be spam divided by the total amount of spam
# 
# Accuracy -> The number of correct mail labels that were given to all test cases

# ## End Notes
# 
# As seen from the results, from using the Random Forest Classifier, we are able to predict the correct ham/spam option with a 98% accuracy with a few  false positives. For this program, the Gradient Boosting Classifier was not used because if you look at the Classifier notes, Gradient does not have a 100% percision. In the case of spam/ham, having false positives if ok, meaning there will be some spam in your ham mail; but having important mails in spam is not good. So with the given situation, the Randon Forest Classifier was the better options. To further imporve the results, in the future we can add more features and try more variales for the classifier parameters.
