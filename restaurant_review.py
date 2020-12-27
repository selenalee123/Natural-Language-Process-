#Natural Language Process

##The purpose of this project is to build a classifier to classify a new given review into positive or negative.
# Classification is a supervised Machine Learning problem. It specifies the class to which data elements belong to and is best
# used when the output has finite and discrete values.

#%%
#Import library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
#import the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting =3)
dataset.head(10)
#%%
#cleaning the texts
# With stemming, words are reduced to their word stems.
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

#print Corpus
#%%
print(corpus)

#%%
#create the bag of words model  a text (such as a sentence or a document) is represented as the bag
# multiset) of its words, disregarding grammar and even word order but keeping multiplicity.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#%%
#split data to training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#%%
#fitiing Naives Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#%%
#Predict the test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#%%

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#print accuracy score
#%%
print(accuracy_score(y_test, y_pred))
#%%
print(classification_report(y_test, y_pred))
#%%
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#print out the condfusion matrix
#%%
print(cm)