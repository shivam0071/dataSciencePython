import numpy as np
import pandas as pd

df = pd.read_csv("TextFiles/smsspamcollection.tsv", sep = "\t")

print(df.head())

# Check for nulls
print(df.isnull().sum())

print(df['label'].value_counts())


from sklearn.model_selection import train_test_split

X = df['message']
y = df['label']


# ctrl + q for hints
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# basic steps
# FIT the vectorizer to data (build the vocab and count the words)
# Transform the text message to vector
# inverse acc to document frequency  tfidftransform
# this all can be done in one step...using tfidfvectorizer


from sklearn.feature_extraction.text import TfidfVectorizer

X_train_tfidf = TfidfVectorizer(X_train)

from sklearn.svm import LinearSVC

clf = LinearSVC()

# clf.fit(X_train_tfidf, y_train)

# and all of this can also be done in  a single step as this is quite common code

from sklearn.pipeline import Pipeline

txt_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

txt_clf.fit(X_train, y_train)

prediction = txt_clf.predict(X_test)

from sklearn import metrics #import classification_report, confusion_matrix, accuracy_score

df = pd.DataFrame(metrics.confusion_matrix(y_test,prediction), index=['ham','spam'], columns=['ham','spam'])
print(df)

print(metrics.classification_report(y_test, prediction))
print(metrics.accuracy_score(y_test,prediction))