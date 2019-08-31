# from sklearn.family import Model (family of model)
# Example -
# from sklearn.linear_model import LinearRegression

# we should split the data into training set and test set

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X,y test_size = 0.3)
# test_size is the percent of data that should be used for testing


# training
# model.fit(X_train, y_train)

# test/ predict

# prediction = model.prediction(X_test)


# Starts here

import numpy as np
import pandas as pd

# Pandas has many methods for reading different type of files
df = pd.read_csv('UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv',sep = "\t")

print(df.head(n=2))
print(df.isnull().sum()) # to check if any column  is missing any entry so we can fix it
print(len(df))

print(df['label'].value_counts())


# import matplotlib.pyplot as plt
#
# plt.xscale('log')
# bins = 1.15**(np.arange(0,50))
# plt.hist(df[df['label']=='ham']['length'],bins=bins,alpha=0.8)
# plt.hist(df[df['label']=='spam']['length'],bins=bins,alpha=0.8)
# plt.legend(('ham','spam'))
# plt.show()
#
# plt.xscale('log')
# bins = 1.5**(np.arange(0,15))
# plt.hist(df[df['label']=='ham']['punct'],bins=bins,alpha=0.8)
# plt.hist(df[df['label']=='spam']['punct'],bins=bins,alpha=0.8)
# plt.legend(('ham','spam'))
# plt.show()


from sklearn.model_selection import  train_test_split
# X feature data

X = df[['length','punct']]

# y is label data

y = df['label']


X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3,random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()