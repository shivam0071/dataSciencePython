import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics


if __name__ == "__main__":
  df = pd.read_csv("TextFiles/moviereviews2.tsv", sep = "\t")
  # print(df.head())
  print(f"Total numbers of Reviews {len(df)}")
  print(f'{df.isnull().sum()}')
  # Drop the Empty reviews

  df.dropna(inplace=True)
  print(f'{df.isnull().sum()}')

  # also check for empty strings
  blanks = []
  for i, lb, rv in df.itertuples():
    if len(rv.strip())< 1:
      blanks.append(i)

  df.drop(blanks, inplace=True)
  print(f"Reviews Left - {len(df)}")

  X = df["review"]
  y = df["label"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

  text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

  text_clf.fit(X_train,y_train)

  predictions = text_clf.predict(X_test)

  print(f"Accuray {metrics.accuracy_score(y_test,predictions)}")
  print(f"{metrics.classification_report(y_test,predictions)}")
  print(metrics.confusion_matrix(y_test,predictions))

  print("\n\n\n\n")

  for rev in ["The movie was really boring,Its sad to see so much money being wasted on such a movie","The movie was mind blowing","The movie was good","the movie is bad","Why would someone even bother making suh non sense movie","it is bullshit"]:
    print(f"Review - {rev} \nSentiment - {text_clf.predict([rev])[0]}")
