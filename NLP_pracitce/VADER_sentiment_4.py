# 7 Sept 2019
# VADER - Valence Aware Dictionary for sEntiment Reasoning
# It is sensitive to both polarity(pos/neg) and intensity(strenght) of emotion
import nltk
# download vader_lexicon

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

str1 = "This is a good movie"

print(sid.polarity_scores(str1))
# {'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.4404}
# compound score is the avg

str2 = "This is the Worst movie that i ever saw"
print(sid.polarity_scores(str2))
# {'neg': 0.369, 'neu': 0.631, 'pos': 0.0, 'compound': -0.6249}
# compound has -ve score

import pandas as pd
df = pd.read_csv("TextFiles/amazonreviews.tsv", sep = '\t')
print(df.head())

# drop the empty
df.dropna(inplace=True)

blanks = []
for i,lb, rv in df.itertuples():
  if type(rv) == 'str':
    if rv.isspace():
      blanks.append(i)


print(df.iloc[0]['review'])
print(sid.polarity_scores(df.iloc[0]['review']))
# {'neg': 0.088, 'neu': 0.669, 'pos': 0.243, 'compound': 0.9454}

# Add a new column to DF about the above score
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
print(df.head())

# we are only concerned with compound scores so

df['compound'] = df['scores'].apply(lambda scores: scores['compound'])
print(df.head())

# now if the compound score is greater than 0 than its pos else negative

df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')
print(df.head())



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(df['label'],df['comp_score']))
print(classification_report(df['label'],df['comp_score']))
print(confusion_matrix(df['label'],df['comp_score']))