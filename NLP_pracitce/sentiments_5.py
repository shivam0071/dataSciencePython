import spacy
from scipy import spatial
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def word_arithmetics(one,two,three):
  nlp = spacy.load('en_core_web_lg')
  cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

  word_one = nlp.vocab[one].vector
  word_two = nlp.vocab[two].vector
  word_three = nlp.vocab[three].vector

  new_vec = word_one - word_two + word_three
  computed_similarities = []

  for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
      if word.is_lower:
        if word.is_alpha:
          similarity = cosine_similarity(new_vec, word.vector)
          computed_similarities.append((word, similarity))

  computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
  print([w[0].text for w in computed_similarities[:10]])

def whats_my_sentiment(pass_feelings_here):
  sid = SentimentIntensityAnalyzer()
  feels = sid.polarity_scores(pass_feelings_here)
  if feels.get('compound') > 0:
    print(f"Feels : '{pass_feelings_here}' is a POSITIVE feeling")
  else:
    print(f"Feels : '{pass_feelings_here}' is a NEGATIVE feeling")

if __name__ == '__main__':
  # word_arithmetics('king','man','woman')
  whats_my_sentiment("Is is too much to ask for?")