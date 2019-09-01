# Word2Vec is a 2 layer Neural Net that processes text
# Input is a text and output is a set of vectors (feature vector for words in that corpus

# the purpose of word2vec is to group the vectors of similar words together in vectorspace
# it detects similarities mathematically
# groups features such as context of Individual word

 # It uses context to predict a target word ( using Continous bag of words CBOW)
 # or using a word to predict a target context, skip-gram


# We can use cosine similarity to measure how similar word vectors are to each other
# cosine similarity is just distance between 2 vectors

# https://skymind.ai/wiki/word2vec


import spacy
nlp = spacy.load('en_core_web_lg')
print(len(nlp.vocab.vectors))

print(nlp(u'lion').vector)


tokens = nlp(u'lion cat pet')

# Tells how the tokens are related to each other
# Iterate through token combinations:
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print(nlp(u'lion').similarity(nlp(u'dandelion')))


# checking if a word is Out of Vacabulary
tokens = nlp(u'dog cat nargle')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)


# We can even do vector arithemetic
from scipy import spatial

# cosine similarity function
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])
# ['king', 'queen', 'prince', 'kings', 'princess', 'royal', 'throne', 'queens', 'monarch', 'kingdom']
