import spacy
;
def read_file(filpath):
  with open(filpath) as f:
    str_text = f.read()

  return str_text

def separate_punc(nlp, doc_text):
  return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

def create_model(vocab_size, seq_len):

  # vocab_size is the no of unique words present in the input files or text
  # seq_len is the #25 , the len of each seq in the input...we have choosen 25 here

  model = Sequential()
  model.add(Embedding(vocab_size, seq_len, input_length = seq_len))
  model.add(LSTM(seq_len * 2,return_sequences= True )) # try to make it as a multiple of seq_len..try 150
  model.add(LSTM(seq_len * 2))  # try to make it as a multiple of seq_len..try 150
  model.add(Dense(50,activation= 'relu'))

  model.add(Dense(vocab_size, activation='softmax'))

  model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

  print(model.summary())

  return model

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_word):
  # seed_text to start off with something
  # num_gen_word : How many words to generate
  output_text = []

  input_text = seed_text # need to provide a line of 25 words then it will give us some output
  # then we take the new word and put it in the end, and remove the first word(creating anew seed
  # and we keep doing it acc to the num_gen_word

  for i in range(num_gen_word):
    encoded_text = tokenizer.texts_to_sequences([input_text])[0] #returns a list so...
    print("ENCODED TEXT -> ",encoded_text)

    pad_encoding = pad_sequences([encoded_text], maxlen = seq_len, truncating = 'pre') # from imports
    # if the seed is short or long...ad it to make 25 words only

    pred_word_ind = model.predict_classes(pad_encoding, verbose = 0)[0]
    # returns an index position
    print("PREDICTED WORD INDEX", pred_word_ind)

    pred_word = tokenizer.index_word[pred_word_ind]
    print("PREDICTED WORD", pred_word)

    input_text += ' '+pred_word

    output_text.append(pred_word)

  return ' '.join(output_text)

if __name__ == "__main__":
  print(read_file("moby_dick_four_chapters.txt"))
  d = read_file("moby_dick_four_chapters.txt")

  nlp = spacy.load("en_core_web_sm", disable = ['parser', 'tagger', 'ner']) # disabling some functionality
  # such that it becomes faster
  nlp.max_length =  1198623 # these are the no of works in whole book

  tokens = separate_punc(nlp, d) # pre process the data....
  print(tokens[:20])

  # we give 25 words and then network predicts 26th

  train_len = 25

  text_sequences = []

  for i in range(train_len,len(tokens)):
    seq = tokens[i - train_len:i]

    text_sequences.append(seq)
    # this creates a list with first 25 words, and the one word over
    # example -
    # ['a b c d e f', 'b c d e f g', 'c d e f g h'] where we have to predict the last word

  print(len(text_sequences),text_sequences[:2])
  # [['call', 'me', 'ishmael', 'some', 'years', 'ago', 'never', 'mind', 'how', 'long', 'precisely', 'having', 'little',
  #   'or'
  #    , 'no', 'money', 'in', 'my', 'purse', 'and', 'nothing', 'particular', 'to', 'interest', 'me'],
  #  ['me', 'ishmael', 'some', 'year
  #   s', 'ago', 'never', 'mind', 'how', 'long', 'precisely', 'having', 'little', ' or ', 'no', 'money', ' in ', 'my', '
  #   purse', ' and ', '
  #   nothing', 'particular', 'to', 'interest', 'me', 'on']]

  from keras.preprocessing.text import Tokenizer
  # to format this into a numercal systm that Keras can understand

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text_sequences)

  sequences = tokenizer.texts_to_sequences(text_sequences)

  print(sequences[:2])
  #same as above but as numbers and these numbers are the IDs for the words
  # [[956, 14, 263, 51, 261, 408, 87, 219, 129, 111, 954, 260, 50, 43, 38, 315, 7, 23, 546, 3, 150, 259, 6, 2712, 14], [14, 263, 5
  # 1, 261, 408, 87, 219, 129, 111, 954, 260, 50, 43, 38, 315, 7, 23, 546, 3, 150, 259, 6, 2712, 14, 24]]

  print(tokenizer.index_word)

  # to check the number of times a word occured
  print(tokenizer.word_counts)
  # 5), ('proudly', 22), ('marched', 21), ('pilot', 11), ('sporting', 7), ('marshal', 2)])


  vocab_size = len(tokenizer.word_counts)
  print(vocab_size) # used for one hot encoding

  # lets convert the seq into matrics..
  # We can do it by numpy
  import numpy as np
  sequences = np.array(sequences)
  print(sequences)
  # [[956   14  263...    6 2712   14]
  #  [14  263   51... 2712   14   24]
  # So given 956   14  263...    6 2712 these words...what word should come next...i.e 14

  # now we need a test train split

  from keras.utils import to_categorical
  X = sequences[:,:-1] # not that its an np array
  # this gives all the numbers except the last number in the row of each matrix
  # for all the numbers...give first to -1

  # similarly for only target
  y = sequences[:,-1]

  # again..for using categorical crossentropy loss function we change the
  # output to one hot encoding

  from keras.utils import to_categorical

  y = to_categorical(y, num_classes= vocab_size + 1)


  seq_len = X.shape[1] # 25

  print(X.shape)


  # lets create the model

  from keras.models import Sequential
  from keras.layers import Dense, LSTM, Embedding


  # LSTM - to deal with sequences
  # Embedding - to deal with the vocab - # 'Turns positive integers (indexes) into dense
  # vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]'
  # 'This layer can only be used as the first layer in a model.'

  # Make a func create_model
  model = create_model(vocab_size + 1, seq_len)

  # + 1 is needed for 0 place

  # trains
  from pickle import dump, load
  # to save the file and load it later

  model.fit(X,y,batch_size = 128, epochs=2 , verbose=1 )
  # batch_size is how many seq we want to pass at a time

  model.save('NLG.h5')

  # save tokenizer as well using pickle
  dump(tokenizer, open('my_tokenizer','wb'))

  # make a generate_text func

  from keras.preprocessing.sequence import pad_sequences


  import random
  random.seed(101)
  random_pick = random.randint(0, len(text_sequences))

  random_seed_text = text_sequences[random_pick]
  seed_text = " ".join(random_seed_text)

  # OR
  # seed_text = "I am going...somethig like the book"
  print("RANDOM SEED TEXT", seed_text)
  print(generate_text(model, tokenizer, seq_len, seed_text,num_gen_word=25))
