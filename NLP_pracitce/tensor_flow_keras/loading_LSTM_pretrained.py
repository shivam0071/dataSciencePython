import spacy


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
    pad_encoding = pad_sequences([encoded_text], maxlen = seq_len, truncating = 'pre') # from imports
    # if the seed is short or long...ad it to make 25 words only
    pred_word_ind = model.predict_classes(pad_encoding, verbose = 0)[0]
    # returns an index position
    pred_word = tokenizer.index_word[pred_word_ind]
    input_text += ' '+pred_word
    output_text.append(pred_word)

  return ' '.join(output_text)

if __name__ == "__main__":
  print(read_file("moby_dick_four_chapters.txt"))
  d = read_file("moby_dick_four_chapters.txt")

  nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])  # disabling some functionality
  # such that it becomes faster
  nlp.max_length = 1198623  # these are the no of works in whole book

  tokens = separate_punc(nlp, d)  # pre process the data....
  print(tokens[:20])

  # we give 25 words and then network predicts 26th

  train_len = 25

  text_sequences = []

  for i in range(train_len, len(tokens)):
    seq = tokens[i - train_len:i]
    text_sequences.append(seq)

  from pickle import load
  tokenizer = load(open('epochBIG', 'rb'))
  vocab_size = len(tokenizer.word_counts)
  print(vocab_size)  # used for one hot encodZing

  seq_len = 25

  # LOAD
  from keras.models import load_model
  model = load_model('epochBIG.h5')


  # make a generate_text func
  from keras.preprocessing.sequence import pad_sequences
  import random
  random.seed(101)
  random_pick = random.randint(0, len(text_sequences))

  random_seed_text = text_sequences[random_pick]
  seed_text = " ".join(random_seed_text)

  # OR
  seed_text = "But here is an artist.He desires to paint you the dreamiest, shadiest, quietest, most enchanting bit of romantic landscape in all"
  print("RANDOM SEED TEXT", seed_text)
  print("Prediction:- ", generate_text(model, tokenizer, seq_len, seed_text,num_gen_word=10))
