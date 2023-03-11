import string
import pandas as pd
import numpy as np
import json
from statistics import mean, median, mode
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

with open("/Users/chuameiyun/Documents/2023 AI Projects/singlish-language-model/dataset/smsCorpus_en_2015.03.09_all.json") as f:
    msg = json.load(f)

text_messages = [i['text'] for i in msg['smsCorpus']['message']]

full_text = [m.get('$') for m in text_messages]

texts = []
for text in full_text:
    texts.append(text_to_word_sequence(str(text)))


texts = texts[:1000]
lengths = [len(text) for text in texts]
sequence_length = mode(lengths)

filtered_texts = [t for t in texts if len(t) >= sequence_length]

vocab = set()
for w in filtered_texts:
    vocab.update(w)
vocab_size = len(vocab) + 1 # for OOV words

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_texts)
sequences = tokenizer.texts_to_sequences(filtered_texts)

tokenizer.word_counts

vocabulary_size = len(tokenizer.word_counts)
vocabulary_size

import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 5, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model

