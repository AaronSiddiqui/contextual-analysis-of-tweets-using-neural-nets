import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding

df = pd.read_csv("../../datasets/sentiment140/sentiment140_clean.csv", index_col="id")
df.text = df.text.astype(str)

model_cbow = KeyedVectors.load("../../models/nlp/w2v_model_cbow.word2vec")
model_sg = KeyedVectors.load("../../models/nlp/w2v_model_sg.word2vec")
# model_dbow = KeyedVectors.load("../../models/nlp/d2v_model_dbow.doc2vec")
# model_dm = KeyedVectors.load("../../models/nlp/d2v_model_dm.doc2vec")

# word = "facebook"
# print(model_cbow.wv.most_similar(word))
# print(model_sg.wv.most_similar(word))
# print(model_dbow.wv.most_similar(word))
# print(model_dm.wv.most_similar(word))

embeddings_index = {}
count = 0
for w in model_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_cbow.wv[w], model_sg.wv[w])

x = df.text
y = df.sentiment

SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = \
    train_test_split(x, y, test_size=0.2, random_state=SEED)

x_validation, x_test, y_validation, y_test = \
    train_test_split(x_validation_and_test, y_validation_and_test,
                     test_size=0.5, random_state=SEED)

max_len = 140
num_words = 20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)

sequences = tokenizer.texts_to_sequences(x_train)
x_train_seq = pad_sequences(sequences, maxlen=max_len)

for x in x_train[:5]:
    print(x)

print()
for s in sequences[:5]:
    print(s)

sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=max_len)

embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print()
print(len(embedding_matrix))

# model_ptw2v = Sequential()
# e = Embedding(num_words, 200, input_length=max_len)
# model_ptw2v.add(e)
# model_ptw2v.add(Flatten())
# model_ptw2v.add(Dense(256, activation='relu'))
# model_ptw2v.add(Dense(1, activation='sigmoid'))
# model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

model_ptw2v = Sequential()
e = Embedding(num_words, 200, weights=[embedding_matrix], input_length=max_len, trainable=True)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

