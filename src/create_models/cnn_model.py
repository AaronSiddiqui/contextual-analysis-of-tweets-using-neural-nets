import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, Input, Activation, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

df = pd.read_csv("../../datasets/sentiment140/sentiment140_clean.csv", index_col="id")
df.text = df.text.astype(str)

model_cbow = KeyedVectors.load("../../models/sentiment_analysis/nlp/word2vec/w2v_cbow.word2vec")
model_sg = KeyedVectors.load("../../models/sentiment_analysis/nlp/word2vec/w2v_sg.word2vec")
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
num_words = 65000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)

# print()
# print(len(tokenizer.word_index.items()))
#
# print()
# print(len(embeddings_index))

sequences = tokenizer.texts_to_sequences(x_train)
x_train_seq = pad_sequences(sequences, maxlen=max_len)

for x in x_train[:5]:
    print(x)

print()
for s in sequences[:5]:
    print(s)

sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=max_len)

embedding_matrix = np.zeros((num_words, 128))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print()
print(len(embedding_matrix))

# model_ann = Sequential()
# model_ann.add(Embedding(num_words, 128, input_length=max_len))
# model_ann.add(Flatten())
# model_ann.add(Dense(256, activation='relu'))
# model_ann.add(Dense(1, activation='sigmoid'))
# model_ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_ann.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=64, verbose=2)
# model_ann.save("../../models/neural_networks/emb_ann.h5")
#
# model_ann = Sequential()
# model_ann.add(Embedding(num_words, 128, weights=[embedding_matrix], input_length=max_len, trainable=True))
# model_ann.add(Flatten())
# model_ann.add(Dense(256, activation='relu'))
# model_ann.add(Dense(1, activation='sigmoid'))
# model_ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_ann.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=64, verbose=2)
# model_ann.save("../../models/neural_networks/w2v_ann.h5")


# model_cnn = Sequential()
# model_cnn.add(Embedding(num_words, 128, input_length=max_len))
# model_cnn.add(Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu', strides=1))
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu', strides=1))
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(GlobalMaxPooling1D())
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(Dense(1, activation='sigmoid'))
# model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_cnn.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=64, verbose=2)
# model_cnn.save("../../models/neural_networks/w2v_cnn.h5")
#
# model_cnn = Sequential()
# model_cnn.add(Embedding(num_words, 128, weights=[embedding_matrix], input_length=max_len, trainable=True))
# model_cnn.add(Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu', strides=1))
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu', strides=1))
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(GlobalMaxPooling1D())
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(Dense(1, activation='sigmoid'))
# model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_cnn.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=64, verbose=2)
# model_cnn.save("../../models/neural_networks/w2v_cnn.h5")



# model_rnn = Sequential()
# model_rnn.add(Embedding(num_words, 128, input_length=max_len))
# model_rnn.add(Bidirectional(LSTM(128)))
# model_rnn.add(Dense(256, activation="relu"))
# model_rnn.add(Dense(1, activation="sigmoid"))
# model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_rnn.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=64, verbose=2)
# model_rnn.save("../../models/neural_networks/emb_rnn.h5")
#
# model_rnn = Sequential()
# model_rnn.add(Embedding(num_words, 128, weights=[embedding_matrix], input_length=max_len, trainable=True))
# model_rnn.add(LSTM(128))
# model_rnn.add(Dense(256, activation="relu"))
# model_rnn.add(Dense(1, activation="sigmoid"))
# model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_rnn.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=64, verbose=2)
# model_rnn.save("../../models/neural_networks/w2v_rnn.h5")



tweet_input = Input(shape=(140,), dtype='int32')

tweet_encoder = Embedding(num_words, 128, weights=[embedding_matrix], input_length=140, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input], outputs=[output])

model.summary()
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), batch_size=64, epochs=5, verbose=2)