from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.layers.embeddings import Embedding


def rnn_01(x_train_seq, x_val_seq, y_train, y_val, num_words, max_len,
           emb_opts=None):
    model = Sequential(
        Embedding(num_words, 128, input_length=max_len, **emb_opts),
        Bidirectional(LSTM(128)),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid")
    )

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_val),
                     epochs=5, batch_size=64, verbose=2)