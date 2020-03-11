from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, Input, Activation, \
    concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model


def cnn_01(x_train_seq, x_val_seq, y_train, y_val, num_words, max_len,
           emb_opts=None):
    model = Sequential(
        Embedding(num_words, 128, input_length=max_len, **emb_opts),
        Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu',
               strides=1),
        Dense(256, activation='relu'),
        Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu',
               strides=1),
        Dense(256, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    )

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_val),
                     epochs=5, batch_size=64, verbose=2)


def cnn_02(x_train_seq, x_val_seq, y_train, y_val, num_words, max_len,
           emb_opts=None):
    tweet_input = Input(shape=(140,), dtype='int32')

    tweet_encoder = Embedding(num_words, 128, input_length=max_len, **emb_opts)\
        (tweet_input)
    bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid',
                           activation='relu', strides=1)(tweet_encoder)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)
    trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid',
                            activation='relu', strides=1)(tweet_encoder)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)
    fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid',
                             activation='relu', strides=1)(tweet_encoder)
    fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch],
                         axis=1)

    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1)(merged)
    output = Activation('sigmoid')(merged)
    model = Model(inputs=[tweet_input], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_val),
                     batch_size=64, epochs=5, verbose=2)
