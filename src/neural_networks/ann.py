from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding


def ann_01(x_train_seq, y_train, x_val_seq, y_val, in_dim, out_dim, in_len,
           emb_opts={}):
    model = Sequential()
    model.add(Embedding(in_dim, out_dim, input_length=in_len, **emb_opts))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_val),
                     epochs=5, batch_size=64, verbose=2)
