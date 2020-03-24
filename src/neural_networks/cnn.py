from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Input, \
    Activation, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from src.neural_networks.train import train_nn

"""Returns an 1D CNN with embedding that I designed"""


def cnn_01(model_path, x_train, y_train, x_val, y_val, in_dim, out_dim, in_len,
           ker_size, emb_opts):
    model = Sequential()
    model.add(Embedding(in_dim, out_dim, input_length=in_len, **emb_opts))
    model.add(Conv1D(filters=out_dim, kernel_size=ker_size, padding="valid",
                     activation="relu", strides=1))
    model.add(Dense(256, activation="relu"))
    model.add(Conv1D(filters=out_dim, kernel_size=ker_size, padding="valid",
                     activation="relu", strides=1))
    model.add(Dense(256, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return train_nn(model, model_path, x_train, y_train, x_val, y_val)


"""Returns a 1D CNN with embedding based on this paper:
   https://arxiv.org/pdf/1510.03820.pdf"""


def cnn_02(model_path, x_train, y_train, x_val, y_val, in_dim, out_dim, in_len,
           emb_opts):
    tweet_input = Input(shape=(in_len,), dtype="int32")

    tweet_encoder = Embedding(in_dim, out_dim, input_length=in_len,
                              **emb_opts)(tweet_input)

    bigram_branch = Conv1D(filters=out_dim, kernel_size=2, padding="valid",
                           activation="relu", strides=1)(tweet_encoder)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)

    trigram_branch = Conv1D(filters=out_dim, kernel_size=3, padding="valid",
                            activation="relu", strides=1)(tweet_encoder)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)

    fourgram_branch = Conv1D(filters=out_dim, kernel_size=4, padding="valid",
                             activation="relu", strides=1)(tweet_encoder)
    fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)

    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch],
                         axis=1)

    merged = Dense(256, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1)(merged)
    output = Activation("sigmoid")(merged)
    model = Model(inputs=[tweet_input], outputs=[output])

    return train_nn(model, model_path, x_train, y_train, x_val, y_val)
