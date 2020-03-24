from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Input, Activation,\
    concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from src.neural_networks.train import train_nn

"""Returns a basic 1D CNN with embedding"""


def cnn_01(model_path, x_train, y_train, x_val, y_val, input_dim, output_dim,
           input_len, num_output_classes, final_act_func, loss, kernel_size,
           embedding_opts):
    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=input_len,
                        **embedding_opts))
    model.add(Conv1D(filters=128, kernel_size=kernel_size, padding="valid",
                     activation="relu", strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(output_dim*2, activation="relu"))
    model.add(Dense(num_output_classes, activation=final_act_func))
    print(model.summary())

    return train_nn(model, model_path, x_train, y_train, x_val, y_val, loss)


"""Returns a 1D CNN with embedding based on this paper:
   https://arxiv.org/pdf/1510.03820.pdf"""


def cnn_02(model_path, x_train, y_train, x_val, y_val, input_dim, output_dim,
           input_len, num_output_classes, final_act_func, loss, embedding_opts):
    tweet_input = Input(shape=(input_dim,), dtype="int32")

    tweet_encoder = Embedding(input_dim, output_dim, input_length=input_len,
                              **embedding_opts)(tweet_input)

    bigram_branch = Conv1D(filters=128, kernel_size=2, padding="valid",
                           activation="relu", strides=1)(tweet_encoder)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)

    trigram_branch = Conv1D(filters=128, kernel_size=3, padding="valid",
                            activation="relu", strides=1)(tweet_encoder)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)

    fourgram_branch = Conv1D(filters=128, kernel_size=4, padding="valid",
                             activation="relu", strides=1)(tweet_encoder)
    fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)

    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch],
                         axis=1)

    merged = Dense(output_dim*2, activation="relu")(merged)
    merged = Dense(num_output_classes)(merged)
    output = Activation(final_act_func)(merged)
    model = Model(inputs=[tweet_input], outputs=[output])

    return train_nn(model, model_path, x_train, y_train, x_val, y_val, loss)
