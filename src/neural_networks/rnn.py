from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential

from sann.neural_networks.train import train_nn

"""Returns a basic LSTM RNN with embedding"""


def rnn_01(model_path, x_train, y_train, x_val, y_val, input_dim, output_dim,
           input_len, num_output_classes, final_act_func, loss,
           embedding_opts):
    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=input_len,
                        **embedding_opts))
    # This was originally a bidirectional LSTM but it took too long to execute
    model.add(LSTM(output_dim))
    model.add(Dense(output_dim * 2, activation="relu"))
    model.add(Dense(num_output_classes, activation=final_act_func))

    return train_nn(model, model_path, x_train, y_train, x_val, y_val, loss)
