from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from src.neural_networks.train import train_nn

"""Returns a basic LSTM RNN with embedding"""


def rnn_01(model_path, x_train, y_train, x_val, y_val, in_dim, out_dim, in_len,
           emb_opts):
    model = Sequential()
    model.add(Embedding(in_dim, out_dim, input_length=in_len, **emb_opts))
    model.add(LSTM(out_dim))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return train_nn(model, model_path, x_train, y_train, x_val, y_val)
