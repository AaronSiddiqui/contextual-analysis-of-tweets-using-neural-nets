from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from sann.neural_networks.train import train_nn

"""Returns a basic MLP with embedding"""


def mlp_01(model_path, x_train, y_train, x_val, y_val, input_dim, output_dim,
           input_len, num_output_classes, final_act_func, loss,
           embedding_opts):
    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=input_len,
                        **embedding_opts))
    model.add(Flatten())
    model.add(Dense(output_dim*2, activation="relu"))
    model.add(Dense(num_output_classes, activation=final_act_func))

    return train_nn(model, model_path, x_train, y_train, x_val, y_val, loss)
