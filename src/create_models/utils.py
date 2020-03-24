import numpy as np
from gensim.models import KeyedVectors
from keras.models import load_model
from os import makedirs, path

"""Creates a directory if it doesn't exist"""


def create_dir_if_nonexist(dir_):
    if not path.exists(dir_):
        print("Creating directory:", dir_)
        makedirs(dir_)


"""Creates and returns an embedding matrix used for training the neural network
   models"""


def create_embedding_matrix(word_emb_map1, word_emb_map2, num_words, vec_size,
                            word_index):
    emb_index = {}
    emb_matrix = np.zeros((num_words, vec_size))

    # Concatenates 2 word embedding mappings
    # e.g. w2v_cbow + w2v_sg
    for w in word_emb_map1.vocab.keys():
        emb_index[w] = np.append(word_emb_map1[w], word_emb_map2[w])

    # Creates an embedding matrix
    for w, i in word_index.items():
        if i < num_words:
            emb_vec = emb_index.get(w)

            if emb_vec is not None:
                emb_matrix[i] = emb_vec

    return emb_matrix


"""Creates/loads and returns a vector model depending on it's existence"""


def create_vec_model(model_type, model_path, vec_cls, **model_key_args):
    if not path.exists(model_path):
        print("Creating", model_type, "model:", model_path)
        model = vec_cls.create_model(model_path, **model_key_args)
    else:
        print(model_type, "model is already created:", model_path)
        model = KeyedVectors.load(model_path)

    return model


"""Creates/loads and returns a neural networks model depending on it's 
   existence"""


def create_nn_model(model_type, model_path, model_func, model_pos_args,
                    model_emb_args={}):
    if not path.exists(model_path):
        print("Creating", model_type, "model:", model_path)
        model = model_func(model_path, *model_pos_args, model_emb_args)
    else:
        print(model_type, "model is already created:", model_path)
        model = load_model(model_path)

    return model


"""Creates/loads and returns the 3 models that we are analysing with a given 
   neural network:
     - Neural network with basic embedding
     - Neural network with Word2Vec
     - Neural network with Doc2Vec"""


def create_models_to_analyse(w2v_emb_matrix, d2v_emb_matrix, nn_type,
                             model_path, model_func, model_pos_args):
    model_emb_type = nn_type + " with basic embedding"
    model_w2v_type = nn_type + " with Word2Vec"
    model_d2v_type = nn_type + " with Doc2Vec"

    model_emb = create_nn_model(model_emb_type, model_path + "_emb.h5",
                                model_func, model_pos_args)

    # Word2Vec embedding matrix is added to this model
    model_w2v = create_nn_model(model_w2v_type, model_path + "_w2v.h5",
                                model_func, model_pos_args,
                                {"weights": [w2v_emb_matrix],
                                 "trainable": True})

    # Doc2Vec embedding matrix is added to this model
    model_d2v = create_nn_model(model_d2v_type, model_path + "_d2v.h5",
                                model_func, model_pos_args,
                                {"weights": [d2v_emb_matrix],
                                 "trainable": True})

    # Returns them as tuples (model_type, model)
    return (model_emb_type, model_emb), (model_w2v_type, model_w2v), \
           (model_d2v_type, model_d2v)
