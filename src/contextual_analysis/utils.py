from os import makedirs, path

import numpy as np
from gensim.models import KeyedVectors
from keras.models import load_model

"""Creates a directory if it doesn't exist"""


def create_dir_if_nonexist(dir_):
    if not path.exists(dir_):
        print("Creating directory:", dir_)
        makedirs(dir_)


"""Creates and returns an embedding matrix used for training the neural network
   models"""


def create_embedding_matrix(word_emb_map, num_words, vec_size,
                            word_index):
    emb_index = {}
    emb_matrix = np.zeros((num_words, vec_size))

    # Creates an embedding index from the word embedding map
    for w in word_emb_map.vocab.keys():
        emb_index[w] = np.array(word_emb_map[w])

    # Creates an embedding matrix
    for w, i in word_index.items():
        if i < num_words:
            emb_vec = emb_index.get(w)

            if emb_vec is not None:
                emb_matrix[i] = emb_vec

    return emb_matrix


"""Creates/loads and returns a Word2Vec model depending on it's existence"""


def create_vec_model(model_type, model_path, vec_cls, **model_kwargs):
    if not path.exists(model_path):
        print("Creating", model_type, "model:", model_path)
        model = vec_cls.create_model(model_path, **model_kwargs)
    else:
        print(model_type, "model is already created:", model_path)
        model = KeyedVectors.load(model_path)

    return model


"""Creates/loads and returns a neural networks model depending on it's 
   existence"""


def create_nn_model(model_type, model_path, model_func, model_posargs,
                    model_emb_args={}):
    if not path.exists(model_path):
        print("Creating", model_type, "model:", model_path)
        model = model_func(model_path, *model_posargs, model_emb_args)
    else:
        print(model_type, "model is already created:", model_path)
        model = load_model(model_path)

    return model


"""Creates/loads and returns the 3 models that we are analysing with a given 
   neural network:
     - Neural network with basic embedding
     - Neural network with Word2Vec CBOW
     - Neural network with Word2Vec SG"""


def create_models_to_analyse(w2v_cbow_emb_matrix, w2v_sg_emb_matrix, nn_type,
                             model_path, model_func, model_posargs):
    model_emb_type = nn_type + " with basic embedding"
    model_emb = create_nn_model(model_emb_type, model_path + "_emb.h5",
                                model_func, model_posargs)

    # Word2Vec CBOW embedding matrix is added to this model
    model_w2v_cbow_type = nn_type + " with Word2Vec CBOW"
    model_w2v_cbow = create_nn_model(model_w2v_cbow_type,
                                     model_path + "_w2v_cbow.h5",
                                     model_func, model_posargs,
                                     {"weights": [w2v_cbow_emb_matrix],
                                      "trainable": True})

    # Word2Vec SG embedding matrix is added to this model
    model_w2v_sg_type = nn_type + " with Word2Vec SG"
    model_w2v_sg = create_nn_model(model_w2v_sg_type,
                                   model_path + "_w2v_sg.h5",
                                   model_func, model_posargs,
                                   {"weights": [w2v_sg_emb_matrix],
                                    "trainable": True})

    # Returns them as tuples (model_type, model)
    return (model_emb_type, model_emb), (model_w2v_cbow_type, model_w2v_cbow), \
           (model_w2v_sg_type, model_w2v_sg)
