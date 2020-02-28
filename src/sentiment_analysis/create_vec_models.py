import pandas as pd
import multiprocessing
from os import path
from src.nlp.word2vec import W2V
from src.nlp.doc2vec import D2V

df = pd.read_csv("../../datasets/sentiment140/sentiment140_clean.csv", index_col="id")
df.text = df.text.astype(str)

directory = "../../models/nlp/"

default_opts = {"sg": 0,
        "size": 100,
        "negative": 5,
        "window": 2,
        "min_count": 2,
        "workers": multiprocessing.cpu_count(),
        "alpha": 0.065,
        "min_alpha": 0.065}

if not path.exists(directory + "w2v_model_cbow.word2vec"):
    word2vec = W2V(df.text, **default_opts)
    word2vec.create_model(epochs=30, learning_rate=0.002)
    model = word2vec.save_model(directory + "w2v_model_cbow.word2vec")

if not path.exists(directory + "w2v_model_sg.word2vec"):
    default_opts["sg"] = 1

    word2vec = W2V(df.text, **default_opts)
    word2vec.create_model(epochs=30, learning_rate=0.002)
    model = word2vec.save_model(directory + "w2v_model_sg.word2vec")

del default_opts["sg"]
del default_opts["size"]
del default_opts["window"]
default_opts["dm"] = 0
default_opts["vector_size"] = 100

if not path.exists(directory + "d2v_model_cbow.doc2vec"):
    doc2vec = D2V(df.text, **default_opts)
    doc2vec.create_model(epochs=30, learning_rate=0.002)
    model = doc2vec.save_model(directory + "d2v_model_dbow.doc2vec")

if not path.exists(directory + "d2v_model_dm.docvec"):
    default_opts["dm"] = 1

    doc2vec = D2V(df.text, **default_opts)
    doc2vec.create_model(epochs=30, learning_rate=0.002)
    model = doc2vec.save_model(directory + "d2v_model_dm.doc2vec")
