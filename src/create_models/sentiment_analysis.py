import pandas as pd
import numpy as np
from os import makedirs, path
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from src.preprocessing.reduce_data import *
from src.preprocessing.clean_data import *
from src.nlp.word2vec import W2V
from src.nlp.doc2vec import D2V
from src.nlp.utils import *
from src.neural_networks.cnn import *


def main():
    print("Creating Sentiment Analysis Models Using the Sentiment 140 Dataset")

    sent140_dir = "../../datasets/sentiment140"
    original_path = sent140_dir + "/sentiment140_original.csv"
    reduced_path = sent140_dir + "/sentiment140_reduced.csv"
    clean_path = sent140_dir + "/sentiment140_clean.csv"

    df = None

    print("\nReducing the dataset...")
    if not path.exists(reduced_path):
        cols = ["sentiment", "id", "date", "query_string", "user", "text"]
        print("Opening the original dataset:", original_path)
        df = pd.read_csv(original_path, encoding="utf-8", header=None,
                         names=cols)

        print("Dropping unnecessary features: id, date, query_string, user")
        df.drop(["id", "date", "query_string", "user"], axis=1, inplace=True)

        n = 200000
        print("Reducing to " + str(n) + " entries")
        ratios = find_class_ratios(df, "sentiment")
        df = reduce_dataset(df, "sentiment", ratios, n)

        print("Creating a new index column")
        df.reset_index(drop=True, inplace=True)
        df.index.name = "id"

        print("Saving reduced dataset:", reduced_path)
        df.to_csv(reduced_path)
    else:
        print("Reduced dataset is already created:", reduced_path)

    print("\nCleaning the dataset...")
    if not path.exists(clean_path):
        print("Opening the reduced dataset:", reduced_path)
        df = pd.read_csv(reduced_path, index_col="id")

        print("Cleaning the data")
        for i in df.index:
            df.at[i, "text"] = clean_tweet(df.at[i, "text"], rem_htags=False)

        # Replaces tweets with empty strings after being processed as null so
        # they can be removed
        # e.g. A tweet that only had a url
        print("Dropping null entries")
        df.text.replace("", np.nan, inplace=True)
        df.dropna(inplace=True)

        print("Creating a new index column")
        df.reset_index(drop=True, inplace=True)
        df.index.name = "id"

        # Change all the positive values to 1 as opposed to 4
        # Required for training the neural networks
        df.loc[df["sentiment"] == 4, "sentiment"] = 1

        # Vec models have trouble processing the text if this isn't explicitly
        # set as string
        df.text = df.text.astype(str)

        print("Saving clean dataset:", clean_path)
        df.to_csv(clean_path)
    else:
        print("Clean dataset is already created:", clean_path)
        df = pd.read_csv(clean_path, index_col="id")

    sa_model_dir = "../../models/sentiment_analysis"
    nn_dir, nlp_dir = sa_model_dir + "/neural_networks", sa_model_dir + "/nlp"
    w2v_dir, d2v_dir = nlp_dir + "/word2vec", nlp_dir + "/doc2vec"

    if not path.exists(nn_dir):
        print("Creating directory:", nn_dir)
        makedirs(nn_dir)

    if not path.exists(w2v_dir):
        print("Creating directory:", w2v_dir)
        makedirs(w2v_dir)

    if not path.exists(d2v_dir):
        print("Creating directory:", d2v_dir)
        makedirs(d2v_dir)

    w2v_cbow_path = w2v_dir + "/w2v_cbow.word2vec"
    w2v_sg_path = w2v_dir + "/w2v_sg.word2vec"
    d2v_dbow_path = d2v_dir + "/d2v_dbow.doc2vec"
    d2v_dm_path = d2v_dir + "/d2v_dm.doc2vec"

    w2v_cbow, w2v_sg, d2v_dbow, d2v_dm = None, None, None, None

    print("\nSplitting the data into training, testing and validation sets")
    x = df.text
    y = df.sentiment

    RS = 12345
    x_train, x_val_test, y_train, y_val_test = \
        train_test_split(x, y, test_size=0.2, random_state=RS)

    x_val, x_test, y_val, y_test = \
        train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=RS)

    print("\nCreating Word2Vec models from training data...")
    if not path.exists(w2v_cbow_path) and not path.exists(w2v_sg_path):
        print("Tokenizing words")
        word2vec = W2V(x_train)

        if not path.exists(w2v_cbow_path):
            print("Creating Word2Vec CBOW model:", w2v_cbow_path)
            w2v_cbow = word2vec.create_model(sg=0)
            w2v_cbow.save(w2v_cbow_path)
        else:
            print("Word2Vec CBOW model is already created:", w2v_cbow_path)
            w2v_cbow = KeyedVectors.load(w2v_cbow_path)

        if not path.exists(w2v_sg_path):
            print("Creating Word2Vec SG model:", w2v_sg_path)
            w2v_sg = word2vec.create_model(sg=1)
            w2v_sg.save(w2v_sg_path)
        else:
            print("Word2Vec SG model is already created:", w2v_sg_path)
            w2v_sg = KeyedVectors.load(w2v_sg_path)
    else:
        print("Word2Vec CBOW model is already created:", w2v_cbow_path)
        w2v_cbow = KeyedVectors.load(w2v_cbow_path)
        print("Word2Vec SG model is already created:", w2v_sg_path)
        w2v_sg = KeyedVectors.load(w2v_sg_path)

    print("\nCreating Doc2Vec models from training data...")
    if not path.exists(d2v_dbow_path) and not path.exists(d2v_dm_path):
        print("Tokenizing words and tagging documents")
        doc2vec = D2V(x_train)

        if not path.exists(d2v_dbow_path):
            print("Creating Doc2Vec DBOW model:", d2v_dbow_path)
            d2v_dbow = doc2vec.create_model(dm=0)
            d2v_dbow.save(d2v_dbow_path)
        else:
            print("Doc2Vec DBOW model is already created:", d2v_dbow_path)
            d2v_dm = KeyedVectors.load(d2v_dbow_path)

        if not path.exists(d2v_dm_path):
            print("Creating Doc2Vec DM model:", d2v_dm_path)
            d2v_dm = doc2vec.create_model(dm=1)
            d2v_dm.save(d2v_dm_path)
        else:
            print("Doc2Vec DM model is already created:", d2v_dm_path)
            d2v_dm = KeyedVectors.load(d2v_dm_path)
    else:
        print("Doc2Vec DBOW model is already created:", d2v_dbow_path)
        d2v_dbow = KeyedVectors.load(d2v_dbow_path)
        print("Doc2Vec DM model is already created:", d2v_dm_path)
        d2v_dm = KeyedVectors.load(d2v_dm_path)

    num_words = 65000
    # The maximum length of a tweet is 280 characters, therefore the maximum
    # number of words is 280/2 = 140
    # e.g. a a a .... a a
    max_len = 140
    vec_size = 128

    print("\nCreating a tokenizer (converting the words into a sequence of "
          "integers) with the training data")
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)

    print("Padding training and validation data")
    sequences = tokenizer.texts_to_sequences(x_train)
    x_train_seq = pad_sequences(sequences, maxlen=max_len)

    sequences_val = tokenizer.texts_to_sequences(x_val)
    x_val_seq = pad_sequences(sequences_val, maxlen=max_len)

    print("Creating an embedding matrix for the Word2Vec models")
    w2v_emb_index = create_embedding_index(w2v_cbow.wv, w2v_sg.wv)
    w2v_emb_matrix = create_embedding_matrix(num_words, vec_size,
                                             tokenizer.word_index,
                                             w2v_emb_index)

    print("Creating an embedding matrix for the Doc2Vec models")
    d2v_emb_index = create_embedding_index(d2v_dbow.wv, d2v_dm.wv)
    d2v_emb_matrix = create_embedding_matrix(num_words, vec_size,
                                             tokenizer.word_index,
                                             d2v_emb_index)

    cnn_dir = nn_dir + "/cnn"
    cnn_emb_path = cnn_dir + "/cnn_emb.h2"
    cnn_w2v_path = cnn_dir + "/cnn_w2v.h2"
    cnn_d2v_path = cnn_dir + "/cnn_d2v.h2"

    cnn_emb, cnn_w2v, cnn_d2v, = None, None, None

    print("\nCreating CNN Models...")
    if not path.exists(cnn_emb_path):
        print("Creating CNN with basic embedding layer")
        cnn_emb = cnn_01(x_train_seq, y_train, x_val_seq, y_val, num_words,
                         vec_size, max_len)
        cnn_emb.save(cnn_emb_path)

    if not path.exists(cnn_w2v_path):
        print("Creating CNN with Word2Vec model")
        cnn_w2v = cnn_01(x_train_seq, y_train, x_val_seq, y_val, num_words,
                         vec_size, max_len, {"weights": [w2v_emb_matrix],
                                             "trainable": True})
        cnn_w2v.save(cnn_w2v_path)

    if not path.exists(cnn_d2v_path):
        print("Creating CNN with Doc2Vec model")
        cnn_d2v = cnn_01(x_train_seq, y_train, x_val_seq, y_val, num_words,
                         vec_size, max_len, {"weights": [d2v_emb_matrix],
                                             "trainable": True})
        cnn_d2v.save(cnn_d2v_path)


if __name__ == "__main__":
    main()
