import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from os import path
from src.create_models.utils import create_dir_if_nonexist, \
    create_embedding_matrix, create_vec_model, create_models_to_analyse
from src.nlp.word2vec import W2V
from src.nlp.doc2vec import D2V
from src.neural_networks.mlp import mlp_01
from src.neural_networks.cnn import cnn_01, cnn_02
from src.neural_networks.rnn import rnn_01
from src.preprocessing.reduce_data import find_feature_ratios, reduce_dataset
from src.preprocessing.clean_data import clean_tweet
from sklearn.model_selection import train_test_split


def main():
    print("Creating Sentiment Analysis Models Using the Sentiment 140 Dataset")

    os.chdir("../..")
    sent140_dir = "datasets/sentiment140"
    sa_model_dir = "models/binary_sentiment_analysis"

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
        ratios = find_feature_ratios(df, "sentiment")
        df = reduce_dataset(df, "sentiment", ratios, n)

        # Change all the positive values to 1 as opposed to 4
        # Required for training the neural networks
        df.loc[df["sentiment"] == 4, "sentiment"] = 1

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

        # Do this again because it was lost when null entries were dropped
        print("Creating a new index column")
        df.reset_index(drop=True, inplace=True)
        df.index.name = "id"

        # Vec models have trouble processing the text if this isn't explicitly
        # set as string
        df.text = df.text.astype(str)

        print("Saving clean dataset:", clean_path)
        df.to_csv(clean_path)
    else:
        print("Clean dataset is already created:", clean_path)
        df = pd.read_csv(clean_path, index_col="id")

    print("\nSplitting the data into training, testing and validation sets")
    x = df.text
    y = df.sentiment

    RS = 12345
    x_train, x_val_test, y_train, y_val_test = \
        train_test_split(x, y, test_size=0.2, random_state=RS)

    x_val, x_test, y_val, y_test = \
        train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=RS)

    nlp_dir = sa_model_dir + "/nlp"
    w2v_dir = nlp_dir + "/word2vec"
    w2v_cbow_path = w2v_dir + "/w2v_cbow.word2vec"
    w2v_sg_path = w2v_dir + "/w2v_sg.word2vec"

    word2vec = None

    create_dir_if_nonexist(w2v_dir)

    print("\nCreating Word2Vec models from training data...")
    if not path.exists(w2v_cbow_path) or not path.exists(w2v_sg_path):
        print("Tokenizing words")
        word2vec = W2V(x_train)

    w2v_cbow = create_vec_model("Word2Vec CBOW", w2v_cbow_path, word2vec, sg=0)
    w2v_sg = create_vec_model("Word2Vec SG", w2v_sg_path, word2vec, sg=1)

    # d2v_dir = nlp_dir + "/doc2vec"
    # d2v_dbow_path = d2v_dir + "/d2v_dbow.doc2vec"
    # d2v_dm_path = d2v_dir + "/d2v_dm.doc2vec"
    #
    # doc2vec = None
    #
    # create_dir_if_nonexist(d2v_dir)
    #
    # print("\nCreating Doc2Vec models from training data...")
    # if not path.exists(d2v_dbow_path) or not path.exists(d2v_dm_path):
    #     print("Tokenizing words and tagging documents")
    #     doc2vec = D2V(x_train)
    #
    # d2v_dbow = create_vec_model("Doc2Vec DBOW", d2v_dbow_path, doc2vec, dm=0)
    # d2v_dm = create_vec_model("Doc2Vec DM", d2v_dm_path, doc2vec, dm=1)

    # word = "facebook"
    # print(w2v_cbow.wv.most_similar(word))
    # print(w2v_sg.wv.most_similar(word))
    # print(d2v_dbow.wv.most_similar(word))
    # print(d2v_dm.wv.most_similar(word))

    print("\nCreating embedding matrices...")
    num_words = len(w2v_cbow.wv.vocab)
    # The maximum length of a tweet is 280 characters, therefore the maximum
    # number of words is 280/2 = 140
    # e.g. a a a .... a a
    max_len = 140
    vec_size = 64

    print("Creating a tokenizer (converting the words into a sequence of "
          "integers) with the training data")
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)

    print("Padding training, testing and validation data")
    sequences_train = tokenizer.texts_to_sequences(x_train)
    x_train_seq = pad_sequences(sequences_train, maxlen=max_len)

    sequences_val = tokenizer.texts_to_sequences(x_val)
    x_val_seq = pad_sequences(sequences_val, maxlen=max_len)

    sequences_test = tokenizer.texts_to_sequences(x_test)
    x_test_seq = pad_sequences(sequences_test, maxlen=max_len)

    print("Creating an embedding matrix for the Word2Vec CBOW model")
    w2v_cbow_emb_matrix = create_embedding_matrix(w2v_cbow.wv, num_words,
                                                  vec_size,
                                                  tokenizer.word_index)

    print("Creating an embedding matrix for the Word2Vec SG model")
    w2v_sg_emb_matrix = create_embedding_matrix(w2v_sg.wv, num_words,
                                                vec_size,
                                                tokenizer.word_index)

    # print("Creating an embedding matrix for the Doc2Vec DBOW model")
    # d2v_dbow_emb_matrix = create_embedding_matrix(d2v_dbow.wv, num_words,
    #                                               vec_size,
    #                                               tokenizer.word_index)
    #
    # print("Creating an embedding matrix for the Doc2Vec DM model")
    # d2v_dm_emb_matrix = create_embedding_matrix(d2v_dm.wv, num_words,
    #                                             vec_size,
    #                                             tokenizer.word_index)

    nn_dir = sa_model_dir + "/neural_networks"

    models = []
    nn_args = [x_train_seq, y_train, x_val_seq, y_val, num_words, vec_size,
               max_len]

    cnn_dir = nn_dir + "/cnn"
    create_dir_if_nonexist(cnn_dir)

    print("\nCreating CNN 01 models...")
    cnn_01_path = cnn_dir + "/cnn_01"
    cnn_01_models = create_models_to_analyse(w2v_cbow_emb_matrix,
                                             w2v_sg_emb_matrix, "CNN 01",
                                             cnn_01_path, cnn_01,
                                             nn_args + [3])  # add kernel size
    models.extend(cnn_01_models)

    print("\nCreating CNN 02 models...")
    cnn_02_path = cnn_dir + "/cnn_02"
    cnn_02_models = create_models_to_analyse(w2v_cbow_emb_matrix,
                                             w2v_sg_emb_matrix, "CNN 02",
                                             cnn_02_path, cnn_02, nn_args)
    models.extend(cnn_02_models)

    mlp_dir = nn_dir + "/mlp"
    create_dir_if_nonexist(mlp_dir)

    print("\nCreating MLP 01 models...")
    mlp_01_path = mlp_dir + "/mlp_01"
    mlp_01_models = create_models_to_analyse(w2v_cbow_emb_matrix,
                                             w2v_sg_emb_matrix, "MLP 01",
                                             mlp_01_path, mlp_01, nn_args)
    models.extend(mlp_01_models)

    rnn_dir = nn_dir + "/rnn"
    create_dir_if_nonexist(rnn_dir)

    print("\nCreating RNN 01 models...")
    rnn_01_path = rnn_dir + "/rnn_01"
    rnn_01_models = create_models_to_analyse(w2v_cbow_emb_matrix,
                                             w2v_sg_emb_matrix, "RNN 01",
                                             rnn_01_path, rnn_01, nn_args)
    models.extend(rnn_01_models)

    print("\nEvaluating models...")
    for m in models:
        loss, acc = m[1].evaluate(x_test_seq, y_test, verbose=0)
        print("Model:", m[0], "\tAccuracy:", acc)


if __name__ == "__main__":
    main()
