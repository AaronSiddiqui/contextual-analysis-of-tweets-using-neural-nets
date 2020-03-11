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

        print("Creating a new index column")
        df.reset_index(drop=True, inplace=True)
        df.index.name = "id"

        n = 200000
        print("Reducing to " + str(n) + " entries")
        ratios = find_class_ratios(df, "sentiment")
        df = reduce_dataset(df, "sentiment", ratios, n)

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

        # Vec Models have trouble processing the text if this isn't explicitly
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

    print("\nCreating Word2Vec Models...")
    if not path.exists(w2v_cbow_path) and not path.exists(w2v_sg_path):
        print("Tokenizing words")
        word2vec = W2V(df.text)

        if not path.exists(w2v_cbow_path):
            print("Creating Word2Vec CBOW model:", w2v_cbow_path)
            w2v_cbow = word2vec.create_model(sg=0, epochs=30, learning_rate=0.002)
            w2v_cbow.save_model(w2v_cbow_path)
        else:
            print("Word2Vec CBOW model is already created:", w2v_cbow_path)
            w2v_cbow = KeyedVectors.load(w2v_cbow_path)

        if not path.exists(w2v_sg_path):
            print("Creating Word2Vec SG model:", w2v_sg_path)
            w2v_sg = word2vec.create_model(sg=1, epochs=30, learning_rate=0.002)
            w2v_sg.save_model(w2v_sg_path)
        else:
            print("Word2Vec SG model is already created:", w2v_sg_path)
            w2v_sg = KeyedVectors.load(w2v_sg_path)
    else:
        print("Word2Vec CBOW model is already created:", w2v_cbow_path)
        w2v_cbow = KeyedVectors.load(w2v_cbow_path)
        print("Word2Vec SG model is already created:", w2v_sg_path)
        w2v_sg = KeyedVectors.load(w2v_sg_path)

    print("\nCreating Doc2Vec Models...")
    if not path.exists(d2v_dbow_path) and not path.exists(d2v_dm_path):
        print("Tokenizing words and tagging documents")
        doc2vec = D2V(df.text)

        if not path.exists(d2v_dbow_path):
            print("Creating Doc2Vec DBOW model:", d2v_dbow_path)
            d2v_dbow = doc2vec.create_model(dm=0, epochs=30, learning_rate=0.002)
            d2v_dbow.save_model(d2v_dbow_path)
        else:
            print("Doc2Vec DBOW model is already created:", d2v_dbow_path)
            d2v_dm = KeyedVectors.load(d2v_dbow_path)

        if not path.exists(d2v_dm_path):
            print("Creating Doc2Vec DM model:", d2v_dm_path)
            d2v_dm = doc2vec.create_model(dm=1, epochs=30, learning_rate=0.002)
            d2v_dm.save_model(d2v_dm_path)
        else:
            print("Doc2Vec DM model is already created:", d2v_dm_path)
            d2v_dm = KeyedVectors.load(d2v_dm_path)
    else:
        print("Doc2Vec DBOW model is already created:", d2v_dbow_path)
        d2v_dbow = KeyedVectors.load(d2v_dbow_path)
        print("Doc2Vec DM model is already created:", d2v_dm_path)
        d2v_dm = KeyedVectors.load(d2v_dm_path)


    embeddings_index = {}
    count = 0
    for w in model_cbow.wv.vocab.keys():
        embeddings_index[w] = np.append(model_cbow.wv[w], model_sg.wv[w])

    x = df.text
    y = df.sentiment

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = \
        train_test_split(x, y, test_size=0.2, random_state=SEED)

    x_validation, x_test, y_validation, y_test = \
        train_test_split(x_validation_and_test, y_validation_and_test,
                         test_size=0.5, random_state=SEED)

    max_len = 140
    num_words = 50000

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)

    # print()
    # print(len(tokenizer.word_index.items()))
    #
    # print()
    # print(len(embeddings_index))

    sequences = tokenizer.texts_to_sequences(x_train)
    x_train_seq = pad_sequences(sequences, maxlen=max_len)

    for x in x_train[:5]:
        print(x)

    print()
    for s in sequences[:5]:
        print(s)

    sequences_val = tokenizer.texts_to_sequences(x_validation)
    x_val_seq = pad_sequences(sequences_val, maxlen=max_len)

    embedding_matrix = np.zeros((num_words, 128))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print()
    print(len(embedding_matrix))


if __name__ == "__main__":
    main()
