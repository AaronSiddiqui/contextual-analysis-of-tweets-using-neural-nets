import pickle
from os import path

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from constants import PROJ_DIR, EMOTION_ENCODER
from src.contextual_analysis.utils import create_dir_if_nonexist, \
    create_embedding_matrix, create_vec_model, create_models_to_analyse
from src.neural_networks.cnn import cnn_01, cnn_02
from src.neural_networks.mlp import mlp_01
from src.neural_networks.rnn import rnn_01
from src.nlp.word2vec import W2V
from src.preprocessing.clean_data import clean_tweet

if __name__ == "__main__":
    print("Creating the Emotion Detection Models Using the Emotion "
          "Classification Dataset")

    # Directories for the datasets and models
    emotion_classification_dir = PROJ_DIR + "datasets/isear"
    ed_model_dir = PROJ_DIR + "models/emotion_detection_2"

    original_path = emotion_classification_dir + "/isear.csv"
    clean_path = emotion_classification_dir + "/isear_clean.csv"

    df = None

    print("\nCleaning the dataset...")
    if not path.exists(clean_path):
        print("Opening the original dataset:", original_path)
        df = pd.read_csv(original_path, header=None, encoding="utf-8")

        df.columns = ["emotion", "text", "NaN"]
        df.drop(["NaN"], axis=1, inplace=True)

        # Emotions are currently represented as strings. These need to be
        # encoded to ints so that they can be trained in the neural network
        print("Encode the strings with ints for the feature emotions")
        for cls, num in EMOTION_ENCODER.items():
            df.loc[df["emotion"] == cls, "emotion"] = num

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

        # Word2Vec models have trouble processing the text if this isn't explicitly
        # set as string
        df.text = df.text.astype(str)

        print("Saving clean dataset:", clean_path)
        df.to_csv(clean_path)
    else:
        print("Clean dataset is already created:", clean_path)
        df = pd.read_csv(clean_path, index_col="id")

    print("\nSplitting the data into training, testing and validation sets")
    x = df.text
    y = df.emotion

    # Converts the ints to binary class matrices so that utilised with the
    # "categorical_crossentropy" loss function in the neural networks
    binary_y = to_categorical(y)

    # Random state constant to ensure the data is always split the same for
    # testing
    RS = 12345
    x_train, x_val_test, y_train, y_val_test = \
        train_test_split(x, binary_y, test_size=0.1, random_state=RS)

    x_val, x_test, y_val, y_test = \
        train_test_split(x_val_test, y_val_test, test_size=0.5,
                         random_state=RS)

    # Some nlp paths and directories to be created
    nlp_dir = ed_model_dir + "/nlp"
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

    print("\nCreating embedding matrices...")
    # When creating the embedding matrices I will be using all the words from
    # the Word2Vec models
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

    tokenizer_path = "models/tokenizers/"
    create_dir_if_nonexist(tokenizer_path)

    print("Saving the tokenizer: ")
    with open(tokenizer_path + "emotion_detection_2_tokenizer.pickle", "wb") as \
            handle:
        pickle.dump(tokenizer, handle)

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

    nn_dir = ed_model_dir + "/neural_networks"

    # List to store the models for evaluation
    # Consists of tuples of the form (model_type, model)
    models = []
    # Default neural network arguments
    nn_args = [x_train_seq, y_train, x_val_seq, y_val, num_words, vec_size,
               max_len, 7, "softmax", "categorical_crossentropy"]

    cnn_dir = nn_dir + "/cnn"
    create_dir_if_nonexist(cnn_dir)

    # From here onwards, the various neural networks are created with basic
    # embedding, Word2Vec CBOW and Word2Vec SG and added to the models list

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

    # Evaluates the accuracy on the test sets
    print("\nEvaluating models...")
    for m in models:
        loss, acc = m[1].evaluate(x_test_seq, y_test, verbose=0)
        print("Model:", m[0], "\tAccuracy:", acc)
