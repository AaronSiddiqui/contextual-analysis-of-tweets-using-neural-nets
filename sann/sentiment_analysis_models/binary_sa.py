import numpy as np
import pandas as pd
import pickle
from constants import PROJ_DIR
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from os import path
from sann.sentiment_analysis_models.utils import create_dir_if_nonexist, \
    create_embedding_matrix, create_vec_model, create_models_to_analyse
from sann.nlp.word2vec import W2V
# from sann.nlp.doc2vec import D2V
from sann.neural_networks.mlp import mlp_01
from sann.neural_networks.cnn import cnn_01, cnn_02
from sann.neural_networks.rnn import rnn_01
from sann.preprocessing.reduce_data import find_feature_ratios, reduce_dataset
from sann.preprocessing.clean_data import clean_tweet
from sklearn.model_selection import train_test_split


def main():
    print("Creating the Binary Sentiment Analysis Model Using the Sentiment 140 "
          "Dataset")

    # Directories for the datasets and models
    sent140_dir = PROJ_DIR + "datasets/sentiment140"
    binary_sa_model_dir = PROJ_DIR + "models/binary_sentiment_analysis"

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

        # I felt 200,000 tweets was an appropriate amount for training without
        # expending my computational resources
        n = 200000
        print("Reducing to " + str(n) + " entries")
        ratios = find_feature_ratios(df, "sentiment")
        df = reduce_dataset(df, "sentiment", ratios, n)

        # Create a new id for each tweet
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

        # Change all the positive values to 1 as opposed to 4
        # Required for training the neural networks
        df.loc[df["sentiment"] == 4, "sentiment"] = 1

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
    y = df.sentiment

    # Random state constant to ensure the data is always split the same for
    # testing
    RS = 12345
    x_train, x_val_test, y_train, y_val_test = \
        train_test_split(x, y, test_size=0.2, random_state=RS)

    x_val, x_test, y_val, y_test = \
        train_test_split(x_val_test, y_val_test, test_size=0.5,
                         random_state=RS)

    # Some nlp paths and directories to be created
    nlp_dir = binary_sa_model_dir + "/nlp"
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
    with open(tokenizer_path + "binary_sa_tokenizer.pickle", "wb") as handle:
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

    # print("Creating an embedding matrix for the Doc2Vec DBOW model")
    # d2v_dbow_emb_matrix = create_embedding_matrix(d2v_dbow.wv, num_words,
    #                                               vec_size,
    #                                               tokenizer.word_index)
    #
    # print("Creating an embedding matrix for the Doc2Vec DM model")
    # d2v_dm_emb_matrix = create_embedding_matrix(d2v_dm.wv, num_words,
    #                                             vec_size,
    #                                             tokenizer.word_index)

    nn_dir = binary_sa_model_dir + "/neural_networks"

    # List to store the models for evaluation
    # Consists of tuples of the form (model_type, model)
    models = []
    # Default neural network arguments
    nn_args = [x_train_seq, y_train, x_val_seq, y_val, num_words, vec_size,
               max_len, 1, "sigmoid", "binary_crossentropy"]

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

    # from sklearn.metrics import roc_curve, auc
    # import matplotlib.pyplot as plt
    #
    # cnn_01_res = models[0][1].predict(x_test_seq)
    # cnn_02_res = models[3][1].predict(x_test_seq)
    # cnn_01_fpr, cnn_01_tpr, cnn_01_threshold = roc_curve(y_test, cnn_01_res)
    # cnn_01_roc_auc = auc(cnn_01_fpr, cnn_01_tpr)
    # cnn_02_fpr, cnn_02_tpr, cnn_02_threshold = roc_curve(y_test, cnn_02_res)
    # cnn_02_roc_auc = auc(cnn_02_fpr, cnn_02_tpr)
    #
    # plt.plot(cnn_01_fpr, cnn_01_tpr, label='CNN 01 with Basic Embedding (Area = %0.3f)' % cnn_01_roc_auc, linewidth=2)
    # plt.plot(cnn_02_fpr, cnn_02_tpr, label='CNN 02 with Basic Embedding (Area = %0.3f)' % cnn_02_roc_auc, linewidth=2)
    # plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve of CNN 01 vs CNN 02 for Binary Sentiment Analysis')
    # plt.legend()
    #
    # pic_dir = "C:/Users/aaron/Dropbox/Final Year Project/figures/results/"
    # plt.savefig(pic_dir + "roc_curve_cnn_01_vs_cnn_02_bsa.png")
    #
    # plt.show()


if __name__ == "__main__":
    main()
