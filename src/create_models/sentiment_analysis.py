from os import path
import pandas as pd
import numpy as np
from src.preprocessing.reduce import *
from src.preprocessing.clean import *

print("Creating Sentiment Analysis Models Using the Sentiment 140 Dataset")

original_path = "../../datasets/sentiment140/sentiment140_original.csv"
reduced_path = "../../datasets/sentiment140/sentiment140_reduced.csv"
clean_path = "../../datasets/sentiment140/sentiment140_clean.csv"
df = None


print("\nReducing the dataset...")
if not path.exists(reduced_path):
    cols = ["sentiment", "id", "date", "query_string", "user", "text"]

    print("Opening the original dataset:", original_path)
    df = pd.read_csv(original_path, encoding="utf-8", header=None, names=cols)

    print("Dropping unnecessary features: id, date, query_string, user")
    df.drop(["id", "date", "query_string", "user"], axis=1, inplace=True)

    print("Creating a new index column")
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"

    n = 200000
    print("Reducing to " + str(n) + " entries")
    ratios = find_class_ratios(df, "sentiment")
    df = reduce_dataset(df, "sentiment", ratios, n)

    df.to_csv(reduced_path)
else:
    print("Reduced dataset is already created:", reduced_path)


print("\nCleaning the dataset...")
if not path.exists(clean_path):
    df = pd.read_csv(reduced_path, index_col="id")

    for i in df.index:
        df.at[i, "text"] = clean_tweet(df.at[i, "text"], rem_htags=False)

    # Replaces tweets with empty strings after being processed as null so they
    # can be removed
    # e.g. A tweet that only had a url
    print("Dropping null entries")
    df.text.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)

    print("Creating a new index column")
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"

    df.to_csv(clean_path)
else:
    print("Clean dataset is already created:", clean_path)


w2v_cbow_path = "../../models/nlp/model_cbow.word2vec"
w2v_sg_path = "../../models/nlp/model_sg.word2vec"
doc2vec_dbow_path = "../../models/nlp/model_dbow.doc2vec"
doc2vec_dm_path = "../../models/nlp/model_dm.doc2vec"
df = pd.read_csv(clean_path, index_col="id")


print("Creating Word2Vec Models")
